#!/usr/bin/env python3
"""Replay every recorded LKA command from the Transit MK5 owner's drives through the
LKA_STEER safety check (opendbc/safety/modes/ford.h) and confirm the check would not
have rejected any of it.

This is Task 6's acceptance gate: the safety limits in ford.h were derived, not
driven, and this replay is the only evidence available without road-testing that the
check does not reject ordinary driving. It is not proof the limits are tight enough,
only that they are not absurdly tight.

Private customer data: TRANSIT_LOGS must point at the directory containing the
extracted lka-can-*.npz routes (never committed to this repo). The suite stays green
and skips on machines without that data.
"""
import os
import numpy as np
import pytest

from opendbc.car.structs import CarParams
from opendbc.car.ford.values import FordSafetyFlags
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerSafety

TRANSIT_LOGS = os.environ.get("TRANSIT_LOGS", "")

# The four recorded routes that carry commanded LKA (Lane_Assist_Data1) frames.
# lka-can-baseline-00000004.npz predates the firmware flash (2026-09-09); the other
# three are post-flash, from two different builds (2026-09-16). All four are replayed.
ROUTES = [
  "lka-can-baseline-00000004.npz",
  "lka-can-00000009--a6c9313e65.npz",
  "lka-can-0000000a--431bb7a1b1.npz",
  "lka-can-00000010.npz",
]

# struct sample_t's rolling window (MAX_SAMPLE_VALS in declarations.h): the number of
# RX updates needed before angle_meas/vehicle_speed fully reflect a newly-set value.
WARMUP_SAMPLES = 6

# extract_lka_can.py appends one row per real Lane_Assist_Data1 TX frame, in
# chronological order, from whichever rlog segments existed for the route -- it
# silently skips a segment that wasn't downloaded (`if not rlog.exists(): continue`),
# which can leave a multi-second-to-multi-minute gap in `t` between two consecutive
# rows despite carState's last-known lat_active/angle carrying over unchanged across
# it. Lane_Assist_Data1 is actually sent at a ~30ms cycle (33 Hz) whether or not it is
# steering, so any gap far larger than that is a capture discontinuity, not real
# continuous transmission. Treat it like a fresh ignition cycle (full safety-state
# reset) rather than asking the check to explain an instantaneous angle change across
# missing data -- that would be testing an artifact of the private log capture, not
# the vehicle.
GAP_RESET_S = 1.0


def _brake_sys_features_checksum(msg):
  # Signal: VehVActlBrk_No_Cs (matches ford_compute_checksum in ford.h). Required: the
  # BrakeSysFeatures RX check is not ignore_checksum, so safety_rx_hook drops the frame
  # (and never updates vehicle_speed) unless this validates.
  addr, dat, bus = msg
  ret = bytearray(dat)
  chksum = dat[0] + dat[1]          # Veh_V_ActlBrk
  chksum += (dat[2] >> 2) & 0xF     # VehVActlBrk_No_Cnt
  chksum += dat[2] >> 6             # VehVActlBrk_D_Qf
  chksum = 0xFF - (chksum & 0xFF)
  ret[3] = chksum
  return addr, ret, bus


@pytest.mark.skipif(not TRANSIT_LOGS, reason="set TRANSIT_LOGS to the outputs directory with the recorded lka-can-*.npz routes")
class TestFordTransitReplay:
  # Every other TestFord*/CarSafetyTest subclass's test_tx_hook_on_wrong_safety_mode
  # (opendbc/safety/tests/common.py) globs all test_*.py files in this directory and
  # reads every `Test*` class's TX_MSGS unconditionally, before any of its own
  # skip-logic runs. This class isn't a CarSafetyTest and has no TX_MSGS of its own;
  # None opts it out of that cross-file check instead of raising AttributeError in
  # every other safety mode's test suite.
  TX_MSGS = None

  def setup_method(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    self.cnt_speed = 0

  def _rx(self, msg):
    return self.safety.safety_rx_hook(msg)

  def _tx(self, msg):
    return self.safety.safety_tx_hook(msg)

  def _rx_pinion(self, angle_deg):
    # Signal: StePinComp_An_Est. Feeds the measured pinion angle into angle_meas, which
    # is the origin the relative LKA request is reconstructed against, so the lateral
    # acceleration bound sees the angle the vehicle actually had, not a default of zero.
    values = {"StePinComp_An_Est": float(angle_deg), "StePinCompAnEst_D_Qf": 3}
    self._rx(self.packer.make_can_msg_safety("SteeringPinion_Data", 0, values))

  def _rx_speed(self, speed_ms):
    # Signal: Veh_V_ActlBrk (kph). ford_rx_hook's UPDATE_VEHICLE_SPEED reads this message,
    # and the ISO lateral-accel bound is speed-dependent (fudged_speed =
    # vehicle_speed.min - 1, floored at 1 m/s). Checksum and counter are
    # both enforced by the RX check (see FORD_COMMON_RX_CHECKS), so both must be correct
    # or the frame is silently dropped and speed never updates.
    values = {
      "Veh_V_ActlBrk": float(speed_ms) * 3.6,
      "VehVActlBrk_D_Qf": 3,
      "VehVActlBrk_No_Cnt": self.cnt_speed % 16,
    }
    self.cnt_speed += 1
    msg = self.packer.make_can_msg_safety("BrakeSysFeatures", 0, values, fix_checksum=_brake_sys_features_checksum)
    self._rx(msg)

  def _lka_msg(self, action, relative_mrad, ramp):
    values = {
      "LkaActvStats_D2_Req": int(action),
      "LaRefAng_No_Req": float(relative_mrad),
      "LaRampType_B_Req": int(ramp),
    }
    return self.packer.make_can_msg_safety("Lane_Assist_Data1", 0, values)

  def _start_session(self, t_s, angle_deg, speed_ms):
    """(Re)initialize safety state at the start of a route, or after a capture gap.

    Matches interface.py for FORD_TRANSIT_MK5: FordFlags.LKA_STEER is always set (it
    steers through Lane_Assist_Data1), and LONG_CONTROL is always set too because the
    platform carries a radar (radarUnavailable is False, so `not radarUnavailable`
    forces the flag regardless of alpha_long). test_ford.py's TestFordTransitLkaSafety
    uses the same combination.

    Warms up angle_meas/vehicle_speed with the current row's own values -- the same
    state a real drive would already be in from ~33Hz heartbeat frames before the
    recorded window starts. libsafety's mock timer (set_timer) starts at 0 and only
    advances when we set it, so it is (re)anchored to this row's real recorded time
    too, keeping the RX timeout checks on real elapsed time.
    """
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, FordSafetyFlags.LKA_STEER | FordSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()
    self.safety.set_timer(int(t_s * 1e6))
    self.cnt_speed = 0
    for _ in range(WARMUP_SAMPLES):
      self._rx_pinion(angle_deg)
      self._rx_speed(speed_ms)
    self.safety.set_controls_allowed(1)

  def test_no_rejections_on_recorded_driving(self):
    total = 0
    rejected = 0
    per_route = {}

    for name in ROUTES:
      path = os.path.join(TRANSIT_LOGS, name)
      # allow_pickle is left at its default (False): both arrays are plain float64/str
      # data (see extract_lka_can.py), so no pickled objects are ever loaded here.
      z = np.load(path)
      data, cols = z["data"], list(z["cols"])
      c = {n: i for i, n in enumerate(cols)}

      route_total = 0
      route_rejected = 0
      last_t = None

      for row in data:
        t_s = float(row[c["t"]])
        angle_deg = float(row[c["angle_deg"]])
        speed_ms = float(row[c["v"]])

        if last_t is None or (t_s - last_t) > GAP_RESET_S:
          self._start_session(t_s, angle_deg, speed_ms)
        else:
          # Real elapsed time, feeding this frame's actual measurements before the TX
          # that depends on them. The mock clock is what the RX timeout/frequency
          # checks read, so it must track real recorded time.
          self.safety.set_timer(int(t_s * 1e6))
          self._rx_pinion(angle_deg)
          self._rx_speed(speed_ms)
          # These are frames from engaged, controls-allowed driving.
          self.safety.set_controls_allowed(1)
        last_t = t_s

        # This is the real message the PSCM actually received at this point in time,
        # active or not -- send it as recorded, heartbeats included, exactly as it would
        # go out onboard. Only frames with lateral control active and an intervention
        # (direction/LkaActvStats_D2_Req != 0, i.e. not idle) are real steering
        # commands, and only those are counted below.
        msg = self._lka_msg(row[c["direction"]], row[c["ref_mrad"]], row[c["ramp"]])
        tx_ok = self._tx(msg)

        if row[c["lat_active"]] == 1 and row[c["direction"]] != 0:
          total += 1
          route_total += 1
          if not tx_ok:
            rejected += 1
            route_rejected += 1

      per_route[name] = (route_total, route_rejected)
      print(f"{name}: {route_total} replayed, {route_rejected} rejected")

    print(f"TOTAL: {total} replayed, {rejected} rejected")

    # A data-loading bug that silently produces zero frames (wrong path, wrong filter,
    # empty array) must not pass as success.
    assert total > 40000, f"only replayed {total} frames across {len(ROUTES)} routes: {per_route}"
    assert rejected == 0, f"{rejected} of {total} recorded frames were rejected: {per_route}"

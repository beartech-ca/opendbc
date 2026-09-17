import copy
import itertools
import math
import random
import unittest

import pytest
from hypothesis import settings, given, strategies as st

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import CarParams
from opendbc.car.fw_versions import build_fw_dict
from opendbc.car.ford import fordcan
from opendbc.car.ford.carcontroller import CarController, TransitLkaState
from opendbc.car.ford.carstate import CarState
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import (CAR, DBC, FW_QUERY_CONFIG, FW_PATTERN, get_platform_codes, FordFlags,
                                     FordSafetyFlags, TRANSIT_LKA_AVAIL_VALUES, TRANSIT_LKA_CONT_ENTER_SPEED,
                                     TRANSIT_LKA_CONT_EXIT_SPEED_HIGH, TRANSIT_LKA_CONT_EXIT_SPEED_LOW,
                                     TRANSIT_LKA_FLAGS_MASK, TransitLkaAvailGate, TransitLkaContinuation,
                                     TransitLkaDirectionSign, TransitLkaIntervention, TransitLkaRamp,
                                     pack_transit_lka_flags, unpack_transit_lka_flags)
from opendbc.car.ford.fingerprints import FW_VERSIONS
from opendbc.testing import parameterized

Ecu = CarParams.Ecu
TransmissionType = CarParams.TransmissionType


ECU_ADDRESSES = {
  Ecu.eps: 0x730,          # Power Steering Control Module (PSCM)
  Ecu.abs: 0x760,          # Anti-Lock Brake System (ABS)
  Ecu.fwdRadar: 0x764,     # Cruise Control Module (CCM)
  Ecu.fwdCamera: 0x706,    # Image Processing Module A (IPMA)
  Ecu.engine: 0x7E0,       # Powertrain Control Module (PCM)
  Ecu.shiftByWire: 0x732,  # Gear Shift Module (GSM)
  Ecu.debug: 0x7D0,        # Accessory Protocol Interface Module (APIM)
}


ECU_PART_NUMBER = {
  Ecu.eps: [
    b"14D003",
  ],
  Ecu.abs: [
    b"2D053",
  ],
  Ecu.fwdRadar: [
    b"14D049",
  ],
  Ecu.fwdCamera: [
    b"14F397",  # Ford Q3
    b"14H102",  # Ford Q4
  ],
}


class TestFordFW(unittest.TestCase):
  def test_fw_query_config(self):
    for (ecu, addr, subaddr) in FW_QUERY_CONFIG.extra_ecus:
      assert ecu in ECU_ADDRESSES, "Unknown ECU"
      assert addr == ECU_ADDRESSES[ecu], "ECU address mismatch"
      assert subaddr is None, "Unexpected ECU subaddress"

  @parameterized("car_model, fw_versions", FW_VERSIONS.items())
  def test_fw_versions(self, car_model, fw_versions):
    for (ecu, addr, subaddr), fws in fw_versions.items():
      assert ecu in ECU_PART_NUMBER, "Unexpected ECU"
      assert addr == ECU_ADDRESSES[ecu], "ECU address mismatch"
      assert subaddr is None, "Unexpected ECU subaddress"

      for fw in fws:
        assert len(fw) == 24, "Expected ECU response to be 24 bytes"

        match = FW_PATTERN.match(fw)
        assert match is not None, f"Unable to parse FW: {fw!r}"
        if match:
          part_number = match.group("part_number")
          assert part_number in ECU_PART_NUMBER[ecu], f"Unexpected part number for {fw!r}"

        codes = get_platform_codes([fw])
        assert 1 == len(codes), f"Unable to parse FW: {fw!r}"

  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    """Ensure function doesn't raise an exception"""
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([
      b"JX6A-14C204-BPL\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"NZ6T-14F397-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"PJ6T-14H102-ABJ\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"LB5A-14C204-EAC\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    assert results == {(b"X6A", b"J"), (b"Z6T", b"N"), (b"J6T", b"P"), (b"B5A", b"L")}

  def test_fuzzy_match(self):
    for platform, fw_by_addr in FW_VERSIONS.items():
      # Ensure there's no overlaps in platform codes
      for _ in range(20):
        car_fw = []
        for ecu, fw_versions in fw_by_addr.items():
          ecu_name, addr, sub_addr = ecu
          fw = random.choice(fw_versions)
          car_fw.append(CarParams.CarFw(ecu=ecu_name, fwVersion=fw, address=addr,
                                        subAddress=0 if sub_addr is None else sub_addr))

        CP = CarParams(carFw=car_fw)
        matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw), CP.carVin, FW_VERSIONS)
        assert matches == {platform}

  def test_match_fw_fuzzy(self):
    offline_fw = {
      (Ecu.eps, 0x730, None): [
        b"L1MC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      (Ecu.abs, 0x760, None): [
        b"L1MC-2D053-BA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"L1MC-2D053-BD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      (Ecu.fwdRadar, 0x764, None): [
        b"LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"LB5T-14D049-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      # We consider all model year hints for ECU, even with different platform codes
      (Ecu.fwdCamera, 0x706, None): [
        b"LB5T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"NC5T-14F397-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
    }
    expected_fingerprint = CAR.FORD_EXPLORER_MK6

    # ensure that we fuzzy match on all non-exact FW with changed revisions
    live_fw = {
      (0x730, None): {b"L1MC-14D003-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x760, None): {b"L1MC-2D053-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x764, None): {b"LB5T-14D049-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x706, None): {b"LB5T-14F397-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
    }
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw})
    assert candidates == {expected_fingerprint}

    # model year hint in between the range should match
    live_fw[(0x706, None)] = {b"MB5T-14F397-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"}
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw,})
    assert candidates == {expected_fingerprint}

    # unseen model year hint should not match
    live_fw[(0x760, None)] = {b"M1MC-2D053-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"}
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw})
    assert len(candidates) == 0, "Should not match new model year hint"


class TestTransitFingerprint(unittest.TestCase):
  # Recorded on the owner's van on 2026-09-16. The ADAS, parkingAdas, and engine (PCM,
  # 0x7E0 - listed in FW_QUERY_CONFIG.extra_ecus) ECUs also answered during that capture,
  # but all three responses have fw.logging=True and build_fw_dict() drops logging entries
  # (opendbc/car/fw_versions.py), so they never participate in fingerprint matching and are
  # intentionally not listed here or in FW_VERSIONS.
  RECORDED = {
    (0x730, None): b'KK21-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x760, None): b'NK41-2D053-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x706, None): b'NK3T-14F397-AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x764, None): b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
  }

  def test_transit_platform_exists(self):
    assert CAR.FORD_TRANSIT_MK5 in FW_VERSIONS
    assert CAR.FORD_TRANSIT_MK5.config.flags & FordFlags.LKA_STEER

  def test_recorded_firmware_is_listed(self):
    listed = FW_VERSIONS[CAR.FORD_TRANSIT_MK5]
    flat = {}
    for (_ecu, addr, _sub), versions in listed.items():
      flat.setdefault(addr, []).extend(versions)
    assert set(flat) == {addr for addr, _sub in self.RECORDED}, "unexpected set of ECU addresses"
    for (addr, _sub), fw in self.RECORDED.items():
      assert fw in flat[addr], f"firmware {fw!r} missing for {hex(addr)}"
      # a single flipped byte must not still match
      corrupted = bytes([fw[0] ^ 0xFF]) + fw[1:]
      assert corrupted not in flat[addr], f"corrupted firmware should not match for {hex(addr)}"


class TestTransitInterface:
  def _params(self, fingerprint_main, car_fw=None, candidate=CAR.FORD_TRANSIT_MK5):
    return CarInterface.get_params(candidate,
                                   {0: fingerprint_main, 2: {}},
                                   car_fw or [], alpha_long=False, is_release=True, docs=False)

  @staticmethod
  def _pscm_asbuilt_fw(tja, lca):
    # interface.py finds the PSCM AsBuilt block-2 response (request 0x22 0xDE01,
    # see ford_asbuilt_block_request/ASBUILT_BLOCKS in values.py) by ecu + request
    # substring, then reads fwVersion[7]/[8] as the TJA/LCA config bytes.
    fw = bytearray(24)
    fw[7] = tja   # Traffic Jam Assist
    fw[8] = lca   # Lane Centering Assist
    return CarParams.CarFw(ecu=Ecu.eps, address=ECU_ADDRESSES[Ecu.eps], request=[b'\x22\xDE\x01'], fwVersion=bytes(fw))

  def test_0x176_means_automatic(self):
    # 0x176 carries PRND and gears 1-10 on this van; it has neither a
    # shiftByWire ECU nor 0x5A, so upstream would call it a manual and impose
    # a 20 mph minimum enable speed.
    ret = self._params({0x176: 8})
    assert ret.transmissionType == TransmissionType.automatic
    assert ret.minEnableSpeed == -1

  def test_no_0x176_stays_manual(self):
    ret = self._params({})
    assert ret.transmissionType == TransmissionType.manual

  def test_not_dashcam_only(self):
    # A PSCM AsBuilt config with non-0xFF TJA/LCA bytes would normally mark a
    # Ford dashcamOnly (it looks like the car lacks the LCA/TJA CAN APIs), but
    # the Transit drives steering via Lane_Assist_Data1 (LKA_STEER) instead of
    # that channel, so the guard must stay off for it even with this firmware
    # present.
    pscm_fw = self._pscm_asbuilt_fw(tja=0x01, lca=0x01)
    ret = self._params({0x176: 8}, car_fw=[pscm_fw])
    assert not ret.dashcamOnly

  def test_dashcam_gate_discriminates_by_lka_steer_flag(self):
    # Proves test_not_dashcam_only isn't vacuous: the exact same PSCM firmware
    # on a Ford platform without FordFlags.LKA_STEER (Escape MK4, which drives
    # the LCA/TJA channel) must still trip dashcamOnly.
    pscm_fw = self._pscm_asbuilt_fw(tja=0x01, lca=0x01)
    ret = self._params({}, car_fw=[pscm_fw], candidate=CAR.FORD_ESCAPE_MK4)
    assert ret.dashcamOnly

  def test_lka_safety_flag_set(self):
    ret = self._params({0x176: 8})
    assert ret.safetyConfigs[-1].safetyParam & FordSafetyFlags.LKA_STEER


class TestTransitLkaMessage:
  def setup_method(self):
    self.packer = CANPacker("ford_lincoln_base_pt")
    # CanBus() bare (no CP, no fingerprint) asserts in CanBusBase.__init__;
    # every other caller in this codebase passes one or the other, so we do
    # the same here rather than loosen CanBus's default for test convenience.
    self.CAN = fordcan.CanBus(None, {0: {}})

  @staticmethod
  def _decode_lane_assist_data1(addr, dat):
    # Register the message via the constructor rather than relying on
    # CANParser's lazy registration on first `parser.vl[...]` access -
    # otherwise the first update() call registers-and-skips the frame
    # instead of parsing it, and vl[...] silently reads back the default 0.
    parser = CANParser("ford_lincoln_base_pt", [("Lane_Assist_Data1", 0)], 0)
    parser.update([(0, [(addr, dat, 0)])])
    return parser.vl["Lane_Assist_Data1"]

  def test_inactive_sends_zero_action(self):
    addr, dat, bus = fordcan.create_transit_lka_msg(self.packer, self.CAN, False, 3.0, 4, 1)
    assert addr == 0x3CA
    assert (dat[0] >> 5) == 0

    # LaRefAng_No_Req has a DBC offset of -102.4 mrad, so an all-zero raw
    # payload decodes to -102.4 mrad, not 0. An earlier revision shipped
    # exactly that and panda blocked every heartbeat frame. Decode through
    # the real DBC (rather than re-deriving the bit layout by hand here) to
    # lock in that the packed *value* is actually 0.
    vals = self._decode_lane_assist_data1(addr, dat)
    assert vals["LaRefAng_No_Req"] == 0.0

  def test_active_sends_requested_action(self):
    addr, dat, _bus = fordcan.create_transit_lka_msg(self.packer, self.CAN, True, 3.0, 4, 1)
    assert (dat[0] >> 5) == 4

    vals = self._decode_lane_assist_data1(addr, dat)
    assert vals["LkaActvStats_D2_Req"] == 4

  def test_angle_is_clipped_to_the_wire_limit(self):
    addr, hi, _b = fordcan.create_transit_lka_msg(self.packer, self.CAN, True, 99.0, 2, 0)
    _a, cap, _b = fordcan.create_transit_lka_msg(self.packer, self.CAN, True, 5.8, 2, 0)
    assert hi == cap

    # The identity check above would still pass if clipping were broken in a
    # way that made both calls pack the same *wrong* value (e.g. always 0
    # mrad). Decode and check against the actual wire limit to rule that out.
    vals = self._decode_lane_assist_data1(addr, hi)
    assert math.isclose(vals["LaRefAng_No_Req"], math.radians(5.8) * 1000.0, abs_tol=0.05)


class TestTransitLateralMotionControlHeartbeat:
  """
  Task 3 made LatCtl_D_Rq != 0 on LateralMotionControl (0x3D3) a transmit
  violation for LKA_STEER platforms, because that check shares limiter state
  with the Lane_Assist_Data1 (0x3CA) angle check. Lateral control must always
  go out on 0x3CA for these platforms; 0x3D3 may only ever carry an inactive
  heartbeat so the PSCM-to-camera link stays alive.
  """

  def _run(self, candidate, steering_angle_deg=10.0, curvature=0.01, frames=20):
    CarInterface = interfaces[candidate]
    car_params = CarInterface.get_params(candidate, {0: {}, 2: {}}, [], alpha_long=False, is_release=True, docs=False)
    car_interface = CarInterface(car_params)
    car_interface.update([])

    CC = structs.CarControl()
    CC.enabled = True
    CC.latActive = True
    CC.actuators.steeringAngleDeg = steering_angle_deg
    CC.actuators.curvature = curvature
    CC = CC.as_reader()

    parser = CANParser("ford_lincoln_base_pt", [], 0)
    lat_ctl_rq_seen = []
    for i in range(frames):
      _, can_sends = car_interface.apply(CC, i)
      for addr, dat, _bus in can_sends:
        if addr == 0x3D3:
          parser.update([(0, [(addr, dat, 0)])])
          lat_ctl_rq_seen.append(parser.vl["LateralMotionControl"]["LatCtl_D_Rq"])

    assert len(lat_ctl_rq_seen) > 0, "LateralMotionControl was never sent"
    return lat_ctl_rq_seen

  def test_lka_steer_platform_never_requests_lateral_on_0x3d3(self):
    lat_ctl_rq_seen = self._run(CAR.FORD_TRANSIT_MK5)
    assert all(v == 0 for v in lat_ctl_rq_seen), \
      f"LKA_STEER platform requested lateral on LateralMotionControl (0x3D3): {lat_ctl_rq_seen}"

  def test_non_lka_steer_platform_keeps_stock_behavior(self):
    # Proves the above isn't vacuous: a non-LKA_STEER platform steers through
    # LateralMotionControl and must still request lateral there when active.
    lat_ctl_rq_seen = self._run(CAR.FORD_BRONCO_SPORT_MK1)
    assert any(v != 0 for v in lat_ctl_rq_seen)


class TestTransitLkaIntervention:
  def test_standard_position_never_escalates(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.SLOW, TransitLkaDirectionSign.POSITIVE_LEFT)
    action, _ = s.update(9.0, 12.0, 0.0)
    assert action == 2

  def test_increasing_position_always_escalates(self):
    s = TransitLkaState(TransitLkaIntervention.INCREASING, TransitLkaRamp.SLOW, TransitLkaDirectionSign.POSITIVE_LEFT)
    action, _ = s.update(0.5, 0.5, 0.0)
    assert action == 1

  def test_preset_latches_on_both_conditions(self):
    s = TransitLkaState(TransitLkaIntervention.PRESET, TransitLkaRamp.SLOW, TransitLkaDirectionSign.POSITIVE_LEFT)
    assert s.update(5.5, 4.0, 0.0)[0] == 2      # req high, desired low -> no
    assert s.update(4.0, 6.0, 0.0)[0] == 2      # desired high, req low -> no
    assert s.update(5.5, 6.0, 0.0)[0] == 1      # both -> escalate
    assert s.update(4.8, 5.0, 0.0)[0] == 1      # inside the hysteresis band -> hold
    assert s.update(4.5, 4.7, 0.0)[0] == 2      # both below exit -> release


class TestTransitLkaRamp:
  def test_preset_enters_on_angle(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.PRESET, TransitLkaDirectionSign.POSITIVE_LEFT)
    assert s.update(1.0, 1.0, 0.0)[1] == 0
    assert s.update(1.9, 1.9, 0.0)[1] == 1

  def test_preset_enters_on_demand_rate_after_the_filter_settles(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.PRESET, TransitLkaDirectionSign.POSITIVE_LEFT)
    for _ in range(40):
      out = s.update(0.2, 0.2, 40.0)
    assert out[1] == 1

  def test_filter_rejects_a_single_rate_spike(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.PRESET, TransitLkaDirectionSign.POSITIVE_LEFT)
    assert s.update(0.2, 0.2, 60.0)[1] == 0

  def test_preset_holds_inside_the_band(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.PRESET, TransitLkaDirectionSign.POSITIVE_LEFT)
    s.update(2.0, 2.0, 0.0)
    assert s.update(1.6, 1.6, 0.0)[1] == 1
    assert s.update(1.4, 1.4, 0.0)[1] == 0


class TestTransitLkaDirectionSign:
  def test_positive_left(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.SLOW, TransitLkaDirectionSign.POSITIVE_LEFT)
    assert s.update(1.0, 1.0, 0.0)[0] == 2
    assert s.update(-1.0, -1.0, 0.0)[0] == 4

  def test_positive_right(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.SLOW, TransitLkaDirectionSign.POSITIVE_RIGHT)
    assert s.update(1.0, 1.0, 0.0)[0] == 4
    assert s.update(-1.0, -1.0, 0.0)[0] == 2

  def test_deadband_gives_no_intervention(self):
    s = TransitLkaState(TransitLkaIntervention.STANDARD, TransitLkaRamp.SLOW, TransitLkaDirectionSign.POSITIVE_LEFT)
    assert s.update(0.05, 0.05, 0.0)[0] == 0


def _build_transit_controller(extra_flags: int = 0):
  candidate = CAR.FORD_TRANSIT_MK5
  # Exactly how car_helpers.get_car builds it: the packed quirk bits go in through
  # get_params, so everything derived from them - safetyConfigs included - is computed
  # there while CarParams is still mutable.
  CP = CarInterface.get_params(candidate, {0: {}, 2: {}}, [], alpha_long=False, is_release=True,
                               docs=False, extra_flags=extra_flags)
  dbc_names = {Bus.pt: DBC[candidate][Bus.pt]}
  return CarController(dbc_names, CP)


class TestTransitLkaParamsPlumbing:
  """The switches must actually be reachable from the openpilot Params
  (card.py packs -> get_car ORs into CP.flags -> CarController unpacks), and must land
  on CP *before* the CarController that reads them is constructed.

  They ride in spare bits of upstream's CarParams.flags rather than in a fork-added
  car.capnp field, so this fork adds no schema ordinal that upstream could later claim
  with a different type. get_params/_get_params carry no extra argument (that was
  removed in "Fix round 3" - it forced all fifteen brand _get_params overrides,
  including the fourteen this fork does not ship, to gain an unused parameter);
  car_helpers.get_car ORs the caller's flags on after CarInterface.get_params returns.
  """

  def test_non_default_setting_reaches_the_state_machine(self):
    # TransitLkaIntervention.INCREASING, packed and ORed onto CP.flags the same way card.py and
    # get_car do, must flip the action the CarController's TransitLkaState emits. If pack
    # or unpack is broken this falls back to TransitLkaIntervention.STANDARD and the assert fails.
    cc = _build_transit_controller(pack_transit_lka_flags(int(TransitLkaIntervention.INCREASING), int(TransitLkaRamp.SLOW),
                                                          int(TransitLkaDirectionSign.POSITIVE_LEFT)))

    action, _ = cc.transit_lka.update(1.0, 1.0, 0.0)
    assert action in (1, 6)
    assert action not in (2, 4)

    action, _ = cc.transit_lka.update(-1.0, -1.0, 0.0)
    assert action in (1, 6)
    assert action not in (2, 4)

  def test_default_setting_reproduces_todays_behaviour(self):
    # No extra flags (the default every other get_car caller still uses) must reproduce
    # the shipped baseline: TransitLkaIntervention.STANDARD, action 2/4.
    cc = _build_transit_controller()

    action, _ = cc.transit_lka.update(1.0, 1.0, 0.0)
    assert action == 2

    action, _ = cc.transit_lka.update(-1.0, -1.0, 0.0)
    assert action == 4


class TestLkaMsgStaysEmptyForEveryOtherFord:
  """fordcan.create_lka_msg must stay the empty frame upstream transmits.

  Lane_Assist_Data1 is the lane-departure-warning frame on the thirteen non-LKA_STEER
  Ford platforms (Escape, Explorer, F-150, Bronco Sport, Maverick, Focus, Ranger,
  Mach-E, Expedition, Lightning and the rest), and carcontroller.py sends it with the
  default call. None of those cars can be tested here, so what they transmit must not
  change: the Transit's steering frame is a separate function for exactly that reason.
  """

  @staticmethod
  def _packer_and_bus():
    # every Ford platform shares this DBC, and CanBus.main is 0 on all non-CAN FD ones
    return CANPacker(DBC[CAR.FORD_TRANSIT_MK5][Bus.pt]), fordcan.CanBus(fingerprint={0: {}, 1: {}, 2: {}})

  def test_default_call_is_byte_identical_to_an_empty_frame(self):
    packer, CAN = self._packer_and_bus()
    expected = packer.make_can_msg("Lane_Assist_Data1", CAN.main, {})
    assert fordcan.create_lka_msg(packer, CAN) == expected
    # spelled out: an empty payload, not merely the same values re-encoded
    assert bytes(expected[1]) == bytes(8)

  def test_transit_frames_are_unchanged_and_are_not_empty(self):
    # The Transit's own frames keep the values they were built with; only the function
    # they come from moved. LdwActvIntns_D_Req = 3 and the explicit zero-value encodings
    # of LaRefAng_No_Req/LaCurvature_No_Calc are what make them differ from empty.
    packer, CAN = self._packer_and_bus()
    inactive = bytes(fordcan.create_transit_lka_msg(packer, CAN)[1])
    active = bytes(fordcan.create_transit_lka_msg(packer, CAN, True, 5.0, 2, 1)[1])
    assert inactive == bytes.fromhex("0380080000000000")
    assert active == bytes.fromhex("43800ed180000000")
    assert inactive != bytes(8)


class TestTransitLkaSettingsClamp:
  """An out-of-range settings param must not take the device down.

  The Params are plain integers, so a value outside each enum's range reaches pack_transit_lka_flags
  and, if it survived, CarController.__init__. A raised ValueError there kills card,
  manager restarts it, and openpilot is unusable until the param is corrected over SSH.
  These switches deliberately ship with no UI, so hand-editing the params is the only
  way to use them and a typo is the expected interaction, not an exotic one.

  The clamp runs on both sides of the bit field: direction sign owns a single bit, so
  7 would otherwise pack down to a valid-looking POSITIVE_RIGHT rather than fall back.
  """

  @pytest.mark.parametrize(("index", "attr", "default"), [
    (0, "intervention", TransitLkaIntervention.STANDARD),
    (1, "ramp", TransitLkaRamp.SLOW),
    (2, "direction", TransitLkaDirectionSign.POSITIVE_LEFT),
  ])
  def test_out_of_range_value_falls_back_to_the_default(self, index, attr, default):
    settings = [0, 0, 0, 0]
    settings[index] = 7  # a plausible typo, outside every one of the three enums
    cc = _build_transit_controller(pack_transit_lka_flags(*settings))
    assert getattr(cc.transit_lka, attr) == default

  @pytest.mark.parametrize(("index", "attr", "default"), [
    (0, "intervention", TransitLkaIntervention.STANDARD),
    (1, "ramp", TransitLkaRamp.SLOW),
  ])
  def test_out_of_range_raw_bits_fall_back_to_the_default(self, index, attr, default):
    # the two-bit fields can hold 3, which is outside TransitLkaIntervention and TransitLkaRamp; unpack must
    # clamp that too rather than trust whatever produced the flags
    raw = [0, 0, 0, 0]
    raw[index] = 3
    flags = (raw[0] << 2) | (raw[1] << 4) | (raw[2] << 6) | (raw[3] << 7)
    cc = _build_transit_controller(flags)
    assert getattr(cc.transit_lka, attr) == default

  def test_in_range_values_are_still_honoured(self):
    # the clamp must not swallow a valid non-default setting
    cc = _build_transit_controller(pack_transit_lka_flags(int(TransitLkaIntervention.PRESET), int(TransitLkaRamp.FAST),
                                                          int(TransitLkaDirectionSign.POSITIVE_RIGHT),
                                                          int(TransitLkaAvailGate.PERMISSIVE)))
    assert cc.transit_lka.intervention == TransitLkaIntervention.PRESET
    assert cc.transit_lka.ramp == TransitLkaRamp.FAST
    assert cc.transit_lka.direction == TransitLkaDirectionSign.POSITIVE_RIGHT


class TestTransitLkaFlagPacking:
  """The five switches live in spare bits of upstream's CarParams.flags.

  That keeps this fork's car.capnp byte-identical to upstream - no ordinal to collide on
  a rebase, no chance of decoding recorded logs against a different upstream @78 - at the
  cost of readability, so the layout in ford/values.py has to be exactly right. These
  tests are what stops a later FordFlags member from silently landing on bits 2-9.
  """

  @staticmethod
  def _all_settings():
    return itertools.product(TransitLkaIntervention, TransitLkaRamp, TransitLkaDirectionSign,
                             TransitLkaAvailGate, TransitLkaContinuation)

  def test_every_combination_round_trips(self):
    for combo in self._all_settings():
      flags = pack_transit_lka_flags(*(int(s) for s in combo))
      assert unpack_transit_lka_flags(flags) == combo, f"{combo}"

  def test_no_combination_disturbs_canfd_or_lka_steer(self):
    static = FordFlags.CANFD | FordFlags.LKA_STEER
    for combo in self._all_settings():
      flags = pack_transit_lka_flags(*(int(s) for s in combo))
      # the packed bits never touch the two static flags, in either direction
      assert flags & static == 0, f"{combo} collides with {static!r}"
      for platform_flags in (0, FordFlags.CANFD, FordFlags.LKA_STEER, static):
        combined = platform_flags | flags
        assert combined & static == platform_flags
        assert unpack_transit_lka_flags(combined) == combo

  def test_defaults_pack_to_zero(self):
    # so that ORing card.py's value onto a non-Ford CarParams.flags is a no-op
    assert pack_transit_lka_flags(int(TransitLkaIntervention.STANDARD), int(TransitLkaRamp.SLOW),
                                  int(TransitLkaDirectionSign.POSITIVE_LEFT), int(TransitLkaAvailGate.STANDARD),
                                  int(TransitLkaContinuation.OFF)) == 0

  def test_layout_constants_agree_with_ford_flags(self):
    assert TRANSIT_LKA_FLAGS_MASK & (FordFlags.CANFD | FordFlags.LKA_STEER) == 0
    assert TRANSIT_LKA_FLAGS_MASK == 0b1111111100  # bits 2-9, as documented in values.py


class TestTransitLkaAvailGate:
  """The switch that decides which LaActAvail_D_Actl reports count as "LKA offered".

  Below roughly 36 km/h this Transit's PSCM reports 1 (LCA_LKA_Suppress_LDW_Avail),
  so openpilot stops commanding and a lowered min-speed calibration would be
  invisible from the car's side whether or not it took effect. PERMISSIVE commands
  through that report so the two can be told apart on the road.
  """

  @staticmethod
  def _carstate(gate):
    candidate = CAR.FORD_TRANSIT_MK5
    CP = CarInterface.get_params(candidate, {0: {0x176: 8}, 2: {}}, [], alpha_long=False, is_release=True,
                                 docs=False, extra_flags=pack_transit_lka_flags(0, 0, 0, int(gate)))
    return CarState(CP)

  @pytest.mark.parametrize(("gate", "expected"), [
    (TransitLkaAvailGate.STANDARD, (2, 3)),
    (TransitLkaAvailGate.PERMISSIVE, (1, 2, 3)),
    (TransitLkaAvailGate.ANY, (0, 1, 2, 3)),
  ])
  def test_each_position_accepts_its_own_reports(self, gate, expected):
    assert self._carstate(gate).lkas_avail_values == expected

  def test_default_is_unchanged_from_the_shipped_behaviour(self):
    # no extra flags at all is what every other caller produces
    candidate = CAR.FORD_TRANSIT_MK5
    CP = CarInterface.get_params(candidate, {0: {0x176: 8}, 2: {}}, [], alpha_long=False, is_release=True, docs=False)
    assert CarState(CP).lkas_avail_values == (2, 3)

  @pytest.mark.parametrize(("gate", "steers_on_1"), [
    (TransitLkaAvailGate.STANDARD, False),
    (TransitLkaAvailGate.PERMISSIVE, True),
  ])
  def test_a_suppressed_report_reaches_the_wire_only_when_permitted(self, gate, steers_on_1):
    """End to end: the gate decides whether a real steering command goes out on 0x3CA.

    TRANSIT_LKA_AVAIL_VALUES alone proves nothing - what matters is that
    CarController stops zeroing the frame, since lka_active is what forces
    LaRefAng_No_Req and LkaActvStats_D2_Req to zero.
    """
    accepted = TRANSIT_LKA_AVAIL_VALUES[gate]
    assert (1 in accepted) == steers_on_1

    packer = CANPacker("ford_lincoln_base_pt")
    CAN = fordcan.CanBus(None, {0: {}})

    # carcontroller: lka_active false means angle 0 and action 0 regardless of the plan
    lka_active = steers_on_1
    apply_angle = 3.0 if lka_active else 0.0
    action = 2 if lka_active else 0
    addr, dat, _bus = fordcan.create_transit_lka_msg(packer, CAN, lka_active, apply_angle, action, 0)

    vals = TestTransitLkaMessage._decode_lane_assist_data1(addr, dat)
    if steers_on_1:
      assert vals["LkaActvStats_D2_Req"] == 2
      assert math.isclose(vals["LaRefAng_No_Req"], math.radians(apply_angle) * 1000.0, abs_tol=0.05)
    else:
      assert vals["LkaActvStats_D2_Req"] == 0
      assert vals["LaRefAng_No_Req"] == 0.0


class TestTransitLkaContinuation:
  """Keeping lateral alive past the PCM's cancel, and keeping longitudinal out of it.

  The PCM leaves Active for Standby at about 4.94 m/s on the way to a stop. Without the
  latch every command stops there, lateral included, so the PSCM is never asked to steer
  below it and the min-speed calibration cannot be exercised at all.
  """

  ENGAGED, STANDBY, OFF = 5, 3, 0

  @staticmethod
  def _carstate(on):
    CP = CarInterface.get_params(CAR.FORD_TRANSIT_MK5, {0: {0x176: 8}, 2: {}}, [],
                                 alpha_long=False, is_release=True, docs=False,
                                 extra_flags=pack_transit_lka_flags(0, 0, 0, 0,
                                   int(TransitLkaContinuation.ON if on else TransitLkaContinuation.OFF)))
    return CarState(CP)

  @staticmethod
  def _step(cs, cruise_state, speed, brake=False):
    """One update of just the latch, driven by the three signals panda also reads."""
    ret = structs.CarState()
    ret.vEgoRaw = speed
    ret.brakePressed = brake
    pcm_engaged = cruise_state in (4, 5)
    standby = cruise_state == 3
    if not cs.lka_continuation:
      cs.lka_continuation = (cs.lka_continuation_enabled and cs.pcm_cruise_engaged_prev and standby and
                             ret.vEgoRaw < TRANSIT_LKA_CONT_ENTER_SPEED and not ret.brakePressed)
    else:
      cs.lka_continuation = (standby and not ret.brakePressed and
                             TRANSIT_LKA_CONT_EXIT_SPEED_LOW < ret.vEgoRaw < TRANSIT_LKA_CONT_EXIT_SPEED_HIGH)
    cs.pcm_cruise_engaged_prev = pcm_engaged
    return pcm_engaged or cs.lka_continuation

  def test_latches_on_the_cancel_and_holds_down_to_walking_pace(self):
    cs = self._carstate(True)
    assert self._step(cs, self.ENGAGED, 12.0)          # engaged, well above the cancel
    assert self._step(cs, self.ENGAGED, 5.2)
    assert self._step(cs, self.STANDBY, 4.9)           # the cancel -> latch
    assert cs.lka_continuation
    for v in (4.0, 3.0, 2.0, 1.0, 0.6):                # 1.0 m/s is the value under test
      assert self._step(cs, self.STANDBY, v), f"dropped at {v} m/s"

  def test_never_latches_from_a_standing_start_in_standby(self):
    # Standby is also "cruise switched on but never set". Entry requires the previous
    # frame to have been genuinely engaged, so that case must not latch.
    cs = self._carstate(True)
    for _ in range(5):
      assert not self._step(cs, self.STANDBY, 3.0)
    assert not cs.lka_continuation

  def test_off_is_the_shipped_behaviour(self):
    cs = self._carstate(False)
    assert self._step(cs, self.ENGAGED, 6.0)
    assert not self._step(cs, self.STANDBY, 4.5)
    assert not cs.lka_continuation

  @pytest.mark.parametrize(("what", "cruise_state", "speed", "brake"), [
    ("brake pressed", STANDBY, 3.0, True),
    ("back above the exit speed", STANDBY, 9.5, False),
    ("cruise re-engaged", ENGAGED, 3.0, False),
    ("cruise switched off", OFF, 3.0, False),
    ("stopped", STANDBY, 0.2, False),
  ])
  def test_every_exit_condition_releases_the_latch(self, what, cruise_state, speed, brake):
    cs = self._carstate(True)
    self._step(cs, self.ENGAGED, 6.0)
    self._step(cs, self.STANDBY, 4.5)
    assert cs.lka_continuation, "precondition: latched"
    self._step(cs, cruise_state, speed, brake)
    assert not cs.lka_continuation, f"latch survived {what}"

  def test_does_not_relatch_after_releasing(self):
    # once released the only way back is a fresh Active -> Standby transition
    cs = self._carstate(True)
    self._step(cs, self.ENGAGED, 6.0)
    self._step(cs, self.STANDBY, 4.5)
    self._step(cs, self.STANDBY, 3.0, brake=True)
    assert not cs.lka_continuation
    for _ in range(5):
      self._step(cs, self.STANDBY, 3.0)
      assert not cs.lka_continuation

  @staticmethod
  def _accdata_while(latched, accel=-2.0):
    """Run the real controller and return what ACCDATA (0x186) actually carried."""
    CP = CarInterface.get_params(CAR.FORD_TRANSIT_MK5, {0: {0x176: 8}, 2: {}}, [],
                                 alpha_long=False, is_release=True, docs=False,
                                 extra_flags=pack_transit_lka_flags(0, 0, 0, 0, int(TransitLkaContinuation.ON)))
    assert CP.openpilotLongitudinalControl, "precondition: this platform runs openpilot long"
    car_interface = CarInterface(CP)
    car_interface.update([])
    car_interface.CS.lka_continuation = latched

    CC = structs.CarControl()
    CC.enabled = True
    CC.longActive = True
    CC.actuators.accel = accel
    CC = CC.as_reader()

    parser = CANParser("ford_lincoln_base_pt", [], 0)
    seen = []
    for i in range(20):
      _, can_sends = car_interface.apply(CC, i)
      for addr, dat, _bus in can_sends:
        if addr == 0x186:
          parser.update([(0, [(addr, dat, 0)])])
          seen.append(dict(parser.vl["ACCDATA"]))
    assert seen, "ACCDATA was never sent"
    return seen

  def test_longitudinal_is_forced_inactive_while_latched(self):
    """The whole reason for choosing lateral-only: no accel or brake request gets out.

    Drives the real CarController rather than re-deriving its expression, and decodes
    the frame it actually emitted.
    """
    latched = self._accdata_while(True)
    for v in latched:
      assert v["Cmbb_B_Enbl"] == 0, "ACC enabled while latched"
      assert v["AccResumEnbl_B_Rq"] == 0
      assert v["AccBrkPrchg_B_Rq"] == 0 and v["AccBrkDecel_B_Rq"] == 0, "brake actuation while latched"
      # panda accepts exactly one AccBrkTot_A_Rq while controls_allowed is false, the
      # inactive value; anything else makes the frame a transmit violation
      assert abs(v["AccBrkTot_A_Rq"]) < 0.005, f"accel {v['AccBrkTot_A_Rq']} is not the inactive value"

  def test_the_same_request_does_get_out_when_not_latched(self):
    # proves the test above is not vacuous - the identical accel goes through normally
    free = self._accdata_while(False)
    assert any(v["AccBrkTot_A_Rq"] < -0.5 for v in free), "braking never requested unlatched"
    assert any(v["Cmbb_B_Enbl"] == 1 for v in free)

  def test_the_switch_reaches_the_safety_layer(self):
    """Unlike the other four switches this one has to be in safetyParam, because panda
    recomputes the latch itself, so get_params has to see it."""
    for on, expected in ((False, False), (True, True)):
      CP = CarInterface.get_params(CAR.FORD_TRANSIT_MK5, {0: {0x176: 8}, 2: {}}, [],
                                   alpha_long=False, is_release=True, docs=False,
                                   extra_flags=pack_transit_lka_flags(0, 0, 0, 0, int(on)))
      got = bool(CP.safetyConfigs[-1].safetyParam & FordSafetyFlags.LKA_CONTINUATION)
      assert got is expected

  def test_constructing_the_interface_leaves_carparams_alone(self):
    """controlsd builds its CarInterface from the carParams *message*, which is a
    read-only capnp reader. An earlier revision set this flag in CarInterface.__init__;
    there it raised AttributeError, controlsd never came up, card sent no CAN, and with
    panda already blocking the camera's ACCDATA the PCM faulted cruise about eleven
    seconds into the drive (recorded routes 00000011 and 00000012). Nothing may be
    written to CarParams outside get_params."""
    CP = CarInterface.get_params(CAR.FORD_TRANSIT_MK5, {0: {0x176: 8}, 2: {}}, [],
                                 alpha_long=False, is_release=True, docs=False,
                                 extra_flags=pack_transit_lka_flags(0, 0, 0, 0, int(TransitLkaContinuation.ON)))
    before = copy.deepcopy(CP)
    CarInterface(CP)
    assert CP.flags == before.flags
    assert [c.safetyParam for c in CP.safetyConfigs] == [c.safetyParam for c in before.safetyConfigs]

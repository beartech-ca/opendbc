import itertools

import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus
from opendbc.car.ford import fordcan
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.transit_lka import TransitLkaState, Intervention, Ramp, DirectionSign, pack_flags, unpack_flags
from opendbc.car.ford.values import CAR, DBC, FordFlags, TRANSIT_LKA_FLAGS_MASK


class TestIntervention:
  def test_standard_position_never_escalates(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.SLOW, DirectionSign.POSITIVE_LEFT)
    action, _ = s.update(9.0, 12.0, 0.0)
    assert action == 2

  def test_increasing_position_always_escalates(self):
    s = TransitLkaState(Intervention.INCREASING, Ramp.SLOW, DirectionSign.POSITIVE_LEFT)
    action, _ = s.update(0.5, 0.5, 0.0)
    assert action == 1

  def test_preset_latches_on_both_conditions(self):
    s = TransitLkaState(Intervention.PRESET, Ramp.SLOW, DirectionSign.POSITIVE_LEFT)
    assert s.update(5.5, 4.0, 0.0)[0] == 2      # req high, desired low -> no
    assert s.update(4.0, 6.0, 0.0)[0] == 2      # desired high, req low -> no
    assert s.update(5.5, 6.0, 0.0)[0] == 1      # both -> escalate
    assert s.update(4.8, 5.0, 0.0)[0] == 1      # inside the hysteresis band -> hold
    assert s.update(4.5, 4.7, 0.0)[0] == 2      # both below exit -> release


class TestRamp:
  def test_preset_enters_on_angle(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.PRESET, DirectionSign.POSITIVE_LEFT)
    assert s.update(1.0, 1.0, 0.0)[1] == 0
    assert s.update(1.9, 1.9, 0.0)[1] == 1

  def test_preset_enters_on_demand_rate_after_the_filter_settles(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.PRESET, DirectionSign.POSITIVE_LEFT)
    for _ in range(40):
      out = s.update(0.2, 0.2, 40.0)
    assert out[1] == 1

  def test_filter_rejects_a_single_rate_spike(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.PRESET, DirectionSign.POSITIVE_LEFT)
    assert s.update(0.2, 0.2, 60.0)[1] == 0

  def test_preset_holds_inside_the_band(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.PRESET, DirectionSign.POSITIVE_LEFT)
    s.update(2.0, 2.0, 0.0)
    assert s.update(1.6, 1.6, 0.0)[1] == 1
    assert s.update(1.4, 1.4, 0.0)[1] == 0


class TestDirectionSign:
  def test_positive_left(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.SLOW, DirectionSign.POSITIVE_LEFT)
    assert s.update(1.0, 1.0, 0.0)[0] == 2
    assert s.update(-1.0, -1.0, 0.0)[0] == 4

  def test_positive_right(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.SLOW, DirectionSign.POSITIVE_RIGHT)
    assert s.update(1.0, 1.0, 0.0)[0] == 4
    assert s.update(-1.0, -1.0, 0.0)[0] == 2

  def test_deadband_gives_no_intervention(self):
    s = TransitLkaState(Intervention.STANDARD, Ramp.SLOW, DirectionSign.POSITIVE_LEFT)
    assert s.update(0.05, 0.05, 0.0)[0] == 0


def _build_controller(extra_flags: int = 0):
  candidate = CAR.FORD_TRANSIT_MK5
  CP = CarInterface.get_params(candidate, {0: {}, 2: {}}, [], alpha_long=False, is_release=True, docs=False)
  # Mirrors what get_car does: OR the caller's packed quirk bits onto the CarParams capnp
  # builder before the CarController (which unpacks them out of CP.flags) is constructed.
  CP.flags |= extra_flags
  dbc_names = {Bus.pt: DBC[candidate][Bus.pt]}
  return CarController(dbc_names, CP)


class TestParamsPlumbing:
  """The switches must actually be reachable from the openpilot Params
  (card.py packs -> get_car ORs into CP.flags -> CarController unpacks), and must land
  on CP *before* the CarController that reads them is constructed.

  They ride in spare bits of upstream's CarParams.flags rather than in a fork-added
  car.capnp field, so this fork adds no schema ordinal that upstream could later claim
  with a different type. get_params/_get_params carry no extra argument (that was
  removed in "Fix round 3" - it forced all fifteen brand _get_params overrides,
  including the fourteen this fork does not ship, to gain an unused parameter);
  car_helpers.get_car ORs the caller's flags on after CarInterface.get_params returns.
  See docs/superpowers/specs/2026-09-16-transit-mk5-lka-port-design.md and
  .superpowers/sdd/task-5-report.md, "Fix round 1" and "Fix round 3".
  """

  def test_non_default_setting_reaches_the_state_machine(self):
    # Intervention.INCREASING, packed and ORed onto CP.flags the same way card.py and
    # get_car do, must flip the action the CarController's TransitLkaState emits. If pack
    # or unpack is broken this falls back to Intervention.STANDARD and the assert fails.
    cc = _build_controller(pack_flags(int(Intervention.INCREASING), int(Ramp.SLOW), int(DirectionSign.POSITIVE_LEFT)))

    action, _ = cc.transit_lka.update(1.0, 1.0, 0.0)
    assert action in (1, 6)
    assert action not in (2, 4)

    action, _ = cc.transit_lka.update(-1.0, -1.0, 0.0)
    assert action in (1, 6)
    assert action not in (2, 4)

  def test_default_setting_reproduces_todays_behaviour(self):
    # No extra flags (the default every other get_car caller still uses) must reproduce
    # the shipped baseline: Intervention.STANDARD, action 2/4.
    cc = _build_controller()

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


class TestSettingsAreClampedNotFatal:
  """An out-of-range settings param must not take the device down.

  The Params are plain integers, so a value outside each enum's range reaches pack_flags
  and, if it survived, CarController.__init__. A raised ValueError there kills card,
  manager restarts it, and openpilot is unusable until the param is corrected over SSH.
  These switches deliberately ship with no UI, so hand-editing the params is the only
  way to use them and a typo is the expected interaction, not an exotic one.

  The clamp runs on both sides of the bit field: direction sign owns a single bit, so
  7 would otherwise pack down to a valid-looking POSITIVE_RIGHT rather than fall back.
  """

  @pytest.mark.parametrize(("index", "attr", "default"), [
    (0, "intervention", Intervention.STANDARD),
    (1, "ramp", Ramp.SLOW),
    (2, "direction", DirectionSign.POSITIVE_LEFT),
  ])
  def test_out_of_range_value_falls_back_to_the_default(self, index, attr, default):
    settings = [0, 0, 0]
    settings[index] = 7  # a plausible typo, outside every one of the three enums
    cc = _build_controller(pack_flags(*settings))
    assert getattr(cc.transit_lka, attr) == default

  @pytest.mark.parametrize(("index", "attr", "default"), [
    (0, "intervention", Intervention.STANDARD),
    (1, "ramp", Ramp.SLOW),
  ])
  def test_out_of_range_raw_bits_fall_back_to_the_default(self, index, attr, default):
    # the two-bit fields can hold 3, which is outside Intervention and Ramp; unpack must
    # clamp that too rather than trust whatever produced the flags
    raw = [0, 0, 0]
    raw[index] = 3
    flags = (raw[0] << 2) | (raw[1] << 4) | (raw[2] << 6)
    cc = _build_controller(flags)
    assert getattr(cc.transit_lka, attr) == default

  def test_in_range_values_are_still_honoured(self):
    # the clamp must not swallow a valid non-default setting
    cc = _build_controller(pack_flags(int(Intervention.PRESET), int(Ramp.FAST), int(DirectionSign.POSITIVE_RIGHT)))
    assert cc.transit_lka.intervention == Intervention.PRESET
    assert cc.transit_lka.ramp == Ramp.FAST
    assert cc.transit_lka.direction == DirectionSign.POSITIVE_RIGHT


class TestFlagPacking:
  """The three switches live in spare bits of upstream's CarParams.flags.

  That keeps this fork's car.capnp byte-identical to upstream - no ordinal to collide on
  a rebase, no chance of decoding recorded logs against a different upstream @78 - at the
  cost of readability, so the layout in ford/values.py has to be exactly right. These
  tests are what stops a later FordFlags member from silently landing on bits 2-6.
  """

  @staticmethod
  def _all_settings():
    return itertools.product(Intervention, Ramp, DirectionSign)

  def test_every_combination_round_trips(self):
    for intervention, ramp, direction in self._all_settings():
      flags = pack_flags(int(intervention), int(ramp), int(direction))
      assert unpack_flags(flags) == (intervention, ramp, direction), f"{intervention}/{ramp}/{direction}"

  def test_no_combination_disturbs_canfd_or_lka_steer(self):
    static = FordFlags.CANFD | FordFlags.LKA_STEER
    for intervention, ramp, direction in self._all_settings():
      flags = pack_flags(int(intervention), int(ramp), int(direction))
      # the packed bits never touch the two static flags, in either direction
      assert flags & static == 0, f"{intervention}/{ramp}/{direction} collides with {static!r}"
      for platform_flags in (0, FordFlags.CANFD, FordFlags.LKA_STEER, static):
        combined = platform_flags | flags
        assert combined & static == platform_flags
        assert unpack_flags(combined) == (intervention, ramp, direction)

  def test_defaults_pack_to_zero(self):
    # so that ORing card.py's value onto a non-Ford CarParams.flags is a no-op
    assert pack_flags(int(Intervention.STANDARD), int(Ramp.SLOW), int(DirectionSign.POSITIVE_LEFT)) == 0

  def test_layout_constants_agree_with_ford_flags(self):
    assert TRANSIT_LKA_FLAGS_MASK & (FordFlags.CANFD | FordFlags.LKA_STEER) == 0
    assert TRANSIT_LKA_FLAGS_MASK == 0b1111100  # bits 2-6, as documented in values.py

from opendbc.car import Bus, structs
from opendbc.car.car_helpers import _apply_transit_lka
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.transit_lka import TransitLkaState, Intervention, Ramp, DirectionSign
from opendbc.car.ford.values import CAR, DBC


def drive(state, samples):
  out = []
  for req, desired, rate in samples:
    out.append(state.update(req, desired, rate))
  return out


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


class TestParamsPlumbing:
  """CP.transitLka must actually be reachable from the openpilot Params switches
  (card.py -> get_car -> car_helpers._apply_transit_lka), and must land on CP
  *before* the CarController that reads it is constructed. get_params/_get_params
  no longer carry a transit_lka argument (removed in "Fix round 3" - it forced all
  fifteen brand _get_params overrides, including the fourteen this fork does not
  ship, to gain an unused parameter). car_helpers.get_car now applies CP.transitLka
  itself, guarded to Ford's LKA_STEER platforms, after CarInterface.get_params
  returns and before constructing the CarInterface/CarController. See
  docs/superpowers/specs/2026-09-16-transit-mk5-lka-port-design.md and
  .superpowers/sdd/task-5-report.md, "Fix round 1" and "Fix round 3".
  """

  @staticmethod
  def _build_controller(transit_lka: 'structs.CarParams.TransitLkaSettings | None'):
    candidate = CAR.FORD_TRANSIT_MK5
    CP = CarInterface.get_params(candidate, {0: {}, 2: {}}, [], alpha_long=False, is_release=True, docs=False)
    # Mirrors what get_car does: apply the Params-sourced switches onto the CarParams
    # capnp builder before the CarController (which reads CP.transitLka) is constructed.
    _apply_transit_lka(CP, transit_lka)
    dbc_names = {Bus.pt: DBC[candidate][Bus.pt]}
    return CarController(dbc_names, CP)

  def test_non_default_setting_reaches_the_state_machine(self):
    # Intervention.INCREASING, routed through _apply_transit_lka the same way get_car
    # does, must flip the action the CarController's TransitLkaState emits. If the
    # guard (or its LKA_STEER check) is broken, this falls back to Intervention.STANDARD
    # and the assert below fails.
    transit_lka = structs.CarParams.TransitLkaSettings(intervention=int(Intervention.INCREASING), ramp=int(Ramp.SLOW),
                                                        directionSign=int(DirectionSign.POSITIVE_LEFT))
    cc = self._build_controller(transit_lka)

    action, _ = cc.transit_lka.update(1.0, 1.0, 0.0)
    assert action in (1, 6)
    assert action not in (2, 4)

    action, _ = cc.transit_lka.update(-1.0, -1.0, 0.0)
    assert action in (1, 6)
    assert action not in (2, 4)

  def test_default_setting_reproduces_todays_behaviour(self):
    # No transit_lka argument (the default every other get_car caller still uses)
    # must reproduce the shipped baseline: Intervention.STANDARD, action 2/4.
    cc = self._build_controller(None)

    action, _ = cc.transit_lka.update(1.0, 1.0, 0.0)
    assert action == 2

    action, _ = cc.transit_lka.update(-1.0, -1.0, 0.0)
    assert action == 4

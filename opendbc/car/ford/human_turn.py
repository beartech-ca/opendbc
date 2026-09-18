"""Manual-steering-override ("human turn") detection.

Ported from BluePilot (BluePilotDev/bluepilot, bp-dev, MIT), where it serves both
Ford lateral strategies. The detection is unchanged; what differs here is the tick
rate and what the caller does with the result.

While the driver holds a sustained turn, lateral is forced inactive rather than
left winding a command the PSCM has to reconcile on release. BluePilot measured
2-3 s of dead time from that reconciliation on a Mach-E PSCM. On this Transit the
command rides Lane_Assist_Data1 and is already relative to the measured angle, so
the stale-command shape is different, but the hand-back is the same problem.
"""
from opendbc.car import DT_CTRL
from opendbc.car.ford.values import CarControllerParams

# Require sustained hands-on AND a large angle (avoids resetting on small wheel nudges in a curve).
HUMAN_TURN_ANGLE_DEG = 45.0
HUMAN_TURN_HOLD_S = 1.5
# When the wheel was ALREADY past HUMAN_TURN_ANGLE_DEG at first contact -- lateral control had it
# turned mid-curve -- the angle condition is pre-satisfied, so a brief corrective nudge would latch
# after only HUMAN_TURN_HOLD_S of light contact and kill steering mid-curve. Require a longer hold
# there before reading it as an intentional takeover. Route 000000bd (2026-07-14) showed the
# discriminator holds on-road: all 7 deliberate turns began with the wheel below the threshold
# (driver wound it up through 45 deg); only mid-maneuver grabs/nudges began beyond it.
HUMAN_TURN_HOLD_PRETURNED_S = 3.0
# This detector is ticked from the Lane_Assist_Data1 branch, which runs at LKA_STEP
# (33 Hz), not STEER_STEP (20 Hz) as in the source. The hold thresholds below are in
# seconds, so the tick length has to match the caller or they are wrong by 5/3.
_STEER_DT = CarControllerParams.LKA_STEP * DT_CTRL  # 33 Hz LKA tick


class HumanTurnDetector:
  """Latches ``active`` once the driver holds real steering pressure AND ``|wheel angle|`` >
  ``HUMAN_TURN_ANGLE_DEG`` continuously for ``HUMAN_TURN_HOLD_S`` — long enough to tell an
  intentional turn from a brief nudge. ``just_released`` pulses True on the first frame after the
  override clears, so a mode can re-seed its command as it re-engages.

  Call ``update`` once per lateral tick while control is active. Modes that want the timer to zero
  on disengage call ``reset`` on their inactive path; modes that want it to persist simply stop
  calling ``update`` (the timer holds its value).
  """

  def __init__(self):
    self.hold_timer_s = 0.0
    self.active = False
    self._active_last = False
    self._pressed_last = False
    self._press_started_preturned = False

  def update(self, enabled: bool, steering_pressed: bool, steering_angle_deg: float) -> bool:
    self._active_last = self.active
    # Was the wheel already past the angle threshold when this press began? If so the driver is
    # touching a wheel that lateral control turned (mid-curve nudge), not driving a turn -- hold
    # the longer HUMAN_TURN_HOLD_PRETURNED_S before latching.
    if steering_pressed and not self._pressed_last:
      self._press_started_preturned = abs(steering_angle_deg) > HUMAN_TURN_ANGLE_DEG
    self._pressed_last = steering_pressed
    if not enabled:
      self.hold_timer_s = 0.0
    elif steering_pressed and abs(steering_angle_deg) > HUMAN_TURN_ANGLE_DEG:
      self.hold_timer_s += _STEER_DT
    else:
      self.hold_timer_s = 0.0
    hold_req = HUMAN_TURN_HOLD_PRETURNED_S if self._press_started_preturned else HUMAN_TURN_HOLD_S
    self.active = self.hold_timer_s >= hold_req
    return self.active

  @property
  def just_released(self) -> bool:
    return self._active_last and not self.active

  def reset(self) -> None:
    self.hold_timer_s = 0.0
    self.active = False
    self._active_last = False
    self._pressed_last = False
    self._press_started_preturned = False

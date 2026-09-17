"""Transit LKA field selection: intervention level, ramp type, direction sign.

All three are switchable so each can be A/B tested on its own. The preset
thresholds come from 46,577 recorded commanded frames across four routes; see
docs/superpowers/specs/2026-09-16-transit-mk5-lka-port-design.md.
"""
import math
from enum import IntEnum

from opendbc.car.carlog import carlog

DT = 1.0 / 33.0                 # Lane_Assist_Data1 is sent at 33Hz
DEADBAND_DEG = 0.1              # below this the wheel counts as centred

INTERV_ENTER_REQ = 5.0
INTERV_ENTER_DESIRED = 5.2
INTERV_EXIT_REQ = 4.6
INTERV_EXIT_DESIRED = 4.8

RAMP_ENTER_REQ = 1.8
RAMP_ENTER_RATE = 15.0
RAMP_EXIT_REQ = 1.5
RAMP_EXIT_RATE = 12.0
RAMP_RATE_TAU = 0.15            # raw demand rate chatters across the band


class Intervention(IntEnum):
  STANDARD = 0
  INCREASING = 1
  PRESET = 2


class Ramp(IntEnum):
  SLOW = 0
  FAST = 1
  PRESET = 2


class DirectionSign(IntEnum):
  POSITIVE_LEFT = 0
  POSITIVE_RIGHT = 1


def coerce_setting(enum_cls: type[IntEnum], value: int) -> IntEnum:
  """Coerce a Params-sourced switch to its enum, falling back to the default.

  These three switches ship with no UI, so hand-editing the params is the only way to
  use them and a typo is the expected interaction. The capnp fields are UInt8, so an
  out-of-range value stores fine and only fails here - and a raised ValueError in
  CarController.__init__ kills card, which manager then restarts forever, leaving the
  device unusable until the param is corrected over SSH. Clamp instead, and say so.

  Member 0 of each enum is the shipped default (STANDARD / SLOW / POSITIVE_LEFT), and
  is also what an unset capnp UInt8 already decodes to.
  """
  try:
    return enum_cls(value)
  except ValueError:
    default = enum_cls(0)
    carlog.warning("transitLka: %s=%s is out of range, falling back to %s", enum_cls.__name__, value, default.name)
    return default


# LkaActvStats_D2_Req: 1 IncrLeft, 2 StandLeft, 4 StandRight, 6 IncrRight
_ACTION = {
  (DirectionSign.POSITIVE_LEFT, False): (2, 4),
  (DirectionSign.POSITIVE_LEFT, True): (1, 6),
  (DirectionSign.POSITIVE_RIGHT, False): (4, 2),
  (DirectionSign.POSITIVE_RIGHT, True): (6, 1),
}


class TransitLkaState:
  def __init__(self, intervention: Intervention, ramp: Ramp, direction: DirectionSign):
    self.intervention = intervention
    self.ramp = ramp
    self.direction = direction
    self.increasing = False
    self.fast = False
    self.rate_filtered = 0.0

  def reset(self) -> None:
    self.increasing = False
    self.fast = False
    self.rate_filtered = 0.0

  def update(self, req_deg: float, desired_deg: float, demand_rate_dps: float) -> tuple[int, int]:
    req, desired = abs(req_deg), abs(desired_deg)

    alpha = 1.0 - math.exp(-DT / RAMP_RATE_TAU)
    self.rate_filtered += alpha * (abs(demand_rate_dps) - self.rate_filtered)

    if self.intervention == Intervention.PRESET:
      if req > INTERV_ENTER_REQ and desired >= INTERV_ENTER_DESIRED:
        self.increasing = True
      elif req < INTERV_EXIT_REQ and desired < INTERV_EXIT_DESIRED:
        self.increasing = False
    else:
      self.increasing = self.intervention == Intervention.INCREASING

    if self.ramp == Ramp.PRESET:
      if req >= RAMP_ENTER_REQ or self.rate_filtered >= RAMP_ENTER_RATE:
        self.fast = True
      elif req < RAMP_EXIT_REQ and self.rate_filtered < RAMP_EXIT_RATE:
        self.fast = False
    else:
      self.fast = self.ramp == Ramp.FAST

    if req_deg > DEADBAND_DEG:
      action = _ACTION[(self.direction, self.increasing)][0]
    elif req_deg < -DEADBAND_DEG:
      action = _ACTION[(self.direction, self.increasing)][1]
    else:
      action = 0

    return action, int(self.fast)

"""Lane-centering trim in the curvature domain.

Ported from BluePilot (BluePilotDev/bluepilot, bp-dev, MIT). The algorithm is
unchanged; only the docstring is rewritten, because the original describes where it
sits inside BluePilot's own controller and that is not where it sits here.

It returns a corrected curvature. The correction is small, bounded, smoothed and
rate-limited, and it is applied to the planner's desired curvature *before*
openpilot's own `clip_curvature`, so it inherits that rate and acceleration limit
rather than bypassing it. Everything downstream - the VehicleModel conversion to a
steering angle, the relative-angle subtraction, panda's magnitude, lateral
acceleration and rate checks - applies to the trimmed value unchanged.

The domain matters. BluePilot records that an earlier version of theirs applied the
same idea to the final angle instead, and it never tracked lane centre: in that
domain the correction has the wrong speed dependence and skips every limiter. That
is also why this is not implemented as a nudge on `actuators.steeringAngleDeg`.

**Model-position fallback (blend, not hard gate).** The target is a blend of the
model's own predicted path (a "no correction" baseline) and the geometric lane
centre, weighted by a laneline confidence score:

- good lane lines (confidence -> 1): target is lane-line centre + offset
- no or poor lane lines (confidence -> 0): target is the model's own path + offset,
  so the correction reduces to just the user's left/right bias rather than dropping
  to zero

The second case is the one that matters on a road with only a centre stripe and a
curbed edge, and on gravel, where there is no reliable pair of lines to sit between
but a bias off the edge is still wanted.

Confidence uses BluePilot's formula and breakpoints unchanged.
"""
import numpy as np
from numpy import interp

from opendbc.car.ford.values import TransitLaneCentering, unpack_transit_lka_flags

# BluePilot's shipped defaults for the two tunables. Kept as constants rather than params:
# the switch is the A/B control this fork needs first, and a second and third knob would have
# to be tuned against drive data that does not exist yet.
DEFAULT_OFFSET_M = 0.0    # positive shifts the target right
DEFAULT_STRENGTH = 0.25   # fraction of the clipped raw correction actually applied

# Laneline confidence blend -- ported verbatim from lateral_curv_ext.py's path_offset blend
# (laneline_width_tolerance / min_laneline_confidence_bp) so both control schemes agree on what
# "good lane lines" means.
_WIDTH_TOLERANCE_BP = (2.4, 2.8, 3.75, 4.25)  # m - the floor is added from StarPilot
_WIDTH_TOLERANCE_V = (0.0, 0.81, 0.81, 0.59) # Adds StarPilot's low edge
_STD_TOLERANCE_BP = (0.3, 0.5) #0.3 is from StarPilot to define bad references
_STD_TOLERANCE_V = (0.81, 0.0) #0.81 is a known good value and allows it to fade to zero as std increases
_CONFIDENCE_BP = (0.6, 0.8)
_CONFIDENCE_V = (0.0, 1.0)

# Speed ramp: reuses curvature-mode's already-tuned centering-authority envelope
# (LC_PID_speed_bp/v in lateral_curv_ext.py) instead of inventing a new one.
_SPEED_RAMP_BP = (0.0, 9.0, 15.0)  # m/s
_SPEED_RAMP_V = (0.0, 0.0, 1.0)

# Lookahead distance for the position-error sample, clipped to a sane range (m).
_LOOKAHEAD_MIN_M = 8.0
_LOOKAHEAD_MAX_M = 35.0

# Hard safety ceiling on the raw correction magnitude (1/m) -- fixed, not a "feel" knob, same
# treatment as _PSCM_SAT_UNWIND_RATE / _soft_roc in lateral_angle_ext.py.
_MAX_RAW_CORRECTION = 0.004

# First-order smoothing time constant (s) -- avoids abrupt jumps in the trim.
_SMOOTH_TAU_S = 0.4

# Rate-of-change limit on the applied correction (1/m per 20 Hz tick), independent of the
# smoothing filter above. The filter alone still has its *fastest* slew immediately after a
# target jump -- its per-tick step is proportional to how far the target moved, so a large,
# sudden confidence swing (e.g. crossing into/out of a too-wide merge lane, or lane lines
# dropping out entirely while the model's own path and the laneline center disagree) can still
# produce a big single-tick correction change even though it's technically "smoothed". This caps
# that step explicitly, the same way path_angle's own soft ROC (lateral_angle_ext.py) and
# curvature mode's LC_path_angle_ROC (lateral_curv_ext.py) rate-limit their outputs. At this
# rate, crossing the full -_MAX_RAW_CORRECTION..+_MAX_RAW_CORRECTION span takes ~2.7s; a more
# typical confidence-transition jump (a fraction of that) resolves proportionally faster.
_CORRECTION_ROC_PER_TICK = 0.00015


class LaneCenterTrim:
  def __init__(self):
    self._correction = 0.0

  def reset(self) -> None:
    self._correction = 0.0

  def update(self, kappa_cmd: float, model, v_ego: float, enabled: bool, offset: float,
             gain: float, lat_active: bool, lane_change: bool) -> float:
    """Returns ``kappa_cmd``, nudged toward (lane-blend target + ``offset``) when active.

    ``offset`` (m): positive shifts the target right, negative left (same sign convention as
    curvature mode's ``custom_path_offset_curv``). Applied whether or not lane lines are usable
    -- see module docstring.
    ``gain`` (0.0-1.0): user-tunable authority -- how much of the (already magnitude-clipped)
    raw correction is actually applied. 0 disables the trim's effect without disabling detection.

    The applied correction is both exponentially smoothed (_SMOOTH_TAU_S) and rate-limited
    (_CORRECTION_ROC_PER_TICK) -- see the constants above -- so a lane-line-confidence
    transition (e.g. a merge lane too wide to be trusted, then narrowing back into range) eases
    into its new target instead of snapping.
    """
    if not enabled or not lat_active or lane_change or model is None:
      self.reset()
      return kappa_cmd

    speed_factor = float(interp(v_ego, _SPEED_RAMP_BP, _SPEED_RAMP_V))
    if speed_factor <= 0.0:
      self.reset()
      return kappa_cmd
    if not (np.isfinite(offset) and np.isfinite(gain)): #stops not a number from getting to steering command; validation of number
      self.reset()
      return kappa_cmd

    valid, raw = self._raw_correction(model, v_ego, offset)
    if not valid:
      # Only true failure case now: model.position itself is unusable, so there's no baseline
      # to offset from at all (see _raw_correction). Lane-line-only failures fall back to the
      # model-position blend instead of landing here.
      self.reset()
      return kappa_cmd

    target = float(np.clip(raw, -_MAX_RAW_CORRECTION, _MAX_RAW_CORRECTION)) * float(np.clip(gain, 0.0, 1.0)) * speed_factor
    alpha = 1.0 - np.exp(-0.05 / _SMOOTH_TAU_S)  # BluePilot lateral tick is 20 Hz (dt=0.05s)
    filtered = float(alpha * target + (1.0 - alpha) * self._correction)
    # Rate-of-change limit -- see _CORRECTION_ROC_PER_TICK. Bounds the correction's own per-tick
    # delta on top of the exponential filter above, so a lane-line-confidence transition can't
    # snap the trim toward its new target in one or two frames.
    self._correction = float(np.clip(filtered, self._correction - _CORRECTION_ROC_PER_TICK,
                                      self._correction + _CORRECTION_ROC_PER_TICK))
    return kappa_cmd + self._correction

  @property
  def correction(self) -> float:
    """Telemetry: last applied correction (1/m)."""
    return self._correction

  def _raw_correction(self, model, v_ego: float, offset: float) -> tuple[bool, float]:
    try:
      pos_x = np.asarray(model.position.x, dtype=float)
      pos_y = np.asarray(model.position.y, dtype=float)
      if pos_x.size < 2 or pos_x.size != pos_y.size:
        return False, 0.0
      if not (np.isfinite(pos_x).all() and np.isfinite(pos_y).all() and np.all(np.diff(pos_x) > 0)):
        return False, 0.0

      lookahead = float(np.clip(v_ego, _LOOKAHEAD_MIN_M, _LOOKAHEAD_MAX_M))
      model_y = float(np.interp(lookahead, pos_x, pos_y))
    except (AttributeError, IndexError, TypeError, ValueError):
      return False, 0.0

    # Laneline contribution, blended in by confidence. scale=0 (model-position-only, i.e. just
    # the user's offset bias) whenever lines are missing, low-confidence, or structurally bad --
    # see _laneline_blend for the confidence formula (ported from lateral_curv_ext.py).
    scale, laneline_center_y = self._laneline_blend(model, lookahead)
    target_y = model_y * (1.0 - scale) + laneline_center_y * scale

    error = (target_y + offset) - model_y
    raw = 2.0 * error / (lookahead ** 2)
    return True, float(raw)

  def _laneline_blend(self, model, lookahead: float) -> tuple[float, float]:
    """Returns (scale, laneline_center_y). scale=0 whenever lanelines can't be trusted (missing,
    low-probability, structurally invalid) -- center_y is unused/meaningless in that case since
    it's weighted out by scale in the caller's blend."""
    try:
      lane_lines = model.laneLines
      probs = model.laneLineProbs
      stds = model.laneLineStds # a line could exist but this verifies how sure it is of where it is
      if len(lane_lines) < 3 or len(probs) < 3 or len(stds) < 3:
        return 0.0, 0.0

      left_x = np.asarray(lane_lines[1].x, dtype=float)
      left_y = np.asarray(lane_lines[1].y, dtype=float)
      right_x = np.asarray(lane_lines[2].x, dtype=float)
      right_y = np.asarray(lane_lines[2].y, dtype=float)
      if (left_x.size < 2 or left_x.size != left_y.size or
          right_x.size < 2 or right_x.size != right_y.size):
        return 0.0, 0.0
      if not (np.isfinite(left_x).all() and np.isfinite(left_y).all() and
              np.isfinite(right_x).all() and np.isfinite(right_y).all()):
        return 0.0, 0.0
      if not (np.all(np.diff(left_x) > 0) and np.all(np.diff(right_x) > 0)):
        return 0.0, 0.0

      left = float(np.interp(lookahead, left_x, left_y))
      right = float(np.interp(lookahead, right_x, right_y))
      width = right - left

      # Same confidence formula as lateral_curv_ext.py's path_offset blend: width-tolerance
      # (penalizes implausibly wide/merging-looking detections) combined with per-line
      # probability via min() -- a single missing/unreliable line (e.g. no line on the curb
      # side, only a center stripe) drags confidence toward 0 on its own.
      width_tolerance = float(np.interp(width, _WIDTH_TOLERANCE_BP, _WIDTH_TOLERANCE_V))
      # StarPilot stopped at 0.3; this fades the std through the table instead.
      std_tolerance = float(np.interp(max(float(stds[1]), float(stds[2])),
                                      _STD_TOLERANCE_BP, _STD_TOLERANCE_V))
      confidence = min(float(probs[1]), float(probs[2]), width_tolerance, std_tolerance) #confidence is the weakest signal
      scale = float(np.clip(np.interp(confidence, _CONFIDENCE_BP, _CONFIDENCE_V), 0.0, 1.0))
      center_y = 0.5 * (left + right)
      return scale, center_y
    except (AttributeError, IndexError, TypeError, ValueError):
      return 0.0, 0.0


def lane_center_trim_for(CP):
  """Returns a LaneCenterTrim when this CarParams asks for one, else None.

  Keeps the flag decoding with the rest of the Ford code; the caller only has to know
  whether it got an object back.
  """
  if CP.brand != "ford":
    return None
  if unpack_transit_lka_flags(CP.flags)[5] != TransitLaneCentering.ON:
    return None
  return LaneCenterTrim()

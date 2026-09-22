import copy
import re
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, IntFlag

from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, uds
from opendbc.car.carlog import carlog
from opendbc.car.lateral import AngleSteeringLimits
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column
from opendbc.car.fw_query_definitions import FwQueryConfig, LiveFwVersions, OfflineFwVersions, Request, StdQueries, p16

Ecu = CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 5        # LateralMotionControl, 20Hz
  LKA_STEP = 3          # Lane_Assist_Data1, 33Hz
  ACC_CONTROL_STEP = 2  # ACCDATA, 50Hz
  LKAS_UI_STEP = 100    # IPMA_Data, 1Hz
  ACC_UI_STEP = 20      # ACCDATA_3, 5Hz
  BUTTONS_STEP = 5      # Steering_Data_FD1, 10Hz, but send twice as fast

  # Driver intervention threshold, Nm. Upstream's 1.0 matches the stock PSCM; this van runs
  # modified PSCM firmware whose own override threshold is 1.5, so leaving this at 1.0 would
  # have openpilot call the driver "holding" while the PSCM is still steering.
  # Only reached through steeringPressed - panda has no driver-torque limit on Ford.
  STEER_DRIVER_ALLOWANCE = 1.5

  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    0.02,  # Max curvature for steering command, m^-1
    # Curvature rate limits
    # Max curvature is limited by the EPS to an equivalent of ~2.0 m/s^2 at all speeds,
    #  however max curvature rate linearly decreases as speed increases:
    #  ~0.009 m^-1/sec at 7 m/s, ~0.002 m^-1/sec at 35 m/s
    # Limit to ~2 m/s^3 up, ~3.3 m/s^3 down at 75 mph and match EPS limit at low speed
    ([5, 25], [0.00045, 0.0001]),
    ([5, 25], [0.00045, 0.00015])
  )
  CURVATURE_ERROR = 0.002  # ~6 degrees at 10 m/s, ~10 degrees at 35 m/s

  ACCEL_MAX = 2.0               # m/s^2 max acceleration
  ACCEL_MIN = -3.5              # m/s^2 max deceleration
  MIN_GAS = -0.5
  INACTIVE_GAS = -5.0

  def __init__(self, CP):
    pass


class FordSafetyFlags(IntFlag):
  LONG_CONTROL = 1
  CANFD = 2
  LKA_STEER = 4
  LKA_CONTINUATION = 8


class FordFlags(IntFlag):
  # Static flags
  CANFD = 1
  # Steers through Lane_Assist_Data1 (0x3CA) because the PSCM ignores LCA/TJA
  LKA_STEER = 2


# CarParams.flags bit layout for Ford. flags is upstream's UInt32 "flags for car specific
# quirks"; the three Transit LKA switches live in spare bits of it rather than in a new
# car.capnp field, so this fork claims no schema ordinal and cannot diverge from upstream's
# wire format. Bit packing costs readability, so the whole layout is spelled out here:
#
#   bit    0  FordFlags.CANFD
#   bit    1  FordFlags.LKA_STEER
#   bits 2-3  Transit LKA intervention   (TransitLkaIntervention,   0-2)
#   bits 4-5  Transit LKA ramp           (TransitLkaRamp,           0-2)
#   bits 6-8  reserved (held the direction sign and the availability gate; see below)
#   bit    9  Transit LKA continuation      (TransitLkaContinuation, 0-1)
#   bit   10  Transit lane centering        (TransitLaneCentering,   0-1)
#   bit   11  Transit human-turn override   (TransitHumanTurn,       0-1)
#   bits 12-31 free
#
# Any new FordFlags member must take a free bit from 12 upwards, never one of bits 2-11;
# test_ford.TestTransitLkaFlagPacking guards that.
TRANSIT_LKA_INTERVENTION_SHIFT = 2
TRANSIT_LKA_INTERVENTION_MASK = 0b11 << TRANSIT_LKA_INTERVENTION_SHIFT      # 0x0C
TRANSIT_LKA_RAMP_SHIFT = 4
TRANSIT_LKA_RAMP_MASK = 0b11 << TRANSIT_LKA_RAMP_SHIFT                      # 0x30
# Bits 6-8 are reserved, not free. Bit 6 was a direction-sign switch and bits 7-8 an
# availability gate; both were removed on 2026-09-18 after the drive that day measured
# them. The gate's PERMISSIVE and ANY positions commanded through the PSCM's own LKA
# suppression, which turned out to *cause* that suppression, so the accepted set is now
# fixed at TRANSIT_LKA_AVAIL_VALUES below. The direction sign never negated the request -
# it only chose which LkaActvStats_D2_Req pair rode along - and the PSCM follows the sign
# of LaRefAng_No_Req and ignores the code's left/right meaning, so the two positions were
# indistinguishable on the road (83.4% against 82.9% of commands followed). The bits stay
# reserved rather than being reused: the recorded routes decode by bit position, and
# keeping the hole leaves them comparable.
TRANSIT_LKA_RESERVED_MASK = 0b111 << 6                                      # 0x1C0
TRANSIT_LKA_CONTINUATION_SHIFT = 9
TRANSIT_LKA_CONTINUATION_MASK = 0b1 << TRANSIT_LKA_CONTINUATION_SHIFT       # 0x200
TRANSIT_LANE_CENTERING_SHIFT = 10
TRANSIT_LANE_CENTERING_MASK = 0b1 << TRANSIT_LANE_CENTERING_SHIFT           # 0x400
TRANSIT_HUMAN_TURN_SHIFT = 11
TRANSIT_HUMAN_TURN_MASK = 0b1 << TRANSIT_HUMAN_TURN_SHIFT                   # 0x800
TRANSIT_LKA_FLAGS_MASK = (TRANSIT_LKA_INTERVENTION_MASK | TRANSIT_LKA_RAMP_MASK |
                          TRANSIT_LKA_RESERVED_MASK |
                          TRANSIT_LKA_CONTINUATION_MASK |
                          TRANSIT_LANE_CENTERING_MASK | TRANSIT_HUMAN_TURN_MASK)


# All three switches are independent so each can be A/B tested on its own. The preset
# thresholds they select come from 46,577 recorded commanded frames across four routes;
# the state machine that applies them is TransitLkaState in ford/carcontroller.py.
class TransitLkaIntervention(IntEnum):
  STANDARD = 0
  INCREASING = 1
  PRESET = 2


class TransitLkaRamp(IntEnum):
  SLOW = 0
  FAST = 1
  PRESET = 2


class TransitLaneCentering(IntEnum):
  """Nudge the planner's curvature toward true lane-line centre before it becomes an angle.

  Ported from BluePilot; see ford/lane_center_trim.py. OFF reproduces today's behaviour
  exactly - the trim is not constructed at all.
  """
  OFF = 0
  ON = 1


class TransitHumanTurn(IntEnum):
  """Force lateral inactive while the driver holds a sustained turn.

  Ported from BluePilot; see ford/human_turn.py. OFF reproduces today's behaviour, where
  a sustained driver turn leaves the command running against the wheel.
  """
  OFF = 0
  ON = 1


class TransitLkaContinuation(IntEnum):
  """Keep steering after the PCM cancels cruise on the way down to a stop.

  The PCM leaves Active for Standby at about 17.8 km/h, panda drops controls_allowed
  and every command stops - lateral included, even though lateral never goes through
  the PCM at all. ON latches a continuation from that transition so Lane_Assist_Data1
  keeps flowing, which is the only way to exercise the PSCM below that speed.

  Lateral only. Longitudinal stays blocked: panda still gates ACCDATA on
  controls_allowed, and CarController forces it inactive while the latch is held, so
  the van coasts exactly as it does today. The failure this rules out is openpilot
  accelerating or braking while the driver's cruise reads Standby.
  """
  OFF = 0
  ON = 1


# Continuation latch thresholds, in m/s off BrakeSysFeatures.Veh_V_ActlBrk. Panda
# recomputes the same latch from the same signal (safety/modes/ford.h); both sides must
# agree or one commands while the other blocks, so these are duplicated there verbatim.
# Entry sits above the ~4.94 m/s cancel and well below normal driving; the exit band is
# wide enough that the two speed paths (openpilot filters, panda does not) cannot
# disagree. The low exit is under the 1.0 m/s the calibration claims, so it does not
# mask what this is meant to test.
TRANSIT_LKA_CONT_ENTER_SPEED = 7.0   # 25.2 km/h
TRANSIT_LKA_CONT_EXIT_SPEED_HIGH = 9.0   # 32.4 km/h
TRANSIT_LKA_CONT_EXIT_SPEED_LOW = 0.5    # 1.8 km/h
TRANSIT_LKA_CRUISE_STANDBY = 3           # CcStat_D_Actl


# LaActAvail_D_Actl values that count as "the PSCM is offering LKA": 3
# "LKA_LCA_LDW_Avail" and 2 "LCA_LKA_Avail_LDW_Suppress", which is what the signal's own
# VAL table says. 1 "LCA_LKA_Suppress_LDW_Avail" and 0 "LCA_LKA_LDW_Suppress" are the
# PSCM reporting that it has suppressed LKA, and commanding through them does not steer:
# over 7843 recorded commanded frames the wheel moved a median of 0.00 deg and followed
# the commanded direction 41.9% of the time, which is chance. Worse, it is self-
# sustaining - with commands sent only on 2/3 the PSCM kept offering LKA for 93.3% of
# engaged time, and on the routes that commanded through suppression that fell to 9-27%
# while their disengaged availability stayed at 83-96%.
TRANSIT_LKA_AVAIL_VALUES: tuple[int, ...] = (2, 3)


def _coerce_transit_lka_setting(enum_cls: type[IntEnum], value: int) -> IntEnum:
  """Coerce a Params-sourced switch to its enum, falling back to the default.

  These three switches ship with no UI, so hand-editing the params is the only way to
  use them and a typo is the expected interaction. A raised ValueError in
  CarController.__init__ kills card, which manager then restarts forever, leaving the
  device unusable until the param is corrected over SSH. Clamp instead, and say so.

  Member 0 of each enum is the shipped default, and is also what unset (all-zero) flag
  bits already decode to.
  """
  try:
    return enum_cls(value)
  except ValueError:
    default = enum_cls(0)
    carlog.warning("transit lka: %s=%s is out of range, falling back to %s", enum_cls.__name__, value, default.name)
    return default


def pack_transit_lka_flags(intervention: int, ramp: int, continuation: int = 0,
                           lane_centering: int = 0, human_turn: int = 0) -> int:
  """Pack the switches into their CarParams.flags bits (layout above).

  Clamping happens here as well as on unpack because the single-bit switches would
  otherwise let a typo'd param survive the round trip looking valid. All-default settings
  pack to 0, so ORing the result into a non-Ford CarParams.flags is a no-op.
  """
  return ((int(_coerce_transit_lka_setting(TransitLkaIntervention, intervention)) << TRANSIT_LKA_INTERVENTION_SHIFT) |
          (int(_coerce_transit_lka_setting(TransitLkaRamp, ramp)) << TRANSIT_LKA_RAMP_SHIFT) |
          (int(_coerce_transit_lka_setting(TransitLkaContinuation, continuation)) << TRANSIT_LKA_CONTINUATION_SHIFT) |
          (int(_coerce_transit_lka_setting(TransitLaneCentering, lane_centering)) << TRANSIT_LANE_CENTERING_SHIFT) |
          (int(_coerce_transit_lka_setting(TransitHumanTurn, human_turn)) << TRANSIT_HUMAN_TURN_SHIFT))


def unpack_transit_lka_flags(flags: int) -> tuple[TransitLkaIntervention, TransitLkaRamp,
                                                  TransitLkaContinuation, TransitLaneCentering,
                                                  TransitHumanTurn]:
  """Unpack the switches from CarParams.flags (layout above).

  The two-bit fields can still hold 3, which is outside both TransitLkaIntervention and
  TransitLkaRamp, so the clamp runs here too rather than trusting whatever produced the
  flags. Bits 6-8 are skipped: they are reserved, and a route recorded before the
  direction sign and the availability gate were removed can still have them set.
  """
  return (_coerce_transit_lka_setting(TransitLkaIntervention, (flags & TRANSIT_LKA_INTERVENTION_MASK) >> TRANSIT_LKA_INTERVENTION_SHIFT),
          _coerce_transit_lka_setting(TransitLkaRamp, (flags & TRANSIT_LKA_RAMP_MASK) >> TRANSIT_LKA_RAMP_SHIFT),
          _coerce_transit_lka_setting(TransitLkaContinuation, (flags & TRANSIT_LKA_CONTINUATION_MASK) >> TRANSIT_LKA_CONTINUATION_SHIFT),
          _coerce_transit_lka_setting(TransitLaneCentering, (flags & TRANSIT_LANE_CENTERING_MASK) >> TRANSIT_LANE_CENTERING_SHIFT),
          _coerce_transit_lka_setting(TransitHumanTurn, (flags & TRANSIT_HUMAN_TURN_MASK) >> TRANSIT_HUMAN_TURN_SHIFT))


class RADAR:
  DELPHI_ESR = 'ford_fusion_2018_adas'
  DELPHI_MRR = 'FORD_CADS'


class Footnote(Enum):
  FOCUS = CarFootnote(
    "Refers only to the Focus Mk4 (C519) available in Europe/China/Taiwan/Australasia, not the Focus Mk3 (C346) in " +
    "North and South America/Southeast Asia.",
    Column.MODEL,
  )


@dataclass
class FordCarDocs(CarDocs):
  package: str = "Co-Pilot360 Assist+"
  hybrid: bool = False
  plug_in_hybrid: bool = False

  def init_make(self, CP: CarParams):
    harness = CarHarness.ford_q4 if CP.flags & FordFlags.CANFD else CarHarness.ford_q3
    self.car_parts = CarParts.common([harness])

    if harness == CarHarness.ford_q4:
      self.setup_video = "https://www.youtube.com/watch?v=uUGkH6C_EQU"

    if CP.carFingerprint in (CAR.FORD_F_150_MK14, CAR.FORD_F_150_LIGHTNING_MK1, CAR.FORD_EXPEDITION_MK4):
      self.setup_video = "https://www.youtube.com/watch?v=MewJc9LYp9M"


@dataclass
class FordPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'ford_lincoln_base_pt',
    Bus.radar: RADAR.DELPHI_MRR,
  })

  def init(self):
    for car_docs in list(self.car_docs):
      if car_docs.hybrid:
        name = f"{car_docs.make} {car_docs.model} Hybrid {car_docs.years}"
        self.car_docs.append(replace(copy.deepcopy(car_docs), name=name))
      if car_docs.plug_in_hybrid:
        name = f"{car_docs.make} {car_docs.model} Plug-in Hybrid {car_docs.years}"
        self.car_docs.append(replace(copy.deepcopy(car_docs), name=name))


@dataclass
class FordCANFDPlatformConfig(FordPlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'ford_lincoln_base_pt',
  })

  def init(self):
    super().init()
    self.flags |= FordFlags.CANFD


@dataclass
class FordLkaPlatformConfig(FordPlatformConfig):
  def init(self):
    super().init()
    self.flags |= FordFlags.LKA_STEER


@dataclass
class FordF150LightningPlatform(FordCANFDPlatformConfig):
  def init(self):
    super().init()

    # Don't show in docs until this issue is resolved. See https://github.com/commaai/openpilot/issues/30302
    self.car_docs = []


class CAR(Platforms):
  FORD_BRONCO_SPORT_MK1 = FordPlatformConfig(
    [FordCarDocs("Ford Bronco Sport 2021-24")],
    CarSpecs(mass=1625, wheelbase=2.67, steerRatio=17.7),
  )
  FORD_ESCAPE_MK4 = FordPlatformConfig(
    [
      FordCarDocs("Ford Escape 2020-22", hybrid=True, plug_in_hybrid=True),
      FordCarDocs("Ford Kuga 2020-23", "Adaptive Cruise Control with Lane Centering", hybrid=True, plug_in_hybrid=True),
    ],
    CarSpecs(mass=1750, wheelbase=2.71, steerRatio=16.7),
  )
  FORD_ESCAPE_MK4_5 = FordCANFDPlatformConfig(
    [
      FordCarDocs("Ford Escape 2023-24", hybrid=True, plug_in_hybrid=True, setup_video="https://www.youtube.com/watch?v=M6uXf4b2SHM"),
      FordCarDocs("Ford Kuga Hybrid 2024", "All"),
      FordCarDocs("Ford Kuga Plug-in Hybrid 2024", "All"),
    ],
    CarSpecs(mass=1750, wheelbase=2.71, steerRatio=16.7),
  )
  FORD_EXPLORER_MK6 = FordPlatformConfig(
    [
      FordCarDocs("Ford Explorer 2020-24", hybrid=True),  # Hybrid: Limited and Platinum only
      FordCarDocs("Lincoln Aviator 2020-24", "Co-Pilot360 Plus", plug_in_hybrid=True),  # Hybrid: Grand Touring only
    ],
    CarSpecs(mass=2050, wheelbase=3.025, steerRatio=16.8),
  )
  FORD_EXPEDITION_MK4 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford Expedition 2022-24", "Co-Pilot360 Assist 2.0", hybrid=False)],
    CarSpecs(mass=2000, wheelbase=3.69, steerRatio=17.0),
  )
  FORD_F_150_MK14 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford F-150 2021-23", "Co-Pilot360 Assist 2.0", hybrid=True)],
    CarSpecs(mass=2000, wheelbase=3.69, steerRatio=17.0),
  )
  FORD_F_150_LIGHTNING_MK1 = FordF150LightningPlatform(
    [FordCarDocs("Ford F-150 Lightning 2022-23", "Co-Pilot360 Assist 2.0")],
    CarSpecs(mass=2948, wheelbase=3.70, steerRatio=16.9),
  )
  FORD_FOCUS_MK4 = FordPlatformConfig(
    [FordCarDocs("Ford Focus 2018", "Adaptive Cruise Control with Lane Centering", footnotes=[Footnote.FOCUS], hybrid=True)],  # mHEV only
    CarSpecs(mass=1350, wheelbase=2.7, steerRatio=15.0),
  )
  FORD_MAVERICK_MK1 = FordPlatformConfig(
    [
      FordCarDocs("Ford Maverick 2022", "LARIAT Luxury", hybrid=True),
      FordCarDocs("Ford Maverick 2023-24", "Co-Pilot360 Assist", hybrid=True),
    ],
    CarSpecs(mass=1650, wheelbase=3.076, steerRatio=17.0),
  )
  FORD_MUSTANG_MACH_E_MK1 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford Mustang Mach-E 2021-24", "All", setup_video="https://www.youtube.com/watch?v=AR4_eTF3b_A")],
    CarSpecs(mass=2200, wheelbase=2.984, steerRatio=17.0),  # TODO: check steer ratio
  )
  FORD_RANGER_MK2 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford Ranger 2024", "Adaptive Cruise Control with Lane Centering", setup_video="https://www.youtube.com/watch?v=2oJlXCKYOy0")],
    CarSpecs(mass=2000, wheelbase=3.27, steerRatio=17.0),
  )
  FORD_TRANSIT_MK5 = FordLkaPlatformConfig(
    [FordCarDocs("Ford Transit 2022-23", "Lane Keeping Aid")],
    # T-350 AWD cargo van, long wheelbase high roof (VIN decode via NHTSA vPIC:
    # 2022 Transit 350, Cargo Van, AWD, 3.5 L V6, GVWR class 2H).
    # mass: 2864. Derivation, because this is NOT the curb weight and the
    #   convention in interfaces.py says CarSpecs.mass should be:
    #     owner-reported curb weight            2500 kg
    #     owner-reported weight as actually     3000 kg  (tools and cargo aboard)
    #       driven
    #     openpilot adds STD_CARGO_KG = 136 kg at interfaces.py:148
    #     so CarSpecs.mass = 3000 - 136        = 2864 kg  -> 3000 kg effective
    #   Using the convention literally would give CarSpecs.mass = 2500 and an
    #   effective 2636 kg, which is 12% under how the van is actually driven.
    #   Mass feeds rotationalInertia and the tire-stiffness scaling; it does NOT
    #   affect slip_factor, where it cancels.
    # wheelbase: owner-confirmed 148 in.
    # steerRatio: 20.9, from a least-squares fit of yaw-rate-derived curvature
    #   against steering angle over 11,658 recorded samples at 36-86 km/h
    #   (r = 0.967). The fit pins steerRatio * wheelbase = 78.43 m; the wheelbase
    #   above splits it. paramsd's own live estimates on three routes were
    #   17.87 / 19.21 / 22.97, which bracket this.
    CarSpecs(mass=2864, wheelbase=3.750, steerRatio=20.9),
  )


# FW response contains a combined software and part number
# A-Z except no I, O or W
# e.g. NZ6A-14C204-AAA
#      1222-333333-444
# 1 = Model year hint (approximates model year/generation)
# 2 = Platform hint
# 3 = Part number
# 4 = Software version
FW_ALPHABET = b'A-HJ-NP-VX-Z'
FW_PATTERN = re.compile(b'^(?P<model_year_hint>[' + FW_ALPHABET + b'])' +
                        b'(?P<platform_hint>[0-9' + FW_ALPHABET + b']{3})-' +
                        b'(?P<part_number>[0-9' + FW_ALPHABET + b']{5,6})-' +
                        b'(?P<software_revision>[' + FW_ALPHABET + b']{2,})\x00*$')


def get_platform_codes(fw_versions: list[bytes] | set[bytes]) -> set[tuple[bytes, bytes]]:
  codes = set()
  for fw in fw_versions:
    match = FW_PATTERN.match(fw)
    if match is not None:
      codes.add((match.group('platform_hint'), match.group('model_year_hint')))

  return codes


def match_fw_to_car_fuzzy(live_fw_versions: LiveFwVersions, vin: str, offline_fw_versions: OfflineFwVersions) -> set[str]:
  candidates: set[str] = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform hint, within model year hint range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws if ecu[0] in PLATFORM_CODE_ECUS}
    for ecu, expected_versions in fws.items():
      addr = ecu[1:]
      # Only check ECUs expected to have platform codes
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      # Expected platform codes & model year hints
      codes = get_platform_codes(expected_versions)
      expected_platform_codes = {code for code, _ in codes}
      expected_model_year_hints = {model_year_hint for _, model_year_hint in codes}

      # Found platform codes & model year hints
      codes = get_platform_codes(live_fw_versions.get(addr, set()))
      found_platform_codes = {code for code, _ in codes}
      found_model_year_hints = {model_year_hint for _, model_year_hint in codes}

      # Check platform code matches for any found versions
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      # Check any model year hint within range in the database. Note that some models have more than one
      # platform code per ECU which we don't consider as separate ranges
      if not any(min(expected_model_year_hints) <= found_model_year_hint <= max(expected_model_year_hints) for
                 found_model_year_hint in found_model_year_hints):
        break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return candidates


# All of these ECUs must be present and are expected to have platform codes we can match
PLATFORM_CODE_ECUS = (Ecu.abs, Ecu.fwdCamera, Ecu.fwdRadar, Ecu.eps)

DATA_IDENTIFIER_FORD_ASBUILT = 0xDE00

ASBUILT_BLOCKS: list[tuple[int, list]] = [
  (1, [Ecu.debug, Ecu.fwdCamera, Ecu.eps]),
  (2, [Ecu.abs, Ecu.debug, Ecu.eps]),
  (3, [Ecu.abs, Ecu.debug, Ecu.eps]),
  (4, [Ecu.debug, Ecu.fwdCamera]),
  (5, [Ecu.debug]),
  (6, [Ecu.debug]),
  (7, [Ecu.debug]),
  (8, [Ecu.debug]),
  (9, [Ecu.debug]),
  (16, [Ecu.debug, Ecu.fwdCamera]),
  (18, [Ecu.fwdCamera]),
  (20, [Ecu.fwdCamera]),
  (21, [Ecu.fwdCamera]),
]


def ford_asbuilt_block_request(block_id: int):
  return bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + p16(DATA_IDENTIFIER_FORD_ASBUILT + block_id - 1)


def ford_asbuilt_block_response(block_id: int):
  return bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + p16(DATA_IDENTIFIER_FORD_ASBUILT + block_id - 1)


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # CAN and CAN FD queries are combined.
    # FIXME: For CAN FD, ECUs respond with frames larger than 8 bytes on the powertrain bus
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.debug, Ecu.engine, Ecu.eps, Ecu.fwdCamera, Ecu.fwdRadar, Ecu.shiftByWire],
      logging=True,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.debug, Ecu.engine, Ecu.eps, Ecu.fwdCamera, Ecu.fwdRadar, Ecu.shiftByWire],
      bus=0,
    ),
    *[Request(
      [StdQueries.TESTER_PRESENT_REQUEST, ford_asbuilt_block_request(block_id)],
      [StdQueries.TESTER_PRESENT_RESPONSE, ford_asbuilt_block_response(block_id)],
      whitelist_ecus=ecus,
      bus=0,
      logging=True,
    ) for block_id, ecus in ASBUILT_BLOCKS],
  ],
  extra_ecus=[
    (Ecu.engine, 0x7e0, None),        # Powertrain Control Module
                                      # Note: We are unlikely to get a response from behind the gateway
    (Ecu.shiftByWire, 0x732, None),   # Gear Shift Module
    (Ecu.debug, 0x7d0, None),         # Accessory Protocol Interface Module
  ],
  # Custom fuzzy fingerprinting function using platform and model year hints
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

DBC = CAR.create_dbc_map()

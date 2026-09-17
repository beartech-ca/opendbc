#pragma once

#include "opendbc/safety/declarations.h"

// Safety-relevant CAN messages for Ford vehicles.
#define FORD_EngBrakeData          0x165U   // RX from PCM, for driver brake pedal and cruise state
#define FORD_EngVehicleSpThrottle  0x204U   // RX from PCM, for driver throttle input
#define FORD_DesiredTorqBrk        0x213U   // RX from ABS, for standstill state
#define FORD_BrakeSysFeatures      0x415U   // RX from ABS, for vehicle speed
#define FORD_EngVehicleSpThrottle2 0x202U   // RX from PCM, for second vehicle speed
#define FORD_Yaw_Data_FD1          0x91U    // RX from RCM, for yaw rate
#define FORD_SteeringPinion_Data   0x07EU   // RX from PSCM, measured steering pinion angle
#define FORD_Steering_Data_FD1     0x083U   // TX by OP, various driver switches and LKAS/CC buttons
#define FORD_ACCDATA               0x186U   // TX by OP, ACC controls
#define FORD_ACCDATA_3             0x18AU   // TX by OP, ACC/TJA user interface
#define FORD_Lane_Assist_Data1     0x3CAU   // TX by OP, Lane Keep Assist
#define FORD_LateralMotionControl  0x3D3U   // TX by OP, Lateral Control message
#define FORD_LateralMotionControl2 0x3D6U   // TX by OP, alternate Lateral Control message
#define FORD_IPMA_Data             0x3D8U   // TX by OP, IPMA and LKAS user interface

// CAN bus numbers.
#define FORD_MAIN_BUS 0U
#define FORD_CAM_BUS  2U

// Set for LKA_STEER platforms, which steer through Lane_Assist_Data1 (the PSCM
// ignores LCA/TJA on these) instead of the LCA/TJA curvature channel.
static bool ford_lka_steer = false;

// Lateral continuation. The PCM leaves CcStat_D_Actl Active for Standby at about
// 4.94 m/s on the way to a stop, pcm_cruise_check drops controls_allowed, and every
// command stops - lateral included, although lateral is a PSCM channel the PCM has no
// part in. When enabled, a latch taken on that exact transition keeps
// Lane_Assist_Data1 permitted below it, and nothing else: ACCDATA, the cruise buttons
// and LateralMotionControl all keep gating on controls_allowed alone, so openpilot
// cannot accelerate or brake while the driver's cruise reads Standby.
//
// The latch is recomputed here rather than trusted from openpilot, and CarState runs
// the identical conditions off the identical signals (opendbc/car/ford/carstate.py and
// the TRANSIT_LKA_CONT_* constants in ford/values.py). The two must stay in step: if
// openpilot latches where panda does not, every steering frame becomes a TX violation.
static bool ford_lka_continuation_enabled = false;
static bool ford_lka_continuation = false;
// m/s, off Veh_V_ActlBrk. Entry sits above the cancel and below normal driving; the
// exit band is wide enough to absorb openpilot filtering the same signal. The low exit
// is under the 1.0 m/s the PSCM calibration claims, so it cannot mask that.
#define FORD_LKA_CONT_ENTER_SPEED     7.0f
#define FORD_LKA_CONT_EXIT_SPEED_HIGH 9.0f
#define FORD_LKA_CONT_EXIT_SPEED_LOW  0.5f
#define FORD_CRUISE_STANDBY           3U

static uint8_t ford_get_counter(const CANPacket_t *msg) {
  uint8_t cnt = 0;
  if (msg->addr == FORD_BrakeSysFeatures) {
    // Signal: VehVActlBrk_No_Cnt
    cnt = (msg->data[2] >> 2) & 0xFU;
  } else if (msg->addr == FORD_Yaw_Data_FD1) {
    // Signal: VehRollYaw_No_Cnt
    cnt = msg->data[5];
  } else {
  }
  return cnt;
}

static uint32_t ford_get_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (msg->addr == FORD_BrakeSysFeatures) {
    // Signal: VehVActlBrk_No_Cs
    chksum = msg->data[3];
  } else if (msg->addr == FORD_Yaw_Data_FD1) {
    // Signal: VehRollYawW_No_Cs
    chksum = msg->data[4];
  } else {
  }
  return chksum;
}

static uint32_t ford_compute_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (msg->addr == FORD_BrakeSysFeatures) {
    chksum += msg->data[0] + msg->data[1];  // Veh_V_ActlBrk
    chksum += msg->data[2] >> 6;                    // VehVActlBrk_D_Qf
    chksum += (msg->data[2] >> 2) & 0xFU;           // VehVActlBrk_No_Cnt
    chksum = 0xFFU - chksum;
  } else if (msg->addr == FORD_Yaw_Data_FD1) {
    chksum += msg->data[0] + msg->data[1];  // VehRol_W_Actl
    chksum += msg->data[2] + msg->data[3];  // VehYaw_W_Actl
    chksum += msg->data[5];                         // VehRollYaw_No_Cnt
    chksum += msg->data[6] >> 6;                    // VehRolWActl_D_Qf
    chksum += (msg->data[6] >> 4) & 0x3U;           // VehYawWActl_D_Qf
    chksum = 0xFFU - chksum;
  } else {
  }
  return chksum;
}

static bool ford_get_quality_flag_valid(const CANPacket_t *msg) {
  bool valid = false;
  if (msg->addr == FORD_BrakeSysFeatures) {
    valid = (msg->data[2] >> 6) == 0x3U;           // VehVActlBrk_D_Qf
  } else if (msg->addr == FORD_EngVehicleSpThrottle2) {
    valid = ((msg->data[4] >> 5) & 0x3U) == 0x3U;  // VehVActlEng_D_Qf
  } else if (msg->addr == FORD_Yaw_Data_FD1) {
    valid = ((msg->data[6] >> 4) & 0x3U) == 0x3U;  // VehYawWActl_D_Qf
  } else if (msg->addr == FORD_SteeringPinion_Data) {
    valid = ((msg->data[5] >> 2) & 0x3U) == 0x3U;  // StePinCompAnEst_D_Qf, 3 = OK
  } else {
  }
  return valid;
}

#define FORD_INACTIVE_CURVATURE 1000U
#define FORD_INACTIVE_CURVATURE_RATE 4096U
#define FORD_INACTIVE_PATH_OFFSET 512U
#define FORD_INACTIVE_PATH_ANGLE 1000U

#define FORD_CANFD_INACTIVE_CURVATURE_RATE 1024U

// Curvature rate limits
#define FORD_LIMITS(limit_lateral_acceleration) {                                               \
  .max_angle = 1000,          /* 0.02 curvature */                                              \
  .angle_deg_to_can = 50000,  /* 1 / (2e-5) rad to can */                                       \
  .max_angle_error = 100,     /* 0.002 * FORD_STEERING_LIMITS.angle_deg_to_can */               \
  .angle_rate_up_lookup = {                                                                     \
    {5., 25., 25.},                                                                             \
    {0.00045, 0.0001, 0.0001}                                                                   \
  },                                                                                            \
  .angle_rate_down_lookup = {                                                                   \
    {5., 25., 25.},                                                                             \
    {0.00045, 0.00015, 0.00015}                                                                 \
  },                                                                                            \
                                                                                                \
  /* no blending at low speed due to lack of torque wind-up and inaccurate current curvature */ \
  .angle_error_min_speed = 10.0,    /* m/s */                                                   \
                                                                                                \
  .angle_is_curvature = (limit_lateral_acceleration),                                           \
  .enforce_angle_error = true,                                                                  \
  .inactive_angle_is_zero = true,                                                               \
}

static const AngleSteeringLimits FORD_STEERING_LIMITS = FORD_LIMITS(false);

static void ford_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == FORD_MAIN_BUS) {
    // Update in motion state from standstill signal
    if (msg->addr == FORD_DesiredTorqBrk) {
      // Signal: VehStop_D_Stat
      vehicle_moving = ((msg->data[3] >> 3) & 0x3U) != 1U;
    }

    // Update vehicle speed
    if (msg->addr == FORD_BrakeSysFeatures) {
      // Signal: Veh_V_ActlBrk
      UPDATE_VEHICLE_SPEED(((msg->data[0] << 8) | msg->data[1]) * 0.01 * KPH_TO_MS);
    }

    // Check vehicle speed against a second source
    if (msg->addr == FORD_EngVehicleSpThrottle2) {
      // Disable controls if speeds from ABS and PCM ECUs are too far apart.
      // Signal: Veh_V_ActlEng
      float filtered_pcm_speed = ((msg->data[6] << 8) | msg->data[7]) * 0.01 * KPH_TO_MS;
      speed_mismatch_check(filtered_pcm_speed);
    }

    // Update vehicle yaw rate
    // LKA_STEER platforms steer through the pinion angle instead (see FORD_SteeringPinion_Data
    // below); angle_meas is shared state and must not be fed two incompatible unit scales.
    if ((msg->addr == FORD_Yaw_Data_FD1) && !ford_lka_steer) {
      // Signal: VehYaw_W_Actl
      // TODO: we should use the speed which results in the closest angle measurement to the desired angle
      float ford_yaw_rate = (((msg->data[2] << 8U) | msg->data[3]) * 0.0002) - 6.5;
      float current_curvature = ford_yaw_rate / SAFETY_MAX(vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR, 0.1);
      // convert current curvature into units on CAN for comparison with desired curvature
      update_sample(&angle_meas, ROUND(current_curvature * FORD_STEERING_LIMITS.angle_deg_to_can));
    }

    // Update measured steering pinion angle, used by LKA_STEER platforms to bound
    // Lane_Assist_Data1 angle requests (see ford_tx_hook)
    if ((msg->addr == FORD_SteeringPinion_Data) && ford_lka_steer) {
      // Signal: StePinComp_An_Est : 22|15@0+ (0.1,-1600) degrees
      // 15 bits, 0.1 deg/bit, -1600 deg offset -> tenths of a degree, matching
      // FORD_LKA_DEG_TO_CAN in ford_tx_hook
      const int pinion_angle = (((msg->data[2] & 0x7FU) << 8) | msg->data[3]) - 16000U;
      update_sample(&angle_meas, pinion_angle);
    }

    // Update gas pedal
    if (msg->addr == FORD_EngVehicleSpThrottle) {
      // Pedal position: (0.1 * val) in percent
      // Signal: ApedPos_Pc_ActlArb
      gas_pressed = (((msg->data[0] & 0x03U) << 8) | msg->data[1]) > 0U;
    }

    // Update brake pedal and cruise state
    if (msg->addr == FORD_EngBrakeData) {
      // Signal: BpedDrvAppl_D_Actl
      brake_pressed = ((msg->data[0] >> 4) & 0x3U) == 2U;

      // Signal: CcStat_D_Actl
      unsigned int cruise_state = msg->data[1] & 0x07U;
      bool cruise_engaged = (cruise_state == 4U) || (cruise_state == 5U);

      // Lateral continuation latch. Evaluated before pcm_cruise_check, which is what
      // consumes and then overwrites cruise_engaged_prev: entry needs the state from
      // the frame before, so that the latch can only be taken on the Active -> Standby
      // transition and never from a standing start in Standby (cruise switched on but
      // never set). Speed is the unfiltered Veh_V_ActlBrk sample, matching CarState.
      const float ford_speed = ((float)vehicle_speed.values[0]) / VEHICLE_SPEED_FACTOR;
      const bool ford_standby = cruise_state == FORD_CRUISE_STANDBY;
      if (!ford_lka_continuation) {
        ford_lka_continuation = ford_lka_continuation_enabled && cruise_engaged_prev && ford_standby &&
                                (ford_speed < FORD_LKA_CONT_ENTER_SPEED) && !brake_pressed;
      } else {
        ford_lka_continuation = ford_standby && !brake_pressed &&
                                (ford_speed > FORD_LKA_CONT_EXIT_SPEED_LOW) &&
                                (ford_speed < FORD_LKA_CONT_EXIT_SPEED_HIGH);
      }

      pcm_cruise_check(cruise_engaged);
    }
  }
}

static bool ford_tx_hook(const CANPacket_t *msg) {
  const LongitudinalLimits FORD_LONG_LIMITS = {
    // acceleration cmd limits (used for brakes)
    // Signal: AccBrkTot_A_Rq
    .max_accel = 5641,       //  1.9999 m/s^s
    .min_accel = 4231,       // -3.4991 m/s^2
    .inactive_accel = 5128,  // -0.0008 m/s^2

    // gas cmd limits
    // Signal: AccPrpl_A_Rq & AccPrpl_A_Pred
    .max_gas = 700,          //  2.0 m/s^2
    .min_gas = 450,          // -0.5 m/s^2
    .inactive_gas = 0,       // -5.0 m/s^2
  };

  bool tx = true;

  // Safety check for ACCDATA accel and brake requests
  if (msg->addr == FORD_ACCDATA) {
    // Signal: AccPrpl_A_Rq
    int gas = ((msg->data[6] & 0x3U) << 8) | msg->data[7];
    // Signal: AccPrpl_A_Pred
    int gas_pred = ((msg->data[2] & 0x3U) << 8) | msg->data[3];
    // Signal: AccBrkTot_A_Rq
    int accel = ((msg->data[0] & 0x1FU) << 8) | msg->data[1];
    // Signal: CmbbDeny_B_Actl
    bool cmbb_deny = (msg->data[4] >> 5) & 1U;

    // Signal: AccBrkPrchg_B_Rq & AccBrkDecel_B_Rq
    bool brake_actuation = ((msg->data[6] >> 6) & 1U) || ((msg->data[6] >> 7) & 1U);

    bool violation = false;
    violation |= longitudinal_accel_checks(accel, FORD_LONG_LIMITS);
    violation |= longitudinal_gas_checks(gas, FORD_LONG_LIMITS);
    violation |= longitudinal_gas_checks(gas_pred, FORD_LONG_LIMITS);

    // Safety check for stock AEB
    violation |= cmbb_deny; // do not prevent stock AEB actuation

    violation |= !get_longitudinal_allowed() && brake_actuation;

    if (violation) {
      tx = false;
    }
  }

  // Safety check for Steering_Data_FD1 button signals
  // Note: Many other signals in this message are not relevant to safety (e.g. blinkers, wiper switches, high beam)
  // which we passthru in OP.
  if (msg->addr == FORD_Steering_Data_FD1) {
    // Violation if resume button is pressed while controls not allowed, or
    // if cancel button is pressed when cruise isn't engaged.
    bool violation = false;
    violation |= ((msg->data[1] >> 0) & 1U) && !cruise_engaged_prev;   // Signal: CcAslButtnCnclPress (cancel)
    violation |= ((msg->data[3] >> 1) & 1U) && !controls_allowed;     // Signal: CcAsllButtnResPress (resume)

    if (violation) {
      tx = false;
    }
  }

  // Safety check for Lane_Assist_Data1 action
  if (msg->addr == FORD_Lane_Assist_Data1) {
    // Signal: LkaActvStats_D2_Req : 7|3@0+ (1,0), i.e. the top 3 bits of byte 0
    unsigned int action = msg->data[0] >> 5;

    if (!ford_lka_steer) {
      // Do not allow steering using Lane_Assist_Data1 (Lane-Departure Aid) on
      // upstream platforms, which steer through LCA/TJA instead. This message
      // must still be sent for Lane Centering to work, and can include values
      // such as the steering angle or lane curvature for debugging, but the
      // action (LkaActvStats_D2_Req) must be set to zero.
      if (action != 0U) {
        tx = false;
      }
    } else {
      // LaRefAng_No_Req is not an absolute position target that openpilot ramps, which is
      // what steer_angle_cmd_checks_vm's per-frame jerk term assumes. It is a RELATIVE
      // correction carrying the whole remaining error, handed to the PSCM, which applies it
      // over its own internal ramp. Rate-limiting the request would measure intent rather
      // than motion, so this platform gets its own check, without the per-frame jerk term:
      // the magnitude of the relative request, the ISO lateral acceleration of the absolute
      // target it reconstructs to, and the real-time cap on how many commands may be sent
      // per interval. There is deliberately no per-frame rate-of-change bound.
      const AngleSteeringParams FORD_LKA_STEERING_PARAMS = {
        .slip_factor = -0.0004472752575630534f,  // calc_slip_factor(VM) for FORD_TRANSIT_MK5
        .steer_ratio = 20.9,
        .wheelbase = 3.75,
      };
      // tenths of a degree, matching the StePinComp_An_Est angle_meas sample
      const float FORD_LKA_DEG_TO_CAN = 10.0f;
      // LaRefAng_No_Req is 12 bits at 0.05 mrad/bit with a -102.4 mrad offset, so a
      // correctly extracted request can only span -5.867..+5.864 deg: +/-5.9 deg, i.e.
      // +/-59, once rounded to tenths of a degree.
      const int FORD_LKA_MAX_REL_ANGLE = 59;
      // Same value steer_angle_cmd_checks_vm uses: highway curves are rolled in the
      // direction of the turn, so the ISO limit gets a superelevation tolerance
      const float FORD_LKA_MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL + (EARTH_G * AVERAGE_ROAD_ROLL);  // ~3.6 m/s^2
      // rt_angle_rate_limit_check reads only .frequency, and touches no limiter state
      // other than rt_angle_msgs/ts_angle_check_last, so it does not reintroduce the
      // desired_angle_last coupling the per-frame jerk term had.
      const AngleSteeringLimits FORD_LKA_RT_LIMITS = {
        .frequency = 33U,  // Lane_Assist_Data1 is sent at 33Hz
      };

      // Only the four intervention requests actuate steering: 1/6 increasing left/right,
      // 2/4 standard left/right. 0 is idle and 3/5 suppress LKA, neither of which steers,
      // and 7 is NotUsed. Whitelist the values we understand rather than guess what the
      // PSCM does with a reserved one (see tesla_tx_hook's steer control type check).
      bool steer_control_enabled = (action == 1U) || (action == 2U) || (action == 4U) || (action == 6U);
      bool valid_action = (action == 0U) || steer_control_enabled;

      // Signal: LaRefAng_No_Req : 19|12@0+ (0.05,-102.4) mrad, i.e. bits [19:8] of the
      // message. It is RELATIVE to the current pinion angle.
      unsigned int raw_rel = ((msg->data[2] & 0x0FU) << 8) | msg->data[3];
      float rel_mrad = ((float)raw_rel * 0.05f) - 102.4f;
      // mrad -> tenths of a degree: (rel_mrad / 1000 rad) * (180/pi deg/rad) * 10 tenths/deg
      int rel_tenths = ROUND(rel_mrad * (1.8f / 3.14159265f));

      // The PSCM ignores the requested angle when the action is idle, and openpilot's
      // heartbeat frame is an all-zero payload, which decodes to the bottom of the signal
      // range (-102.4 mrad), not to zero. Zero the relative term so an idle frame carries
      // no steering request at all.
      if (!steer_control_enabled) {
        rel_tenths = 0;
      }

      // The only place the continuation latch is consulted. Every other branch of this
      // hook keeps gating on controls_allowed alone, which is what confines the latch
      // to lateral: it can never permit an ACCDATA or cruise-button transmission.
      const bool lat_allowed = controls_allowed || ford_lka_continuation;

      bool violation = false;

      if (lat_allowed && steer_control_enabled) {
        // *** relative request magnitude limit ***
        // The signal encoding already bounds this on the car side; the bound is repeated
        // here to catch panda's own bit extraction of LaRefAng_No_Req being wrong, a class
        // of defect this port has hit twice. It guards the extraction, not the car.
        violation |= safety_max_limit_check(rel_tenths, FORD_LKA_MAX_REL_ANGLE, -FORD_LKA_MAX_REL_ANGLE);

        // *** ISO lateral accel limit on the reconstructed absolute target ***
        // Identical in effect to steer_angle_cmd_checks_vm's lateral acceleration term:
        // same fudged speed, same vehicle model helpers, same ceiling. This is what stops
        // a sustained maximum relative request from winding the wheel up indefinitely.
        const int desired_angle = angle_meas.values[0] + rel_tenths;
        const float fudged_speed = SAFETY_MAX((vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.0, 1.0);
        const float curvature_factor = get_curvature_factor(fudged_speed, FORD_LKA_STEERING_PARAMS);
        const float max_curvature = FORD_LKA_MAX_LATERAL_ACCEL / (fudged_speed * fudged_speed);
        const float max_angle = get_angle_from_curvature(max_curvature, curvature_factor, FORD_LKA_STEERING_PARAMS);
        const int max_angle_can = (int)((max_angle * FORD_LKA_DEG_TO_CAN) + 1.0f);

        violation |= safety_max_limit_check(desired_angle, max_angle_can, -max_angle_can);

        // *** angle real time rate limit check ***
        // The two bounds above constrain each frame in isolation. The argument that the
        // PSCM's own ramp bounds wheel motion was measured against a 33Hz command stream,
        // so a fault emitting 0x3CA far faster would hand the PSCM many times the
        // correction per unit time with every individual frame still inside both.
        violation |= rt_angle_rate_limit_check(FORD_LKA_RT_LIMITS);
      }

      // No steering request allowed when lateral control is not allowed
      violation |= !lat_allowed && steer_control_enabled;

      violation |= !valid_action;

      if (violation) {
        tx = false;
      }
    }
  }

  // Safety check for LateralMotionControl action
  if (msg->addr == FORD_LateralMotionControl) {
    // Signal: LatCtl_D_Rq
    bool steer_control_enabled = ((msg->data[4] >> 2) & 0x7U) != 0U;
    unsigned int raw_curvature = (msg->data[0] << 3) | (msg->data[1] >> 5);
    unsigned int raw_curvature_rate = ((msg->data[1] & 0x1FU) << 8) | msg->data[2];
    unsigned int raw_path_angle = (msg->data[3] << 3) | (msg->data[4] >> 5);
    unsigned int raw_path_offset = (msg->data[5] << 2) | (msg->data[6] >> 6);

    // These signals are not yet tested with the current safety limits
    bool violation = (raw_curvature_rate != FORD_INACTIVE_CURVATURE_RATE) || (raw_path_angle != FORD_INACTIVE_PATH_ANGLE) || (raw_path_offset != FORD_INACTIVE_PATH_OFFSET);

    if (ford_lka_steer) {
      // LKA_STEER platforms steer through Lane_Assist_Data1; the PSCM ignores LCA/TJA here.
      // This message stays transmittable so the camera heartbeat to the PSCM survives, but
      // it must never carry a steering request. The curvature angle check is not run here:
      // angle_meas holds the pinion angle in tenths of a degree on these platforms, not a
      // curvature, so the curvature limits would be applied to the wrong unit scale.
      violation |= steer_control_enabled;
    } else {
      // Check angle error and steer_control_enabled
      int desired_curvature = raw_curvature - FORD_INACTIVE_CURVATURE;  // /FORD_STEERING_LIMITS.angle_deg_to_can to get real curvature
      violation |= steer_angle_cmd_checks(desired_curvature, steer_control_enabled, FORD_STEERING_LIMITS);
    }

    if (violation) {
      tx = false;
    }
  }

  // Safety check for LateralMotionControl2 action
  if (msg->addr == FORD_LateralMotionControl2) {
    static const AngleSteeringLimits FORD_CANFD_STEERING_LIMITS = FORD_LIMITS(true);

    // Signal: LatCtl_D2_Rq
    bool steer_control_enabled = ((msg->data[0] >> 4) & 0x7U) != 0U;
    unsigned int raw_curvature = (msg->data[2] << 3) | (msg->data[3] >> 5);
    unsigned int raw_curvature_rate = (msg->data[6] << 3) | (msg->data[7] >> 5);
    unsigned int raw_path_angle = ((msg->data[3] & 0x1FU) << 6) | (msg->data[4] >> 2);
    unsigned int raw_path_offset = ((msg->data[4] & 0x3U) << 8) | msg->data[5];

    // These signals are not yet tested with the current safety limits
    bool violation = (raw_curvature_rate != FORD_CANFD_INACTIVE_CURVATURE_RATE) || (raw_path_angle != FORD_INACTIVE_PATH_ANGLE) || (raw_path_offset != FORD_INACTIVE_PATH_OFFSET);

    if (ford_lka_steer) {
      // No platform sets both LKA_STEER and CANFD today, but ford_init accepts the
      // combination. Same reasoning as LateralMotionControl above: no steering request,
      // and no curvature angle check against a pinion-angle angle_meas.
      violation |= steer_control_enabled;
    } else {
      // Check angle error and steer_control_enabled
      int desired_curvature = raw_curvature - FORD_INACTIVE_CURVATURE;  // /FORD_STEERING_LIMITS.angle_deg_to_can to get real curvature
      violation |= steer_angle_cmd_checks(desired_curvature, steer_control_enabled, FORD_CANFD_STEERING_LIMITS);
    }

    if (violation) {
      tx = false;
    }
  }

  return tx;
}

static safety_config ford_init(uint16_t param) {
  // warning: quality flags are not yet checked in openpilot's CAN parser,
  // this may be the cause of blocked messages
  #define FORD_COMMON_RX_CHECKS \
    {.msg = {{FORD_BrakeSysFeatures, 0, 8, 50U, .max_counter = 15U}, { 0 }, { 0 }}},                                        \
    /* FORD_EngVehicleSpThrottle2 has a counter that either randomly skips or by 2, likely ECU bug */                      \
    /* Some hybrid models also experience a bug where this checksum mismatches for one or two frames */                    \
    /* under heavy acceleration with ACC. It has been confirmed that the Bronco Sport's camera only */                     \
    /* disallows ACC for bad quality flags, not counters or checksums, so we match that */                                 \
    {.msg = {{FORD_EngVehicleSpThrottle2, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true}, { 0 }, { 0 }}},      \
    {.msg = {{FORD_Yaw_Data_FD1, 0, 8, 100U, .max_counter = 255U}, { 0 }, { 0 }}},                                          \
    /* These messages have no counter or checksum */                                                                       \
    {.msg = {{FORD_EngBrakeData, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
    {.msg = {{FORD_EngVehicleSpThrottle, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
    {.msg = {{FORD_DesiredTorqBrk, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},

  static RxCheck ford_rx_checks[] = {
    FORD_COMMON_RX_CHECKS
  };

  // LKA_STEER platforms additionally require the measured pinion angle to bound
  // Lane_Assist_Data1 angle requests; scoped to these platforms only so a missing
  // or stale SteeringPinion_Data message can't disable controls on other Fords.
  static RxCheck ford_lka_rx_checks[] = {
    FORD_COMMON_RX_CHECKS
    // The pinion angle is what bounds the Lane_Assist_Data1 command, so StePinCompAnEst_D_Qf
    // is checked: an uninitialised or degraded PSCM estimate must not become the origin a
    // steering command is measured from. StePinAn_No_Cs and StePinAn_No_Cnt are not checked
    // because this PSCM does not transmit them: both are a constant zero across 856k frames
    // of Transit MK5 capture, matching the DBC's "Signal not transmitted on gas variants".
    // Enforcing the counter would invalidate every frame and permanently disable controls.
    // .ignore_checksum is inert rather than load-bearing: neither ford_get_checksum nor
    // ford_compute_checksum has a case for this message, so both return 0 and the
    // comparison would pass whether or not it is ignored. It is set for intent.
    {.msg = {{FORD_SteeringPinion_Data, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true}, { 0 }, { 0 }}},
  };

  #define FORD_COMMON_TX_MSGS \
    {FORD_Steering_Data_FD1, 0, 8, .check_relay = false}, \
    {FORD_Steering_Data_FD1, 2, 8, .check_relay = false}, \
    {FORD_ACCDATA_3, 0, 8, .check_relay = true},          \
    {FORD_Lane_Assist_Data1, 0, 8, .check_relay = true},  \
    {FORD_IPMA_Data, 0, 8, .check_relay = true},          \

  static const CanMsg FORD_CANFD_LONG_TX_MSGS[] = {
    FORD_COMMON_TX_MSGS
    {FORD_ACCDATA, 0, 8, .check_relay = true},
    {FORD_LateralMotionControl2, 0, 8, .check_relay = true},
  };

  static const CanMsg FORD_CANFD_STOCK_TX_MSGS[] = {
    FORD_COMMON_TX_MSGS
    {FORD_LateralMotionControl2, 0, 8, .check_relay = true},
  };

  static const CanMsg FORD_LONG_TX_MSGS[] = {
    FORD_COMMON_TX_MSGS
    {FORD_ACCDATA, 0, 8, .check_relay = true},
    {FORD_LateralMotionControl, 0, 8, .check_relay = true},
  };

  const uint16_t FORD_PARAM_CANFD = 2;
  const bool ford_canfd = GET_FLAG(param, FORD_PARAM_CANFD);

  const uint16_t FORD_PARAM_LKA_STEER = 4;
  ford_lka_steer = GET_FLAG(param, FORD_PARAM_LKA_STEER);

  // Lateral continuation, and only on a platform that steers through Lane_Assist_Data1
  // in the first place - it permits nothing anywhere else. The latch itself always
  // starts clear, so a mode change cannot inherit one taken before it.
  const uint16_t FORD_PARAM_LKA_CONTINUATION = 8;
  ford_lka_continuation_enabled = ford_lka_steer && GET_FLAG(param, FORD_PARAM_LKA_CONTINUATION);
  ford_lka_continuation = false;

  bool ford_longitudinal = false;

#ifdef ALLOW_DEBUG
  const uint16_t FORD_PARAM_LONGITUDINAL = 1;
  ford_longitudinal = GET_FLAG(param, FORD_PARAM_LONGITUDINAL);
#endif

  // Longitudinal is the default for CAN, and optional for CAN FD w/ ALLOW_DEBUG
  ford_longitudinal = !ford_canfd || ford_longitudinal;

  safety_config ret;
  if (ford_lka_steer) {
    if (ford_canfd) {
      ret = ford_longitudinal ? BUILD_SAFETY_CFG(ford_lka_rx_checks, FORD_CANFD_LONG_TX_MSGS) : \
                                BUILD_SAFETY_CFG(ford_lka_rx_checks, FORD_CANFD_STOCK_TX_MSGS);
    } else {
      ret = BUILD_SAFETY_CFG(ford_lka_rx_checks, FORD_LONG_TX_MSGS);
    }
  } else {
    if (ford_canfd) {
      ret = ford_longitudinal ? BUILD_SAFETY_CFG(ford_rx_checks, FORD_CANFD_LONG_TX_MSGS) : \
                                BUILD_SAFETY_CFG(ford_rx_checks, FORD_CANFD_STOCK_TX_MSGS);
    } else {
      ret = BUILD_SAFETY_CFG(ford_rx_checks, FORD_LONG_TX_MSGS);
    }
  }
  return ret;
}

const safety_hooks ford_hooks = {
  .init = ford_init,
  .rx = ford_rx_hook,
  .tx = ford_tx_hook,
  .get_counter = ford_get_counter,
  .get_checksum = ford_get_checksum,
  .compute_checksum = ford_compute_checksum,
  .get_quality_flag_valid = ford_get_quality_flag_valid,
};

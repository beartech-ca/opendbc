#!/usr/bin/env python3
import math
import numpy as np
import os
import pathlib
import random
import re
import unittest

import pytest

import opendbc.safety.tests.common as common
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY
from opendbc.car.ford.carcontroller import AVERAGE_ROAD_ROLL, MAX_LATERAL_ACCEL
from opendbc.car.ford.values import FordSafetyFlags
from opendbc.car.lateral import ISO_LATERAL_ACCEL
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerSafety

MSG_EngBrakeData = 0x165           # RX from PCM, for driver brake pedal and cruise state
MSG_EngVehicleSpThrottle = 0x204   # RX from PCM, for driver throttle input
MSG_BrakeSysFeatures = 0x415       # RX from ABS, for vehicle speed
MSG_EngVehicleSpThrottle2 = 0x202  # RX from PCM, for second vehicle speed
MSG_Yaw_Data_FD1 = 0x91            # RX from RCM, for yaw rate
MSG_SteeringPinion_Data = 0x07E    # RX from PSCM, measured steering pinion angle
MSG_Steering_Data_FD1 = 0x083      # TX by OP, various driver switches and LKAS/CC buttons
MSG_ACCDATA = 0x186                # TX by OP, ACC controls
MSG_ACCDATA_3 = 0x18A              # TX by OP, ACC/TJA user interface
MSG_Lane_Assist_Data1 = 0x3CA      # TX by OP, Lane Keep Assist
MSG_LateralMotionControl = 0x3D3   # TX by OP, Lateral Control message
MSG_LateralMotionControl2 = 0x3D6  # TX by OP, alternate Lateral Control message
MSG_IPMA_Data = 0x3D8              # TX by OP, IPMA and LKAS user interface


def checksum(msg):
  addr, dat, bus = msg
  ret = bytearray(dat)

  if addr == MSG_Yaw_Data_FD1:
    chksum = dat[0] + dat[1]  # VehRol_W_Actl
    chksum += dat[2] + dat[3]  # VehYaw_W_Actl
    chksum += dat[5]  # VehRollYaw_No_Cnt
    chksum += dat[6] >> 6  # VehRolWActl_D_Qf
    chksum += (dat[6] >> 4) & 0x3  # VehYawWActl_D_Qf
    chksum = 0xff - (chksum & 0xff)
    ret[4] = chksum

  elif addr == MSG_BrakeSysFeatures:
    chksum = dat[0] + dat[1]  # Veh_V_ActlBrk
    chksum += (dat[2] >> 2) & 0xf  # VehVActlBrk_No_Cnt
    chksum += dat[2] >> 6  # VehVActlBrk_D_Qf
    chksum = 0xff - (chksum & 0xff)
    ret[3] = chksum

  elif addr == MSG_EngVehicleSpThrottle2:
    chksum = (dat[2] >> 3) & 0xf  # VehVActlEng_No_Cnt
    chksum += (dat[4] >> 5) & 0x3  # VehVActlEng_D_Qf
    chksum += dat[6] + dat[7]  # Veh_V_ActlEng
    chksum = 0xff - (chksum & 0xff)
    ret[1] = chksum

  return addr, ret, bus


class Buttons:
  CANCEL = 0
  RESUME = 1
  TJA_TOGGLE = 2


# Ford safety has four different configurations tested here:
#  * CAN with openpilot longitudinal
#  * CAN FD with stock longitudinal
#  * CAN FD with openpilot longitudinal

class TestFordSafetyBase(common.CarSafetyTest):
  STANDSTILL_THRESHOLD = 1
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                                 MSG_LateralMotionControl2, MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                               MSG_LateralMotionControl2, MSG_IPMA_Data]}

  STEER_MESSAGE = 0

  # Curvature control limits
  DEG_TO_CAN = 50000  # 1 / (2e-5) rad to can
  MAX_CURVATURE = 0.02
  MAX_CURVATURE_ERROR = 0.002
  CURVATURE_ERROR_MIN_SPEED = 10.0  # m/s

  ANGLE_RATE_BP = [5., 25., 25.]
  ANGLE_RATE_UP = [0.00045, 0.0001, 0.0001]  # windup limit
  ANGLE_RATE_DOWN = [0.00045, 0.00015, 0.00015]  # unwind limit

  cnt_speed = 0
  cnt_speed_2 = 0
  cnt_yaw_rate = 0

  packer: CANPackerSafety
  safety: libsafety_py.LibSafety

  def get_canfd_curvature_limits(self, speed):
    # Round it in accordance with the safety
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(speed, 1) ** 2)
    curvature_accel_limit_lower = int(curvature_accel_limit * self.DEG_TO_CAN - 1) / self.DEG_TO_CAN
    curvature_accel_limit_upper = int(curvature_accel_limit * self.DEG_TO_CAN + 1) / self.DEG_TO_CAN
    return curvature_accel_limit_lower, curvature_accel_limit_upper

  def _set_prev_desired_angle(self, t):
    t = round(t * self.DEG_TO_CAN)
    self.safety.set_desired_angle_last(t)

  def _reset_curvature_measurement(self, curvature, speed):
    for _ in range(6):
      self._rx(self._speed_msg(speed))
      self._rx(self._yaw_rate_msg(curvature, speed))

  # Driver brake pedal
  def _user_brake_msg(self, brake: bool):
    # brake pedal and cruise state share same message, so we have to send
    # the other signal too
    enable = self.safety.get_controls_allowed()
    values = {
      "BpedDrvAppl_D_Actl": 2 if brake else 1,
      "CcStat_D_Actl": 5 if enable else 0,
    }
    return self.packer.make_can_msg_safety("EngBrakeData", 0, values)

  # ABS vehicle speed
  def _speed_msg(self, speed: float, quality_flag=True):
    values = {"Veh_V_ActlBrk": speed * 3.6, "VehVActlBrk_D_Qf": 3 if quality_flag else 0, "VehVActlBrk_No_Cnt": self.cnt_speed % 16}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_safety("BrakeSysFeatures", 0, values, fix_checksum=checksum)

  # PCM vehicle speed
  def _speed_msg_2(self, speed: float, quality_flag=True):
    # Ford relies on speed for driver curvature limiting, so it checks two sources
    values = {"Veh_V_ActlEng": speed * 3.6, "VehVActlEng_D_Qf": 3 if quality_flag else 0, "VehVActlEng_No_Cnt": self.cnt_speed_2 % 16}
    self.__class__.cnt_speed_2 += 1
    return self.packer.make_can_msg_safety("EngVehicleSpThrottle2", 0, values, fix_checksum=checksum)

  # Standstill state
  def _vehicle_moving_msg(self, speed: float):
    values = {"VehStop_D_Stat": 1 if speed <= self.STANDSTILL_THRESHOLD else random.choice((0, 2, 3))}
    return self.packer.make_can_msg_safety("DesiredTorqBrk", 0, values)

  # Current curvature
  def _yaw_rate_msg(self, curvature: float, speed: float, quality_flag=True):
    values = {"VehYaw_W_Actl": curvature * speed, "VehYawWActl_D_Qf": 3 if quality_flag else 0,
              "VehRollYaw_No_Cnt": self.cnt_yaw_rate % 256}
    self.__class__.cnt_yaw_rate += 1
    return self.packer.make_can_msg_safety("Yaw_Data_FD1", 0, values, fix_checksum=checksum)

  # Drive throttle input
  def _user_gas_msg(self, gas: float):
    values = {"ApedPos_Pc_ActlArb": gas}
    return self.packer.make_can_msg_safety("EngVehicleSpThrottle", 0, values)

  # Cruise status
  def _pcm_status_msg(self, enable: bool):
    # brake pedal and cruise state share same message, so we have to send
    # the other signal too
    brake = self.safety.get_brake_pressed_prev()
    values = {
      "BpedDrvAppl_D_Actl": 2 if brake else 1,
      "CcStat_D_Actl": 5 if enable else 0,
    }
    return self.packer.make_can_msg_safety("EngBrakeData", 0, values)

  # LKAS command
  def _lkas_command_msg(self, action: int):
    values = {
      "LkaActvStats_D2_Req": action,
    }
    return self.packer.make_can_msg_safety("Lane_Assist_Data1", 0, values)

  # Cruise control buttons
  def _acc_button_msg(self, button: int, bus: int):
    values = {
      "CcAslButtnCnclPress": 1 if button == Buttons.CANCEL else 0,
      "CcAsllButtnResPress": 1 if button == Buttons.RESUME else 0,
      "TjaButtnOnOffPress": 1 if button == Buttons.TJA_TOGGLE else 0,
    }
    return self.packer.make_can_msg_safety("Steering_Data_FD1", bus, values)

  def test_rx_hook(self):
    # checksum, counter, and quality flag checks
    for quality_flag in [True, False]:
      for msg_type in ["speed", "speed_2", "yaw"]:
        self.safety.set_controls_allowed(True)
        # send multiple times to verify counter checks
        for _ in range(10):
          if msg_type == "speed":
            msg = self._speed_msg(0, quality_flag=quality_flag)
          elif msg_type == "speed_2":
            msg = self._speed_msg_2(0, quality_flag=quality_flag)
          elif msg_type == "yaw":
            msg = self._yaw_rate_msg(0, 0, quality_flag=quality_flag)

          self.assertEqual(quality_flag, self._rx(msg))
          self.assertEqual(quality_flag, self.safety.get_controls_allowed())

        # Mess with checksum to make it fail, checksum is not checked for 2nd speed
        msg[0].data[3] = 0  # Speed checksum & half of yaw signal
        should_rx = msg_type == "speed_2" and quality_flag
        self.assertEqual(should_rx, self._rx(msg))
        self.assertEqual(should_rx, self.safety.get_controls_allowed())

  def test_prevent_lkas_action(self):
    self.safety.set_controls_allowed(1)
    self.assertFalse(self._tx(self._lkas_command_msg(1)))

    self.safety.set_controls_allowed(0)
    self.assertFalse(self._tx(self._lkas_command_msg(1)))

  def test_acc_buttons(self):
    for allowed in (0, 1):
      self.safety.set_controls_allowed(allowed)
      for enabled in (True, False):
        self._rx(self._pcm_status_msg(enabled))
        self.assertTrue(self._tx(self._acc_button_msg(Buttons.TJA_TOGGLE, 2)))

    for allowed in (0, 1):
      self.safety.set_controls_allowed(allowed)
      for bus in (0, 2):
        self.assertEqual(allowed, self._tx(self._acc_button_msg(Buttons.RESUME, bus)))

    for enabled in (True, False):
      self._rx(self._pcm_status_msg(enabled))
      for bus in (0, 2):
        self.assertEqual(enabled, self._tx(self._acc_button_msg(Buttons.CANCEL, bus)))


class TestFordCurvatureSteeringBase(TestFordSafetyBase):
  """Tests for the LCA/TJA curvature channel. Platforms that steer through
  Lane_Assist_Data1 inherit TestFordSafetyBase directly and skip these."""

  # LCA command
  def _lat_ctl_msg(self, enabled: bool, path_offset: float, path_angle: float, curvature: float, curvature_rate: float):
    if self.STEER_MESSAGE == MSG_LateralMotionControl:
      values = {
        "LatCtl_D_Rq": 1 if enabled else 0,
        "LatCtlPathOffst_L_Actl": path_offset,     # Path offset [-5.12|5.11] meter
        "LatCtlPath_An_Actl": path_angle,          # Path angle [-0.5|0.5235] radians
        "LatCtlCurv_NoRate_Actl": curvature_rate,  # Curvature rate [-0.001024|0.00102375] 1/meter^2
        "LatCtlCurv_No_Actl": curvature,           # Curvature [-0.02|0.02094] 1/meter
      }
      return self.packer.make_can_msg_safety("LateralMotionControl", 0, values)
    elif self.STEER_MESSAGE == MSG_LateralMotionControl2:
      values = {
        "LatCtl_D2_Rq": 1 if enabled else 0,
        "LatCtlPathOffst_L_Actl": path_offset,     # Path offset [-5.12|5.11] meter
        "LatCtlPath_An_Actl": path_angle,          # Path angle [-0.5|0.5235] radians
        "LatCtlCrv_NoRate2_Actl": curvature_rate,  # Curvature rate [-0.001024|0.001023] 1/meter^2
        "LatCtlCurv_No_Actl": curvature,           # Curvature [-0.02|0.02094] 1/meter
      }
      return self.packer.make_can_msg_safety("LateralMotionControl2", 0, values)

  def test_angle_measurements(self):
    """Tests rx hook correctly parses the curvature measurement from the vehicle speed and yaw rate"""
    for speed in np.arange(0.5, 40, 0.5):
      for curvature in np.arange(0, self.MAX_CURVATURE * 2, 2e-3):
        self._rx(self._speed_msg(speed))
        for c in (curvature, -curvature, 0, 0, 0, 0):
          self._rx(self._yaw_rate_msg(c, speed))

        self.assertEqual(self.safety.get_angle_meas_min(), round(-curvature * self.DEG_TO_CAN))
        self.assertEqual(self.safety.get_angle_meas_max(), round(curvature * self.DEG_TO_CAN))

        self._rx(self._yaw_rate_msg(0, speed))
        self.assertEqual(self.safety.get_angle_meas_min(), round(-curvature * self.DEG_TO_CAN))
        self.assertEqual(self.safety.get_angle_meas_max(), 0)

        self._rx(self._yaw_rate_msg(0, speed))
        self.assertEqual(self.safety.get_angle_meas_min(), 0)
        self.assertEqual(self.safety.get_angle_meas_max(), 0)

  def test_max_lateral_acceleration(self):
    # Ford CAN FD can achieve a higher max lateral acceleration than CAN so we limit curvature based on speed
    step = 1 / self.DEG_TO_CAN
    for speed in np.arange(0, 40, 0.5):
      # Clip so we test curvature limiting at low speed due to low max curvature
      _, curvature_accel_limit_upper = self.get_canfd_curvature_limits(speed)
      curvature_accel_limit_upper = np.clip(curvature_accel_limit_upper, -self.MAX_CURVATURE, self.MAX_CURVATURE)

      # Test boundary curvature values around the limit, rounded to CAN precision
      lower = curvature_accel_limit_upper * 0.8
      upper = min(curvature_accel_limit_upper * 1.2, self.MAX_CURVATURE)
      test_curvatures = {round(c * self.DEG_TO_CAN) / self.DEG_TO_CAN
                         for c in self._boundary_values([curvature_accel_limit_upper], lower, upper, step)
                         if 0 <= c <= self.MAX_CURVATURE}

      for sign in (-1, 1):
        for curvature in sorted(test_curvatures):
          curvature = sign * curvature
          self.safety.set_controls_allowed(True)
          self._set_prev_desired_angle(curvature)
          self._reset_curvature_measurement(curvature, speed)

          should_tx = abs(curvature) <= curvature_accel_limit_upper
          self.assertEqual(should_tx, self._tx(self._lat_ctl_msg(True, 0, 0, curvature, 0)))

  def test_steer_allowed(self):
    path_offsets = np.arange(-5.12, 5.11, 2.5).round()
    path_angles = np.arange(-0.5, 0.5235, 0.25).round(1)
    curvature_rates = np.arange(-0.001024, 0.00102375, 0.001).round(3)
    curvatures = np.arange(-0.02, 0.02094, 0.01).round(2)

    for speed in (self.CURVATURE_ERROR_MIN_SPEED - 1,
                  self.CURVATURE_ERROR_MIN_SPEED + 1):
      _, curvature_accel_limit_upper = self.get_canfd_curvature_limits(speed)
      for controls_allowed in (True, False):
        for steer_control_enabled in (True, False):
          for path_offset in path_offsets:
            for path_angle in path_angles:
              for curvature_rate in curvature_rates:
                for curvature in curvatures:
                  self.safety.set_controls_allowed(controls_allowed)
                  self._set_prev_desired_angle(curvature)
                  self._reset_curvature_measurement(curvature, speed)

                  should_tx = path_offset == 0 and path_angle == 0 and curvature_rate == 0
                  # when request bit is 0, only allow curvature of 0 since the signal range
                  # is not large enough to enforce it tracking measured
                  should_tx = should_tx and (controls_allowed if steer_control_enabled else curvature == 0)

                  # Only CAN FD has the max lateral acceleration limit
                  if self.STEER_MESSAGE == MSG_LateralMotionControl2:
                    should_tx = should_tx and abs(curvature) <= curvature_accel_limit_upper

                  with self.subTest(controls_allowed=controls_allowed, steer_control_enabled=steer_control_enabled,
                                    path_offset=float(path_offset), path_angle=float(path_angle), curvature_rate=float(curvature_rate),
                                    curvature=float(curvature)):
                    self.assertEqual(should_tx, self._tx(self._lat_ctl_msg(steer_control_enabled, path_offset, path_angle, curvature, curvature_rate)))

  def test_curvature_rate_limits(self):
    """
    When the curvature error is exceeded, commanded curvature must start moving towards meas respecting rate limits.
    Since safety allows higher rate limits to avoid false positives, we need to allow a lower rate to move towards meas.
    """
    self.safety.set_controls_allowed(True)
    # safety fudges the speed (1 m/s) and rate limits (1 CAN unit) to avoid false positives
    small_curvature = 1 / self.DEG_TO_CAN  # significant small amount of curvature to cross boundary

    for speed in np.arange(0, 40, 0.5):
      curvature_accel_limit_lower, curvature_accel_limit_upper = self.get_canfd_curvature_limits(speed)
      limit_command = speed > self.CURVATURE_ERROR_MIN_SPEED
      # ensure our limits match the safety's rounded limits
      max_delta_up = int(np.interp(speed - 1, self.ANGLE_RATE_BP, self.ANGLE_RATE_UP) * self.DEG_TO_CAN + 1) / self.DEG_TO_CAN
      max_delta_up_lower = int(np.interp(speed + 1, self.ANGLE_RATE_BP, self.ANGLE_RATE_UP) * self.DEG_TO_CAN - 1) / self.DEG_TO_CAN

      max_delta_down = int(np.interp(speed - 1, self.ANGLE_RATE_BP, self.ANGLE_RATE_DOWN) * self.DEG_TO_CAN + 1 + 1e-3) / self.DEG_TO_CAN
      max_delta_down_lower = int(np.interp(speed + 1, self.ANGLE_RATE_BP, self.ANGLE_RATE_DOWN) * self.DEG_TO_CAN - 1 + 1e-3) / self.DEG_TO_CAN

      up_cases = (self.MAX_CURVATURE_ERROR * 2, [
        (not limit_command, 0, 0),
        (not limit_command, 0, max_delta_up_lower - small_curvature),
        (True, 1e-9, max_delta_down),  # TODO: safety should not allow down limits at 0
        (not limit_command, 1e-9, max_delta_up_lower),  # TODO: safety should not allow down limits at 0
        (True, 0, max_delta_up_lower),
        (True, 0, max_delta_up),
        (False, 0, max_delta_up + small_curvature),
        # stay at boundary limit
        (True, self.MAX_CURVATURE_ERROR - small_curvature, self.MAX_CURVATURE_ERROR - small_curvature),
        # 1 unit below boundary limit
        (not limit_command, self.MAX_CURVATURE_ERROR - small_curvature * 2, self.MAX_CURVATURE_ERROR - small_curvature * 2),
        # shouldn't allow command to move outside the boundary limit if last was inside
        (not limit_command, self.MAX_CURVATURE_ERROR - small_curvature, self.MAX_CURVATURE_ERROR - small_curvature * 2),
      ])

      down_cases = (self.MAX_CURVATURE - self.MAX_CURVATURE_ERROR * 2, [
        (not limit_command, self.MAX_CURVATURE, self.MAX_CURVATURE),
        (not limit_command, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta_down_lower + small_curvature),
        (True, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta_down_lower),
        (True, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta_down),
        (False, self.MAX_CURVATURE, self.MAX_CURVATURE - max_delta_down - small_curvature),
      ])

      for sign in (-1, 1):
        for angle_meas, cases in (up_cases, down_cases):
          self._reset_curvature_measurement(sign * angle_meas, speed)
          for should_tx, initial_curvature, desired_curvature in cases:

            # Only CAN FD has the max lateral acceleration limit
            if self.STEER_MESSAGE == MSG_LateralMotionControl2:
              if should_tx:
                # can not send if the curvature is above the max lateral acceleration
                should_tx = should_tx and abs(desired_curvature) <= curvature_accel_limit_upper
              else:
                # if desired curvature violates driver curvature error, it can only send if
                # the curvature is being limited by max lateral acceleration
                should_tx = should_tx or curvature_accel_limit_lower <= abs(desired_curvature) <= curvature_accel_limit_upper

            # small curvature ensures we're using up limits. at 0, safety allows down limits to allow to account for rounding errors
            curvature_offset = small_curvature if initial_curvature == 0 else 0
            self._set_prev_desired_angle(sign * (curvature_offset + initial_curvature))
            self.assertEqual(should_tx, self._tx(self._lat_ctl_msg(True, 0, 0, sign * (curvature_offset + desired_curvature), 0)))


class TestFordCANFDStockSafety(TestFordCurvatureSteeringBase):
  STEER_MESSAGE = MSG_LateralMotionControl2

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl2, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                               MSG_IPMA_Data]}

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, FordSafetyFlags.CANFD)
    self.safety.init_tests()


class TestFordLongitudinalSafetyBase(TestFordCurvatureSteeringBase):
  MAX_ACCEL = 2.0  # accel is used for brakes, but openpilot can set positive values
  MIN_ACCEL = -3.5
  INACTIVE_ACCEL = 0.0

  MAX_GAS = 2.0
  MIN_GAS = -0.5
  INACTIVE_GAS = -5.0

  # ACC command
  def _acc_command_msg(self, gas: float, brake: float, brake_actuation: bool, cmbb_deny: bool = False):
    values = {
      "AccPrpl_A_Rq": gas,                              # [-5|5.23] m/s^2
      "AccPrpl_A_Pred": gas,                            # [-5|5.23] m/s^2
      "AccBrkTot_A_Rq": brake,                          # [-20|11.9449] m/s^2
      "AccBrkPrchg_B_Rq": 1 if brake_actuation else 0,  # Pre-charge brake request: 0=No, 1=Yes
      "AccBrkDecel_B_Rq": 1 if brake_actuation else 0,  # Deceleration request: 0=Inactive, 1=Active
      "CmbbDeny_B_Actl": 1 if cmbb_deny else 0,         # [0|1] deny AEB actuation
    }
    return self.packer.make_can_msg_safety("ACCDATA", 0, values)

  def test_stock_aeb(self):
    # Test that CmbbDeny_B_Actl is never 1, it prevents the ABS module from actuating AEB requests from ACCDATA_2
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for cmbb_deny in (True, False):
        should_tx = not cmbb_deny
        self.assertEqual(should_tx, self._tx(self._acc_command_msg(self.INACTIVE_GAS, self.INACTIVE_ACCEL, controls_allowed, cmbb_deny)))
        should_tx = controls_allowed and not cmbb_deny
        self.assertEqual(should_tx, self._tx(self._acc_command_msg(self.MAX_GAS, self.MAX_ACCEL, controls_allowed, cmbb_deny)))

  def test_gas_safety_check(self):
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for gas in np.concatenate((np.arange(self.MIN_GAS - 2, self.MAX_GAS + 2, 0.05), [self.INACTIVE_GAS])):
        gas = round(gas, 2)  # floats might not hit exact boundary conditions without rounding
        should_tx = (controls_allowed and self.MIN_GAS <= gas <= self.MAX_GAS) or gas == self.INACTIVE_GAS
        self.assertEqual(should_tx, self._tx(self._acc_command_msg(gas, self.INACTIVE_ACCEL, controls_allowed)))

  def test_brake_safety_check(self):
    brake_values = self._boundary_values([self.MIN_ACCEL, self.MAX_ACCEL, self.INACTIVE_ACCEL],
                                         self.MIN_ACCEL - 2, self.MAX_ACCEL + 2, 0.05)
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for brake_actuation in (True, False):
        for brake in brake_values:
          should_tx = (controls_allowed and self.MIN_ACCEL <= brake <= self.MAX_ACCEL) or brake == self.INACTIVE_ACCEL
          should_tx = should_tx and (controls_allowed or not brake_actuation)
          self.assertEqual(should_tx, self._tx(self._acc_command_msg(self.INACTIVE_GAS, brake, brake_actuation)))


class TestFordLongitudinalSafety(TestFordLongitudinalSafetyBase):
  STEER_MESSAGE = MSG_LateralMotionControl

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA, 0], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                               MSG_IPMA_Data]}

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    # Make sure we enforce long safety even without long flag for CAN
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, 0)
    self.safety.init_tests()

  def test_max_lateral_acceleration(self):
    # CAN does not limit curvature from lateral acceleration
    pass


class TestFordCANFDLongitudinalSafety(TestFordLongitudinalSafetyBase):
  STEER_MESSAGE = MSG_LateralMotionControl2

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA, 0], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl2, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl2,
                               MSG_IPMA_Data]}

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, FordSafetyFlags.LONG_CONTROL | FordSafetyFlags.CANFD)
    self.safety.init_tests()


class TestFordTransitLkaSafety(TestFordSafetyBase):
  """
  Tests for the LKA_STEER platforms (e.g. Transit MK5), which steer through
  Lane_Assist_Data1 (0x3CA) directly instead of the LCA/TJA curvature channel.
  Inherits TestFordSafetyBase directly, not TestFordCurvatureSteeringBase,
  since the curvature-channel tests don't apply here.
  """
  STEER_MESSAGE = MSG_Lane_Assist_Data1

  TX_MSGS = [
    [MSG_Steering_Data_FD1, 0], [MSG_Steering_Data_FD1, 2], [MSG_ACCDATA, 0], [MSG_ACCDATA_3, 0], [MSG_Lane_Assist_Data1, 0],
    [MSG_LateralMotionControl, 0], [MSG_IPMA_Data, 0],
  ]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                                 MSG_IPMA_Data)}

  FWD_BLACKLISTED_ADDRS = {2: [MSG_ACCDATA, MSG_ACCDATA_3, MSG_Lane_Assist_Data1, MSG_LateralMotionControl,
                               MSG_IPMA_Data]}

  cnt_lka_cmd = 0

  def setUp(self):
    self.packer = CANPackerSafety("ford_lincoln_base_pt")
    self.safety = libsafety_py.libsafety
    # Matches interface.py: LKA_STEER platforms always set LONG_CONTROL too,
    # since FORD_TRANSIT_MK5 has a radar and is not CAN FD.
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford, FordSafetyFlags.LKA_STEER | FordSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()
    # The LKA command is subject to a real time send-rate limit, so the mock clock has
    # to advance across the frames a test sends, exactly as onboard time would.
    self.__class__.cnt_lka_cmd = 0
    self.safety.set_timer(0)

  # Must match the FORD_LKA_* constants and FORD_LKA_STEERING_PARAMS in ford.h
  LKA_DEG_TO_CAN = 10
  LKA_FREQUENCY = 33  # Hz, must match FORD_LKA_RT_LIMITS.frequency
  LKA_SLIP_FACTOR = -0.0004472752575630534
  LKA_STEER_RATIO = 20.9
  LKA_WHEELBASE = 3.75

  # The safety's own lateral acceleration ceiling, which is not the car side's
  SAFETY_MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)

  # LaRefAng_No_Req is 12 bits at 0.05 mrad/bit with a -102.4 mrad offset
  LKA_REL_ANGLE_MIN_MRAD = -102.4
  LKA_REL_ANGLE_MAX_MRAD = 102.35
  LKA_MAX_REL_ANGLE_CAN = 59  # +/-5.9 deg in tenths, must match FORD_LKA_MAX_REL_ANGLE

  # LkaActvStats_D2_Req values that request steering: 1/6 increasing left/right
  # intervention, 2/4 standard left/right. 0 is idle, 3/5 suppress, 7 is NotUsed.
  LKA_STEERING_ACTIONS = (1, 2, 4, 6)

  def _fudged_speed(self, speed: float) -> float:
    # the safety fudges the speed down by 1 m/s and floors it at 1 m/s
    return max(speed - 1.0, 1.0)

  def _curvature_factor(self, speed: float) -> float:
    fudged_speed = self._fudged_speed(speed)
    return 1. / (1. - (self.LKA_SLIP_FACTOR * fudged_speed ** 2)) / self.LKA_WHEELBASE

  def _max_lka_angle_deg(self, speed: float) -> float:
    """The ISO lateral acceleration ceiling on an absolute angle command, in degrees."""
    max_curvature = self.SAFETY_MAX_LATERAL_ACCEL / (self._fudged_speed(speed) ** 2)
    return math.degrees(max_curvature * self.LKA_STEER_RATIO / self._curvature_factor(speed))

  # Measured pinion angle. StePinAn_No_Cs and StePinAn_No_Cnt are left at zero to match
  # what the Transit's PSCM actually transmits; see test_rx_hook_pinion_has_no_counter.
  def _pinion_angle_msg(self, angle_deg: float, quality_flag: bool = True):
    values = {
      "StePinComp_An_Est": angle_deg,
      "StePinCompAnEst_D_Qf": 3 if quality_flag else 0,
    }
    return self.packer.make_can_msg_safety("SteeringPinion_Data", 0, values)

  def _reset_pinion_measurement(self, angle_deg: float):
    for _ in range(6):
      self._rx(self._pinion_angle_msg(angle_deg))

  def _reset_speed_measurement(self, speed: float):
    for _ in range(6):
      self._rx(self._speed_msg(speed))
      self._rx(self._speed_msg_2(speed))

  # LKA command: action + angle relative to the current pinion angle.
  # Advances the mock clock one 33Hz frame per command by default, the way the real
  # onboard clock does; pass increment_timer=False to exercise the real time rate limit.
  def _lka_angle_msg(self, action: int, relative_mrad: float, increment_timer: bool = True):
    values = {
      "LkaActvStats_D2_Req": action,
      "LaRefAng_No_Req": relative_mrad,
    }
    if increment_timer:
      self.safety.set_timer(self.cnt_lka_cmd * int(1e6 / self.LKA_FREQUENCY))
      self.__class__.cnt_lka_cmd += 1
    return self.packer.make_can_msg_safety("Lane_Assist_Data1", 0, values)

  def _lka_heartbeat_msg(self):
    # The worst case the safety has to accept when nothing is steering: an all-zero
    # payload, in which raw zero in LaRefAng_No_Req decodes to -102.4 mrad, not 0. It is
    # what fordcan.create_lka_msg sends on every other Ford platform; the Transit's own
    # inactive frame (create_transit_lka_msg) packs the value 0.0 explicitly instead.
    return self.packer.make_can_msg_safety("Lane_Assist_Data1", 0, {})

  # LCA/TJA message. The PSCM ignores it on this platform, but openpilot still sends it
  # at 20Hz, so it stays TX-whitelisted as part of the camera heartbeat.
  def _lat_ctl_msg(self, enabled: bool, curvature: float = 0.):
    values = {
      "LatCtl_D_Rq": 1 if enabled else 0,
      "LatCtlPathOffst_L_Actl": 0.,
      "LatCtlPath_An_Actl": 0.,
      "LatCtlCurv_NoRate_Actl": 0.,
      "LatCtlCurv_No_Actl": curvature,
    }
    return self.packer.make_can_msg_safety("LateralMotionControl", 0, values)

  def test_prevent_lkas_action(self):
    # Lane_Assist_Data1 is the real LKA channel here, so a non-zero action is not
    # unconditionally blocked. Only the four intervention requests are whitelisted:
    # the suppress values and the reserved one are rejected rather than guessed at.
    for action in range(8):
      for allowed in (0, 1):
        self.safety.set_controls_allowed(allowed)
        self._reset_pinion_measurement(0.)
        should_tx = (action == 0) or (action in self.LKA_STEERING_ACTIONS and allowed)
        with self.subTest(action=action, allowed=allowed):
          self.assertEqual(should_tx, self._tx(self._lka_angle_msg(action, 0.)))

  def test_heartbeat_frame_is_never_blocked(self):
    # The all-zero heartbeat payload decodes to a -5.9 deg relative request, which would
    # become a real steering request unless the relative term is forced to zero whenever
    # the action is not a steering request.
    for allowed in (0, 1):
      for angle in (0., 12.3, -12.3):
        self.safety.set_controls_allowed(allowed)
        self._reset_pinion_measurement(angle)
        with self.subTest(allowed=allowed, angle=angle):
          self.assertTrue(self._tx(self._lka_heartbeat_msg()))

  def test_inactive_frame_allowed_at_large_wheel_angles(self):
    # StePinComp_An_Est is a steering-wheel-side angle spanning +/-1600 deg. An idle frame
    # carries no steering request at all, so it must stay transmittable however far the
    # wheel is turned: roundabouts, junctions and parking manoeuvres.
    for angle in (89., 91., -400., 1000., -1595.):
      for allowed in (0, 1):
        self.safety.set_controls_allowed(allowed)
        self._reset_pinion_measurement(angle)
        with self.subTest(angle=angle, allowed=allowed):
          self.assertTrue(self._tx(self._lka_heartbeat_msg()))

  def test_action_blocked_when_controls_not_allowed(self):
    self.safety.set_controls_allowed(0)
    self._reset_pinion_measurement(0.)
    assert not self._tx(self._lka_angle_msg(2, 20.0))

  def test_small_request_allowed_when_controls_allowed(self):
    self.safety.set_controls_allowed(1)
    self._reset_pinion_measurement(0.)
    assert self._tx(self._lka_angle_msg(2, 20.0))

  def test_max_lateral_acceleration(self):
    # BOUND 2. The reconstructed absolute target (measured pinion angle + relative request)
    # is bounded by the vehicle model's lateral acceleration ceiling: ~52 deg of wheel at
    # 20 m/s, ~26 deg at 30 m/s. This is the bound that stops a sustained maximum relative
    # request from winding the wheel up indefinitely.
    for speed in (20., 30.):
      max_angle_can = int(self._max_lka_angle_deg(speed) * self.LKA_DEG_TO_CAN) + 1
      for sign in (-1, 1):
        for angle_can, should_tx in ((max_angle_can, True), (max_angle_can + 1, False)):
          angle = sign * angle_can / self.LKA_DEG_TO_CAN
          self.safety.set_controls_allowed(1)
          self._reset_speed_measurement(speed)
          self._reset_pinion_measurement(angle)
          with self.subTest(speed=speed, angle=angle):
            self.assertEqual(should_tx, self._tx(self._lka_angle_msg(2, 0.)))

  def test_relative_request_magnitude(self):
    # BOUND 1. LaRefAng_No_Req is a relative correction, and its 12-bit encoding already
    # confines it to -5.867..+5.864 deg. The bound is repeated in the safety to catch
    # panda's own bit extraction of the field being wrong, so what it must not do is sit
    # below the encodable range: every value the signal can carry has to survive it.
    # Checked at a standstill, where bound 2 is far too loose to bind.
    self.safety.set_controls_allowed(1)
    self._reset_speed_measurement(0.)
    self._reset_pinion_measurement(0.)
    for raw in range(4096):
      mrad = self.LKA_REL_ANGLE_MIN_MRAD + (raw * 0.05)
      with self.subTest(raw=raw):
        self.assertTrue(self._tx(self._lka_angle_msg(2, mrad)))
        rel_can = round(math.degrees(mrad / 1000.) * self.LKA_DEG_TO_CAN)
        self.assertLessEqual(abs(rel_can), self.LKA_MAX_REL_ANGLE_CAN)

  def test_relative_request_magnitude_bound_is_present(self):
    """BOUND 1, continued: the bound is still in the source.

    The bound sits exactly at the ceiling of the signal's own encoding, so no CAN frame
    can reach it while the extraction is correct: it is unreachable defence in depth
    against the extraction itself being wrong, and no behavioural test can therefore fail
    on its deletion. What can be pinned is that it is still there, which is the risk a
    check nothing exercises actually carries. The behavioural coverage of the defect it
    guards lives in test_neighbouring_signal_does_not_move_the_request; this is only a
    presence check, matched loosely so ordinary reformatting of ford.h does not break it.
    """
    ford_h = (pathlib.Path(__file__).parents[1] / "modes" / "ford.h").read_text()
    constant = rf"FORD_LKA_MAX_REL_ANGLE\s*=\s*{self.LKA_MAX_REL_ANGLE_CAN}\s*;"
    check = r"safety_max_limit_check\(\s*rel_tenths\s*,\s*FORD_LKA_MAX_REL_ANGLE\s*,\s*-\s*FORD_LKA_MAX_REL_ANGLE\s*\)"
    self.assertRegex(ford_h, re.compile(constant + ".*" + check, re.DOTALL))

  def test_neighbouring_signal_does_not_move_the_request(self):
    # BOUND 1's failure mode. LaCurvature_No_Calc (15|12@0+) shares byte 2 with
    # LaRefAng_No_Req (19|12@0+): the curvature occupies the top nibble, the relative
    # angle request the bottom one. An extraction that lost the nibble mask would read the
    # curvature into the angle request. Pin that the curvature cannot change the outcome,
    # and that when it does leak the magnitude bound is what sees it: the largest value it
    # could inject is ~181 deg of relative request, three orders of magnitude past the
    # 5.9 deg ceiling.
    self.safety.set_controls_allowed(1)
    for speed in (0., 20.):
      self._reset_speed_measurement(speed)
      self._reset_pinion_measurement(0.)
      for curvature in (-0.01024, 0., 0.01023):
        values = {"LkaActvStats_D2_Req": 2, "LaRefAng_No_Req": 20.0, "LaCurvature_No_Calc": curvature}
        msg = self.packer.make_can_msg_safety("Lane_Assist_Data1", 0, values)
        with self.subTest(speed=speed, curvature=curvature):
          self.assertTrue(self._tx(msg))

  def test_no_rate_of_change_bound(self):
    # Deliberate design decision, recorded so it is not silently reintroduced.
    # LaRefAng_No_Req is not a position target openpilot ramps: it is the whole remaining
    # correction, and the PSCM applies it over its own internal ramp. A per-frame step
    # limit on the request therefore measures intent, not motion, and there is none here.
    # Alternating between the extremes of the signal on consecutive frames must transmit.
    self.safety.set_controls_allowed(1)
    self._reset_speed_measurement(20.)
    self._reset_pinion_measurement(0.)
    for i in range(20):
      mrad = self.LKA_REL_ANGLE_MAX_MRAD if (i % 2) == 0 else self.LKA_REL_ANGLE_MIN_MRAD
      self.assertTrue(self._tx(self._lka_angle_msg(2, mrad)), f"frame {i} blocked")

  def test_command_send_rate_is_bounded(self):
    # There is no per-frame rate-of-change bound (above), and the reason the two
    # remaining bounds are enough is that the PSCM's own ramp bounds the wheel motion a
    # single relative request can produce - and that ramp was measured against a 33Hz
    # command stream. A device-side fault emitting 0x3CA far faster would hand the PSCM
    # many times the correction per unit time with every individual frame still inside
    # both bounds, so the number of commands per RT interval is capped as well.
    self.safety.set_timer(0)
    self.safety.set_controls_allowed(1)
    self._reset_speed_measurement(20.)
    self._reset_pinion_measurement(0.)
    max_rt_msgs = int(self.LKA_FREQUENCY * common.RT_INTERVAL / 1e6 * 1.2 + 1)  # 1.2x buffer

    # commands blasted without the clock advancing stop being accepted
    for i in range(max_rt_msgs * 2):
      with self.subTest(i=i):
        self.assertEqual(i <= max_rt_msgs, self._tx(self._lka_angle_msg(2, 0., increment_timer=False)))

    # one microsecond under the interval is still the same window
    self.safety.set_timer(common.RT_INTERVAL - 1)
    for _ in range(5):
      self.assertFalse(self._tx(self._lka_angle_msg(2, 0., increment_timer=False)))

    # crossing the interval resets the window on the next command
    self.safety.set_timer(common.RT_INTERVAL)
    self.assertFalse(self._tx(self._lka_angle_msg(2, 0., increment_timer=False)))
    for _ in range(5):
      self.assertTrue(self._tx(self._lka_angle_msg(2, 0., increment_timer=False)))

  def test_measured_angle_sample_is_required(self):
    # Regression test for the trap: if the pinion angle sample were never updated by the
    # rx hook, the desired angle would be computed relative to a stale/zero measurement
    # and a large actual wheel angle would slip through bound 2.
    self.safety.set_controls_allowed(1)
    self._reset_speed_measurement(20.)
    self._rx(self._pinion_angle_msg(900.0))
    # A single rx is enough for update_sample to move the current value,
    # even before the 6-sample window is entirely full of the new angle.
    assert not self._tx(self._lka_angle_msg(2, 20.0))

  def test_lateral_motion_control_carries_no_steering_request(self):
    # LCA/TJA cannot steer this PSCM, so the message may only go out inactive
    for allowed in (0, 1):
      self.safety.set_controls_allowed(allowed)
      self.assertTrue(self._tx(self._lat_ctl_msg(False)))
      self.assertFalse(self._tx(self._lat_ctl_msg(True)))

  def test_lateral_motion_control_does_not_disturb_lka_state(self):
    # openpilot sends 0x3D3 at 20Hz alongside the LKA command. Its curvature angle check
    # must not run on this platform: angle_meas holds the pinion angle in tenths of a
    # degree here, not a curvature, so the curvature limits would be applied to the wrong
    # unit scale and would reject the interleaved LKA commands.
    self.safety.set_controls_allowed(1)
    self._reset_speed_measurement(15.)
    self._reset_pinion_measurement(30.)
    for i in range(8):
      self.assertTrue(self._tx(self._lat_ctl_msg(False)), f"0x3D3 blocked on frame {i}")
      self.assertTrue(self._tx(self._lka_angle_msg(2, 5.0)), f"0x3CA blocked on frame {i}")

  def test_rx_hook_pinion_quality_flag(self):
    # The pinion angle is the origin every steering command is measured from, so an
    # uninitialised or degraded PSCM estimate must not be accepted.
    for quality_flag in (True, False):
      self.safety.set_controls_allowed(True)
      for _ in range(10):
        self.assertEqual(quality_flag, self._rx(self._pinion_angle_msg(0., quality_flag=quality_flag)))
        self.assertEqual(quality_flag, self.safety.get_controls_allowed())

  def test_rx_hook_pinion_has_no_counter(self):
    # This PSCM does not transmit StePinAn_No_Cs or StePinAn_No_Cnt: both are a constant
    # zero across 856k captured frames, and the DBC says the checksum is "not transmitted
    # on gas variants". Checking the counter would invalidate the message after
    # MAX_WRONG_COUNTERS frames and permanently disable controls on the van, which is why
    # .ignore_counter is set. The accompanying .ignore_checksum is inert rather than
    # load-bearing: neither ford_get_checksum nor ford_compute_checksum has a 0x07E case,
    # so both return 0 for this message and the comparison would pass either way.
    self.safety.set_controls_allowed(True)
    for _ in range(4 * common.MAX_WRONG_COUNTERS):
      assert self._rx(self._pinion_angle_msg(0.))
      assert self.safety.get_controls_allowed()

  def test_lateral_motion_control_2_is_gated_on_canfd_lka(self):
    # No platform sets both LKA_STEER and CANFD today, but ford_init accepts the
    # combination, and LateralMotionControl2 would then run a curvature check against a
    # pinion-angle angle_meas too.
    self.safety.set_safety_hooks(CarParams.SafetyModel.ford,
                                 FordSafetyFlags.LKA_STEER | FordSafetyFlags.CANFD)
    self.safety.init_tests()
    values = {
      "LatCtl_D2_Rq": 0,
      "LatCtlPathOffst_L_Actl": 0.,
      "LatCtlPath_An_Actl": 0.,
      "LatCtlCrv_NoRate2_Actl": 0.,
      "LatCtlCurv_No_Actl": 0.,
    }
    self.safety.set_controls_allowed(1)
    self._reset_pinion_measurement(30.)
    assert self._tx(self.packer.make_can_msg_safety("LateralMotionControl2", 0, values))
    # the curvature check must not have run, so the LKA command is still accepted
    assert self._tx(self._lka_angle_msg(2, 5.0))

    values["LatCtl_D2_Rq"] = 1
    assert not self._tx(self.packer.make_can_msg_safety("LateralMotionControl2", 0, values))


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


@pytest.mark.skipif(not TRANSIT_LOGS, reason="set TRANSIT_LOGS to the outputs directory with the recorded lka-can-*.npz routes")
class TestFordTransitReplay:
  """Replay every recorded LKA command from the Transit MK5 owner's drives through the
  LKA_STEER safety check (opendbc/safety/modes/ford.h) and confirm the check would not
  have rejected any of it.

  The safety limits in ford.h were derived, not driven, and this replay is the only
  evidence available without road-testing that the check does not reject ordinary
  driving. It is not proof the limits are tight enough, only that they are not absurdly
  tight.

  Private customer data: TRANSIT_LOGS must point at the directory containing the
  extracted lka-can-*.npz routes (never committed to this repo). The suite stays green
  and skips on machines without that data.
  """

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
    msg = self.packer.make_can_msg_safety("BrakeSysFeatures", 0, values, fix_checksum=checksum)
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


if __name__ == "__main__":
  unittest.main()

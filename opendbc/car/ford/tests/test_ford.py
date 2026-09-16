import random
import unittest

from hypothesis import settings, given, strategies as st

from opendbc.car.structs import CarParams
from opendbc.car.fw_versions import build_fw_dict
from opendbc.car.ford.values import CAR, FW_QUERY_CONFIG, FW_PATTERN, get_platform_codes, FordFlags
from opendbc.car.ford.fingerprints import FW_VERSIONS
from opendbc.testing import parameterized

Ecu = CarParams.Ecu


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
  Ecu.engine: [
    b"14C204",
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
  # Recorded on the owner's van on 2026-09-16. The ADAS and parkingAdas ECUs also answered
  # during that capture, but both responses have fw.logging=True and build_fw_dict() drops
  # logging entries (opendbc/car/fw_versions.py), so they never participate in fingerprint
  # matching and are intentionally not listed here.
  RECORDED = {
    (0x730, None): b'KK21-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x760, None): b'NK41-2D053-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x7E0, None): b'NK41-14C204-AFD\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x706, None): b'NK3T-14F397-AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    (0x764, None): b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
  }

  def test_transit_platform_exists(self):
    assert CAR.FORD_TRANSIT_MK5 in FW_VERSIONS
    assert CAR.FORD_TRANSIT_MK5.config.flags & FordFlags.LKA_STEER

  def test_recorded_firmware_is_listed(self):
    listed = FW_VERSIONS[CAR.FORD_TRANSIT_MK5]
    flat = {}
    for (_ecu, addr, sub), versions in listed.items():
      flat.setdefault(addr, []).extend(versions)
    assert set(flat) == {addr for addr, _sub in self.RECORDED}, "unexpected set of ECU addresses"
    for (addr, _sub), fw in self.RECORDED.items():
      assert fw in flat[addr], f"firmware {fw!r} missing for {hex(addr)}"
      # a single flipped byte must not still match
      corrupted = bytes([fw[0] ^ 0xFF]) + fw[1:]
      assert corrupted not in flat[addr], f"corrupted firmware should not match for {hex(addr)}"


from opendbc.car.structs import CarParams
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.values import FordSafetyFlags

TransmissionType = CarParams.TransmissionType


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


import math

from opendbc.can import CANPacker, CANParser
from opendbc.car.ford import fordcan


class TestTransitLkaMessage:
  def setup_method(self):
    self.packer = CANPacker("ford_lincoln_base_pt")
    # CanBus() bare (no CP, no fingerprint) asserts in CanBusBase.__init__;
    # every other caller in this codebase passes one or the other, so we do
    # the same here rather than loosen CanBus's default for test convenience.
    self.CAN = fordcan.CanBus(None, {0: {}})

  def _decode_lane_assist_data1(self, addr, dat):
    # Register the message via the constructor rather than relying on
    # CANParser's lazy registration on first `parser.vl[...]` access -
    # otherwise the first update() call registers-and-skips the frame
    # instead of parsing it, and vl[...] silently reads back the default 0.
    parser = CANParser("ford_lincoln_base_pt", [("Lane_Assist_Data1", 0)], 0)
    parser.update([(0, [(addr, dat, 0)])])
    return parser.vl["Lane_Assist_Data1"]

  def test_inactive_sends_zero_action(self):
    addr, dat, bus = fordcan.create_lka_msg(self.packer, self.CAN, False, 3.0, 4, 1)
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
    addr, dat, _bus = fordcan.create_lka_msg(self.packer, self.CAN, True, 3.0, 4, 1)
    assert (dat[0] >> 5) == 4

    vals = self._decode_lane_assist_data1(addr, dat)
    assert vals["LkaActvStats_D2_Req"] == 4

  def test_angle_is_clipped_to_the_wire_limit(self):
    addr, hi, _b = fordcan.create_lka_msg(self.packer, self.CAN, True, 99.0, 2, 0)
    _a, cap, _b = fordcan.create_lka_msg(self.packer, self.CAN, True, 5.8, 2, 0)
    assert hi == cap

    # The identity check above would still pass if clipping were broken in a
    # way that made both calls pack the same *wrong* value (e.g. always 0
    # mrad). Decode and check against the actual wire limit to rule that out.
    vals = self._decode_lane_assist_data1(addr, hi)
    assert math.isclose(vals["LaRefAng_No_Req"], math.radians(5.8) * 1000.0, abs_tol=0.05)


from opendbc.car import structs
from opendbc.car.car_helpers import interfaces


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

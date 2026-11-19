"""E1-E4: clock, IMU timing + attitude, NMEA/UBX, trajectory, frame ring."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from roadflag_edge.camera.ring import FrameRing
from roadflag_edge.clock import Clock
from roadflag_edge.gps.neo6m import Fix, NmeaAssembler, cfg_rate, nmea_checksum_ok, ubx
from roadflag_edge.gps.trajectory import InsufficientTrajectory, NoFix, Trajectory, offset_latlon
from roadflag_edge.imu.mpu6050 import G, SampleClock
from roadflag_edge.imu.orientation import GravityTracker, MountCalibration

SEC = 1_000_000_000


# ------------------------------------------------------------------ clock
def test_clock_offset_median_and_mapping():
    c = Clock()
    base = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)
    for k in range(10):
        jitter = (3 if k % 2 else -2) * 1_000_000           # +-3 ms arrival jitter
        c.add_gps_fix(5 * SEC + k * SEC // 5 + jitter, base + timedelta(seconds=k * 0.2))
    assert c.spread_ms() == pytest.approx(5.0)
    utc = c.mono_to_utc(5 * SEC)
    assert abs((utc - base).total_seconds()) < 0.004


def test_clock_without_fix_gives_none():
    assert Clock().mono_to_utc(123) is None


# -------------------------------------------------------------------- IMU
def test_sample_clock_absorbs_late_reads_and_drift():
    period = 5_000_000 * 1.01                                # chip 1 % slow
    sc = SampleClock(5_000_000)
    rng = np.random.default_rng(0)
    idx = 0
    for _ in range(400):                                     # 20 s of 50 ms drains
        idx += 10
        true_t = 1_000 * SEC + (idx - 1) * period
        sc.add(idx - 1, int(true_t + rng.uniform(0.2e6, 8e6)))   # reads are always late
    t = sc.times(idx - 10, 10)
    truth = 1_000 * SEC + np.arange(idx - 10, idx) * period
    assert np.max(np.abs(t - truth)) < 1.0e6                # within 1 ms
    assert sc.period_ns() == pytest.approx(period, rel=1e-3)


def test_mount_calibration_and_vertical_accel_on_tilted_mount():
    # sensor tilted 30 degrees about x; parked reading is gravity along the tilted up axis
    ang = np.radians(30)
    up = np.array([0, np.sin(ang), np.cos(ang)])
    rng = np.random.default_rng(1)
    parked = up * G + rng.normal(0, 0.02, (2000, 3))
    cal = MountCalibration.from_parked(parked, np.zeros((2000, 3)))
    assert np.allclose(cal.up_body, up, atol=0.01)
    tr = GravityTracker(cal)
    bump = np.tile(up * G, (200, 1))
    bump[100] += up * 15.0                                   # 15 m/s^2 vertical hit
    bump[100] += np.array([4.0, 0, 0])                       # plus some braking (horizontal)
    az, horiz = tr.update(bump, np.zeros((200, 3)))
    assert az[100] == pytest.approx(15.0, abs=0.3)
    assert np.max(np.abs(np.delete(az, 100))) < 0.2
    assert horiz[100] == pytest.approx(4.0, abs=0.3)


def test_mount_calibration_rejects_disconnected_sensor():
    with pytest.raises(ValueError):
        MountCalibration.from_parked(np.zeros((100, 3)), np.zeros((100, 3)))


# -------------------------------------------------------------------- GPS
def _nmea(body: str) -> str:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"${body}*{cs:02X}"


def test_nmea_rmc_gga_assemble_into_fix():
    asm = NmeaAssembler(latency_ns=80_000_000)
    rmc = _nmea("GPRMC,104107.40,A,3723.3460,N,12638.5800,E,16.33,92.0,191126,,,A")
    gga = _nmea("GPGGA,104107.40,3723.3460,N,12638.5800,E,1,08,1.1,20.0,M,18.0,M,,")
    assert nmea_checksum_ok(rmc) and not nmea_checksum_ok(rmc[:-1] + "0")
    assert asm.feed(rmc, 10 * SEC) is None
    fix = asm.feed(gga, 10 * SEC + 5_000_000)
    assert fix.valid and fix.quality == 1 and fix.sats == 8
    assert fix.lat == pytest.approx(37.38910, abs=1e-5) and fix.lon == pytest.approx(126.643, abs=1e-5)
    assert fix.speed_mps == pytest.approx(8.40, abs=0.01)
    assert fix.t_ns == 10 * SEC - 80_000_000
    assert fix.utc == datetime(2026, 11, 19, 10, 41, 7, 400000, tzinfo=timezone.utc)


def test_nmea_no_fix_indoors():
    asm = NmeaAssembler(0)
    asm.feed(_nmea("GPRMC,,V,,,,,,,,,,N"), 1)
    fix = asm.feed(_nmea("GPGGA,,,,,,0,00,99.99,,,,,,"), 2)
    assert fix is not None and not fix.valid


def test_ubx_checksum_known_message():
    # CFG-RATE 200 ms, navRate 1, timeRef GPS (the well-known u-center bytes)
    assert cfg_rate(200).hex() == "b56206080600c80001000100de6a"
    assert ubx(0x06, 0x08, bytes.fromhex("c80001000100")) == cfg_rate(200)


def _traj(speeds, dt=0.2, t0=100 * SEC, lat=37.0, lon=126.0, heading=90.0):
    tr = Trajectory(keep_s=60)
    pos = 0.0
    for k, v in enumerate(speeds):
        if k:
            pos += 0.5 * (speeds[k - 1] + v) * dt
        la, lo = offset_latlon(lat, lon, heading, pos)
        tr.add(Fix(t0 + int(k * dt * SEC), la, lo, v, heading, 1.0, 1, None))
    return tr


def test_time_before_distance_constant_speed():
    tr = _traj([10.0] * 20)
    end = 100 * SEC + 3 * SEC
    assert (end - tr.time_before_distance(end, 4.0)) / SEC == pytest.approx(0.4, abs=1e-6)


def test_time_before_distance_accelerating():
    speeds = [5.0 + 0.2 * k for k in range(30)]               # 5 m/s accelerating 1 m/s^2
    tr = _traj(speeds)
    end = 100 * SEC + int(4.1 * SEC)
    t_f = tr.time_before_distance(end, 4.0)
    # check by integrating v(t) = 5 + t over [t_f, end]
    a, b = (t_f - 100 * SEC) / SEC, 4.1
    assert (5 * b + b * b / 2) - (5 * a + a * a / 2) == pytest.approx(4.0, abs=1e-6)


def test_trajectory_refuses_gaps_and_extrapolation():
    tr = _traj([10.0] * 10)
    with pytest.raises(InsufficientTrajectory):
        tr.time_before_distance(100 * SEC + SEC // 10, 4.0)    # only 1 m of history
    with pytest.raises(InsufficientTrajectory):
        tr.state_at(200 * SEC)
    gap = Trajectory()
    gap.add(Fix(0, 37, 126, 10, 0, 1, 1, None))
    gap.add(Fix(2 * SEC, 37, 126, 10, 0, 1, 1, None))
    with pytest.raises(InsufficientTrajectory):
        gap.state_at(SEC)


def test_bad_hdop_is_no_fix():
    tr = Trajectory(max_hdop=5)
    tr.add(Fix(0, 37, 126, 10, 0, 7.5, 1, None))
    tr.add(Fix(SEC // 5, 37, 126, 10, 0, 7.5, 1, None))
    with pytest.raises(NoFix):
        tr.fix_quality_at(SEC // 10)


def test_position_interpolates_and_antenna_offset():
    tr = _traj([10.0] * 10)
    st = tr.state_at(100 * SEC + SEC // 2)                    # 5 m east of start
    la, lo = offset_latlon(37.0, 126.0, 90.0, 5.0)
    assert st["lat"] == pytest.approx(la, abs=1e-8) and st["lon"] == pytest.approx(lo, abs=1e-8)
    la2, lo2 = offset_latlon(la, lo, 90.0, 1.5)
    assert (lo2 - lo) * 111_320 * np.cos(np.radians(37)) == pytest.approx(1.5, rel=0.01)


# ----------------------------------------------------------------- camera
def test_frame_ring_nearest_neighbours_and_edge():
    ring = FrameRing(5, 8, 4)
    for k in range(7):                                       # 7 pushed, 5 kept: seq 2..6
        ring.push(k * 41_666_667, k, np.full((6, 8), k, np.uint8))
    assert len(ring) == 5 and ring.span()[0] == 2 * 41_666_667
    f, nb, err, edge = ring.nearest(4 * 41_666_667 + 3_000_000)
    assert f.seq == 4 and err == -3_000_000 and not edge and [n.seq for n in nb] == [3, 5]
    assert f.yuv[0, 0] == 4
    ring.push(7 * 41_666_667, 7, np.full((6, 8), 7, np.uint8))  # overwrite: copies stay intact
    assert f.yuv[0, 0] == 4
    *_, edge = ring.nearest(0)
    assert edge


def test_frame_ring_counts_dropped_frames():
    ring = FrameRing(10, 2, 2)
    for seq in (1, 2, 5, 6):
        ring.push(seq * 1000, seq, np.zeros((3, 2), np.uint8))
    assert ring.dropped == 2

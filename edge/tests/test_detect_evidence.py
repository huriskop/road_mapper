"""E5-E6: trigger, localisation and frame selection on synthetic signals."""
import numpy as np
import pytest

from roadflag_edge.camera.ring import FrameRing
from roadflag_edge.detect.detector import Detector, DetectorConfig
from roadflag_edge.detect.localize import ImuRing, localize
from roadflag_edge.evidence.select import EvidenceConfig, EvidenceStatus, select_evidence
from roadflag_edge.gps.neo6m import Fix
from roadflag_edge.gps.trajectory import Trajectory, offset_latlon

FS = 200
SEC = 1_000_000_000
DT = SEC // FS


def hit(n, at, amp=15.0, f=12.0, decay=0.08):
    """Damped ringing starting at sample `at` (what a wheel hit looks like after the suspension)."""
    x = np.zeros(n)
    k = np.arange(n - at) / FS
    x[at:] = amp * np.sin(2 * np.pi * f * k) * np.exp(-k / decay)
    return x


def run(det, az, speed=8.0, t0=0, batch=10):
    t = t0 + np.arange(len(az), dtype=np.int64) * DT
    out = []
    for i in range(0, len(az), batch):
        out += det.process(t[i:i + batch], az[i:i + batch], speed)
    return out, t


def test_single_impulse_one_candidate():
    az = hit(4 * FS, 2 * FS)
    det = Detector()
    cands, t = run(det, az)
    assert len(cands) == 1
    c = cands[0]
    assert c.done.is_set()                    # refractory over before the data ends
    assert t[2 * FS] <= c.t_d <= t[2 * FS] + int(0.25 * SEC)
    assert not c.axle2_seen and c.score > 1


def test_two_axles_one_wheelbase_apart_is_one_event():
    v = 8.0
    gap = int(2.7 / v * FS)                   # rear axle 2.7 m later
    az = hit(5 * FS, 2 * FS) + hit(5 * FS, 2 * FS + gap, amp=10)
    det = Detector()
    cands, _ = run(det, az, speed=v)
    assert len(cands) == 1 and cands[0].axle2_seen


def test_two_separate_potholes_are_two_events():
    az = hit(8 * FS, 2 * FS) + hit(8 * FS, 5 * FS)
    cands, _ = run(Detector(), az)
    assert len(cands) == 2


def test_noise_and_slow_body_motion_do_not_trigger():
    from scipy import signal

    rng = np.random.default_rng(3)
    n = 60 * FS
    t = np.arange(n) / FS
    rough = signal.sosfilt(signal.butter(2, (0.5, 20), "bandpass", fs=FS, output="sos"), rng.normal(0, 1, n))
    rough *= 1.0 / rough.std()                                       # rough road, 1 m/s^2 rms
    az = rough + rng.normal(0, 0.05, n) + 2.0 * np.sin(2 * np.pi * 0.3 * t)   # + sensor noise + pitch
    cands, _ = run(Detector(), az)
    assert cands == []


def test_parked_door_slam_ignored():
    det = Detector()
    cands, _ = run(det, hit(3 * FS, FS), speed=0.0)
    assert cands == [] and det.dropped_parked == 1


def test_filter_state_reset_between_cases():
    det = Detector()
    run(det, np.full(FS, 50.0))               # big DC step leaves filter state behind
    det.reset()
    cands, _ = run(det, np.zeros(2 * FS))
    assert cands == []


def test_speed_raises_threshold():
    cfg = DetectorConfig(tau_a0=6, tau_a1=0.5, tau_j0=1e9)
    az = hit(4 * FS, 2 * FS, amp=11.0)                             # ~7.6 m/s^2 after the high-pass
    assert len(run(Detector(cfg), az, speed=2.0)[0]) == 1          # tau 7
    assert len(run(Detector(cfg), az, speed=20.0)[0]) == 0         # tau 16


def test_localize_finds_impact_without_filter_delay():
    az = hit(2 * FS, FS) + np.random.default_rng(0).normal(0, 0.3, 2 * FS)
    t = np.arange(2 * FS, dtype=np.int64) * DT
    loc = localize(t, az, t_d=int(1.2 * SEC))
    # first peak of a 12 Hz sine is a quarter period (~21 ms) after onset
    assert abs(loc["t_i"] - (SEC + int(SEC / 48))) <= 2 * DT
    assert loc["peak_a"] == pytest.approx(15 * np.exp(-1 / 48 / 0.08), rel=0.15)   # damped first peak


def test_imu_ring_wraps_and_windows():
    r = ImuRing(seconds=1.0, fs=FS)
    for k in range(5):
        t = np.arange(k * 100, (k + 1) * 100, dtype=np.int64)
        r.push(t, t.astype(float), np.zeros(100), np.zeros(100), 1)
    w = r.window(350, 420)
    assert list(w["t"]) == list(range(350, 421)) and r.newest_ns() == 499


# ---------------------------------------------------------------- evidence
def make_world(speed=8.0, fps=24.0, ring_s=5.0, drop=()):
    tr = Trajectory(keep_s=60)
    for k in range(60):                                    # 12 s at 5 Hz, heading east
        la, lo = offset_latlon(37.0, 126.0, 90.0, speed * k * 0.2)
        tr.add(Fix(k * SEC // 5, la, lo, speed, 90.0, 1.0, 1, None))
    ring = FrameRing(int(ring_s * fps), 4, 2)
    period = SEC / fps
    k = 0
    while k * period < 10 * SEC:
        if k not in drop:
            ring.push(int(k * period), k, np.zeros((3, 4), np.uint8))
        k += 1
    return tr, ring


def test_evidence_available_picks_frame_4m_back():
    tr, ring = make_world()
    t_i = int(9.0 * SEC)
    ev = select_evidence(t_i, tr, ring, EvidenceConfig(lookahead_m=4.0, antenna_to_front_wheel_m=1.5))
    assert ev.status == EvidenceStatus.AVAILABLE
    assert (t_i - ev.t_f) / SEC == pytest.approx(0.5, abs=1e-6)   # 4 m at 8 m/s
    assert abs(ev.alignment_error_ms) <= 21                        # half a frame
    assert len(ev.neighbours) == 2
    # defect = vehicle at t_i (72 m east) + 1.5 m antenna offset
    la, lo = offset_latlon(37.0, 126.0, 90.0, 72 + 1.5)
    assert ev.defect_latlon == pytest.approx((la, lo), abs=1e-7)


def test_evidence_low_speed():
    tr, ring = make_world(speed=1.0)
    assert select_evidence(int(9 * SEC), tr, ring, EvidenceConfig()).status == EvidenceStatus.LOW_SPEED


def test_evidence_buffer_miss_when_target_older_than_ring():
    tr, ring = make_world(speed=2.5)                  # 4 m back = 1.6 s
    ev = select_evidence(int(6.0 * SEC), tr, ring, EvidenceConfig())   # ring holds 5..10 s
    assert ev.status == EvidenceStatus.BUFFER_MISS


def test_evidence_alignment_rejected_on_dropped_frames():
    target_frame = round(8.5 * 24)            # t_f = 8.5 s
    tr, ring = make_world(drop=set(range(target_frame - 2, target_frame + 3)))
    ev = select_evidence(int(9.0 * SEC), tr, ring, EvidenceConfig())
    assert ev.status == EvidenceStatus.ALIGNMENT_REJECTED and abs(ev.alignment_error_ms) > 30


def test_evidence_no_gps_and_insufficient_trajectory_and_no_camera():
    tr, ring = make_world()
    assert select_evidence(int(30 * SEC), tr, ring, EvidenceConfig()).status == EvidenceStatus.NO_GPS_FIX
    ev = select_evidence(int(0.3 * SEC), tr, ring, EvidenceConfig())
    assert ev.status == EvidenceStatus.INSUFFICIENT_TRAJECTORY and ev.defect_latlon is not None
    assert select_evidence(int(9 * SEC), tr, None, EvidenceConfig()).status == EvidenceStatus.NO_CAMERA_DATA

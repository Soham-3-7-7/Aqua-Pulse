"""
Adaptive Software-Defined Sonar (SDS) Payload - Full Pipeline Simulation
=========================================================================
Runs out-of-the-box in Google Colab (needs only numpy + matplotlib).

Pipeline simulated, one iteration per "ping cycle":

    Sensor frame (bytes + CRC-16/Modbus)
        -> TelemetryGatekeeper  (timeout / CRC / NaN / range / rate-of-change,
                                 drop whole frame, fall back to Last_Valid)
        -> Physics              (Mackenzie speed of sound c, attenuation index D)
        -> AdaptiveModulationEngine (3-tier priority, hysteresis, debounce)
        -> Waveform synthesis   (LFM chirp / geometric sweep / Barker-13 BPSK)

A "naive" baseline (no gatekeeper, no hysteresis) runs alongside so the
benefit of the gatekeeper is visible in the plots.

Scenarios (400 pings, 100 each):
    A  Clear water, low noise          -> LFM chirp      350-450 kHz @ 50 %
    B  Deep / muddy water, low noise   -> Geometric      105-135 kHz @ 100 %
    C  High ambient noise (> 0.7)      -> Barker-13      @ 100 %
    D  Corrupted packets / spikes      -> Gatekeeper drops them, mode unchanged

Note on "drive power": the drive level is applied as a DAC amplitude scale
factor (0.5 or 1.0). If you want true acoustic power, power ~ amplitude^2.
"""

from __future__ import annotations

import math
import struct
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================
SEED = 7
N_PINGS = 400
PING_PERIOD_S = 0.5
SCENARIO_EDGES = [0, 100, 200, 300, 400]
SCENARIO_NAMES = ["A: Clear, low noise", "B: Deep/muddy, low noise",
                  "C: High ambient noise", "D: Corrupted packets"]
SCEN_COLORS = ["#e8f1fb", "#e8f5e9", "#f3e9f7", "#fff1e0"]

FIELDS = ("T", "Turb", "Sal", "Depth")
LIMITS = {"T": (-2.0, 40.0), "Turb": (0.0, 1000.0),
          "Sal": (0.0, 11476.0), "Depth": (0.0, 300.0)}
# Max plausible change per ping (rate-of-change / delta limits)
MAX_DELTA = {"T": 1.0, "Turb": 150.0, "Sal": 400.0, "Depth": 12.0}
SAFE_DEFAULT = {"T": 15.0, "Turb": 100.0, "Sal": 10000.0, "Depth": 50.0}
MAX_CONSEC_DROPS = 10          # raise TELEMETRY_STALE beyond this

# Decision-engine thresholds (enter / exit = hysteresis)
NOISE_ENTER, NOISE_EXIT = 0.70, 0.60
D_ENTER, D_EXIT = 0.40, 0.35
N_CONFIRM, MIN_DWELL = 3, 5

# Waveform modes
LFM, EXP, BARKER = "LFM", "EXP", "BARKER"
MODE_LEVEL = {LFM: 1, EXP: 2, BARKER: 3}
LEVEL_MODE = {v: k for k, v in MODE_LEVEL.items()}
MODE_LABEL = {LFM: "LFM Chirp", EXP: "Geometric Sweep", BARKER: "Barker-13 BPSK"}
MODE_COLOR = {LFM: "#1f77b4", EXP: "#2ca02c", BARKER: "#8e44ad"}
DRIVE = {LFM: 0.5, EXP: 1.0, BARKER: 1.0}          # drive level per mode
RED, BLUE = "#d62728", "#1f77b4"

# Acoustic parameters (sample rate chosen for clean visualisation only)
FS = 5e6                        # 5 MSPS simulation rate
PULSE_S = 260e-6                # all pulses 260 us
LFM_F0, LFM_F1 = 350e3, 450e3
EXP_F0, EXP_F1 = 105e3, 135e3
BARKER_FC = 200e3
BARKER_CYCLES_PER_CHIP = 4      # 4 cycles @ 200 kHz = 20 us/chip, 13 chips = 260 us
BARKER_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

# Faults injected in Scenario D  {ping_index: fault_type}
FAULT_SCHEDULE = {305: "crc_subtle", 315: "range", 325: "nan", 335: "timeout",
                  345: "delta", 346: "delta", 347: "delta",
                  360: "crc_gross", 375: "delta"}


# =============================================================================
# 2. TELEMETRY FRAMING, CRC & THE INTEGRITY GATEKEEPER
# =============================================================================
def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS (poly 0xA001 reflected, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def build_frame(T: float, turb: float, sal: float, depth: float) -> bytes:
    """Pack 4 float32 features + CRC-16 into an 18-byte telemetry frame."""
    payload = struct.pack("<4f", T, turb, sal, depth)
    return payload + struct.pack("<H", crc16_modbus(payload))


def parse_raw_values(frame):
    """Decode payload ignoring CRC (used for plotting / naive baseline)."""
    if frame is None or len(frame) != 18:
        return (math.nan,) * 4
    return struct.unpack("<4f", frame[:16])


class TelemetryGatekeeper:
    """
    High-bandwidth telemetry integrity gatekeeper.

    A frame is accepted only if ALL checks pass:
        1. arrived before timeout            5. every field inside sensor range
        2. correct length                    6. change vs Last_Valid within
        3. CRC-16 matches                       the per-ping delta limit
        4. no NaN / Inf
    Otherwise the WHOLE frame is dropped (atomic - no partial update) and
    Last_Valid_Telemetry is returned so downstream stages never see bad data.
    The delta limit widens by (1 + consecutive_drops) so a genuine, sustained
    change can re-sync instead of being rejected forever.
    """

    def __init__(self, limits=LIMITS, max_delta=MAX_DELTA,
                 max_drops=MAX_CONSEC_DROPS):
        self.limits, self.max_delta, self.max_drops = limits, max_delta, max_drops
        self.last_valid = None            # Last_Valid_Telemetry
        self.consecutive_drops = 0
        self.total_frames = 0
        self.total_drops = 0
        self.stale = False                # TELEMETRY_STALE flag
        self.reasons = Counter()

    def _validate(self, frame):
        """Return (values_dict, None) if valid, else (None, reason)."""
        if frame is None:
            return None, "TIMEOUT"
        if len(frame) != 18:
            return None, "LENGTH"
        payload = frame[:16]
        (rx_crc,) = struct.unpack("<H", frame[16:])
        if crc16_modbus(payload) != rx_crc:
            return None, "CRC_FAIL"
        vals = dict(zip(FIELDS, struct.unpack("<4f", payload)))
        if any(not math.isfinite(v) for v in vals.values()):
            return None, "NAN_INF"
        for k in FIELDS:
            lo, hi = self.limits[k]
            if not lo <= vals[k] <= hi:
                return None, "RANGE"
        if self.last_valid is not None:
            scale = 1 + self.consecutive_drops
            for k in FIELDS:
                if abs(vals[k] - self.last_valid[k]) > self.max_delta[k] * scale:
                    return None, "DELTA"
        return vals, None

    def process(self, frame):
        """
        Returns (telemetry_used, accepted, drop_reason).
        telemetry_used is always a clean dict: the new frame if valid,
        otherwise Last_Valid_Telemetry (or safe defaults before first valid frame).
        """
        self.total_frames += 1
        vals, reason = self._validate(frame)
        if vals is not None:                                   # commit atomically
            self.last_valid = vals
            self.consecutive_drops = 0
            self.stale = False
            return dict(vals), True, None
        self.total_drops += 1                                  # drop entire frame
        self.consecutive_drops += 1
        self.reasons[reason] += 1
        self.stale = self.consecutive_drops > self.max_drops
        fallback = self.last_valid if self.last_valid else SAFE_DEFAULT
        return dict(fallback), False, reason


# =============================================================================
# 3. ENVIRONMENTAL PHYSICS
# =============================================================================
def speed_of_sound(T: float, sal_ppm: float, depth_m: float) -> float:
    """
    Simplified Mackenzie equation (m/s).
    T in degC, salinity converted ppm -> PSU (ppm/1000, first-order estimate),
    depth in metres.
    """
    S = sal_ppm / 1000.0
    return (1448.96 + 4.591 * T - 0.05304 * T ** 2 + 2.374e-4 * T ** 3
            + 1.340 * (S - 35.0) + 0.0163 * depth_m)


def attenuation_index(turb_ntu: float, depth_m: float) -> float:
    """Acoustic attenuation index D = 0.7*(Turb/1000) + 0.3*(Depth/300), clamped 0..1."""
    return float(np.clip(0.7 * (turb_ntu / 1000.0) + 0.3 * (depth_m / 300.0), 0.0, 1.0))


# =============================================================================
# 4. 3-TIER ADAPTIVE MODULATION ENGINE
# =============================================================================
class AdaptiveModulationEngine:
    """
    Priority 1: interference (noise > 0.7) or host SET_MODE_CODED -> Barker-13
                (committed immediately, 100 % drive)
    Priority 2: no interference, D <  0.4                        -> LFM  @ 50 %
    Priority 3: no interference, D >= 0.4                        -> EXP  @ 100 %
    Hysteresis (separate enter/exit thresholds) + debounce
    (N_CONFIRM consecutive cycles and MIN_DWELL pings) prevent mode chattering
    for Priority 2/3 changes.
    """

    def __init__(self):
        self.mode = None                  # None -> cold start commits immediately
        self.pending, self.confirm, self.dwell = None, 0, 0
        self.in_interference = False
        self.in_high_atten = False

    def _commit(self, target):
        self.mode, self.pending, self.confirm, self.dwell = target, None, 0, 0

    def update(self, D: float, noise: float, host_cmd=None):
        """Run one decision cycle. Returns (mode, drive_level)."""
        # --- hysteresis-updated condition flags ---
        if noise > NOISE_ENTER:
            self.in_interference = True
        elif noise < NOISE_EXIT:
            self.in_interference = False
        if D >= D_ENTER:
            self.in_high_atten = True
        elif D < D_EXIT:
            self.in_high_atten = False

        # --- priority resolution ---
        if host_cmd == "SET_MODE_CODED" or self.in_interference:
            target, immediate = BARKER, True
        elif self.in_high_atten:
            target, immediate = EXP, False
        else:
            target, immediate = LFM, False

        # --- commit / debounce ---
        if self.mode is None or (immediate and target != self.mode):
            self._commit(target)
        elif target != self.mode:
            if target == self.pending:
                self.confirm += 1
            else:
                self.pending, self.confirm = target, 1
            if self.confirm >= N_CONFIRM and self.dwell >= MIN_DWELL:
                self._commit(target)
        else:
            self.pending, self.confirm = None, 0
        self.dwell += 1
        return self.mode, DRIVE[self.mode]


class NaiveBaseline:
    """Reference pipeline: trusts raw data, stateless thresholds, no hysteresis."""

    def __init__(self):
        self.prev = dict(SAFE_DEFAULT)

    def process(self, frame, noise):
        vals = parse_raw_values(frame)
        if any(not math.isfinite(v) for v in vals):        # only guards NaN/timeout
            vals = tuple(self.prev[k] for k in FIELDS)
        self.prev = dict(zip(FIELDS, vals))
        c = speed_of_sound(self.prev["T"], self.prev["Sal"], self.prev["Depth"])
        D = attenuation_index(self.prev["Turb"], self.prev["Depth"])
        mode = BARKER if noise > NOISE_ENTER else (EXP if D >= D_ENTER else LFM)
        return c, D, mode


# =============================================================================
# 5. WAVEFORM GENERATION
# =============================================================================
def _taper(n: int, alpha: float = 0.1) -> np.ndarray:
    """Tukey-style edge taper to reduce spectral leakage of the chirps."""
    w = np.ones(n)
    m = int(alpha * (n - 1) / 2)
    if m > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m], w[-m:] = ramp, ramp[::-1]
    return w


def generate_lfm(fs, dur, f0, f1):
    """Linear FM: f(t) = f0 + k t, phase = 2*pi*(f0 t + k t^2 / 2)."""
    t = np.arange(int(round(fs * dur))) / fs
    k = (f1 - f0) / dur
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2)) * _taper(t.size)


def generate_geometric(fs, dur, f0, f1):
    """
    Geometric (exponential) sweep: f(t) = f0 * r^(t/dur), r = f1/f0.
    Phase = 2*pi*f0*dur/ln(r) * (r^(t/dur) - 1); dwells longer at low frequency.
    """
    t = np.arange(int(round(fs * dur))) / fs
    r = f1 / f0
    phase = 2 * np.pi * f0 * dur / math.log(r) * (r ** (t / dur) - 1.0)
    return np.sin(phase) * _taper(t.size)


def generate_barker13(fs, fc, cycles_per_chip):
    """Barker-13 BPSK: fixed carrier, +/-180 deg phase flip per chip."""
    spc = int(round(fs * cycles_per_chip / fc))            # samples per chip
    t = np.arange(13 * spc) / fs
    return np.repeat(BARKER_13, spc) * np.sin(2 * np.pi * fc * t)


def build_pulse_bank():
    """Pre-synthesise unit-amplitude pulses (stand-in for wave tables in flash)."""
    bank = {LFM: generate_lfm(FS, PULSE_S, LFM_F0, LFM_F1),
            EXP: generate_geometric(FS, PULSE_S, EXP_F0, EXP_F1),
            BARKER: generate_barker13(FS, BARKER_FC, BARKER_CYCLES_PER_CHIP)}
    bank["t"] = np.arange(bank[LFM].size) / FS
    return bank


def synthesize(mode, drive, bank):
    """Select the pre-built pulse and apply the drive level (DMA -> DAC stand-in)."""
    return bank[mode] * drive


# =============================================================================
# 6. DUMMY DATASET (4 SCENARIOS) & FAULT INJECTION
# =============================================================================
def generate_ground_truth(rng):
    """Smooth ground-truth environment + noise floor for all 4 scenarios."""
    idx = np.arange(N_PINGS)
    wp = [0, 100, 120, 200, 250, 280, 300, N_PINGS]        # scenario waypoints

    def profile(values):
        return np.interp(idx, wp, values)

    truth = {
        "T": profile([22, 22, 12, 12, 12, 20, 20, 20]) + rng.normal(0, 0.05, N_PINGS),
        "Turb": np.clip(profile([30, 30, 700, 700, 700, 30, 30, 30])
                        + rng.normal(0, 3, N_PINGS), 0, None),
        "Sal": profile([9500, 9500, 10500, 10500, 10500, 9800, 9800, 9800])
               + rng.normal(0, 20, N_PINGS),
        "Depth": np.clip(profile([20, 20, 180, 180, 180, 30, 30, 30])
                         + rng.normal(0, 0.3, N_PINGS), 0, None),
    }
    in_c = (idx >= 200) & (idx < 300)
    noise = np.where(in_c, rng.normal(0.85, 0.05, N_PINGS),
                     rng.normal(0.15, 0.03, N_PINGS))
    return truth, np.clip(noise, 0.0, 1.0)


def make_frame(vals, fault=None):
    """Build a frame; optionally corrupt it in one of several realistic ways."""
    T, turb, sal, depth = vals
    if fault == "timeout":
        return None                                        # sensor never answered
    if fault == "range":                                   # valid CRC, absurd values
        turb, sal = 5000.0, 60000.0
    elif fault == "nan":                                   # valid CRC, NaN payload
        turb = float("nan")
    elif fault == "delta":                                 # in-range but impossible jump
        T, turb, sal, depth = T + 15.0, 950.0, 11300.0, 290.0
    frame = bytearray(build_frame(T, turb, sal, depth))
    if fault == "crc_subtle":                              # 1-bit flip (value x2 or /2)
        frame[6] ^= 0x80
    elif fault == "crc_gross":                             # multi-bit corruption
        frame[7] ^= 0x3F
    return bytes(frame)


# =============================================================================
# 7. SIMULATION LOOP
# =============================================================================
def run_simulation():
    """Stream every ping through gatekeeper -> physics -> decision -> synthesis."""
    rng = np.random.default_rng(SEED)
    truth, noise = generate_ground_truth(rng)
    bank = build_pulse_bank()
    gk, engine, naive = TelemetryGatekeeper(), AdaptiveModulationEngine(), NaiveBaseline()

    keys = ["c_g", "D_g", "mode_g", "drive_g", "c_n", "D_n", "mode_n",
            "accepted", "stale", "synth_ok"]
    log = {k: [] for k in keys}
    log["raw"] = {k: [] for k in FIELDS}
    log["used"] = {k: [] for k in FIELDS}
    log["reason"] = []

    for n in range(N_PINGS):
        vals = tuple(float(truth[k][n]) for k in FIELDS)
        frame = make_frame(vals, FAULT_SCHEDULE.get(n))

        # --- Step 1: integrity gatekeeper ---
        used, accepted, reason = gk.process(frame)
        # --- Step 2: physics on validated telemetry only ---
        c = speed_of_sound(used["T"], used["Sal"], used["Depth"])
        D = attenuation_index(used["Turb"], used["Depth"])
        # --- Step 3: 3-tier decision engine ---
        mode, drive = engine.update(D, float(noise[n]))
        # --- Step 4: waveform synthesis (never interrupted by dropped packets) ---
        pulse = synthesize(mode, drive, bank)

        # --- naive comparison pipeline ---
        c_n, D_n, mode_n = naive.process(frame, float(noise[n]))

        raw = parse_raw_values(frame)
        for k, r in zip(FIELDS, raw):
            log["raw"][k].append(r)
            log["used"][k].append(used[k])
        log["c_g"].append(c); log["D_g"].append(D)
        log["mode_g"].append(MODE_LEVEL[mode]); log["drive_g"].append(drive)
        log["c_n"].append(c_n); log["D_n"].append(D_n)
        log["mode_n"].append(MODE_LEVEL[mode_n])
        log["accepted"].append(accepted); log["stale"].append(gk.stale)
        log["synth_ok"].append(bool(np.any(pulse != 0)))
        log["reason"].append(reason)

    out = {k: np.array(v) for k, v in log.items() if k not in ("raw", "used", "reason")}
    out["raw"] = {k: np.array(v) for k, v in log["raw"].items()}
    out["used"] = {k: np.array(v) for k, v in log["used"].items()}
    out["reason"] = log["reason"]
    out["noise"] = noise
    out["t"] = np.arange(N_PINGS) * PING_PERIOD_S
    return out, gk


# =============================================================================
# 8. TEXT SUMMARY
# =============================================================================
def print_summary(log, gk):
    """Console report with per-scenario results and pass/fail checks."""
    expected = [LFM, EXP, BARKER, LFM]
    print("=" * 84)
    print("SDS PIPELINE SIMULATION SUMMARY")
    print("=" * 84)
    print(f"{'Scenario':<27}{'Dominant mode':<18}{'Drive':<8}{'Drops':<7}"
          f"{'Switches (gated/naive)':<24}{'Check'}")
    print("-" * 84)
    for i, name in enumerate(SCENARIO_NAMES):
        s = slice(SCENARIO_EDGES[i], SCENARIO_EDGES[i + 1])
        lv = log["mode_g"][s]
        vals, counts = np.unique(lv, return_counts=True)
        dom = LEVEL_MODE[int(vals[counts.argmax()])]
        drops = int((~log["accepted"][s]).sum())
        sw_g = int(np.count_nonzero(np.diff(lv)))
        sw_n = int(np.count_nonzero(np.diff(log["mode_n"][s])))
        settled = log["mode_g"][SCENARIO_EDGES[i + 1] - 20:SCENARIO_EDGES[i + 1]]
        ok = np.all(settled == MODE_LEVEL[expected[i]])
        print(f"{name:<27}{MODE_LABEL[dom]:<18}{DRIVE[dom] * 100:>4.0f}%   {drops:<7}"
              f"{f'{sw_g} / {sw_n}':<24}{'PASS' if ok else 'FAIL'}")
    print("-" * 84)

    w = slice(310, N_PINGS)                                # after C->D handover
    g_sw = int(np.count_nonzero(np.diff(log["mode_g"][w])))
    n_sw = int(np.count_nonzero(np.diff(log["mode_n"][w])))
    c_g = np.ptp(log["c_g"][w]); c_n = np.ptp(log["c_n"][w])
    print(f"Frames processed          : {gk.total_frames}")
    print(f"Frames dropped            : {gk.total_drops}  {dict(gk.reasons)}")
    print(f"TELEMETRY_STALE raised    : {bool(log['stale'].any())}")
    print(f"Pulses synthesised        : {int(log['synth_ok'].sum())}/{N_PINGS} "
          f"(synthesis never interrupted: {bool(log['synth_ok'].all())})")
    print(f"Scenario D false mode flips: gatekeeper={g_sw}  naive={n_sw}")
    print(f"Scenario D speed-of-sound swing (peak-to-peak): "
          f"gatekeeper={c_g:.2f} m/s  naive={c_n:.1f} m/s")
    print("=" * 84)


# =============================================================================
# 9. VISUALISATION
# =============================================================================
def set_plot_style():
    """Consistent, publication-style Matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True,
        "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _shade(ax, label=False):
    """Colour the background of each scenario region."""
    for i in range(4):
        x0 = SCENARIO_EDGES[i] * PING_PERIOD_S
        x1 = SCENARIO_EDGES[i + 1] * PING_PERIOD_S
        ax.axvspan(x0, x1, color=SCEN_COLORS[i], alpha=0.7, lw=0, zorder=0)
        if label:
            ax.text((x0 + x1) / 2, 0.96, SCENARIO_NAMES[i],
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=9, fontweight="bold")


def _mark_drops(ax, t, drop):
    """Faint red vertical lines at every dropped packet."""
    for td in t[drop]:
        ax.axvline(td, color=RED, alpha=0.3, lw=1, zorder=1)


def plot_telemetry(log):
    """Plot 1: sensor input telemetry, valid data vs dropped packets."""
    t, acc = log["t"], log["accepted"]
    drop = ~acc
    panels = [("T", "Temperature (°C)", (0, 40)),
              ("Turb", "Turbidity (NTU)", (0, 1000)),
              ("Sal", "Salinity (ppm)", (0, 12000)),
              ("Depth", "Depth (m)", (0, 300))]
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    for i, (ax, (key, ylabel, ylim)) in enumerate(zip(axes[:4], panels)):
        _shade(ax, label=(i == 0))
        _mark_drops(ax, t, drop)
        ax.plot(t, log["used"][key], color=BLUE, lw=1.5, drawstyle="steps-post",
                label="Gatekeeper output (valid / Last_Valid)", zorder=3)
        raw = log["raw"][key]
        bad = drop & np.isfinite(raw)
        ax.plot(t[bad], np.clip(raw[bad], *ylim), "x", color=RED, ms=7, mew=1.8,
                label="Dropped packet (raw value, clipped to axis)", zorder=4)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83))
    ax = axes[4]
    _shade(ax)
    ax.plot(t, log["noise"], color="0.15", lw=1.1, label="Ambient noise floor (ADC A3)")
    ax.axhline(NOISE_ENTER, color=RED, ls="--", lw=1, label="Enter interference (0.70)")
    ax.axhline(NOISE_EXIT, color="darkorange", ls=":", lw=1.2, label="Exit interference (0.60)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Noise floor (0-1)")
    ax.set_xlabel(f"Time (s)   [ping period = {PING_PERIOD_S} s]")
    ax.legend(loc="center left", ncol=1)
    fig.suptitle("Plot 1: Sensor input telemetry: valid data vs dropped packets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_modes(log):
    """Plot 2: physics outputs, selected waveform mode and drive power."""
    t, drop = log["t"], ~log["accepted"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.1, 0.8]})
    for ax in axes:
        _shade(ax)
        _mark_drops(ax, t, drop)

    ax = axes[0]                                           # speed of sound
    ax.plot(t, log["c_n"], color="0.5", lw=1, ls="--", label="Naive (raw data, no gatekeeper)")
    ax.plot(t, log["c_g"], color=BLUE, lw=1.6, label="Gatekeeper (Last_Valid fallback)")
    ax.set_ylabel("Speed of sound c (m/s)")
    ax.legend(loc="upper right")

    ax = axes[1]                                           # attenuation index
    ax.plot(t, np.clip(log["D_n"], 0, 1.05), color="0.5", lw=1, ls="--",
            label="Naive D (clipped)")
    ax.plot(t, log["D_g"], color=BLUE, lw=1.6, label="Gatekeeper D")
    ax.axhline(D_ENTER, color="k", ls="--", lw=0.9)
    ax.axhline(D_EXIT, color="k", ls=":", lw=0.9)
    ax.text(N_PINGS * PING_PERIOD_S * 0.995, D_ENTER + 0.02, "D = 0.40 (enter)",
            ha="right", fontsize=8)
    ax.text(N_PINGS * PING_PERIOD_S * 0.995, D_EXIT - 0.07, "D = 0.35 (exit)",
            ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attenuation index D")
    ax.legend(loc="upper left")

    ax = axes[2]                                           # selected mode
    ax.step(t, log["mode_n"], where="post", color="0.5", lw=1.6, ls="--",
            label="Naive (no gatekeeper / hysteresis)")
    ax.step(t, log["mode_g"], where="post", color="k", lw=1.2, alpha=0.6)
    ax.scatter(t, log["mode_g"], s=18, zorder=3,
               c=[MODE_COLOR[LEVEL_MODE[int(l)]] for l in log["mode_g"]])
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["LFM\n350-450 kHz", "Geometric\n105-135 kHz", "Barker-13\nBPSK"])
    ax.set_ylim(0.5, 3.6)
    ax.set_ylabel("Selected waveform")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MODE_COLOR[m], label=MODE_LABEL[m])
               for m in (LFM, EXP, BARKER)]
    handles.append(plt.Line2D([], [], color="0.5", ls="--", lw=1.6, label="Naive baseline"))
    ax.legend(handles=handles, loc="upper center", ncol=4)

    ax = axes[3]                                           # drive power
    pct = log["drive_g"] * 100
    ax.fill_between(t, pct, step="post", color="#4c72b0", alpha=0.45)
    ax.step(t, pct, where="post", color="#4c72b0", lw=1.6)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Drive level (%)")
    ax.set_xlabel(f"Time (s)   [ping period = {PING_PERIOD_S} s]")

    axes[0].text(0.5, 1.02, "  ".join(f"{n}" for n in SCENARIO_NAMES),
                 transform=axes[0].transAxes, ha="center", fontsize=9)
    fig.suptitle("Plot 2: Selected waveform mode & drive power over time",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_waveforms(bank):
    """Plot 3: time-domain waveforms (full pulse + zoom) for all three modes."""
    modes = [LFM, EXP, BARKER]
    zoom = {LFM: (0, 12), EXP: (0, 45), BARKER: (80, 120)}
    t_us = bank["t"] * 1e6
    fig, axes = plt.subplots(3, 2, figsize=(12, 9),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    for row, m in enumerate(modes):
        y = bank[m] * DRIVE[m]
        ax, axz = axes[row]
        ax.plot(t_us, y, color=MODE_COLOR[m], lw=0.6)
        ax.axhline(DRIVE[m], color="k", ls=":", lw=0.8)
        ax.axhline(-DRIVE[m], color="k", ls=":", lw=0.8)
        ax.axvspan(*zoom[m], color="gold", alpha=0.3, lw=0)
        ax.set_ylim(-1.55, 1.55)
        ax.set_ylabel("Amplitude (norm.)")
        ax.set_title(f"{MODE_LABEL[m]}, drive {DRIVE[m] * 100:.0f} %", loc="left")
        if m == BARKER:                                    # chip boundaries + signs
            chip_us = PULSE_S * 1e6 / 13
            for k, s in enumerate(BARKER_13):
                ax.axvline(k * chip_us, color="0.5", ls=":", lw=0.7)
                ax.text((k + 0.5) * chip_us, 1.3, "+" if s > 0 else "\u2212",
                        ha="center", va="center", fontsize=11, fontweight="bold")
        lo, hi = zoom[m]
        sel = (t_us >= lo) & (t_us <= hi)
        axz.plot(t_us[sel], y[sel], "-o", color=MODE_COLOR[m], lw=1.2, ms=2.5)
        axz.set_ylim(-1.3, 1.3)
        axz.set_title("Zoom (yellow region)", loc="left", fontsize=10)
        if m == BARKER:
            axz.axvline(100, color=RED, ls="--", lw=1.2)
            axz.text(100.8, 1.12, "180° phase flip", color=RED, fontsize=9)
    axes[2, 0].set_xlabel("Time (µs)")
    axes[2, 1].set_xlabel("Time (µs)")
    fig.suptitle("Plot 3: Output time-domain waveforms for all three modes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_spectral(bank):
    """Plot 4: spectrogram of a LFM -> Geometric -> Barker ping train + spectra."""
    gap = np.zeros(int(round(100e-6 * FS)))
    train = np.concatenate([synthesize(LFM, DRIVE[LFM], bank), gap,
                            synthesize(EXP, DRIVE[EXP], bank), gap,
                            synthesize(BARKER, DRIVE[BARKER], bank)])
    Pxx, f, tt = mlab.specgram(train, NFFT=256, Fs=FS, noverlap=240, pad_to=4096)
    dB = 10 * np.log10(Pxx + 1e-15)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    mesh = ax1.pcolormesh(tt * 1e6, f / 1e3, dB, shading="auto", cmap="magma",
                          vmin=dB.max() - 50, vmax=dB.max())
    ax1.set_ylim(0, 600)
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Frequency (kHz)")
    ax1.grid(False)
    pulse_us, gap_us = PULSE_S * 1e6, 100.0
    starts = [0, pulse_us + gap_us, 2 * (pulse_us + gap_us)]
    labels = ["LFM 350→450 kHz\n@ 50 %", "Geometric 105→135 kHz\n@ 100 %",
              "Barker-13 @ 200 kHz\n@ 100 %"]
    for s0, lab in zip(starts, labels):
        ax1.text(s0 + pulse_us / 2, 585, lab, color="white", ha="center", va="top",
                 fontsize=9, fontweight="bold")
    fig.colorbar(mesh, ax=ax1, label="Power (dB)", pad=0.01)
    ax1.set_title("Spectrogram of a three-mode ping train (frequency sweeps per mode)",
                  loc="left")

    n_fft = 2 ** 15
    freq_khz = np.fft.rfftfreq(n_fft, 1 / FS) / 1e3
    spectra = {m: np.abs(np.fft.rfft(synthesize(m, DRIVE[m], bank), n=n_fft))
               for m in (LFM, EXP, BARKER)}
    ref = max(s.max() for s in spectra.values())
    for m, s in spectra.items():
        ax2.plot(freq_khz, 20 * np.log10(s / ref + 1e-6), color=MODE_COLOR[m], lw=1.3,
                 label=f"{MODE_LABEL[m]} (drive {DRIVE[m] * 100:.0f} %)")
    ax2.set_xlim(0, 600)
    ax2.set_ylim(-70, 3)
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Magnitude (dB re max)")
    ax2.set_title("Magnitude spectra (as transmitted, including drive scaling)", loc="left")
    ax2.legend(loc="upper right")
    fig.suptitle("Plot 4: Frequency spectrum / spectrogram across modes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    set_plot_style()
    log, gk = run_simulation()
    bank = build_pulse_bank()
    print_summary(log, gk)
    plot_telemetry(log)       # Plot 1
    plot_modes(log)           # Plot 2
    plot_waveforms(bank)      # Plot 3
    plot_spectral(bank)       # Plot 4


if __name__ == "__main__":
    main()

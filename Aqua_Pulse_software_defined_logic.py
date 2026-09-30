import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate

# ---------------------------------------------------------------
# 1. Sine lookup table (10-bit index, Q15 amplitude) -- same table
#    the firmware would store in flash.
# ---------------------------------------------------------------
LUT_BITS = 10
LUT_SIZE = 1 << LUT_BITS
sine_lut = np.round(32767 * np.sin(2 * np.pi * np.arange(LUT_SIZE + 1) / LUT_SIZE)).astype(np.int32)

def lut_sin(phase_u32):
    """32-bit phase -> interpolated Q15 sine, same math as the firmware."""
    idx = (phase_u32 >> (32 - LUT_BITS)) & (LUT_SIZE - 1)
    frac = (phase_u32 >> (32 - LUT_BITS - 16)) & 0xFFFF
    s0 = sine_lut[idx]
    s1 = sine_lut[idx + 1]
    return s0 + (((s1 - s0) * frac) >> 16)

# ---------------------------------------------------------------
# 2. Windows (Q15), matching the firmware window tables
# ---------------------------------------------------------------
def window_table(kind, n):
    x = np.linspace(0, 1, n, endpoint=False)
    if kind == "rect":
        w = np.ones(n)
    elif kind == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * x)
    elif kind == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * x)
    elif kind == "blackman":
        w = 0.42 - 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)
    elif kind == "tukey":
        alpha = 0.25
        w = np.ones(n)
        taper = int(alpha * n / 2)
        t = np.linspace(0, np.pi, taper)
        w[:taper] = 0.5 * (1 - np.cos(t))
        w[-taper:] = w[:taper][::-1]
    else:
        raise ValueError(kind)
    return np.round(w * 32767).astype(np.int32)

# ---------------------------------------------------------------
# 3. Barker-13 phase code
# ---------------------------------------------------------------
BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

# ---------------------------------------------------------------
# 4. Ping synthesis -- mirrors the firmware loop
# ---------------------------------------------------------------
def synth_ping(mode, fs, f0, f1, T, amplitude, window="hann"):
    N = int(round(fs * T))
    win = window_table(window, N)
    buf = np.zeros(N, dtype=np.int32)
    A_dac = int(round(amplitude * 2047))

    phase = np.uint64(0)
    if mode == "lfm":
        ftw = np.uint64(round((f0 / fs) * (1 << 32)))
        dftw = np.uint64(round(((f1 - f0) / T) / (fs ** 2) * (1 << 32)))
        for n in range(N):
            s = lut_sin(int(phase) & 0xFFFFFFFF)
            buf[n] = 2048 + (((s * win[n]) >> 15) * A_dac >> 15)
            phase = (phase + ftw) & 0xFFFFFFFF
            ftw = (ftw + dftw) & 0xFFFFFFFF

    elif mode == "geometric":
        ftw = round((f0 / fs) * (1 << 32))
        q = (f1 / f0) ** (1.0 / N)
        for n in range(N):
            s = lut_sin(int(phase) & 0xFFFFFFFF)
            buf[n] = 2048 + (((s * win[n]) >> 15) * A_dac >> 15)
            phase = (phase + int(ftw)) & 0xFFFFFFFF
            ftw = ftw * q

    elif mode == "coded":
        fc = 0.5 * (f0 + f1)
        ftw = round((fc / fs) * (1 << 32))
        chip_len = N // len(BARKER13)
        for n in range(N):
            chip = min(n // chip_len, len(BARKER13) - 1)
            phase_off = 0 if BARKER13[chip] > 0 else (1 << 31)
            s = lut_sin((int(phase) + phase_off) & 0xFFFFFFFF)
            buf[n] = 2048 + (((s * win[n]) >> 15) * A_dac >> 15)
            phase = (phase + ftw) & 0xFFFFFFFF
    else:
        raise ValueError(mode)

    return buf, N

# ---------------------------------------------------------------
# 5. Two scenarios from the adaptation-logic rules (Plan A, /5 scaled)
# ---------------------------------------------------------------
FS = 850_000  # Hz, Plan A sample rate on DAC1

scenarios = {
    "clear_reef_lfm":     dict(mode="lfm",       f0=30_000, f1=130_000, T=1.0e-3,  amplitude=0.44),
    "muddy_estuary_lfm":  dict(mode="lfm",       f0=21_000, f1=27_000,  T=10.0e-3, amplitude=0.86),
    "clear_reef_geom":    dict(mode="geometric", f0=30_000, f1=130_000, T=1.0e-3,  amplitude=0.44),
    "muddy_estuary_coded":dict(mode="coded",     f0=21_000, f1=27_000,  T=10.0e-3, amplitude=0.86),
}

# ---------------------------------------------------------------
# 6. Run, plot, and save one figure per scenario
# ---------------------------------------------------------------
for name, cfg in scenarios.items():
    buf, N = synth_ping(fs=FS, window="hann", **cfg)
    t = np.arange(N) / FS
    sig = buf.astype(float) - 2048.0  # remove DC offset for analysis

    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    fig.suptitle(f"{name}  |  mode={cfg['mode']}  N={N} samples  fs={FS/1e3:.0f} kHz")

    ax[0].plot(t * 1e3, sig)
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel("DAC code (AC)")
    ax[0].set_title("Time domain")

    f, tt, Sxx = spectrogram(sig, fs=FS, nperseg=128, noverlap=96)
    ax[1].pcolormesh(tt * 1e3, f / 1e3, 10 * np.log10(Sxx + 1e-9), shading="auto")
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("frequency (kHz)")
    ax[1].set_title("Spectrogram (STFT)")

    corr = correlate(sig, sig, mode="full")
    corr = corr / np.max(np.abs(corr))
    lag = (np.arange(len(corr)) - (len(corr) - 1) // 2) / FS * 1e3
    ax[2].plot(lag, corr)
    ax[2].set_xlabel("lag (ms)")
    ax[2].set_ylabel("normalised autocorrelation")
    ax[2].set_title("Matched-filter response (pulse compression)")

    fig.tight_layout()
    fig.savefig(f"/content/{name}.png", dpi=140)
    print(f"{name}: N={N} samples, duration={N/FS*1e3:.2f} ms, "
          f"peak={np.max(np.abs(sig)):.0f} DAC codes -> saved {name}.png")

print("\nDone. Open the PNG files to see each waveform, its spectrogram, "
      "and its pulse-compression peak.")

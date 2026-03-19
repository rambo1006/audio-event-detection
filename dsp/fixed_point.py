import numpy as np
import matplotlib.pyplot as plt
from numpy_mel import mel_numpy

SCALE = 32768

def to_q15(x):
    scaled = x * SCALE
    rounded = np.round(scaled)
    clipped = np.clip(rounded, -32768, 32767)
    return clipped.astype(np.int16)

def from_q15(x):
    return x.astype(np.float32) / SCALE

def mel_fixed_point(audio_path):
    import librosa
    import scipy.signal

    SR = 16000
    N_FFT = 512
    HOP = 160
    N_MELS = 64
    FMAX = 8000

    
    y, sr = librosa.load(audio_path, sr=SR)

    
    y = y / (np.max(np.abs(y)) + 1e-10)

    
    y = np.pad(y, N_FFT // 2, mode='reflect')

    
    num_frames = 1 + (len(y) - N_FFT) // HOP
    frames = np.stack(
        [y[i*HOP : i*HOP + N_FFT] for i in range(num_frames)],
        axis=1
    )

    
    window = scipy.signal.get_window('hann', N_FFT, fftbins=True)
    windowed = frames * window[:, None]

    
    spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=0)) ** 2

    
    mel_fb = librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX
    )
    mel_spec = mel_fb @ spectrum

    
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    
    mel_normalized = mel_db / 80.0

    
    mel_q15 = to_q15(mel_normalized)

    
    mel_reconstructed = from_q15(mel_q15) * 80.0

    return mel_db, mel_reconstructed

def compare_float_vs_fixed(audio_path):
    print("computing float32 mel spectrogram...")
    float_spec = mel_numpy(audio_path)

    print("computing Q15 fixed point mel spectrogram...")
    float_original, fixed_spec = mel_fixed_point(audio_path)

    # match shapes
    min_frames = min(float_spec.shape[1], fixed_spec.shape[1])
    float_spec = float_spec[:, :min_frames]
    fixed_spec = fixed_spec[:, :min_frames]

    # calculate SNR
    noise = float_spec - fixed_spec
    snr = 10 * np.log10(np.var(float_spec) / (np.var(noise) + 1e-10))

    print(f"\n─── RESULTS ───────────────────────────")
    print(f"Float32 range:     {float_spec.min():.2f} to {float_spec.max():.2f}")
    print(f"Fixed point range: {fixed_spec.min():.2f} to {fixed_spec.max():.2f}")
    print(f"SNR float vs Q15:  {snr:.2f} dB")
    print(f"Theoretical max Q15 SNR: ~90 dB")
    print(f"───────────────────────────────────────\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(float_spec, aspect='auto', origin='lower')
    axes[0].set_title('Float32 (software)')
    axes[1].imshow(fixed_spec, aspect='auto', origin='lower')
    axes[1].set_title('Q15 Fixed Point (simulated FPGA)')
    diff = np.abs(float_spec - fixed_spec)
    axes[2].imshow(diff, aspect='auto', origin='lower')
    axes[2].set_title('Difference (quantization error)')

    plt.tight_layout()
    plt.savefig('../results/fixed_point_comparison.png')
    plt.show()
    print("saved to results/fixed_point_comparison.png")

    return snr

if __name__ == "__main__":
    snr = compare_float_vs_fixed("test.wav")
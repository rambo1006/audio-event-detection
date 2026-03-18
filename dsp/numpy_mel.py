import numpy as np
import librosa
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
FMAX = 8000

def mel_librosa(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, fmax=FMAX
    )
    return librosa.power_to_db(S, ref=np.max)

def mel_numpy(audio_path):
    import scipy.signal
    y, sr = librosa.load(audio_path, sr=SR)
    
    # pad exactly like librosa
    y = np.pad(y, N_FFT // 2, mode='reflect')
    
    # build frames
    num_frames = 1 + (len(y) - N_FFT) // HOP
    frames = np.stack([y[i*HOP : i*HOP + N_FFT] for i in range(num_frames)], axis=1)
    
    # use scipy hann window — matches librosa internally
    window = scipy.signal.get_window('hann', N_FFT, fftbins=True)
    windowed = frames * window[:, None]
    
    # FFT
    spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=0)) ** 2
    
    # mel filterbank
    mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX)
    mel_spec = mel_fb @ spectrum
    
    return librosa.power_to_db(mel_spec, ref=np.max)

def compare_and_plot(audio_path):
    ref = mel_librosa(audio_path)
    custom = mel_numpy(audio_path)

    # fix shape mismatch
    min_frames = min(ref.shape[1], custom.shape[1])
    ref = ref[:, :min_frames]
    custom = custom[:, :min_frames]

    # calculate SNR
    noise = ref - custom
    snr = 10 * np.log10(np.var(ref) / np.var(noise))
    print(f"SNR between librosa and numpy: {snr:.2f} dB")
    print(f"Target: >25 dB  |  Your result: {snr:.2f} dB")

    if snr > 25:
        print("PASS - numpy implementation matches librosa")
    else:
        print("needs debugging - SNR too low")

    # plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(ref, aspect='auto', origin='lower')
    axes[0].set_title('Librosa reference')
    axes[1].imshow(custom, aspect='auto', origin='lower')
    axes[1].set_title('NumPy implementation')
    plt.tight_layout()
    plt.savefig('../results/mel_comparison.png')
    plt.show()
    print("Plot saved to results/mel_comparison.png")

if __name__ == "__main__":
    compare_and_plot("test.wav")
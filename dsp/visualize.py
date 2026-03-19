import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
FMAX = 8000

DATA_PATH = "../data"

def get_audio_file(keyword):
    folder = os.path.join(DATA_PATH, keyword)
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    return os.path.join(folder, files[0])

def compute_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, fmax=FMAX
    )
    return librosa.power_to_db(S, ref=np.max)

def plot_single(keyword):
    audio_path = get_audio_file(keyword)
    mel = compute_mel(audio_path)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        mel, sr=SR, hop_length=HOP,
        x_axis='time', y_axis='mel',
        fmax=FMAX, cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram — "{keyword}"')
    plt.tight_layout()
    plt.savefig(f'../results/spectrogram_{keyword}.png', dpi=150)
    plt.show()
    print(f"saved results/spectrogram_{keyword}.png")

def plot_comparison():
    keywords = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()

    for i, keyword in enumerate(keywords):
        audio_path = get_audio_file(keyword)
        mel = compute_mel(audio_path)

        librosa.display.specshow(
            mel, sr=SR, hop_length=HOP,
            x_axis='time', y_axis='mel',
            fmax=FMAX, cmap='magma', ax=axes[i]
        )
        axes[i].set_title(f'"{keyword}"', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    fig.suptitle('Mel Spectrograms — Google Speech Commands', fontsize=14)
    plt.tight_layout()
    plt.savefig('../results/spectrogram_comparison.png', dpi=150)
    plt.show()
    print("saved results/spectrogram_comparison.png")

if __name__ == "__main__":
    print("plotting single spectrogram...")
    plot_single("yes")

    print("plotting comparison grid...")
    plot_comparison()

    print("\nDone! Check results/ folder for images.")
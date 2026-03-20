import sys
import os
sys.path.append('model')
sys.path.append('data')

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

from train import SimpleCNN
from dataset import KEYWORDS

MODEL_PATH = "results/best_model.pt"
SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
FMAX = 8000
DURATION = 1.0

def load_model():
    model = SimpleCNN(num_classes=len(KEYWORDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def preprocess(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_length = int(SR * DURATION)
    if waveform.shape[1] < target_length:
        pad = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :target_length]

    mel_transform = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, f_max=FMAX
    )
    db_transform = T.AmplitudeToDB(top_db=80)

    mel = mel_transform(waveform)
    mel_db = db_transform(mel)
    mel_db = torch.clamp(mel_db, -80, 0)
    mel_db = (mel_db + 80) / 80

    return mel_db.unsqueeze(0)

def predict(model, audio_path):
    spec = preprocess(audio_path)

    with torch.no_grad():
        output = model(spec)
        probs = torch.softmax(output, dim=1)
        top3_probs, top3_idx = torch.topk(probs, 3)

    print(f"\naudio file: {os.path.basename(audio_path)}")
    print(f"{'─'*35}")
    for i, (prob, idx) in enumerate(zip(top3_probs[0], top3_idx[0])):
        bar = '█' * int(prob.item() * 30)
        print(f"#{i+1} {KEYWORDS[idx]:<12} {prob.item()*100:5.1f}%  {bar}")
    print(f"{'─'*35}")
    print(f"prediction: {KEYWORDS[top3_idx[0][0]]}")
    return KEYWORDS[top3_idx[0][0]]

def run_on_dataset_samples():
    from data.dataset import SpeechCommandsDataset
    DATA_PATH = "data"

    print("\n" + "="*40)
    print("DEMO — Audio Event Detection Pipeline")
    print("="*40)

    test_keywords = ['yes', 'no', 'stop', 'go', 'up', 'down']
    model = load_model()

    correct = 0
    total = len(test_keywords)

    for keyword in test_keywords:
        folder = os.path.join(DATA_PATH, keyword)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        audio_path = os.path.join(folder, files[0])
        predicted = predict(model, audio_path)
        if predicted == keyword:
            correct += 1
            print(f"correct!")
        else:
            print(f"wrong — expected '{keyword}'")

    print(f"\n{'='*40}")
    print(f"demo accuracy: {correct}/{total} = {correct/total*100:.0f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"file not found: {audio_path}")
            sys.exit(1)
        model = load_model()
        predict(model, audio_path)
    else:
        run_on_dataset_samples()
import glob
import re
import pdb
import torch.nn as nn
import torch
import librosa
import soundfile as sf
from hparams import hparams, set_hparams
import numpy as np
import pyloudnorm as pyln
from model import WaveGlowMelHF
from utils import load_ckpt

def run(model, wav, sigma=1.0):
    wav = torch.Tensor(wav).reshape(1, -1).cuda()
    output = np.array(model.infer(wav, sigma=sigma)[0].cpu().detach())
    output = output.reshape(-1)
    return output

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2

def load_wav(wav_fn):
    wav, sr = librosa.core.load(wav_fn, sr=hparams['sampling_rate'] // 2)
    print(wav.shape, sr, hparams['sampling_rate'])

    if hparams['loud_norm']:
        print('LOUD NORM!', flush=True)
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    fft_size = hparams['fft_size']
    hop_size = hparams['hop_size']
    win_length = hparams['win_size']
    fmin = hparams['fmin']
    fmax = hparams['fmax']
    sample_rate = hparams['sampling_rate']
    num_mels = hparams['num_mels']
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window='hann', pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc
    eps = 1e-10
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]
    return wav, sr

if __name__ == '__main__':
    set_hparams()

    model = WaveGlowMelHF(**hparams['waveglow_config']).cuda()

    #pdb.set_trace()

    load_ckpt(model, 'checkpoints/glowmultin/model_ckpt_best.pt')
    model.eval()

    fns = ['p225_001_lr.wav']
    print(load_wav('1_pred_p225_001_lr.wav')[1])
    sigma = 1
    for lr_fn in fns:
        lr, sr = load_wav(lr_fn)
        print(f'sampling rate (lr) = {sr}')
        print(f'lr.shape = {lr.shape}', flush=True)
        with torch.no_grad():
            pred = run(model, lr, sigma=sigma)
        print(lr.shape, pred.shape)
        pred_fn = f'{sigma}_pred_{lr_fn}'
        print(f'sampling rate = {sr * 2}')
        sf.write(open(pred_fn, 'wb'), pred, sr * 2)



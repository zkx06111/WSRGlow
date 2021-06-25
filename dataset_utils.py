from hparams import hparams
import numpy as np
import torch
import utils
import glob
import librosa
import os
import importlib
import audio
from skimage.transform import resize
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
import librosa
import numpy as np
import pyloudnorm as pyln
import re
import json
from collections import OrderedDict
import pickle
from copy import deepcopy

PUNCS = '!,.?;:'

int16_max = (2 ** 15) - 1

def trim_long_silences(path, sr=None, return_raw_wav=False, norm=True, vad_max_silence_length=12):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :param vad_max_silence_length: Maximum number of consecutive silent frames a segment can have.
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    sampling_rate = 16000
    wav_raw, sr = librosa.core.load(path, sr=sr)

    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav_raw)
        wav_raw = pyln.normalize.loudness(wav_raw, loudness, -20.0)
        if np.abs(wav_raw).max() > 1.0:
            wav_raw = wav_raw / np.abs(wav_raw).max()

    wav = librosa.resample(wav_raw, sr, sampling_rate, res_type='kaiser_best')

    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width = 8

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    audio_mask = resize(audio_mask, (len(wav_raw),)) > 0
    if return_raw_wav:
        return wav_raw, audio_mask, sr
    return wav_raw[audio_mask], audio_mask, sr

def process_utterance(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-10,
                      sample_rate=22050,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg'):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(
        sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == 'pwg':
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    else:
        assert False, f'"{vocoder}" is not in ["pwg"].'

    l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    if not return_linear:
        return wav, mel
    else:
        spc = audio.amp_to_db(spc)
        spc = audio.normalize(spc, {'min_level_db': min_level_db})
        return wav, mel, spc

def wav2spec(wav_fn, return_linear=False):
    res = process_utterance(
        wav_fn, fft_size=hparams['fft_size'],
        hop_size=hparams['hop_size'],
        win_length=hparams['win_size'],
        num_mels=hparams['audio_num_mel_bins'],
        fmin=hparams['fmin'],
        fmax=hparams['fmax'],
        sample_rate=hparams['audio_sample_rate'],
        loud_norm=hparams['loud_norm'],
        min_level_db=hparams['min_level_db'],
        return_linear=return_linear, vocoder='pwg')
    if return_linear:
        return res[0], res[1].T, res[2].T  # [T, 80], [T, n_fft]
    else:
        return res[0], res[1].T


class IndexedDataset:
    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], hparams['max_frames'])
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(
                    np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))

class SRDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.is_infer = prefix == 'test'
        self.batch_max_samples = 0 if self.is_infer else hparams['max_samples']
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)

        sample = {
            'id': index,
            'item_name': item['item_name'],
            'wav': item['wav'],
            'lr_wav': item['resampled_wav']
        }

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = []
        item_name = []
        resample_ratio = float(self.hparams['binarization_args']['resample_ratio'])
        hr_batch = []
        lr_batch = []
        lr_mag_batch = []
        lr_pha_batch = []
        hr_mag_batch = []
        hr_pha_batch = []
        n_fft = hparams['n_fft']
        for (idx, s) in enumerate(samples):
            id.append(s['id'])
            item_name.append(s['item_name'])
            hr, lr = s['wav'], s['lr_wav']
            if len(hr) > self.batch_max_samples:
                batch_max_samples = len(hr) if self.is_infer else self.batch_max_samples
                lr_max_samples = int(resample_ratio * batch_max_samples)
                lr_start_step = np.random.randint(0, len(lr) - lr_max_samples + 1)
                # print(hr.shape)
                hr_ = hr[int(lr_start_step / resample_ratio): int((lr_start_step + lr_max_samples) / resample_ratio)]
                lr_ = lr[lr_start_step: lr_start_step + lr_max_samples]
                Dlow = librosa.stft(lr_, n_fft=n_fft // 2)
                lr_mag = np.abs(Dlow)
                lr_pha = np.angle(Dlow)
                D = librosa.stft(hr_, n_fft=n_fft)
                hr_mag = np.abs(D)
                hr_pha = np.angle(D)
            else:
                print(f'Removed short sample from batch (length={len(hr)}).')
                continue
            hr_batch += [torch.FloatTensor(hr_)]
            lr_batch += [torch.FloatTensor(lr_)]
            lr_mag_batch += [torch.FloatTensor(lr_mag).t()]
            lr_pha_batch += [torch.FloatTensor(lr_pha).t()]
            hr_mag_batch += [torch.FloatTensor(hr_mag).t()]
            hr_pha_batch += [torch.FloatTensor(hr_pha).t()]

        hr_batch = utils.collate_1d(hr_batch, 0)
        lr_batch = utils.collate_1d(lr_batch, 0)
        lr_mag_batch = utils.collate_2d(lr_mag_batch, 0).permute(0, 2, 1)
        lr_pha_batch = utils.collate_2d(lr_pha_batch, 0).permute(0, 2, 1)
        hr_mag_batch = utils.collate_2d(hr_mag_batch, 0).permute(0, 2, 1)
        hr_pha_batch = utils.collate_2d(hr_pha_batch, 0).permute(0, 2, 1)

        return {
            'wavs': hr_batch,
            'nsamples': len(samples),
            'resampled_wavs': lr_batch,
            'item_name': item_name,
            'lr_mags': lr_mag_batch,
            'lr_phas': lr_pha_batch,
            'hr_mags': hr_mag_batch,
            'hr_phas': hr_pha_batch
        }

    def load_test_inputs(self, test_input_dir):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            ph = txt = tg_fn = ''
            wav_fn = wav_fn
            encoder = None
            item = binarizer_cls.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes


import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
import librosa  # 수정된 부분: librosa.filters.mel 함수 직접 사용

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window

    key = str(fmax) + '_' + str(y.device)
    if key not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)  # 수정된 부분
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.abs(spec) + 1e-9  # 복소수 magnitude + 안정성

    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio).unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)
                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
"""
class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, input_wavs_dir=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.input_wavs_dir = input_wavs_dir


    def __getitem__(self, index):
        filename = self.audio_files[index]

        if not self.fine_tuning:
            # fine_tuning이 False일 때는 기존 로직을 유지하거나,
            # 이 경우도 미리 전처리된 데이터를 사용하도록 통일하는 것이 좋습니다.
            # (아래는 기존 로직을 그대로 둔 예시)
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
            audio = torch.FloatTensor(audio).unsqueeze(0)

            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            
            # Generator의 입력으로 사용할 mel
            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
            
            # Loss 계산에 사용할 mel
            mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                      self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                      center=False)
            
            return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

        else: # fine_tuning이 True인 경우 (핵심 수정 부분)
            # 1. 미리 계산된 멜 스펙트로그램 경로 생성 및 로드
            # relative_wav_path_from_root = os.path.relpath(filename, self.input_wavs_dir)
            # .wav 확장자 완전히 제거            

            # mel_file_name_with_subdirs = os.path.splitext(relative_wav_path_from_root)[0] + '.npy'
            # mel_full_load_path = os.path.join(self.base_mels_path, mel_file_name_with_subdirs)
            # 기준을 정확히 input_wavs_dir로 잡아야 함
            dataset_folder_name = os.path.basename(self.input_wavs_dir)

            relative_wav_path_from_root = os.path.relpath(os.path.abspath(filename), os.path.abspath(self.input_wavs_dir))
            relative_path, _ = os.path.splitext(relative_wav_path_from_root)
            mel_file_name_with_subdirs = relative_path + '.npy'

            # 경로를 조합할 때 dataset_folder_name을 중간에 추가해줍니다.
            mel_full_load_path = os.path.join(self.base_mels_path, dataset_folder_name, mel_file_name_with_subdirs)


            
            # 이 mel이 Generator의 입력(x)이자, loss 계산용 ground truth(y_mel)가 됩니다.
            mel = torch.from_numpy(np.load(mel_full_load_path))
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            # 2. 원본 오디오 로드 및 정규화
            audio, sampling_rate = load_wav(filename)
            audio = torch.FloatTensor(audio / MAX_WAV_VALUE).unsqueeze(0)

            # 3. 멜 스펙트로그램과 오디오를 동일한 세그먼트로 자르기
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)

            if self.split and audio.size(1) >= self.segment_size:
                # 멜 길이를 기준으로 자를 위치를 랜덤하게 선택
                if mel.size(2) > frames_per_seg:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    
                    # 선택된 멜 위치에 해당하는 오디오 세그먼트 추출
                    audio_start = mel_start * self.hop_size
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else: # 멜 길이가 세그먼트보다 짧으면 그대로 사용하고 패딩
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            else: # split=False 이거나 오디오가 세그먼트보다 짧을 경우
                mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            # 4. 불필요한 mel_spectrogram 계산 제거!
            # mel_loss = mel_spectrogram(audio, ...) # <- 이 줄을 반드시 삭제해야 합니다.

            # Generator 입력용 mel과 loss 계산용 mel을 동일한 것으로 반환합니다.
            # train.py에서는 네 번째 반환값을 y_mel로 사용합니다.
            return (mel.squeeze(), audio.squeeze(0), filename, mel.squeeze())


    def __len__(self):
        return len(self.audio_files)


"""


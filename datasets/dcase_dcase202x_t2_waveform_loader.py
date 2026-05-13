import os

import librosa
import numpy as np
import soundfile as sf
import torch

from datasets import loader_common as com


class DCASE202XT2WaveformLoader(torch.utils.data.Dataset):
    """File-level two-channel waveform loader for MED-ASD."""

    def __init__(
        self,
        root: str,
        dataset_name,
        section_keyword,
        machine_type: str = "ToyCar",
        section_ids=None,
        train=True,
        data_type="dev",
        use_id=None,
        is_auto_download=False,
        sample_rate=16000,
        audio_samples=160000,
    ):
        super().__init__()
        self.section_ids = section_ids or []
        self.use_id = use_id or []
        self.machine_type = machine_type
        self.sample_rate = sample_rate
        self.audio_samples = audio_samples

        target_dir = os.getcwd() + "/" + root + "raw/" + machine_type
        dir_name = "train" if train else "test"
        self.mode = data_type == "dev"
        if train:
            dir_name = "train"
        elif os.path.exists(f"{target_dir}/test_rename"):
            dir_name = "test_rename"
            self.mode = True
        else:
            dir_name = "test"

        target_dir = os.path.abspath(f"{target_dir}/")
        if is_auto_download:
            com.download_raw_data(
                target_dir=target_dir,
                dir_name=dir_name,
                machine_type=machine_type,
                data_type=data_type,
                dataset=dataset_name,
                root=root,
            )
        elif not os.path.exists(f"{target_dir}/{dir_name}"):
            raise FileNotFoundError(
                f"{target_dir}/{dir_name} is not directory and do not use auto download. "
                "please download dataset or using auto download."
            )

        section_names = [f"{section_keyword}_{section_id}" for section_id in self.section_ids]
        unique_section_names = np.unique(section_names)

        self.files = []
        self.y_true = []
        self.condition = []
        for section_name in unique_section_names:
            files, labels, condition = com.file_list_generator(
                target_dir=target_dir,
                section_name=section_name,
                unique_section_names=unique_section_names,
                dir_name=dir_name,
                mode=self.mode,
                train=train,
            )
            self.files.extend(list(files))
            if labels is None:
                self.y_true.extend([-1] * len(files))
            else:
                self.y_true.extend([int(label) for label in labels])
            self.condition.extend(condition)

        if len(self.use_id) > 0:
            keep = []
            for idx, cond in enumerate(self.condition):
                section_idx = int(np.argmax(cond))
                section_id = int(self.section_ids[section_idx])
                if section_id in self.use_id:
                    keep.append(idx)
            self.files = [self.files[idx] for idx in keep]
            self.y_true = [self.y_true[idx] for idx in keep]
            self.condition = [self.condition[idx] for idx in keep]

        self.basenames = [os.path.basename(path) for path in self.files]
        self.n_vectors_ea_file = 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        waveform = self._load_two_channel(self.files[index])
        return (
            waveform,
            self.y_true[index],
            np.asarray(self.condition[index], dtype=np.float32),
            self.basenames[index],
            index,
        )

    def _load_two_channel(self, file_path):
        wav, sr = sf.read(file_path, always_2d=True)
        wav = wav.astype(np.float32).T

        if wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate, axis=1)

        if wav.shape[1] < self.audio_samples:
            pad = self.audio_samples - wav.shape[1]
            wav = np.pad(wav, ((0, 0), (0, pad)), mode="constant")
        else:
            wav = wav[:, : self.audio_samples]

        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak
        return torch.from_numpy(wav.astype(np.float32))

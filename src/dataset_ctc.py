import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class CTCEditDistanceDataset(Dataset):
    def __init__(self, file_path: str, max_seq_len: int = 100):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.data = []
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                entry = f[key]
                orig = np.array(entry['original'])
                edit = np.array(entry['edited'])
                self.data.append((orig, edit))

    def __len__(self) -> int:
        return len(self.data)

    def _process_seq(self, seq: np.ndarray) -> tuple:
        seq_len = len(seq)
        ctc_seq = seq + 1
        pad_len = self.max_seq_len - seq_len
        if pad_len < 0:
            ctc_seq = ctc_seq[:self.max_seq_len]
            valid_len = self.max_seq_len
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        else:
            ctc_seq = np.pad(ctc_seq, (0, pad_len), 'constant', constant_values=0)
            valid_len = seq_len
            zeros_t = torch.zeros(seq_len, dtype=torch.bool)
            ones_t = torch.ones(pad_len, dtype=torch.bool)
            mask = torch.cat([zeros_t, ones_t])
        return torch.tensor(ctc_seq, dtype=torch.long), mask, valid_len

    def __getitem__(self, idx: int) -> tuple:
        orig, edit = self.data[idx]
        orig_seq, orig_mask, orig_len = self._process_seq(orig)
        edit_seq, edit_mask, edit_len = self._process_seq(edit)
        return edit_seq, edit_mask, edit_len, orig_seq, orig_mask, orig_len
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EditDistanceDataset(Dataset):
    def __init__(self, file_path: str, max_seq_len: int):
        super().__init__()
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.keys = []
        with h5py.File(self.file_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def _pad_and_mask(self, seq: np.ndarray) -> tuple:
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len
        if pad_len < 0:
            seq = seq[:self.max_seq_len]
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        else:
            seq = np.pad(seq, (0, pad_len), 'constant', constant_values=0)
            zeros_t = torch.zeros(seq_len, dtype=torch.bool)
            ones_t = torch.ones(pad_len, dtype=torch.bool)
            mask = torch.cat([zeros_t, ones_t])
        return torch.tensor(seq, dtype=torch.float32), mask

    def __getitem__(self, idx: int) -> tuple:
        key = self.keys[idx]
        with h5py.File(self.file_path, 'r') as f:
            entry = f[key]
            original = np.array(entry['original'])
            edited = np.array(entry['edited'])
            distance = np.array(entry['distance'])
        orig_seq, orig_mask = self._pad_and_mask(original)
        edit_seq, edit_mask = self._pad_and_mask(edited)
        dist_tensor = torch.tensor(distance, dtype=torch.float32)
        return orig_seq, orig_mask, edit_seq, edit_mask, dist_tensor

if __name__ == "__main__":
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = EditDistanceDataset(dataset_path, max_seq_len=100)
    orig_seq, orig_mask, edit_seq, edit_mask, distance = dataset[0]
    print(orig_seq.shape)
    print(orig_mask.shape)
    print(edit_seq.shape)
    print(distance.item())

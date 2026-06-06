import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TripletEditDistanceDataset(Dataset):
    def __init__(self, file_path: str, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.data = []
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                entry = f[key]
                anchor = np.array(entry['anchor'])
                positive = np.array(entry['positive'])
                negative = np.array(entry['negative'])
                self.data.append((anchor, positive, negative))

    def __len__(self) -> int:
        return len(self.data)

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
        anchor, positive, negative = self.data[idx]
        anc_seq, anc_mask = self._pad_and_mask(anchor)
        pos_seq, pos_mask = self._pad_and_mask(positive)
        neg_seq, neg_mask = self._pad_and_mask(negative)
        return anc_seq, anc_mask, pos_seq, pos_mask, neg_seq, neg_mask

if __name__ == "__main__":
    dataset_path = "../dataset/triplet_datasets_K_90_num_seq_10000.h5"
    dataset = TripletEditDistanceDataset(dataset_path, max_seq_len=100)
    anc_seq, anc_mask, pos_seq, pos_mask, neg_seq, neg_mask = dataset[0]
    print(anc_seq.shape)
    print(pos_seq.shape)
    print(neg_seq.shape)

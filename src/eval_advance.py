import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from model import SiameseTransformer

class LegacyEditDistanceDataset(Dataset):
    def __init__(self, file_path: str, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.data = []
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                entry = f[key]
                orig = np.array(entry['original'])
                edit = np.array(entry['edited'])
                dist = entry['distance'][()]
                self.data.append((orig, edit, dist))

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
        orig, edit, dist = self.data[idx]
        orig_seq, orig_mask = self._pad_and_mask(orig)
        edit_seq, edit_mask = self._pad_and_mask(edit)
        return orig_seq, orig_mask, edit_seq, edit_mask, dist

def evaluate_extended_metrics() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = LegacyEditDistanceDataset(dataset_path, max_seq_len=100)
    subset_size = 10000
    if len(dataset) > subset_size:
        subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = SiameseTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    true_dists = []
    emb_dists = []
    print("Extracting representations for extended metrics...")
    with torch.no_grad():
        for orig_seq, orig_mask, edit_seq, edit_mask, dist in dataloader:
            orig_seq = orig_seq.to(device)
            orig_mask = orig_mask.to(device)
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            orig_emb = model(orig_seq, src_key_padding_mask=orig_mask)
            edit_emb = model(edit_seq, src_key_padding_mask=edit_mask)
            batch_emb_dist = F.pairwise_distance(orig_emb, edit_emb).cpu().numpy()
            true_dists.extend(dist.cpu().numpy())
            emb_dists.extend(batch_emb_dist)
    true_arr = np.array(true_dists)
    emb_arr = np.array(emb_dists)
    pearson_corr, _ = pearsonr(true_arr, emb_arr)
    spearman_corr, _ = spearmanr(true_arr, emb_arr)
    slope, intercept = np.polyfit(emb_arr, true_arr, 1)
    pred_arr = slope * emb_arr + intercept
    rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))
    mae = np.mean(np.abs(pred_arr - true_arr))
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"Spearman Correlation: {spearman_corr:.3f}")
    print("Hamming Loss: N/A (Encoder-only architecture)")

if __name__ == "__main__":
    evaluate_extended_metrics()
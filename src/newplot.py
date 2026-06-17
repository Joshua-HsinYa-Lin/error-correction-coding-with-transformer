import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
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

def generate_plots() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = LegacyEditDistanceDataset(dataset_path, max_seq_len=100)
    subset_size = 2000
    subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=256, shuffle=False)
    model = SiameseTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    embeddings = []
    distances = []
    true_dists = []
    emb_dists = []
    print("Extracting representations...")
    with torch.no_grad():
        for orig_seq, orig_mask, edit_seq, edit_mask, dist in dataloader:
            orig_seq = orig_seq.to(device)
            orig_mask = orig_mask.to(device)
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            orig_emb = model(orig_seq, src_key_padding_mask=orig_mask)
            edit_emb = model(edit_seq, src_key_padding_mask=edit_mask)
            batch_emb_dist = F.pairwise_distance(orig_emb, edit_emb).cpu().numpy()
            embeddings.append(edit_emb.cpu().numpy())
            distances.append(dist.cpu().numpy())
            true_dists.extend(dist.cpu().numpy())
            emb_dists.extend(batch_emb_dist)
    embeddings_arr = np.concatenate(embeddings, axis=0)
    distances_arr = np.concatenate(distances, axis=0)
    plt.figure(figsize=(10, 8))
    plt.scatter(true_dists, emb_dists, alpha=0.3, s=15, color='purple')
    plt.xlabel("True Edit Distance")
    plt.ylabel("Euclidean Embedding Distance")
    plt.title("Siamese Transformer: Metric Space Correlation")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/distance_scatter.png")
    print("Scatter plot saved to plots/distance_scatter.png")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_arr)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=distances_arr, cmap='viridis', alpha=0.7, s=15)
    cbar = plt.colorbar(scatter)
    cbar.set_label('True Edit Distance')
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of Siamese Embedding Manifold")
    plt.savefig("plots/pca_clusters.png")
    print("PCA plot saved to plots/pca_clusters.png")

if __name__ == "__main__":
    generate_plots()
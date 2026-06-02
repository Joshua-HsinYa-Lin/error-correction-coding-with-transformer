import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataset import EditDistanceDataset
from model import Seq2SeqTransformer

def run_pca() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = EditDistanceDataset(dataset_path, max_seq_len=100)
    subset_size = 2000
    if len(dataset) > subset_size:
        subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    model = Seq2SeqTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    embeddings = []
    distances = []
    print("Extracting high-dimensional encoder representations...")
    with torch.no_grad():
        for orig_seq, orig_mask, edit_seq, edit_mask, dist in dataloader:
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            src_exp = edit_seq.unsqueeze(-1)
            src_emb = model.src_proj(src_exp)
            src_pos = model.pos_encoder(src_emb)
            memory = model.transformer.encoder(src_pos, src_key_padding_mask=edit_mask)
            memory_pooled = memory.mean(dim=1)
            embeddings.append(memory_pooled.cpu().numpy())
            distances.append(dist.cpu().numpy())
    embeddings_arr = np.concatenate(embeddings, axis=0)
    distances_arr = np.concatenate(distances, axis=0)
    print("Executing Principal Component Analysis...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_arr)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=distances_arr, cmap='viridis', alpha=0.7, s=15)
    cbar = plt.colorbar(scatter)
    cbar.set_label('True Edit Distance')
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of Transformer Encoder Representations")
    plt.savefig("pca_clusters.png")
    print("PCA plot successfully saved to pca_clusters.png")

if __name__ == "__main__":
    run_pca()
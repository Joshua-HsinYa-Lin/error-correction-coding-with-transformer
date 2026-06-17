import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import TripletEditDistanceDataset
from model import SiameseTransformer

def plot_distance_distributions() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/triplet_datasets_K_90_num_seq_10000.h5"
    dataset = TripletEditDistanceDataset(dataset_path, max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = SiameseTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    all_pos_dists = []
    all_neg_dists = []
    print("Extracting embedding distances...")
    with torch.no_grad():
        for anc_seq, anc_mask, pos_seq, pos_mask, neg_seq, neg_mask in dataloader:
            anc_seq = anc_seq.to(device)
            anc_mask = anc_mask.to(device)
            pos_seq = pos_seq.to(device)
            pos_mask = pos_mask.to(device)
            neg_seq = neg_seq.to(device)
            neg_mask = neg_mask.to(device)
            anc_emb = model(anc_seq, src_key_padding_mask=anc_mask)
            pos_emb = model(pos_seq, src_key_padding_mask=pos_mask)
            neg_emb = model(neg_seq, src_key_padding_mask=neg_mask)
            pos_dist = F.pairwise_distance(anc_emb, pos_emb).cpu().numpy()
            neg_dist = F.pairwise_distance(anc_emb, neg_emb).cpu().numpy()
            all_pos_dists.extend(pos_dist)
            all_neg_dists.extend(neg_dist)
    plt.figure(figsize=(10, 6))
    plt.hist(all_pos_dists, bins=50, alpha=0.6, color='blue', label='Positive Pairs (Correlated)')
    plt.hist(all_neg_dists, bins=50, alpha=0.6, color='red', label='Negative Pairs (Corrupted)')
    plt.axvline(np.mean(all_pos_dists), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(all_neg_dists), color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Euclidean Distance in 256-D Space")
    plt.ylabel("Frequency")
    plt.title("Siamese Transformer: Embedding Distance Separation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/distance_histogram.png")
    print("Distance histogram successfully saved to plots/distance_histogram.png")

if __name__ == "__main__":
    plot_distance_distributions()
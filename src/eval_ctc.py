import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_ctc import CTCEditDistanceDataset
from model import CTCTransformer
from ctc_decode import ctc_greedy_decode, sample_metrics

def evaluate_ctc() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = CTCEditDistanceDataset(dataset_path, max_seq_len=100)
    subset_size = 10000
    if len(dataset) > subset_size:
        subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = CTCTransformer(num_classes=3)
    model.load_state_dict(torch.load("ctc_model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    upsample_factor = 2
    ber_values = []
    hamming_values = []
    block_errors = []
    edit_distances = []
    print("Decoding sequences and scoring...")
    with torch.no_grad():
        for edit_seq, edit_mask, edit_len, orig_seq, orig_mask, orig_len in dataloader:
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            logits = model(edit_seq, src_key_padding_mask=edit_mask)
            for b in range(edit_seq.size(0)):
                valid_time = int(edit_len[b]) * upsample_factor
                sample_logits = logits[b:b + 1, :valid_time, :]
                decoded = ctc_greedy_decode(sample_logits, blank=0)[0]
                pred_bits = [c - 1 for c in decoded]
                target_length = int(orig_len[b])
                true_bits = (orig_seq[b, :target_length] - 1).tolist()
                metrics = sample_metrics(pred_bits, true_bits)
                ber_values.append(metrics["bit_error_rate"])
                hamming_values.append(metrics["hamming_loss"])
                block_errors.append(metrics["block_error"])
                edit_distances.append(metrics["edit_distance"])
    bit_error_rate = float(np.mean(ber_values))
    block_error_rate = float(np.mean(block_errors))
    hamming_loss = float(np.mean(hamming_values))
    mean_edit_distance = float(np.mean(edit_distances))
    print(f"Evaluation complete on {len(ber_values)} samples")
    print(f"Bit Error Rate: {bit_error_rate:.4f}")
    print(f"Block Error Rate: {block_error_rate:.4f}")
    print(f"Hamming Loss: {hamming_loss:.4f}")
    print(f"Mean Reconstruction Edit Distance: {mean_edit_distance:.4f}")
    plt.figure(figsize=(8, 6))
    bars = ["Bit Error Rate", "Block Error Rate", "Hamming Loss"]
    heights = [bit_error_rate, block_error_rate, hamming_loss]
    plt.bar(bars, heights, color=["steelblue", "indianred", "seagreen"])
    plt.ylabel("Rate")
    plt.title("CTCTransformer Recovery Error Rates")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig("plots/ctc_error_rates.png")
    print("Error rate chart saved to plots/ctc_error_rates.png")
    plt.figure(figsize=(8, 6))
    plt.hist(edit_distances, bins=40, color="slateblue", alpha=0.8)
    plt.xlabel("Reconstruction Edit Distance (predicted vs original)")
    plt.ylabel("Frequency")
    plt.title("CTCTransformer Reconstruction Edit Distance Distribution")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/ctc_edit_distance_histogram.png")
    print("Edit distance histogram saved to plots/ctc_edit_distance_histogram.png")

if __name__ == "__main__":
    evaluate_ctc()

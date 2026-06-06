import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from dataset import EditDistanceDataset
from model import Seq2SeqTransformer
from collections import defaultdict

def levenshtein_distance(seq1: list, seq2: list) -> int:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(matrix[x-1, y] + 1, matrix[x, y-1] + 1, matrix[x-1, y-1])
            else:
                matrix[x, y] = min(matrix[x-1, y] + 1, matrix[x, y-1] + 1, matrix[x-1, y-1] + 1)
    return matrix[size_x - 1, size_y - 1]

def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = EditDistanceDataset(dataset_path, max_seq_len=100)
    subset_size = 12800
    if len(dataset) > subset_size:
        subset_indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    model = Seq2SeqTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    distance_stats = defaultdict(lambda: {'bit_errors': 0, 'total_bits': 0, 'block_errors': 0, 'total_blocks': 0})
    all_true_distances = []
    all_pred_distances = []
    
    print("Starting dynamic autoregressive evaluation...")
    with torch.no_grad():
        for batch_idx, (orig_seq, orig_mask, edit_seq, edit_mask, dist_tensor) in enumerate(dataloader):
            orig_seq = orig_seq.to(device)
            orig_mask = orig_mask.to(device)
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            batch_size = edit_seq.size(0)
            max_len = orig_seq.size(1)
            
            tgt_input = torch.full((batch_size, 1), -1.0, dtype=torch.float32, device=device)
            for step in range(max_len):
                seq_len = tgt_input.size(1)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                logits = model(edit_seq, tgt_input, src_key_padding_mask=edit_mask, tgt_mask=causal_mask)
                next_token_logits = logits[:, -1]
                next_token = (next_token_logits > 0.0).float().unsqueeze(1)
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                
            predictions = tgt_input[:, 1:]
            
            for i in range(batch_size):
                valid_len = (~orig_mask[i]).sum().item()
                true_seq_list = orig_seq[i][:valid_len].cpu().tolist()
                pred_seq_list = predictions[i][:valid_len].cpu().tolist()
                
                correct_bits = sum(1 for a, b in zip(true_seq_list, pred_seq_list) if a == b)
                bit_errors = valid_len - correct_bits
                true_dist = int(dist_tensor[i].item())
                
                distance_stats[true_dist]['bit_errors'] += bit_errors
                distance_stats[true_dist]['total_bits'] += valid_len
                distance_stats[true_dist]['block_errors'] += 1 if bit_errors > 0 else 0
                distance_stats[true_dist]['total_blocks'] += 1
                
                pred_dist = levenshtein_distance(true_seq_list, pred_seq_list)
                all_true_distances.append(true_dist)
                all_pred_distances.append(pred_dist)
                
            print(f"Evaluated Batch {batch_idx + 1}/{len(dataloader)}")
            
    dist_keys = sorted(distance_stats.keys())
    ber_curve = [distance_stats[d]['bit_errors'] / max(1, distance_stats[d]['total_bits']) for d in dist_keys]
    bler_curve = [distance_stats[d]['block_errors'] / max(1, distance_stats[d]['total_blocks']) for d in dist_keys]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dist_keys, ber_curve, marker='o', color='blue', label='Transformer BER')
    plt.plot(dist_keys, bler_curve, marker='x', color='red', label='Transformer BLER')
    plt.xlabel("True Edit Distance (Corruption Severity)")
    plt.ylabel("Error Rate")
    plt.title("Dynamic Error Rates vs Sequence Complexity")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/dynamic_error_rates.png")
    print("Dynamic error curve saved to plots/dynamic_error_rates.png")
    
    true_arr = np.array(all_true_distances)
    pred_arr = np.array(all_pred_distances)
    
    jitter_x = true_arr + np.random.uniform(-0.3, 0.3, size=true_arr.shape)
    jitter_y = pred_arr + np.random.uniform(-0.3, 0.3, size=pred_arr.shape)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(jitter_x, jitter_y, alpha=0.3, s=5)
    plt.plot([min(true_arr), max(true_arr)], [min(true_arr), max(true_arr)], 'r')
    plt.xlabel("True Edit Distance")
    plt.ylabel("Predicted Sequence Edit Distance")
    plt.title("Transformer ECC: True vs Predicted Edit Distance (Jittered)")
    plt.savefig("plots/distance_scatter_jittered.png")
    print("Jittered scatter plot saved to plots/distance_scatter_jittered.png")

if __name__ == "__main__":
    evaluate()

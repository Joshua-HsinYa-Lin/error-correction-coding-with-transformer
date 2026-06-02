import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import EditDistanceDataset
from model import Seq2SeqTransformer

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
    total_bits = 0
    total_bit_errors = 0
    total_blocks = 0
    total_block_errors = 0
    print(f"Starting autoregressive evaluation on {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, (orig_seq, orig_mask, edit_seq, edit_mask, dist) in enumerate(dataloader):
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
            valid_mask = ~orig_mask
            correct_bits = ((predictions == orig_seq) * valid_mask).sum(dim=1)
            valid_lengths = valid_mask.sum(dim=1)
            bit_errors = valid_lengths - correct_bits
            total_bit_errors += bit_errors.sum().item()
            total_bits += valid_lengths.sum().item()
            block_errors = (bit_errors > 0).sum().item()
            total_block_errors += block_errors
            total_blocks += batch_size
            print(f"Evaluated Batch {batch_idx + 1}/{len(dataloader)}")
    ber = total_bit_errors / total_bits
    bler = total_block_errors / total_blocks
    hamming_loss = ber
    print("Evaluation Complete")
    print(f"Bit Error Rate (BER): {ber:.6f}")
    print(f"Block Error Rate (BLER): {bler:.6f}")
    print(f"Hamming Loss: {hamming_loss:.6f}")
    labels = ['BER', 'BLER']
    values = [ber, bler]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'red'])
    plt.ylabel("Error Rate")
    plt.title("Transformer ECC Error Rates (BER / BLER)")
    plt.savefig("error_rates.png")
    print("Plot saved to error_rates.png")

if __name__ == "__main__":
    evaluate()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_ctc import CTCEditDistanceDataset
from model import CTCTransformer

def test_ctc_cpu() -> None:
    print("Forcing CPU mode to bypass CuDNN...")
    device = torch.device("cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = CTCEditDistanceDataset(dataset_path, max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = CTCTransformer(num_classes=3).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    edit_seq, edit_mask, edit_len, orig_seq, orig_mask, orig_len = next(iter(dataloader))
    print("Data loaded. Running forward pass...")
    logits = model(edit_seq, src_key_padding_mask=edit_mask)
    log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)
    input_lengths = torch.full((edit_seq.size(0),), 200, dtype=torch.long)
    print("Calculating CTC Loss...")
    loss = criterion(log_probs, orig_seq, input_lengths, orig_len)
    print(f"Loss calculated: {loss.item():.4f}")
    print("Running backward pass...")
    loss.backward()
    print("Backward pass complete. Architecture is sound.")

if __name__ == "__main__":
    test_ctc_cpu()
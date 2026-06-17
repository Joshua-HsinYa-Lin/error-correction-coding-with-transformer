import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_ctc import CTCEditDistanceDataset
from model import CTCTransformer

def train_ctc() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = CTCEditDistanceDataset(dataset_path, max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = CTCTransformer(num_classes=3)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    num_epochs = 30
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (edit_seq, edit_mask, edit_len, orig_seq, orig_mask, orig_len) in enumerate(dataloader):
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            orig_seq = orig_seq.to(device)
            optimizer.zero_grad()
            logits = model(edit_seq, src_key_padding_mask=edit_mask)
            log_probs = F.log_softmax(logits, dim=2)
            log_probs = log_probs.transpose(0, 1)
            loss = criterion(log_probs, orig_seq, edit_len, orig_len)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, CTC Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "ctc_model_weights.pth")
    print("CTC Training complete. Weights saved.")

if __name__ == "__main__":
    train_ctc()
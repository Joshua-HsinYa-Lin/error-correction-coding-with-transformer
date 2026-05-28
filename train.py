import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EditDistanceDataset
from model import Seq2SeqTransformer

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = EditDistanceDataset(dataset_path, max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = Seq2SeqTransformer()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    num_epochs = 2
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for orig_seq, orig_mask, edit_seq, edit_mask, dist in dataloader:
            orig_seq = orig_seq.to(device)
            orig_mask = orig_mask.to(device)
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            optimizer.zero_grad()
            seq_len = edit_seq.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            causal_mask = causal_mask.to(device)
            logits = model(orig_seq, edit_seq, src_key_padding_mask=orig_mask, tgt_key_padding_mask=edit_mask)
            loss_matrix = criterion(logits, edit_seq)
            valid_mask = ~edit_mask
            masked_loss = loss_matrix * valid_mask
            loss = masked_loss.sum() / valid_mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    train()

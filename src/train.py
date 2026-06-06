import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TripletEditDistanceDataset
from model import SiameseTransformer

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/triplet_datasets_K_90_num_seq_10000.h5"
    dataset = TripletEditDistanceDataset(dataset_path, max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = SiameseTransformer()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    num_epochs = 2
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (anc_seq, anc_mask, pos_seq, pos_mask, neg_seq, neg_mask) in enumerate(dataloader):
            anc_seq = anc_seq.to(device)
            anc_mask = anc_mask.to(device)
            pos_seq = pos_seq.to(device)
            pos_mask = pos_mask.to(device)
            neg_seq = neg_seq.to(device)
            neg_mask = neg_mask.to(device)
            optimizer.zero_grad()
            anc_emb = model(anc_seq, src_key_padding_mask=anc_mask)
            pos_emb = model(pos_seq, src_key_padding_mask=pos_mask)
            neg_emb = model(neg_seq, src_key_padding_mask=neg_mask)
            loss = criterion(anc_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Current Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    train()
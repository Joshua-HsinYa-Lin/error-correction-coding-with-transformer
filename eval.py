import torch
import torch.nn as nn
from dataset import EditDistanceDataset
from model import Seq2SeqTransformer

def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = EditDistanceDataset(dataset_path, max_seq_len=100)
    model = Seq2SeqTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    total_mse = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for orig_seq, orig_mask, edit_seq, edit_mask, dist in dataloader:
            orig_seq = orig_seq.to(device)
            orig_mask = orig_mask.to(device)
            edit_seq = edit_seq.to(device)
            edit_mask = edit_mask.to(device)
            dist = dist.to(device)
            logits = model(orig_seq, edit_seq, src_key_padding_mask=orig_mask, tgt_key_padding_mask=edit_mask)
            predictions = logits.mean(dim=1)
            mse = mse_loss(predictions, dist)
            mae = mae_loss(predictions, dist)
            total_mse += mse.item()
            total_mae += mae.item()
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    print("Evaluation Complete")
    print(f"MSE: {avg_mse:.4f}")
    print(f"MAE: {avg_mae:.4f}")

if __name__ == "__main__":
    evaluate()

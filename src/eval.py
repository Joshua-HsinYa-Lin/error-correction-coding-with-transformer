import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TripletEditDistanceDataset
from model import SiameseTransformer

def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/triplet_datasets_K_90_num_seq_10000.h5"
    dataset = TripletEditDistanceDataset(dataset_path, max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    model = SiameseTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    total_pos_dist = 0.0
    total_neg_dist = 0.0
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
            pos_dist = F.pairwise_distance(anc_emb, pos_emb)
            neg_dist = F.pairwise_distance(anc_emb, neg_emb)
            correct = (pos_dist < neg_dist).sum().item()
            total_correct += correct
            total_samples += anc_seq.size(0)
            total_pos_dist += pos_dist.sum().item()
            total_neg_dist += neg_dist.sum().item()
    accuracy = total_correct / total_samples
    avg_pos_dist = total_pos_dist / total_samples
    avg_neg_dist = total_neg_dist / total_samples
    print(f"Evaluation Complete on {total_samples} samples")
    print(f"Triplet Accuracy: {accuracy:.4f}")
    print(f"Average Positive Distance: {avg_pos_dist:.4f}")
    print(f"Average Negative Distance: {avg_neg_dist:.4f}")

if __name__ == "__main__":
    evaluate()
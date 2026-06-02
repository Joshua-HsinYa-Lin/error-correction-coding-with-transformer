import torch
from dataset import EditDistanceDataset
from model import Seq2SeqTransformer

def test_leakage() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset/edit_distance_dataset_K_90.h5"
    dataset = EditDistanceDataset(dataset_path, max_seq_len=100)
    orig_seq, orig_mask, edit_seq, edit_mask, dist = dataset[0]
    orig_seq = orig_seq.unsqueeze(0).to(device)
    orig_mask = orig_mask.unsqueeze(0).to(device)
    edit_seq = edit_seq.unsqueeze(0).to(device)
    edit_mask = edit_mask.unsqueeze(0).to(device)
    model = Seq2SeqTransformer()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    with torch.no_grad():
        leak_logits = model(orig_seq, edit_seq, src_key_padding_mask=orig_mask, tgt_key_padding_mask=edit_mask)
        leak_preds = (leak_logits > 0.0).float()
        blank_tgt = torch.zeros_like(edit_seq).to(device)
        blank_logits = model(orig_seq, blank_tgt, src_key_padding_mask=orig_mask, tgt_key_padding_mask=edit_mask)
        blank_preds = (blank_logits > 0.0).float()
        print(f"Ground Truth: {edit_seq[0, :10]}")
        print(f"Prediction (Leaked Target): {leak_preds[0, :10]}")
        print(f"Prediction (Blank Target): {blank_preds[0, :10]}")

if __name__ == "__main__":
    test_leakage()
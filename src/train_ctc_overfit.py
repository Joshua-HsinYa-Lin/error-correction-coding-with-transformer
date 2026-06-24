import itertools
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CTCTransformer
from ctc_decode import ctc_greedy_decode, sample_metrics

def process(seq: np.ndarray, max_len: int = 100) -> tuple:
    seq_len = len(seq)
    ctc_seq = seq + 1
    if seq_len > max_len:
        ctc_seq = ctc_seq[:max_len]
        valid_len = max_len
        mask = np.zeros(max_len, dtype=bool)
    else:
        pad_len = max_len - seq_len
        ctc_seq = np.pad(ctc_seq, (0, pad_len), "constant", constant_values=1)
        valid_len = seq_len
        mask = np.concatenate([np.zeros(seq_len, dtype=bool), np.ones(pad_len, dtype=bool)])
    return ctc_seq, mask, valid_len

def load_subset(file_path: str, num_samples: int) -> tuple:
    edit_seqs = []
    edit_masks = []
    edit_lens = []
    orig_seqs = []
    orig_lens = []
    orig_bits = []
    with h5py.File(file_path, "r") as f:
        for key in itertools.islice(f.keys(), num_samples):
            entry = f[key]
            original = np.array(entry["original"])
            edited = np.array(entry["edited"])
            e_seq, e_mask, e_len = process(edited)
            o_seq, o_mask, o_len = process(original)
            edit_seqs.append(e_seq)
            edit_masks.append(e_mask)
            edit_lens.append(e_len)
            orig_seqs.append(o_seq)
            orig_lens.append(o_len)
            orig_bits.append(original.tolist())
    edit_seq = torch.tensor(np.stack(edit_seqs), dtype=torch.long)
    edit_mask = torch.tensor(np.stack(edit_masks), dtype=torch.bool)
    edit_len = torch.tensor(edit_lens, dtype=torch.long)
    orig_seq = torch.tensor(np.stack(orig_seqs), dtype=torch.long)
    orig_len = torch.tensor(orig_lens, dtype=torch.long)
    return edit_seq, edit_mask, edit_len, orig_seq, orig_len, orig_bits

def decode_ber(logits: torch.Tensor, edit_len: torch.Tensor, orig_bits: list) -> float:
    bers = []
    for b in range(logits.size(0)):
        valid_time = int(edit_len[b]) * 2
        decoded = ctc_greedy_decode(logits[b:b + 1, :valid_time, :], blank=0)[0]
        pred_bits = [c - 1 for c in decoded]
        bers.append(sample_metrics(pred_bits, orig_bits[b])["bit_error_rate"])
    return float(np.mean(bers))

def overfit() -> None:
    device = torch.device("cpu")
    num_samples = 128
    edit_seq, edit_mask, edit_len, orig_seq, orig_len, orig_bits = load_subset("dataset/edit_distance_dataset_K_90.h5", num_samples)
    print(f"Loaded {num_samples} samples for overfit test")
    model = CTCTransformer(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    for step in range(300):
        optimizer.zero_grad()
        logits = model(edit_seq, src_key_padding_mask=edit_mask)
        log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)
        input_lengths = edit_len * 2
        loss = criterion(log_probs, orig_seq, input_lengths, orig_len)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step == 299:
            model.eval()
            with torch.no_grad():
                eval_logits = model(edit_seq, src_key_padding_mask=edit_mask)
            ber = decode_ber(eval_logits, edit_len, orig_bits)
            model.train()
            print(f"Step {step:4d}, CTC Loss: {loss.item():.4f}, Train-subset BER: {ber:.4f}")

if __name__ == "__main__":
    overfit()

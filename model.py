import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos_emb = self.pe[:, :seq_len, :]
        return x + pos_emb

class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 4, num_decoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1, out_dim: int = 1):
        #d_model is the dimension of the embedding space
        super().__init__()
        self.src_proj = nn.Linear(1, d_model)
        self.tgt_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src_exp = src.unsqueeze(-1)
        src_emb = self.src_proj(src_exp)
        src_pos = self.pos_encoder(src_emb)
        tgt_exp = tgt.unsqueeze(-1)
        tgt_emb = self.tgt_proj(tgt_exp)
        tgt_pos = self.pos_encoder(tgt_emb)
        out = self.transformer(src=src_pos, tgt=tgt_pos, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.out_proj(out)
        return logits.squeeze(-1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer().to(device)
    dummy_src = torch.randn(2, 100).to(device)
    dummy_tgt = torch.randn(2, 100).to(device)
    dummy_src_mask = torch.zeros(2, 100, dtype=torch.bool).to(device)
    dummy_tgt_mask = torch.zeros(2, 100, dtype=torch.bool).to(device)
    output = model(dummy_src, dummy_tgt, src_key_padding_mask=dummy_src_mask, tgt_key_padding_mask=dummy_tgt_mask)
    print(output.shape)

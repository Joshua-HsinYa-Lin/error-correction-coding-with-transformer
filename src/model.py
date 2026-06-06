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

class SiameseTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.src_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src_exp = src.unsqueeze(-1)
        src_emb = self.src_proj(src_exp)
        src_pos = self.pos_encoder(src_emb)
        out = self.transformer_encoder(src=src_pos, src_key_padding_mask=src_key_padding_mask)
        out_pooled = out.mean(dim=1)
        return out_pooled
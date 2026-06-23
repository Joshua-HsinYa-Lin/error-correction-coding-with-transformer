import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CTCTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.src_embedding = nn.Embedding(num_classes, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.ctc_head = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src_emb = self.src_embedding(src)
        src_pos = self.pos_encoder(src_emb)
        enc_out = self.transformer_encoder(src=src_pos, src_key_padding_mask=src_key_padding_mask)
        enc_out_transposed = enc_out.transpose(1, 2)
        enc_out_upsampled = F.interpolate(enc_out_transposed, scale_factor=2.0, mode='nearest')
        enc_out_restored = enc_out_upsampled.transpose(1, 2)
        logits = self.ctc_head(enc_out_restored)
        return logits
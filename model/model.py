import torch
import torch.nn as nn
import math

# ------------------------- GRU 生成器 -------------------------
class GeneratorGRU(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear_2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.linear_3 = nn.Linear(hidden_dim // 4, out_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=device)

        out, _ = self.gru(x, h0)                 # (B, T, hidden_dim)
        last_feature = self.dropout(out[:, -1])  # (B, hidden_dim)
        return self.linear_3(self.linear_2(self.linear_1(last_feature)))


# ------------------------- LSTM 生成器 -------------------------
class GeneratorLSTM(nn.Module):
    def __init__(self, input_size, out_size,
                 hidden_size=128, num_layers=1, dropout=0.1):
        super().__init__()

        self.depth_conv = nn.Conv1d(input_size, input_size, kernel_size=3,
                                    padding=1, groups=input_size)
        self.point_conv = nn.Conv1d(input_size, input_size, kernel_size=1)
        self.act = nn.ReLU()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x, hidden=None):
        # (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.act(self.point_conv(self.depth_conv(x)))
        x = x.permute(0, 2, 1)                   # back to (B, T, F)

        lstm_out, hidden = self.lstm(x, hidden)  # (B, T, hidden_size)
        return self.linear(lstm_out[:, -1])      # (B, out_size)


# ------------------------- 位置编码 -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.encoding[:, :x.size(1)]


# ------------------------- Transformer 生成器 -------------------------
class GeneratorTransformer(nn.Module):
    def __init__(self, input_dim, feature_size=128, num_layers=2,
                 num_heads=8, dropout=0.1, output_len=1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=num_heads,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer,
                                                         num_layers=num_layers)

        self.decoder = nn.Linear(feature_size, output_len)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)

    # ---- 修改 ①：直接返回 bool 掩码 ----
    @staticmethod
    def _generate_square_subsequent_mask(seq_len, device):
        """
        True → 被屏蔽；False → 可见
        shape: (seq_len, seq_len)
        """
        return torch.triu(torch.ones(seq_len, seq_len,
                                     dtype=torch.bool, device=device), 1)

    def forward(self, src, src_mask=None):
        """
        src: (B, T, input_dim)
        """
        seq_len = src.size(1)
        src = self.pos_encoder(self.input_projection(src))

        # ---- 修改 ②：把 device 直接传给 mask ----
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len,
                                                             src.device)

        output = self.transformer_encoder(src, src_mask)  # (B, T, D)
        return self.decoder(output[:, -1])                # 只取最后一步

import torch
from torch import nn


class AirQualityModel(nn.Module):
    def __init__(
        self,
        num_provinces,
        seq_len=30,
        seq_input_dim=17,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        embed_dim=16,
    ):
        super().__init__()

        self.province_embed = nn.Embedding(num_provinces, embed_dim)

        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 7),
        )

    def forward(self, province_ids, seq_data):
        _, (hidden, _) = self.lstm(seq_data)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        province_emb = self.province_embed(province_ids)
        combined = torch.cat([last_hidden, province_emb], dim=-1)
        prediction = self.fc(combined)
        return prediction


if __name__ == "__main__":
    num_provinces = 31

    model = AirQualityModel(num_provinces=num_provinces)

    province_ids = torch.tensor([0, 1, 2])
    seq_data = torch.randn(3, 30, 17)

    pred = model(province_ids, seq_data)
    print(f"预测结果形状: {pred.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

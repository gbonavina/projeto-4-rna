import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """(B, T, F) sequence -> (B, n_classes) logits.

    A bidirectional LSTM reads the whole clip (valid here: classification uses
    the full utterance, there is no "future" to leak). Its outputs are
    summarised with mean + max pooling over time, so the decision uses every
    frame, not just the last one.
    """

    def __init__(self, input_size, hidden_size, num_layers, n_classes,
                 dropout=0.0, bidirectional=True):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(self.input_norm(x))  # (B, T, out_dim)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)
        return self.head(pooled)

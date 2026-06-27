import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels     = in_channels
        padding              = kernel_size // 2

        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates    = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, H, W, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, H, W, device=device),
            torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for spatio-temporal climate prediction.

    Encoder: processes input sequence (in_channels=3)
    Decoder: generates future predictions (all hidden_channels)
    """

    def __init__(
        self,
        in_channels:     int = 3,
        hidden_channels: int = 64,
        kernel_size:     int = 3,
        num_layers:      int = 2,
        pred_len:        int = 3,
        out_channels:    int = 3
    ):
        super().__init__()
        self.num_layers      = num_layers
        self.pred_len        = pred_len
        self.out_channels    = out_channels
        self.hidden_channels = hidden_channels

        # ── Encoder cells ─────────────────────────────────────────────────────
        # Layer 0: raw input (in_channels=3)
        # Layer 1+: hidden state from previous layer (hidden_channels)
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = in_channels if i == 0 else hidden_channels
            self.encoder_cells.append(
                ConvLSTMCell(cell_in, hidden_channels, kernel_size)
            )

        # ── Decoder cells ─────────────────────────────────────────────────────
        # ALL decoder layers receive hidden_channels sized input
        # because decoder feeds hidden states between layers
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_cells.append(
                ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
            )

        # ── Output projection ─────────────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)
        )

        self.dropout = nn.Dropout2d(p=0.2)

        # Print architecture summary
        enc_params = sum(p.numel() for c in self.encoder_cells for p in c.parameters())
        dec_params = sum(p.numel() for c in self.decoder_cells for p in c.parameters())
        out_params = sum(p.numel() for p in self.output_conv.parameters())
        print(f"   Encoder params : {enc_params:,}")
        print(f"   Decoder params : {dec_params:,}")
        print(f"   Output  params : {out_params:,}")
        print(f"   Total   params : {enc_params + dec_params + out_params:,}")

    def forward(self, x):
        """
        x      : (batch, seq_len, in_channels, H, W)
        returns: (batch, pred_len, out_channels, H, W)
        """
        batch, seq_len, C, H, W = x.shape
        device = x.device

        # ── ENCODER ───────────────────────────────────────────────────────────
        # Initialize encoder hidden states
        enc_hidden = [
            cell.init_hidden(batch, H, W, device)
            for cell in self.encoder_cells
        ]

        for t in range(seq_len):
            layer_input = x[:, t]              # (batch, in_channels, H, W)
            for i, cell in enumerate(self.encoder_cells):
                h, c           = enc_hidden[i]
                h, c           = cell(layer_input, h, c)
                enc_hidden[i]  = (h, c)
                layer_input    = h             # pass hidden to next layer

        # ── DECODER ───────────────────────────────────────────────────────────
        # Initialize decoder hidden states from encoder final states
        dec_hidden = [
            (enc_hidden[i][0].clone(), enc_hidden[i][1].clone())
            for i in range(self.num_layers)
        ]

        predictions  = []

        # First decoder input: last encoder hidden state of final layer
        decoder_input = enc_hidden[-1][0]      # (batch, hidden_channels, H, W)

        for _ in range(self.pred_len):
            layer_input = decoder_input

            for i, cell in enumerate(self.decoder_cells):
                h, c           = dec_hidden[i]
                h, c           = cell(layer_input, h, c)
                dec_hidden[i]  = (h, c)
                layer_input    = h

            out = self.dropout(layer_input)
            out = self.output_conv(out)        # (batch, out_channels, H, W)
            predictions.append(out)

            # Next decoder input = last decoder layer hidden state
            decoder_input = dec_hidden[-1][0]

        return torch.stack(predictions, dim=1) # (batch, pred_len, out_channels, H, W)


if __name__ == "__main__":
    print("🧪 Testing ConvLSTM architecture...\n")
    model  = ConvLSTM(in_channels=3, hidden_channels=64, num_layers=2, pred_len=3)
    dummy  = torch.randn(2, 7, 3, 29, 33)
    output = model(dummy)
    print(f"\n✅ Input  shape : {dummy.shape}")
    print(f"✅ Output shape : {output.shape}")
    print(f"✅ Architecture test passed!")
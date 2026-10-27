# Trajectory (emotion) scorer -- the proposal's main model: a 1-layer GRU,
# 64 hidden units, over the full prefix of per-turn emotion features. Trained
# many-to-many (a score at every turn, not just the end) so it can be used
# for the earliness metric, with the same per-turn weak labels as the
# message-level baseline.
import numpy as np
import torch
import torch.nn as nn


class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x: (batch, max_len, input_dim) padded; lengths: (batch,)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return self.out(h).squeeze(-1)  # (batch, max_len) raw logits


class TrajectoryModel:
    def __init__(self, input_dim=15, hidden_dim=64, epochs=30, lr=1e-2, seed=0):
        torch.manual_seed(seed)
        self.net = TrajectoryGRU(input_dim, hidden_dim)
        self.epochs = epochs
        self.lr = lr

    def fit(self, feature_seqs, label_seqs):
        # feature_seqs: list of (len_i, input_dim) arrays; label_seqs: list of (len_i,) arrays
        lengths = [len(f) for f in feature_seqs]
        max_len = max(lengths)
        dim = feature_seqs[0].shape[1]
        n = len(feature_seqs)

        X = torch.zeros(n, max_len, dim)
        Y = torch.zeros(n, max_len)
        mask = torch.zeros(n, max_len)
        for i, (f, y) in enumerate(zip(feature_seqs, label_seqs)):
            L = len(f)
            X[i, :L] = torch.tensor(f, dtype=torch.float32)
            Y[i, :L] = torch.tensor(y, dtype=torch.float32)
            mask[i, :L] = 1.0
        lengths_t = torch.tensor(lengths)

        # class-balance the loss: real turns and most legitimate pools push
        # the label distribution heavily toward 0, so weight positive turns
        # by the actual negative:positive ratio in this training set
        n_pos = (Y * mask).sum().clamp(min=1.0)
        n_neg = mask.sum() - n_pos
        pos_weight = (n_neg / n_pos).clamp(min=1.0)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.net(X, lengths_t)
            loss = (loss_fn(logits, Y) * mask).sum() / mask.sum()
            loss.backward()
            opt.step()
        return self

    def running_score(self, turn_feats: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(turn_feats, dtype=torch.float32).unsqueeze(0)
            logits = self.net(x, torch.tensor([len(turn_feats)]))
            return torch.sigmoid(logits).squeeze(0).numpy()

"""LSTM variants for the Slot 2 research (PyTorch; research only, not used by the slot).

Family B      : LSTM direction classifier on sequences of daily OHLC features.
Family C-LSTM : LSTM that outputs the position directly, trained on -Sharpe with a
                turnover penalty (Lim, Zohren & Roberts 2019), on DMN momentum features.

Same leakage rules as the slot: sequences end at t and use data up to the close of t;
targets are built inside fit() from the train frame; scaling uses train statistics;
early stopping holds out the last 15% of sequences in time order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from qqq import config
from qqq import features as F
from qqq.strategies.base import Strategy
from qqq.strategies.neural_net import direction_features, momentum_features, sharpe_loss

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


class _Net(torch.nn.Module):
    def __init__(self, n_in, hidden, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.sigmoid(self.head(self.drop(h[:, -1, :]))).squeeze(-1)


def _sharpe_loss_t(a, x, cost):
    da = torch.diff(a, prepend=a[:1])
    R = a * x - cost * torch.abs(da)
    return -np.sqrt(252) * R.mean() / (R.std(unbiased=False) + 1e-9)


class LSTMStrategy(Strategy):
    name = "nn_lstm"

    def __init__(self, mode="direction", seq_len=20, hidden=16, turnover_bps=5.0,
                 threshold="base", n_seeds=3, max_epochs=150, patience=20,
                 seed=config.RANDOM_SEED):
        hidden = hidden[0] if isinstance(hidden, (tuple, list)) else hidden
        self.mode, self.seq_len, self.hidden = mode, int(seq_len), int(hidden)
        self.turnover_bps, self.threshold = float(turnover_bps), threshold
        self.n_seeds, self.max_epochs = n_seeds, max_epochs
        self.patience, self.seed = patience, seed

    def _feat(self, prices):
        if self.mode == "direction":
            return direction_features(prices, 1)        # today's daily features; LSTM adds memory
        return momentum_features(prices)

    def _sequences(self, Z: np.ndarray):
        L = self.seq_len
        idx = np.arange(L - 1, len(Z))
        return np.stack([Z[i - L + 1:i + 1] for i in idx]), idx

    def fit(self, train_prices):
        X = self._feat(train_prices)
        nxt = F.forward_return(train_prices, 1)
        ok = X.notna().all(axis=1)
        first = ok.idxmax()
        X, nxt = X.loc[first:], nxt.loc[first:]
        self.mu, self.sd = X.mean(), X.std(ddof=0).replace(0, 1)
        Z = ((X - self.mu) / self.sd).clip(-5, 5).to_numpy(dtype=np.float32)
        seqs, idx = self._sequences(Z)
        x = nxt.to_numpy()[idx]
        keep = ~np.isnan(x)                               # last train row has no target
        seqs, x = seqs[keep], x[keep]
        self.threshold_ = float((x > 0).mean()) if self.threshold == "base" else self.threshold
        y = (x > 0).astype(np.float32) if self.mode == "direction" else x.astype(np.float32)
        n_hold = int(len(seqs) * 0.15)
        Xt, yt = torch.tensor(seqs[:-n_hold]), torch.tensor(y[:-n_hold])
        Xh, yh = torch.tensor(seqs[-n_hold:]), torch.tensor(y[-n_hold:])
        cost = self.turnover_bps / 1e4
        self.nets = []
        for k in range(self.n_seeds):
            torch.manual_seed(self.seed + k)
            net = _Net(Z.shape[1], self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_state, wait = np.inf, None, 0
            for _ in range(self.max_epochs):
                net.train()
                opt.zero_grad()
                a = net(Xt)
                loss = (torch.nn.functional.binary_cross_entropy(a, yt) if self.mode == "direction"
                        else _sharpe_loss_t(a, yt, cost))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                net.eval()
                with torch.no_grad():
                    ah = net(Xh)
                    bce = torch.nn.functional.binary_cross_entropy
                    hl = (bce(ah, yh) if self.mode == "direction"
                          else _sharpe_loss_t(ah, yh, cost)).item()
                if hl < best - 1e-6:
                    state = {k2: v.clone() for k2, v in net.state_dict().items()}
                    best, best_state, wait = hl, state, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        return self

    def raw_output(self, prices):
        X = self._feat(prices)
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=prices.index)
        if ok.sum() < self.seq_len:
            return out
        first = ok.idxmax()
        Z = ((X.loc[first:] - self.mu) / self.sd).clip(-5, 5).to_numpy(dtype=np.float32)
        seqs, idx = self._sequences(Z)
        with torch.no_grad():
            preds = np.mean([n(torch.tensor(seqs)).numpy() for n in self.nets], axis=0)
        out.loc[X.loc[first:].index[idx]] = preds
        return out

    def generate_positions(self, prices):
        y = self.raw_output(prices)
        pos = (y > self.threshold_).astype(float) if self.mode == "direction" else y.fillna(0.0)
        return pos.where(y.notna(), 0.0).clip(0, 1).rename("position")


def grid_lstm_direction():
    for L in (20, 63):
        yield {"mode": "direction", "seq_len": L, "hidden": 16}


def grid_lstm_sharpe():
    for L in (20, 63):
        yield {"mode": "sharpe", "seq_len": L, "hidden": 16, "turnover_bps": 25.0}


def make_lstm(cfg):
    return LSTMStrategy(**cfg)


_ = sharpe_loss  # NumPy twin of _sharpe_loss_t, kept for reference

# AME-Network

Here’s a sketch of an **AME layer** implemented as a PyTorch module.
It behaves like an RNN cell: takes input **xₜ**, hidden state **θₜ**, coefficients **Cₜ**, and evolves them with AME dynamics.

---

# 🔹 AME Layer (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMECell(nn.Module):
    def __init__(self, input_dim, hidden_dim, coeff_dim, hidden_size=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeff_dim = coeff_dim

        # networks controlling drift, noise scale, dither, macro
        self.f_net = nn.Sequential(
            nn.Linear(coeff_dim + input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_dim)
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(coeff_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_dim)
        )
        self.d_net = nn.Sequential(
            nn.Linear(coeff_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_dim)
        )
        self.m_net = nn.Sequential(
            nn.Linear(coeff_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_dim)
        )

        # coefficient update network
        self.g_net = nn.Sequential(
            nn.Linear(coeff_dim + hidden_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, coeff_dim)
        )

    def forward(self, x, theta, C):
        """
        x     : input at time t (batch × input_dim)
        theta : hidden state at time t (batch × hidden_dim)
        C     : coefficients at time t (batch × coeff_dim)
        """
        batch = x.size(0)

        # concatenate input with coefficients for drift
        drift_input = torch.cat([x, C], dim=-1)
        drift = self.f_net(drift_input)

        # reparameterized noise
        sigma = F.softplus(self.sigma_net(C)) + 1e-3
        noise = sigma * torch.randn_like(theta)

        # dither term
        dither = torch.sin(self.d_net(C))

        # macro term (gated)
        macro = torch.tanh(self.m_net(C))

        # hidden update
        theta_new = theta + drift + noise + dither + macro

        # coefficient update (feedback uses hidden)
        coeff_input = torch.cat([theta_new, C], dim=-1)
        C_new = C + self.g_net(coeff_input)

        return theta_new, C_new
```

---

# 🔹 Wrapping into a Layer

```python
class AMELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, coeff_dim):
        super().__init__()
        self.cell = AMECell(input_dim, hidden_dim, coeff_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)  # map back to output

    def forward(self, x_seq):
        """
        x_seq : sequence (batch × time × input_dim)
        """
        batch, T, _ = x_seq.shape
        theta = torch.zeros(batch, self.cell.hidden_dim, device=x_seq.device)
        C = torch.zeros(batch, self.cell.coeff_dim, device=x_seq.device)

        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            theta, C = self.cell(x_t, theta, C)
            y_t = self.readout(theta)
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # batch × time × input_dim
```

---

## How to use

```python
# Example: sequence modeling
batch, T, input_dim = 16, 50, 8
hidden_dim, coeff_dim = 32, 16

model = AMELayer(input_dim, hidden_dim, coeff_dim)
x_seq = torch.randn(batch, T, input_dim)
y_seq = model(x_seq)  # output sequence
print(y_seq.shape)  # (16, 50, 8)
```

---

## Notes

* **θ (hidden state)** = memory / representation.
* **C (coefficients)** = adaptive internal parameters, evolving online.
* Noise, dither, and macro shifts keep dynamics chaotic/adaptive.
* This can replace an RNN/GRU/LSTM cell in sequence models.

---

✅ With this, **AME is a neural network**: a chaotic, adaptive recurrent layer.

Do you want me to also show how to **train this AME layer on a real task** (e.g. sequence prediction on sine waves or character-level text) to demonstrate it learning?

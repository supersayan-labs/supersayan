# Supersayan

A high‑performance Python library that integrates Fully Homomorphic Encryption (FHE) into PyTorch workflows. Supersayan provides:

- A client/server path to offload selected layers to a remote FHE server (hybrid mode).
- Zero‑copy tensor bridges across PyTorch, NumPy, CuPy, and Julia.

The Julia backend is managed via `juliacall` and initialized automatically on first import, or explicitly with a helper CLI.

## Installation

From PyPI:

```bash
pip install supersayan
# or with uv
uv add supersayan
```

Default behavior:
- After install, importing `supersayan` triggers a one‑time Julia backend setup automatically. No extra step is needed for typical users.

Optional (CI/Docker or troubleshooting):

```bash
# Manually initialize the Julia backend if you want explicit control
supersayan-setup
```

Advanced control (not recommended for regular users):
- To prevent network access during import in CI images, set `SUPERSAYAN_SKIP_JULIA_SETUP=1` and run `supersayan-setup` in a controlled build step.

Notes:
- GPU support uses CuPy when available; code runs on CPU otherwise.

## Hybrid Remote Inference

Run the TCP server:

```bash
python scripts/run_server.py --host 127.0.0.1 --port 8000 --models-dir /tmp/supersayan/models
```

Use the client to execute only selected layers remotely in FHE while keeping other ops local:

```python
import torch
import torch.nn as nn
from supersayan.core.types import SupersayanTensor
from supersayan.remote import SupersayanClient

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4*28*28, 10)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.view(x.size(0), -1))

cnn = SmallCNN().eval()
client = SupersayanClient(
    server_url="127.0.0.1:8000",
    torch_model=cnn,
    fhe_modules=[nn.Conv2d, nn.Linear],  # offload these layers
)

x = SupersayanTensor(torch.randn(1, 1, 28, 28))
y = client(x)
```

For a runnable example of the TCP server, see `scripts/run_server.py`.

Supported offloaded layers: `nn.Linear`, `nn.Conv2d`.

## Tensors and Interop

- `SupersayanTensor(data, device=...)` accepts `torch.Tensor`, `numpy.ndarray`, or `cupy.ndarray` and preserves dtype float32.
- Helpers: `SupersayanTensor.zeros(...)`, `ones(...)`, `randn(...)`.
- Interop: `.to_numpy()`, `.to_dlpack()`, and zero‑copy conversion to Julia via `.to_julia()`.

## Project Layout

- `src/supersayan/` core, layers, remote client/server, Julia backend
- `scripts/` runnable examples (`run_server.py`)

## Troubleshooting

- If Julia setup fails on first import, run `supersayan-setup` manually.
- In CI or headless environments, set `SUPERSAYAN_SKIP_JULIA_SETUP=1` during import and run `supersayan-setup` explicitly in a build step.

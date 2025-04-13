# 🚀 Supersayan

**Supersayan** is a Python frontend for [Julia SupersayanTFHE](https://github.com/supersayan-org/SupersayanTFHE), enabling **Fully Homomorphic Encryption (FHE)** with a PyTorch-style API for neural networks.

---

## 🧱 Prerequisites

To use Supersayan, you’ll need the following installed:

- Python **3.9+**
- Julia (**latest stable version** recommended)

### 📦 Install Julia

- **macOS** (Homebrew):
  ```bash
  brew install julia
  ```

- **Linux** (APT):
  ```bash
  sudo apt install julia
  ```

---

## 🛠️ Setup

### 1. Sync Dependencies

Install all required Python packages:
```bash
uv sync
```

---

## 🚀 Usage

Coming soon: usage examples and tutorials for building privacy-preserving neural networks on encrypted data using the Supersayan API.

---

## 📅 Development Roadmap

### 🔜 Next Week

- [ ] Build **hybrid server-side architecture**
- [ ] Run **initial performance benchmarks**
- [ ] Implement **2D convolution** following concepts from the [Orion FHE paper](https://eprint.iacr.org/2023/1314)

### 📌 Follow-Up Features

- [ ] Add **sparsity support**
- [ ] Apply **compression techniques** (e.g., pruning)

### 🎯 Final Steps (Optional)

- [ ] Add **GPU acceleration** support

## Tests
To run the tests, execute the following command:
```bash
python -m pytest tests/ -v
```

## Roadmap
- Implement double hoisting for Conv2D as described in the Orion paper
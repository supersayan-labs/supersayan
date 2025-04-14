# 🚀 Supersayan

**Supersayan** is a comprehensive project that includes both a Python frontend and a Julia backend. The Python frontend provides a PyTorch-style API for neural networks, while the Julia backend, known as [Julia SupersayanTFHE](https://github.com/bonsainoodle/supersayan), enables **Fully Homomorphic Encryption (FHE)** capabilities. This allows for the integration of FHE into machine learning operations, ensuring the security and privacy of sensitive data.

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

## 🚀 Benchmarks

To run the benchmarks, execute the following command:
```bash
pytest benchmarks/ -v
```
After running the benchmarks, you can compare the results using:
```bash
pytest-benchmark compare last mysession
```

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
pytest tests/ -v
```

## Roadmap
- Implement double hoisting for Conv2D as described in the Orion paper
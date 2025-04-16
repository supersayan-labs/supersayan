# SuperSayan: Privacy-Preserving Neural Networks

## Technical Documentation for SACS Lab Meeting

<p align="center">
  <img src="https://media.cleanshot.cloud/media/1430/vshfcLTG4wGYwExCbXE3Kr86t1Ltxew5cVR61U9N.jpeg?Expires=1744817223&Signature=ZStTSYhMUeivc0nboaO57wjoDdcLKZoTWn1Y1FwpB~Syiud0crfRjTxgzvDuQlStA-XSmOMRbDS3i1mao-L5k4CbsSWKMU8huxDCb-9CRZaPeBt2Tq5TFT5afuToFLIZjvvPJoI9GTcngDT~jBSG3WBnTeO5qP7qdZwNvelf6K-XOT59uezhuy9PCZ91y0VdhNExpzP0syE~4UAKRMz0-pcjeeyBCcRiG6YpefgtC-2~FHRGn0vyYiQ-fz~lZCnaUB1N7qne7GAYus6ae1xUcKqdI33RYz5WNELQkQWUWQJv9PzOJ89sJtf7QYV6CtTJrusMA7uSEhBjHAmPPI6T9g__&Key-Pair-Id=K269JMAT9ZF4GZ" width="150" height="150" alt="SuperSayan Logo">
</p>

***

## 📋 Project Overview

SuperSayan is a Python library that implements privacy-preserving neural networks using Fully Homomorphic Encryption (FHE). It enables machine learning computations directly on encrypted data, ensuring privacy throughout the entire processing pipeline.

The system combines a PyTorch-style API with an optimized Julia backend to provide both usability and performance for privacy-preserving machine learning.

***

## 🏗️ System Architecture

SuperSayan follows a hybrid architecture with three main components:

1. **Python Frontend**: PyTorch-style API for defining, training, and deploying models
2. **Julia Backend**: Optimized implementation of FHE operations
3. **Client-Server Architecture**: Distributed secure computation system

<p align="center">
  <pre>
  ┌─────────────────┐     ┌─────────────────┐
  │                 │     │                 │
  │  Python Client  │────▶│  Julia Server   │
  │  (PyTorch API)  │◀────│  (TFHE Engine)  │
  │                 │     │                 │
  └─────────────────┘     └─────────────────┘
         ▲                       ▲
         │                       │
         │                       │
  ┌──────┴──────────┐    ┌───────┴─────────┐
  │ Encryption Keys │    │ FHE Operations  │
  │ Model Definition│    │ Secure Compute  │
  └─────────────────┘    └─────────────────┘
  </pre>
</p>

***

## 🧮 Core Components

### 1. Encryption System

- Fully Homomorphic Encryption (FHE)
- Key generation, encryption, and decryption functionality
- Noise management for numerical stability (not implemented - no bootstrapping yet)

### 2. Neural Network Layers

- **Linear (Fully Connected) Layers**
  
  - Matrix-vector multiplications on encrypted data

- **Convolutional Layers**
  
  - Standard Conv2d implementation
  - Support for stride, padding, and kernel configurations
  - Optimized Conv2dOrion
    - Toeplitz matrices
    - Optimized BSGS (Baby-Step Giant-Step) algorithm

### 3. Client-Server System (Hybrid inference)

- Client-side encryption and model preparation
- Server-side secure computation (never sees plaintext)
- Support for pure FHE or hybrid execution models

***

## ✨ Key Features & Capabilities

### Privacy-Preserving Computation

- End-to-end encryption for sensitive data
- Computations performed directly on encrypted data
- Server never sees plaintext inputs or model parameters

### PyTorch Integration

- Familiar API compatible with PyTorch
- Easy conversion of existing models
- Support for hybrid models (mix of encrypted/unencrypted operations)

### Optimized FHE Operations

- BSGS algorithm reducing complexity from O(n) to O(√n) - need to implement double hoisting for this
- Multi-threading for performance
- Advanced Orion optimization techniques:
  - Toeplitz encoding for efficient convolutions
  - Single-shot multiplexing for stride operations

***

## 💻 Technical Implementation Details

### Optimized Dot Products

- Standard dot product: O(n) complexity
- BSGS optimization: O(√n) complexity
- Thread-parallel implementation for batch inference

### Advanced Convolutions

- Toeplitz matrix representation
- Single-shot rotations for stride operations
- Double-hoisting technique (in development)

***

## 🔄 Execution Models

### Pure FHE Mode

- All computations performed on encrypted data
- Maximum privacy guarantees
- Higher computational overhead

### Hybrid Mode

- Selected layers (e.g., Linear) run in FHE
- Non-linear operations (e.g., ReLU) run locally
- Balance between privacy and performance

***

## 📊 Performance Benchmarks

![Performance Benchmarks](https://media.cleanshot.cloud/media/1430/Qou1d3tRQw68evUd4jQE5IhCsUfF2THQ0NWKiWYR.jpeg?Expires=1744817179&Signature=H4RsSQm33hIMexeOTIy20qca2x2HjBXMzVXKoBL4SkgTXnmcHi8o1k7MPKj8F3sWz6VRgbeFqy6OCGCxWWtjQ8GD9qNOXQpveeUuF1H6fwIjJSgY5znlTqxYd1Wz5eeL~E0E4qQEM6o8xbmRod98Band71KtmIMw-6322HQIvNSoaRu~L~OY2YkpWMFDhOxlb8OT~j3GN1sa5bw7GtM8k~3QrwgZQP6JLqfnGYBZ6z5xiyJ1WwobCtuI1y5xPkwr64NDAXSma7BkYvjaxxvxYQPDB8nXIdtQzs4l4mEGBMwOkzzqhrXH36s8dMC5rVbjVl2Rj0o~PpI6~46huUCWcg__&Key-Pair-Id=K269JMAT9ZF4GZ)

*Note: Measurements from pytest benchmark tests*

***

## 📈 Current Progress

### Completed Features

- ✅ Core FHE operations (encryption, decryption, add, multiply)
- ✅ Linear layer implementation with BSGS optimization
- ✅ Conv2d standard implementation
- ✅ Conv2dOrion implementation with Toeplitz optimization
- ✅ Client-server architecture for distributed computation

### In Development

- 🔄 Double-hoisting technique for Conv2d
- 🔄 Cache persistence improvements
- 🔄 Caching system for performance improvements
- 🔄 Sparse matrix support for performance gains
- 🔄 Prune and ensemble benchmark
- 🔄 GPU support

***

## 🔗 References & Resources

1. Orion FHE Paper: [Orion: Private Neural Networks via Sublinear Homomorphic Encryption](https://arxiv.org/pdf/2311.03470)

***

<p align="center">
  <b>SACS Lab - Scalable Computing Systems Laboratory</b><br>
  April 16, 2025
</p>
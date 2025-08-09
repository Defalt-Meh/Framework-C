# FRAMEWORK-C

*High-performance C-based Neural Network Library with Python Integration*

---

## 📖 Overview

FRAMEWORK-C is a lightweight, ISO C99 neural network library designed for seamless integration into Python workflows via the C API. It provides core primitives for building, training, and evaluating multi-layer perceptrons, while allowing you to choose between system BLAS or a portable fallback for matrix operations.

Key benefits:

* **Pure C implementation** for forward/backward propagation, gradient updates, and evaluation
* **Flexible BLAS support**: Accelerate (macOS), OpenBLAS/System BLAS, or built-in portable fallback (ELAS)
* **Zero-copy NumPy bridge**: expose `NNbuild`, `NNtrain`, `NNinfer`, `NNload`, and `NNsave` to Python
* **High performance**: benchmarks show competitive inference & training times vs. PyTorch/TensorFlow
* **Modular design**: easily swap kernels, tune tile sizes, or integrate custom optimized routines

---

## 🚀 Features

* **Neural Network Primitives**

  * Build arbitrary MLP architectures: customizable input, hidden, and output layer sizes
  * Fast & precise sigmoid activation variants
  * SGD training loop with mean squared error (MSE) loss and accuracy metrics

* **Matrix Operations**

  * Level-1 & Level-2 routines: dot, saxpy, copy, scale
  * GEMM via `cblas_sgemm` (system BLAS) or `elas_gemm` fallback
  * Optional OpenMP parallelization for multi-core acceleration

* **Python Extension**

  * Wraps core C functions in a Python module for easy scripting
  * Accepts NumPy arrays without extra data copies

* **Benchmark Suite**

  * Demo script on the Semeion digits dataset (2.9 MB)
  * Reports loss, accuracy, and inference/training timing comparisons

---

## ⚙️ Requirements

* **C Compiler** with C99 support (GCC, Clang, MSVC)
* **Python ≥ 3.8** (tested up to 3.13)
* **NumPy**
* **BLAS Library** (one of):

  * Apple Accelerate (macOS)
  * OpenBLAS / system BLAS (Linux/Windows)
  * *or* use the bundled portable fallback (ELAS) by defining `ELAS_LOCAL`
* **Optional**: OpenMP flags (`-fopenmp` or `/openmp`) to enable threading

---

## 📥 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/framework-c.git
   cd framework-c
   ```
2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\\Scripts\\activate
   ```
3. **Install Python dependencies**

   ```bash
   pip install --upgrade pip setuptools wheel numpy
   ```
4. **(Optional) Force portable fallback**
   In `setup.py`, add `-DELAS_LOCAL` to `CFLAGS` or set environment variable:

   ```bash
   export CFLAGS="-DELAS_LOCAL"
   ```

---

## 🔧 Usage

Under your activated virtual environment, execute:

```bash
source venv/bin/activate        # Activate virtualenv
python setup.py clean --all      # Remove previous build artifacts
python setup.py build_ext --inplace  # Compile the C extension in-place
python benchmark.py              # Run training & inference benchmarks
deactivate                       # Exit virtualenv
```

* **`clean --all`**: Cleans up `build/`, `dist/`, and shared libraries (`*.so`/`.pyd`).
* **`build_ext --inplace`**: Compiles and places the extension module alongside Python files for import.
* **`benchmark.py`**: Trains a 2-layer MLP on the Semeion dataset, printing loss, accuracy, and timing vs. PyTorch/TensorFlow.

---

## 📂 Project Structure

```
framework-c/
│                
├── src/                   # Core C implementation
│   ├── data_split.h
│   ├── data_split.c
│   ├── elas.c             # Portable BLAS fallback (ELAS)
│   ├── elas.c
│   ├── model_selection.c
│   ├── model_selection.c
│   ├── my_module.c
│   ├── nn.c
│   ├── nn.c
│   ├── utils.c
│   └── utils.c
├── benchmark_mnist.py      
├── benchmark_semeion.py 
├── setup.py               # Build script (setuptools Extension)
├── README.md              # Project documentation
└── LICENSE                # MIT License
```

---

## 📊 Benchmark Results

*Sample (100-run average) on x86\_64 single-core:*

| Metric           |     FRAMEWORK-C |        PyTorch |     TensorFlow |
| ---------------- | --------------: | -------------: | -------------: |
| Test Accuracy    |          92.16% |         88.40% |         91.00% |
| Single Inference |          1.8 ms |         7.3 ms |         5.2 ms |
| Batch Inference  | 0.036 ms/sample | 0.05 ms/sample | 0.04 ms/sample |

*(Results vary by hardware & BLAS choice.)*

---

## 🛠 Configuration & Tuning

* **Tile sizes**: Adjust default tiles in `elas.h` for L1/L2 cache fitting.
* **Threading**: Compile with `-fopenmp` or `/openmp` to enable OpenMP in BLAS routines.
* **Activations**: Swap `fast_sigmoid` in `nn.c` for standard `sigmoidf` if needed.
* **Hyperparameters**: Modify `benchmark.py` (learning rate, batch size, epochs) for your dataset.

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Implement changes & add tests/examples
4. Submit a pull request with a clear description

Please follow the existing C coding style and include Doxygen comments for new functions.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

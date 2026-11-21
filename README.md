# 🧠 tinynn-cpp — a neural network from scratch in C++17

A small, header-only, dependency-free library that implements a **fully-connected neural network with
backpropagation** to understand deep learning from first principles: matrices, activations, loss,
gradients and mini-batch gradient descent.

## Features
- `Matrix` class (matmul, transpose, broadcasting, element-wise ops) — no external libraries
- Dense layers with `Linear`, `Sigmoid`, `Tanh`, `ReLU` activations and Xavier-style initialization
- Mean squared error loss, mini-batch SGD with shuffling
- **Gradient checking** (analytic vs. numerical gradients) in the test-suite
- Examples: XOR classification and `sin(x)` regression
- CMake + CTest, no test framework required

## Build & run
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

./build/xor
./build/regression
```

## Usage
```cpp
#include "tinynn/network.hpp"
using namespace tinynn;

Matrix X(4, 2, {0,0, 0,1, 1,0, 1,1});
Matrix Y(4, 1, {0, 1, 1, 0});

Network net(/*seed=*/42);
net.add(2, 4, Activation::Tanh).add(4, 1, Activation::Sigmoid);
auto history = net.fit(X, Y, /*epochs=*/5000, /*lr=*/0.8);
Matrix prediction = net.predict(X);
```

## How backprop is implemented
For a layer `A = f(XW + b)` the backward pass computes
`dZ = dA ⊙ f'(A)`, `dW = Xᵀ·dZ / n`, `db = Σrows(dZ) / n`, and returns `dX = dZ·Wᵀ`.
`tests/test_tinynn.cpp` verifies these formulas against central finite differences.

## Project structure
```
├── include/tinynn/matrix.hpp    # dense matrix
├── include/tinynn/network.hpp   # layers, activations, training
├── examples/                    # xor.cpp, regression.cpp
└── tests/test_tinynn.cpp        # unit tests + gradient check
```

## Roadmap
- Softmax + cross-entropy loss, momentum/Adam optimizers
- MNIST loader example
- SIMD / OpenMP matmul and a benchmark against Eigen

## Author
Nur Bilge Özcan · [GitHub](https://github.com/BILGEOZCAN) · [LinkedIn](https://linkedin.com/in/nur-bilge-özcan-123975257)

Licensed under the MIT License.

#pragma once
// A tiny fully-connected neural network with backpropagation, written from scratch.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "matrix.hpp"

namespace tinynn {

enum class Activation { Linear, Sigmoid, Tanh, ReLU };

inline double activate(Activation a, double z) {
    switch (a) {
        case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-z));
        case Activation::Tanh: return std::tanh(z);
        case Activation::ReLU: return z > 0.0 ? z : 0.0;
        default: return z;
    }
}

/// Derivative expressed in terms of the activation *output* (cheap, no need to keep z).
inline double derivativeFromOutput(Activation a, double out) {
    switch (a) {
        case Activation::Sigmoid: return out * (1.0 - out);
        case Activation::Tanh: return 1.0 - out * out;
        case Activation::ReLU: return out > 0.0 ? 1.0 : 0.0;
        default: return 1.0;
    }
}

struct Layer {
    Matrix W, b;      // weights (in x out), bias (1 x out)
    Activation act;
    Matrix input, output, dW, db;  // caches from the last forward/backward pass

    Layer(std::size_t in, std::size_t out, Activation a, std::mt19937& rng)
        : W(Matrix::random(in, out, rng, std::sqrt(6.0 / static_cast<double>(in + out)))),
          b(1, out, 0.0), act(a) {}

    Matrix forward(const Matrix& x) {
        input = x;
        output = x.matmul(W).addRowVector(b).apply([this](double z) { return activate(act, z); });
        return output;
    }

    /// dA: gradient of the loss w.r.t. this layer's output. Returns gradient w.r.t. its input.
    Matrix backward(const Matrix& dA) {
        Matrix dZ = dA.hadamard(output.apply([this](double o) { return derivativeFromOutput(act, o); }));
        const double invN = 1.0 / static_cast<double>(dA.rows());
        dW = input.transpose().matmul(dZ) * invN;
        db = dZ.sumRows() * invN;
        return dZ.matmul(W.transpose());
    }
};

class Network {
public:
    explicit Network(unsigned seed = 42) : rng_(seed) {}

    Network& add(std::size_t in, std::size_t out, Activation a) {
        layers_.emplace_back(in, out, a, rng_);
        return *this;
    }

    Matrix forward(const Matrix& x) {
        Matrix a = x;
        for (auto& l : layers_) a = l.forward(a);
        return a;
    }
    Matrix predict(const Matrix& x) { return forward(x); }

    /// Mean over samples of 0.5 * sum_k (prediction - target)^2.
    double loss(const Matrix& x, const Matrix& y) {
        const Matrix diff = forward(x) - y;
        double s = 0.0;
        for (double v : diff.data()) s += 0.5 * v * v;
        return s / static_cast<double>(x.rows());
    }

    /// Forward + backward pass; fills dW/db of every layer.
    void backprop(const Matrix& x, const Matrix& y) {
        Matrix grad = forward(x) - y;  // d(loss)/d(output) for the loss above (per sample)
        for (std::size_t i = layers_.size(); i-- > 0;) grad = layers_[i].backward(grad);
    }

    void step(double lr) {
        for (auto& l : layers_) {
            l.W = l.W - l.dW * lr;
            l.b = l.b - l.db * lr;
        }
    }

    /// Mini-batch gradient descent. Returns the loss after every epoch.
    std::vector<double> fit(const Matrix& x, const Matrix& y, int epochs, double lr, std::size_t batch = 0) {
        const std::size_t n = x.rows();
        if (batch == 0 || batch > n) batch = n;
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<double> history;
        history.reserve(static_cast<std::size_t>(epochs));
        for (int e = 0; e < epochs; ++e) {
            std::shuffle(idx.begin(), idx.end(), rng_);
            for (std::size_t s = 0; s < n; s += batch) {
                const std::vector<std::size_t> part(idx.begin() + static_cast<std::ptrdiff_t>(s),
                                                    idx.begin() + static_cast<std::ptrdiff_t>(std::min(s + batch, n)));
                backprop(x.rowsAt(part), y.rowsAt(part));
                step(lr);
            }
            history.push_back(loss(x, y));
        }
        return history;
    }

    std::vector<Layer>& layers() { return layers_; }

private:
    std::mt19937 rng_;
    std::vector<Layer> layers_;
};

}  // namespace tinynn

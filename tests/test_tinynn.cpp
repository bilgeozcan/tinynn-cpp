// Dependency-free test runner (no external framework needed).
#include <cmath>
#include <iostream>
#include <random>

#include "tinynn/network.hpp"

static int failures = 0;
#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            ++failures;                                                                  \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << "  " #cond << "\n"; \
        }                                                                                \
    } while (0)

using namespace tinynn;

static void testMatrixOps() {
    const Matrix a(2, 2, {1, 2, 3, 4}), b(2, 2, {5, 6, 7, 8});
    const Matrix c = a.matmul(b);
    CHECK(c(0, 0) == 19 && c(0, 1) == 22 && c(1, 0) == 43 && c(1, 1) == 50);
    CHECK(a.transpose()(0, 1) == 3);
    const Matrix s = a.sumRows();
    CHECK(s.rows() == 1 && s(0, 0) == 4 && s(0, 1) == 6);
    const Matrix biased = a.addRowVector(Matrix(1, 2, {10, 20}));
    CHECK(biased(1, 0) == 13 && biased(1, 1) == 24);
    bool threw = false;
    try { Matrix(2, 3).matmul(Matrix(2, 3)); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// Compare analytic gradients with central finite differences.
static void testGradientCheck() {
    std::mt19937 rng(1);
    Network net(3);
    net.add(3, 4, Activation::Tanh).add(4, 3, Activation::Sigmoid).add(3, 2, Activation::Linear);
    const Matrix X = Matrix::random(5, 3, rng, 1.0);
    const Matrix Y = Matrix::random(5, 2, rng, 1.0);

    net.backprop(X, Y);
    std::vector<Matrix> dW, db;
    for (auto& l : net.layers()) { dW.push_back(l.dW); db.push_back(l.db); }

    const double eps = 1e-6;
    double worst = 0.0;
    for (std::size_t li = 0; li < net.layers().size(); ++li) {
        Layer& l = net.layers()[li];
        for (std::size_t i = 0; i < l.W.rows(); ++i)
            for (std::size_t j = 0; j < l.W.cols(); ++j) {
                const double orig = l.W(i, j);
                l.W(i, j) = orig + eps; const double lp = net.loss(X, Y);
                l.W(i, j) = orig - eps; const double lm = net.loss(X, Y);
                l.W(i, j) = orig;
                worst = std::max(worst, std::fabs((lp - lm) / (2 * eps) - dW[li](i, j)));
            }
        for (std::size_t j = 0; j < l.b.cols(); ++j) {
            const double orig = l.b(0, j);
            l.b(0, j) = orig + eps; const double lp = net.loss(X, Y);
            l.b(0, j) = orig - eps; const double lm = net.loss(X, Y);
            l.b(0, j) = orig;
            worst = std::max(worst, std::fabs((lp - lm) / (2 * eps) - db[li](0, j)));
        }
    }
    std::cout << "max gradient error: " << worst << "\n";
    CHECK(worst < 1e-6);
}

static void testLearnsXor() {
    const Matrix X(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    const Matrix Y(4, 1, {0, 1, 1, 0});
    Network net(42);
    net.add(2, 4, Activation::Tanh).add(4, 1, Activation::Sigmoid);
    const auto history = net.fit(X, Y, 5000, 0.8);
    CHECK(history.back() < history.front());
    const Matrix P = net.predict(X);
    for (std::size_t i = 0; i < 4; ++i) CHECK((P(i, 0) > 0.5) == (Y(i, 0) > 0.5));
}

int main() {
    testMatrixOps();
    testGradientCheck();
    testLearnsXor();
    if (failures) { std::cerr << failures << " check(s) failed\n"; return 1; }
    std::cout << "all tests passed\n";
    return 0;
}

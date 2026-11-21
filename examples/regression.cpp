// Fit y = sin(x) on [-3, 3] with a 1-16-1 network (tanh hidden, linear output).
#include <cmath>
#include <iomanip>
#include <iostream>

#include "tinynn/network.hpp"

int main() {
    using namespace tinynn;
    const std::size_t n = 100;
    Matrix X(n, 1), Y(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(n - 1);
        X(i, 0) = x;
        Y(i, 0) = std::sin(x);
    }

    Network net(7);
    net.add(1, 16, Activation::Tanh).add(16, 1, Activation::Linear);
    const auto history = net.fit(X, Y, 3000, 0.05, 20);

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "loss epoch 1:    " << history.front() << "\n";
    std::cout << "loss epoch 3000: " << history.back() << "\n\n";
    const Matrix P = net.predict(X);
    for (std::size_t i = 0; i < n; i += 20)
        std::cout << "x=" << std::setw(8) << X(i, 0) << "  sin(x)=" << std::setw(8) << Y(i, 0)
                  << "  net=" << std::setw(8) << P(i, 0) << "\n";
    return 0;
}

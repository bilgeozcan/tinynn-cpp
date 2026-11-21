// Learn the XOR function with a 2-4-1 network.
#include <iomanip>
#include <iostream>

#include "tinynn/network.hpp"

int main() {
    using namespace tinynn;
    const Matrix X(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    const Matrix Y(4, 1, {0, 1, 1, 0});

    Network net(42);
    net.add(2, 4, Activation::Tanh).add(4, 1, Activation::Sigmoid);

    const auto history = net.fit(X, Y, 5000, 0.8);
    std::cout << "final loss: " << history.back() << "\n\n";

    const Matrix P = net.predict(X);
    std::cout << std::fixed << std::setprecision(3);
    for (std::size_t i = 0; i < X.rows(); ++i)
        std::cout << X(i, 0) << " XOR " << X(i, 1) << " -> " << P(i, 0) << "  (target " << Y(i, 0) << ")\n";
    return 0;
}

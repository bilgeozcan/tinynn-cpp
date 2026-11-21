# GitHub Copilot prompt (paste into Copilot Chat in VS Code, Agent mode)

You are helping me polish an educational open-source project: **tinynn-cpp**, a header-only C++17 neural
network library (Matrix, dense layers, backprop, SGD, gradient-check tests, XOR/regression examples).
I am a computer engineer who works with PyTorch/TensorFlow and wants to show I understand the fundamentals.

Work step by step; after each step build with CMake, run `ctest`, and show a diff before applying changes:
1. Review for bugs, undefined behavior, needless copies and API design issues. Fix them while keeping tests green.
2. Add Softmax activation + categorical cross-entropy loss (with a numerically stable implementation) and
   extend the gradient-check tests to cover it.
3. Add optimizers as a strategy class: SGD with momentum and Adam. Add tests proving they reduce loss on XOR.
4. Add an MNIST example (`examples/mnist.cpp`) that loads IDX files from a path given on the command line
   (do not commit the dataset) and prints test accuracy.
5. Add model save/load (simple binary or text format) with a round-trip test.
6. Add Google Benchmark (via CMake FetchContent, optional target) to benchmark `matmul` and parallelize it
   with OpenMP when available. Report speedups only from real measurements, never invent numbers.
7. Add `.clang-format`, clang-tidy config, GitHub Actions CI (gcc, clang, MSVC; Debug with sanitizers and
   Release), and improve the README with badges and a Mermaid diagram of the forward/backward pass.
Keep the library header-only, C++17, and free of mandatory external dependencies.

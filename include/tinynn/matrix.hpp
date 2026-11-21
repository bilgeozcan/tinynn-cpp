#pragma once
// Minimal dense row-major matrix used by tinynn.
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace tinynn {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}
    Matrix(std::size_t rows, std::size_t cols, std::vector<double> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows * cols) throw std::invalid_argument("Matrix: data size mismatch");
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const std::vector<double>& data() const { return data_; }

    double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    const double& operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    Matrix matmul(const Matrix& o) const {
        if (cols_ != o.rows_) throw std::invalid_argument("matmul: shape mismatch");
        Matrix r(rows_, o.cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k) {
                const double a = (*this)(i, k);
                for (std::size_t j = 0; j < o.cols_; ++j) r(i, j) += a * o(k, j);
            }
        return r;
    }

    Matrix transpose() const {
        Matrix r(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) r(j, i) = (*this)(i, j);
        return r;
    }

    Matrix operator+(const Matrix& o) const { return zip(o, [](double a, double b) { return a + b; }); }
    Matrix operator-(const Matrix& o) const { return zip(o, [](double a, double b) { return a - b; }); }
    Matrix hadamard(const Matrix& o) const { return zip(o, [](double a, double b) { return a * b; }); }
    Matrix operator*(double s) const { return apply([s](double a) { return a * s; }); }

    /// Add a (1 x cols) row vector to every row (bias broadcasting).
    Matrix addRowVector(const Matrix& row) const {
        if (row.rows_ != 1 || row.cols_ != cols_) throw std::invalid_argument("addRowVector: shape mismatch");
        Matrix r = *this;
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) r(i, j) += row(0, j);
        return r;
    }

    /// Sum over rows -> (1 x cols).
    Matrix sumRows() const {
        Matrix r(1, cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) r(0, j) += (*this)(i, j);
        return r;
    }

    template <class F>
    Matrix apply(F f) const {
        Matrix r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = f(data_[i]);
        return r;
    }

    /// Select rows by index (used for mini-batches).
    Matrix rowsAt(const std::vector<std::size_t>& idx) const {
        Matrix r(idx.size(), cols_);
        for (std::size_t i = 0; i < idx.size(); ++i)
            for (std::size_t j = 0; j < cols_; ++j) r(i, j) = (*this)(idx[i], j);
        return r;
    }

    static Matrix random(std::size_t rows, std::size_t cols, std::mt19937& rng, double limit) {
        std::uniform_real_distribution<double> dist(-limit, limit);
        Matrix r(rows, cols);
        for (auto& v : r.data_) v = dist(rng);
        return r;
    }

private:
    template <class F>
    Matrix zip(const Matrix& o, F f) const {
        if (rows_ != o.rows_ || cols_ != o.cols_) throw std::invalid_argument("shape mismatch");
        Matrix r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) r.data_[i] = f(data_[i], o.data_[i]);
        return r;
    }

    std::size_t rows_ = 0, cols_ = 0;
    std::vector<double> data_;
};

}  // namespace tinynn

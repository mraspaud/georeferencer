#include <algorithm>
#include <array>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Locating the covariance maximum between pixels, as in Khlopenkov, Trishchenko and Luo
// (2010), "Achieving subpixel georeferencing accuracy in the Canadian AVHRR processing
// system", IEEE TGRS 48(4), section VI. A parabola through the peak and its neighbours
// along each axis first estimates where the maximum lies. Close to the grid node, a
// quadratic is fitted to the 3 x 3 neighbourhood (their eq. 10). Between two pixels along
// one axis, the box is widened to four samples along that axis and a cubic term added
// (eq. 11); between four pixels, a cubic surface is fitted to a 4 x 4 box (eq. 12). In
// every case the box straddles the expected maximum, the samples nearest it weighted
// four, the sides two and the corners one.

namespace
{
// One axis of the fitting box: where it starts relative to the peak pixel, how many
// samples it holds, where its centre lies, and the weight of each sample.
struct BoxAxis
{
    int first;
    int count;
    double centre;
    std::array<double, 4> weight;
};

BoxAxis box_axis(int side)
{
    if (side > 0)
        return {-1, 4, 0.5, {1.0, 2.0, 2.0, 1.0}};   // maximum beyond the peak pixel
    if (side < 0)
        return {-2, 4, -0.5, {1.0, 2.0, 2.0, 1.0}};  // maximum before it
    return {-1, 3, 0.0, {1.0, 2.0, 1.0, 0.0}};       // maximum close to it
}

// The polynomial terms, cubic ones first, then x^2, y^2, xy, x, y, 1.
int polynomial_terms(double x, double y, int side_x, int side_y, double *terms)
{
    int n = 0;
    if (side_x != 0)
        terms[n++] = x * x * x;
    if (side_y != 0)
        terms[n++] = y * y * y;
    if (side_x != 0 || side_y != 0)
    {
        terms[n++] = x * x * y;
        terms[n++] = x * y * y;
    }
    for (double t : {x * x, y * y, x * y, x, y, 1.0})
        terms[n++] = t;
    return n;
}
} // namespace

std::tuple<float, float> fit_around_the_maximum(const Eigen::MatrixXf &res, int dy, int dx, int side_x, int side_y)
{
    const BoxAxis ax = box_axis(side_x), ay = box_axis(side_y);
    double terms[10];
    const int n_terms = polynomial_terms(0.0, 0.0, side_x, side_y, terms);

    Eigen::MatrixXd A(ax.count * ay.count, n_terms);
    Eigen::VectorXd b(ax.count * ay.count);
    int rows = 0;
    for (int j = 0; j < ay.count; ++j)
    {
        for (int i = 0; i < ax.count; ++i)
        {
            const int xi = dx + ax.first + i, yi = dy + ay.first + j;
            if (xi < 0 || xi >= res.cols() || yi < 0 || yi >= res.rows())
                continue;
            const double w = ax.weight[i] * ay.weight[j];
            polynomial_terms(ax.first + i - ax.centre, ay.first + j - ay.centre, side_x, side_y, terms);
            for (int t = 0; t < n_terms; ++t)
                A(rows, t) = w * terms[t];
            b[rows++] = w * res(yi, xi);
        }
    }
    if (rows < n_terms)
        return {static_cast<float>(dx), static_cast<float>(dy)};

    const Eigen::VectorXd p = A.topRows(rows).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b.head(rows));
    const int q = n_terms - 6; // where the quadratic part begins
    const Eigen::Matrix2d hessian = (Eigen::Matrix2d() << 2 * p[q], p[q + 2], p[q + 2], 2 * p[q + 1]).finished();
    const Eigen::Vector2d gradient(p[q + 3], p[q + 4]);
    if (std::abs(hessian.determinant()) < 1e-6)
        return {static_cast<float>(dx), static_cast<float>(dy)};

    const Eigen::Vector2d offset = -hessian.inverse() * gradient;
    return {static_cast<float>(dx + ax.centre + offset[0]), static_cast<float>(dy + ay.centre + offset[1])};
}

std::tuple<float, float> subpixel_maximum(const Eigen::MatrixXf &res, int dy, int dx)
{
    if (dx < 1 || dx >= res.cols() - 1 || dy < 1 || dy >= res.rows() - 1)
    {
        return {static_cast<float>(dx), static_cast<float>(dy)};
    }

    const float x_vals[3] = {res(dy, dx - 1), res(dy, dx), res(dy, dx + 1)};
    const float a_x = (x_vals[2] + x_vals[0] - 2 * x_vals[1]) / 2.0f;
    const float b_x = (x_vals[2] - x_vals[0]) / 2.0f;
    const float x0 = a_x < -1e-6f ? std::clamp(dx - b_x / (2 * a_x), 0.0f, float(res.cols() - 1)) : dx;

    const float y_vals[3] = {res(dy - 1, dx), res(dy, dx), res(dy + 1, dx)};
    const float a_y = (y_vals[2] + y_vals[0] - 2 * y_vals[1]) / 2.0f;
    const float b_y = (y_vals[2] - y_vals[0]) / 2.0f;
    const float y0 = a_y < -1e-6f ? std::clamp(dy - b_y / (2 * a_y), 0.0f, float(res.rows() - 1)) : dy;

    const int side_x = std::abs(x0 - dx) > 0.25f ? (x0 > dx ? 1 : -1) : 0;
    const int side_y = std::abs(y0 - dy) > 0.25f ? (y0 > dy ? 1 : -1) : 0;
    return fit_around_the_maximum(res, dy, dx, side_x, side_y);
}

Eigen::MatrixXf calculate_covariance_matrix(const Eigen::Ref<const Eigen::MatrixXf> &P,
                                            const Eigen::Ref<const Eigen::MatrixXf> &Q,
                                            int max_displacement)
{
    if (P.rows() != Q.rows() || P.cols() != Q.cols() || P.rows() != P.cols())
    {
        throw std::invalid_argument("P and Q must be square matrices of the same size.");
    }

    int N = P.rows();
    int D = 2 * max_displacement + 1;
    float normalization_factor = 1.0f / (N * N);

    float P_mean = P.sum() * normalization_factor;

    Eigen::MatrixXf covariance_matrix = Eigen::MatrixXf::Zero(D, D);

    for (int dy = -max_displacement; dy <= max_displacement; ++dy)
    {
        for (int dx = -max_displacement; dx <= max_displacement; ++dx)
        {
            int y_start = std::max(0, -dy);
            int y_end = std::min(N, N - dy);
            int x_start = std::max(0, -dx);
            int x_end = std::min(N, N - dx);

            float sum_PQ = 0.0f;
            float sum_Q_shifted = 0.0f;

            for (int y = y_start; y < y_end; ++y)
            {
                for (int x = x_start; x < x_end; ++x)
                {
                    float p_val = P(y, x);
                    float q_val = Q(y + dy, x + dx);
                    sum_PQ += p_val * q_val;
                    sum_Q_shifted += q_val;
                }
            }
            float term1 = sum_PQ * normalization_factor;
            float term2 = P_mean * (sum_Q_shifted * normalization_factor);

            int row = dy + max_displacement;
            int col = dx + max_displacement;
            covariance_matrix(row, col) = term1 - term2;
        }
    }

    float min_cov = covariance_matrix.minCoeff();
    float max_cov = covariance_matrix.maxCoeff();
    if (max_cov > min_cov)
        covariance_matrix = 255 * (covariance_matrix.array() - min_cov) / (max_cov - min_cov);
    else
        covariance_matrix.setZero();

    return covariance_matrix;
}

Eigen::MatrixXf laplacian_operator(const Eigen::Ref<const Eigen::MatrixXf> &image, int s = 1)
{
    int rows = image.rows(), cols = image.cols();

    if (rows <= 2 * s || cols <= 2 * s)
    {
        throw std::invalid_argument("Image dimensions must be larger than 2 * s.");
    }

    Eigen::MatrixXf laplacian = Eigen::MatrixXf::Zero(rows, cols);

#pragma omp parallel for
    for (int y = s; y < rows - s; ++y)
    {
        for (int x = s; x < cols - s; ++x)
        {
            float d2x = (image(y, x - s) - 2 * image(y, x) + image(y, x + s));
            float d2y = (image(y - s, x) - 2 * image(y, x) + image(y + s, x));

            laplacian(y, x) = (d2x + d2y) / (2 * s);
        }
    }

    return laplacian;
}

std::vector<std::tuple<float, float>> calculate_covariance_displacement(const std::vector<std::pair<int, int>> &swath_coords,
                                                                        const Eigen::MatrixXf &image,
                                                                        const Eigen::MatrixXf &reference_image,
                                                                        int N = 48,
                                                                        int max_displacement = 24)
{
    std::vector<std::tuple<float, float>> displacements(swath_coords.size());
    int half_N = N / 2;
    const int s{1};
    Eigen::MatrixXf lap_image = laplacian_operator(image, s);
    Eigen::MatrixXf lap_ref_image = laplacian_operator(reference_image, s);

#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < swath_coords.size(); ++i)
        {
            const auto &coord = swath_coords[i];
            int y = coord.first, x = coord.second;
            displacements[i] = {-100, -100};
            if (y - half_N < s || y + half_N > lap_image.rows() - s ||
                x - half_N < s || x + half_N > lap_image.cols() - s)
            {
                continue;
            }

            Eigen::MatrixXf image_block = lap_image.block(y - half_N, x - half_N, N, N);
            Eigen::MatrixXf ref_image_block = lap_ref_image.block(y - half_N, x - half_N, N, N);

            if (ref_image_block.size() == 0)
            {
                continue;
            }

            Eigen::MatrixXf res = calculate_covariance_matrix(image_block, ref_image_block, max_displacement);

            Eigen::MatrixXf::Index maxRow, maxCol;
            res.maxCoeff(&maxRow, &maxCol);
            float apex_val = res(maxRow, maxCol);
            if (apex_val < 254.5f)
            {
                continue;
            }
            int dy = maxRow, dx = maxCol;
            float mean_cov = res.mean();
            const int R = 2;
            std::vector<float> circ;
            circ.reserve(100);
            for (int t = 0; t < 100; ++t)
            {
                double theta = 2 * M_PI * t / 100.0;
                int y = static_cast<int>(std::round(maxRow + R * std::sin(theta)));
                int x = static_cast<int>(std::round(maxCol + R * std::cos(theta)));
                if (y >= 0 && y < res.rows() && x >= 0 && x < res.cols())
                    circ.push_back(res(y, x));
            }
            float CL = circ.empty() ? 0.f : *std::max_element(circ.begin(), circ.end());

            float score = mean_cov + CL;
            if (score < 237)
            {
                auto [x0, y0] = subpixel_maximum(res, dy, dx);
                x0 -= max_displacement;
                y0 -= max_displacement;
                if (std::abs(y0) < max_displacement && std::abs(x0) < max_displacement)
                    displacements[i] = {y0, x0};
            }
        }
    }
    return displacements;
}

PYBIND11_MODULE(displacement_calc, m)
{
    m.doc() = "Optimized image processing functions using Eigen";
    m.def("calculate_covariance_matrix", &calculate_covariance_matrix, "Calculate covariance matrix");
    m.def("laplacian_operator", &laplacian_operator, "Apply Laplacian operator");
    m.def("calculate_covariance_displacement", &calculate_covariance_displacement, "Calculate covariance displacement");
    m.def("subpixel_maximum", &subpixel_maximum, "Locate the covariance maximum between pixels, as (x, y)");
}

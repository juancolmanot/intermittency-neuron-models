// Rulkov 1D map: numerical characteristic relation ⟨l⟩(ε)
// Type-I intermittency via tangent bifurcation
// Usage: rulkov_scaling <c_lam> [n_eps] [n_steps]
//   c_lam: laminar window half-width (e.g. 0.1 or 0.01)
//   n_eps: number of ε points (default 30)
//   n_steps: iterations per ε (default 10000000)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

static const double ALPHA = 4.8;
static const double GAMMA_C = -2.93141;
static const double FIXED_POINT = -1.76493;

inline double rulkov_map(double x, double gamma) {
    return ALPHA / (1.0 + x * x) + gamma;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <c_lam> [n_eps] [n_steps]\n", argv[0]);
        return 1;
    }

    double c_lam = std::atof(argv[1]);
    int n_eps = (argc > 2) ? std::atoi(argv[2]) : 30;
    long n_steps = (argc > 3) ? std::atol(argv[3]) : 10000000L;
    long transient = 100000;

    double exp_min = -11.0, exp_max = -5.0;

    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    // Output file
    std::ostringstream fname;
    fname << "../datafiles/rulkov_scaling_c=" << std::fixed << std::setprecision(2) << c_lam << ".csv";
    std::ofstream out(fname.str());
    out << "# Rulkov 1D scaling: alpha=" << ALPHA << " gamma_c=" << GAMMA_C
        << " fp=" << FIXED_POINT << " c=" << c_lam << "\n";
    out << "# epsilon,gamma,mean_l,max_l,N_episodes,a2,a1,a0\n";

    fprintf(stdout, "%3s  %12s  %12s  %10s  %10s  %10s  %8s\n",
            "i", "epsilon", "gamma", "mean_l", "max_l", "N_eps", "time_s");
    fprintf(stdout, "%s\n", std::string(76, '=').c_str());
    fflush(stdout);

    for (int k = 0; k < n_eps; k++) {
        double exponent = exp_min + (exp_max - exp_min) * k / (n_eps - 1);
        double eps = std::exp(exponent);
        double gamma = GAMMA_C + eps;

        auto t0 = std::chrono::high_resolution_clock::now();

        double x = dist(rng);

        // Transient
        for (long n = 0; n < transient; n++) {
            x = rulkov_map(x, gamma);
        }

        // Collect laminar episodes + local map data
        std::vector<double> laminar_lengths;
        laminar_lengths.reserve(100000);

        // Local map fit data
        double fit_width = 0.15;
        std::vector<double> x_fit, x1_fit;
        x_fit.reserve(5000);
        x1_fit.reserve(5000);

        bool in_laminar = false;
        long start_lam = 0;
        double xp = x;

        for (long n = 0; n < n_steps; n++) {
            double x1 = rulkov_map(xp, gamma);

            double dx = xp - FIXED_POINT;
            double dx1 = x1 - FIXED_POINT;

            if (!in_laminar) {
                if (std::fabs(dx) > c_lam && std::fabs(dx1) <= c_lam) {
                    in_laminar = true;
                    start_lam = n + 1;
                }
            } else {
                if (std::fabs(dx1) > c_lam) {
                    in_laminar = false;
                    long length = n + 1 - start_lam;
                    if (length > 0)
                        laminar_lengths.push_back(static_cast<double>(length));
                }
            }

            // Collect local map data near fixed point
            if (std::fabs(dx) <= fit_width && std::fabs(dx1) <= fit_width && x_fit.size() < 5000) {
                x_fit.push_back(dx);
                x1_fit.push_back(dx1);
            }

            xp = x1;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Statistics
        double mean_l = 0, max_l = 0;
        size_t ne = laminar_lengths.size();
        if (ne > 0) {
            mean_l = std::accumulate(laminar_lengths.begin(), laminar_lengths.end(), 0.0) / ne;
            max_l = *std::max_element(laminar_lengths.begin(), laminar_lengths.end());
        }

        // Quadratic fit: x1 = a2*x^2 + a1*x + a0 (least squares)
        double a2 = 0, a1 = 0, a0 = 0;
        if (x_fit.size() >= 10) {
            size_t N = x_fit.size();
            double sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
            double sy = 0, sxy = 0, sx2y = 0;
            for (size_t i = 0; i < N; i++) {
                double xi = x_fit[i], yi = x1_fit[i];
                double xi2 = xi * xi;
                sx += xi; sx2 += xi2; sx3 += xi2 * xi; sx4 += xi2 * xi2;
                sy += yi; sxy += xi * yi; sx2y += xi2 * yi;
            }
            // Solve 3x3 normal equations
            double n_d = static_cast<double>(N);
            // [sx4 sx3 sx2] [a2]   [sx2y]
            // [sx3 sx2 sx ] [a1] = [sxy ]
            // [sx2 sx  n  ] [a0]   [sy  ]
            double M[3][4] = {
                {sx4, sx3, sx2, sx2y},
                {sx3, sx2, sx,  sxy},
                {sx2, sx,  n_d, sy}
            };
            // Gaussian elimination
            for (int col = 0; col < 3; col++) {
                int piv = col;
                for (int row = col + 1; row < 3; row++)
                    if (std::fabs(M[row][col]) > std::fabs(M[piv][col])) piv = row;
                std::swap(M[col], M[piv]);
                for (int row = col + 1; row < 3; row++) {
                    double f = M[row][col] / M[col][col];
                    for (int j = col; j < 4; j++) M[row][j] -= f * M[col][j];
                }
            }
            a0 = M[2][3] / M[2][2];
            a1 = (M[1][3] - M[1][2] * a0) / M[1][1];
            a2 = (M[0][3] - M[0][2] * a0 - M[0][1] * a1) / M[0][0];
        }

        fprintf(stdout, "%3d  %12.6e  %12.8f  %10.2f  %10.0f  %10zu  %7.1f\n",
                k + 1, eps, gamma, mean_l, max_l, ne, elapsed);
        fflush(stdout);

        out << std::scientific << std::setprecision(8)
            << eps << "," << std::fixed << std::setprecision(8) << gamma << ","
            << std::fixed << std::setprecision(4) << mean_l << "," << max_l << ","
            << ne << ","
            << std::scientific << std::setprecision(8) << a2 << "," << a1 << "," << a0 << "\n";
    }

    out.close();
    fprintf(stdout, "\nSaved: %s\n", fname.str().c_str());
    return 0;
}

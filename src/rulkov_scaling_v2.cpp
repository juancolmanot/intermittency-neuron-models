// Rulkov 1D map: numerical characteristic relation ⟨l⟩(ε)
// Type-I intermittency via tangent bifurcation
// Usage: rulkov_scaling_v2 <c_lam> [n_eps] [n_ics] [target_episodes]
//
// Epsilon range: log10(eps) in [-10, -5]
// gamma_c and fp computed from exact tangency conditions.

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
static const double GAMMA_C = -2.931400258306671;
static const double FIXED_POINT = -1.765458470442114;

inline double rulkov_map(double x, double gamma) {
    return ALPHA / (1.0 + x * x) + gamma;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <c_lam> [n_eps] [n_ics] [target_episodes]\n", argv[0]);
        return 1;
    }

    double c_lam = std::atof(argv[1]);
    int n_eps = (argc > 2) ? std::atoi(argv[2]) : 30;
    int n_ics = (argc > 3) ? std::atoi(argv[3]) : 20;
    int target_episodes = (argc > 4) ? std::atoi(argv[4]) : 500;

    double log10_min = -10.0, log10_max = -5.0;
    long transient = 500000;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    std::ostringstream fname;
    fname << "../datafiles/rulkov_scaling_v2_c=" << std::fixed << std::setprecision(2) << c_lam << ".csv";
    std::ofstream out(fname.str());
    out << "# Rulkov 1D scaling v2: alpha=" << ALPHA
        << " gamma_c=" << std::setprecision(15) << GAMMA_C
        << " fp=" << FIXED_POINT << " c=" << std::setprecision(2) << c_lam
        << " n_ics=" << n_ics << " target_eps=" << target_episodes << "\n";
    out << "# log10_eps,epsilon,gamma,mean_l,max_l,std_l,N_episodes,a2,a1,a0\n";

    fprintf(stdout, "%3s  %8s  %12s  %16s  %10s  %10s  %10s  %10s  %8s\n",
            "i", "log10eps", "epsilon", "gamma", "mean_l", "max_l", "std_l", "N_eps", "time_s");
    fprintf(stdout, "%s\n", std::string(110, '=').c_str());
    fflush(stdout);

    for (int k = 0; k < n_eps; k++) {
        double log10_eps = log10_min + (log10_max - log10_min) * k / (n_eps - 1);
        double eps = std::pow(10.0, log10_eps);
        double gamma = GAMMA_C + eps;

        // Adaptive: <l> ~ pi/sqrt(a*eps) for type-I with a ~ 0.6
        double est_mean_l = M_PI / std::sqrt(0.6 * eps);
        long n_steps_per_ic = std::max(2000000L,
            static_cast<long>(static_cast<double>(target_episodes) * est_mean_l * 3.0));
        if (n_steps_per_ic > 2000000000L)
            n_steps_per_ic = 2000000000L;

        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<double> all_laminar_lengths;
        all_laminar_lengths.reserve(static_cast<size_t>(target_episodes) * static_cast<size_t>(n_ics));

        std::vector<double> x_fit, x1_fit;
        x_fit.reserve(50000);
        x1_fit.reserve(50000);

        for (int ic = 0; ic < n_ics; ic++) {
            double x = dist(rng);

            for (long n = 0; n < transient; n++) {
                x = rulkov_map(x, gamma);
            }

            bool in_laminar = false;
            long start_lam = 0;
            double xp = x;
            double fit_width = 0.15;

            for (long n = 0; n < n_steps_per_ic; n++) {
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
                            all_laminar_lengths.push_back(static_cast<double>(length));
                    }
                }

                if (ic == 0 && std::fabs(dx) <= fit_width && std::fabs(dx1) <= fit_width
                    && x_fit.size() < 50000) {
                    x_fit.push_back(dx);
                    x1_fit.push_back(dx1);
                }

                xp = x1;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        double mean_l = 0, max_l = 0, std_l = 0;
        size_t ne = all_laminar_lengths.size();
        if (ne > 0) {
            mean_l = std::accumulate(all_laminar_lengths.begin(), all_laminar_lengths.end(), 0.0)
                     / static_cast<double>(ne);
            max_l = *std::max_element(all_laminar_lengths.begin(), all_laminar_lengths.end());
            double var = 0;
            for (auto ll : all_laminar_lengths) {
                double diff = ll - mean_l;
                var += diff * diff;
            }
            std_l = std::sqrt(var / static_cast<double>(ne));
        }

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
            double n_d = static_cast<double>(N);
            double M[3][4] = {
                {sx4, sx3, sx2, sx2y},
                {sx3, sx2, sx,  sxy},
                {sx2, sx,  n_d, sy}
            };
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

        fprintf(stdout, "%3d  %8.2f  %12.6e  %16.12f  %10.2f  %10.0f  %10.2f  %10zu  %7.1f\n",
                k + 1, log10_eps, eps, gamma, mean_l, max_l, std_l, ne, elapsed);
        fflush(stdout);

        out << std::fixed << std::setprecision(4) << log10_eps << ","
            << std::scientific << std::setprecision(10)
            << eps << "," << std::fixed << std::setprecision(12) << gamma << ","
            << std::fixed << std::setprecision(4) << mean_l << "," << max_l << ","
            << std_l << "," << ne << ","
            << std::scientific << std::setprecision(10) << a2 << "," << a1 << "," << a0 << "\n";
        out.flush();
    }

    out.close();
    fprintf(stdout, "\nSaved: %s\n", fname.str().c_str());
    return 0;
}

// Collect reinjection data for Rulkov 1D map at given epsilon and c
// Usage: rulkov_reinjection <c_lam> <epsilon> [n_steps]
// Outputs: reinjection points (shifted by fixed point) to stdout

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

static const double ALPHA = 4.8;
static const double GAMMA_C = -2.93141;
static const double FIXED_POINT = -1.76493;

inline double rulkov_map(double x, double gamma) {
    return ALPHA / (1.0 + x * x) + gamma;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <c_lam> <epsilon> [n_steps]\n", argv[0]);
        return 1;
    }

    double c_lam = std::atof(argv[1]);
    double eps = std::atof(argv[2]);
    long n_steps = (argc > 3) ? std::atol(argv[3]) : 50000000L;
    long transient = 200000;
    double gamma = GAMMA_C + eps;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    double x = dist(rng);

    for (long n = 0; n < transient; n++)
        x = rulkov_map(x, gamma);

    bool in_laminar = false;
    long count = 0;

    printf("# Reinjection data: c=%.6f eps=%.8e gamma=%.10f\n", c_lam, eps, gamma);
    printf("# x_reinjected (shifted by fp)\n");

    double xp = x;
    for (long n = 0; n < n_steps; n++) {
        double x1 = rulkov_map(xp, gamma);
        double dx = xp - FIXED_POINT;
        double dx1 = x1 - FIXED_POINT;

        if (!in_laminar) {
            if (std::fabs(dx) > c_lam && std::fabs(dx1) <= c_lam) {
                in_laminar = true;
                printf("%.12e\n", dx1);
                count++;
            }
        } else {
            if (std::fabs(dx1) > c_lam) {
                in_laminar = false;
            }
        }
        xp = x1;
    }

    fprintf(stderr, "Collected %ld reinjection points\n", count);
    return 0;
}

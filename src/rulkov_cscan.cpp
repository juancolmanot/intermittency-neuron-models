// Reinjection collector for the Rulkov fast map, parameterised in the laminar
// half-width c. Same protocol as supplementary/src/rulkov_reinjection.cpp but with
// gamma_c and x* to machine precision and a reinjection-count target.
// Usage: rulkov_cscan <c> <eps> <target_reinj> <max_steps> > out.dat
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

static const double ALPHA   = 4.8;
static const double GAMMA_C = -2.931400258306671;
static const double FP      = -1.765458470442114;

inline double F(double x, double g) { return ALPHA / (1.0 + x * x) + g; }

int main(int argc, char* argv[]) {
    if (argc < 5) { fprintf(stderr, "usage: %s c eps target max_steps\n", argv[0]); return 1; }
    const double c      = atof(argv[1]);
    const double eps    = atof(argv[2]);
    const long   target = atol(argv[3]);
    const long   maxst  = atol(argv[4]);
    const double g      = GAMMA_C + eps;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    double x = dist(rng);
    for (long n = 0; n < 200000; n++) x = F(x, g);

    printf("# c=%.6f eps=%.8e gamma=%.15f fp=%.15f\n", c, eps, g, FP);
    printf("# x_reinjected (shifted by fp)\n");
    bool in_lam = std::fabs(x - FP) <= c;
    long count = 0, n = 0;
    for (; n < maxst && count < target; n++) {
        x = F(x, g);
        const double dx = x - FP;
        const bool now_in = std::fabs(dx) <= c;
        if (now_in && !in_lam) { printf("%.15e\n", dx); count++; }
        in_lam = now_in;
    }
    fprintf(stderr, "c=%g collected %ld reinjections in %ld steps\n", c, count, n);
    return 0;
}

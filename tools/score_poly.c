/* score-poly: Murphy-E scorer for SNFS/GNFS polynomial candidates.
 *
 * Reads a GGNFS-format poly (n:, cK:, Y0:, Y1: lines; a full .job file
 * works — everything else is ignored) and prints the msieve polyselect
 * scores: skew (E-optimal, found by golden section over log-skew),
 * Murphy E, size, alpha and real roots.  This is the same analyze_poly
 * path msieve uses to rank its own polys — and the same computation the
 * phi.kyleaskine.com scorer runs client-side as WASM — so its E values
 * are directly comparable with factordb's reported `e` and with the site
 * (whose ranking normalizes E * 1.6^(6 - degree)).
 *
 * usage: score-poly <poly-or-job-file> [fixed-skew]
 *
 * CPU only; links libmsieve.a for the scoring routines but never
 * touches a GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gmp.h>
#include <ms_common.h>
#include "gnfs/poly/poly.h"

#define MAX_LINE 4096

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s <polyfile> [fixed-skew]\n", argv[0]);
		return 1;
	}
	FILE *fp = fopen(argv[1], "r");
	if (!fp) { perror(argv[1]); return 1; }

	char buf[MAX_LINE];
	mpz_t y0, y1, coeffs[MAX_POLY_DEGREE + 1];
	mpz_inits(y0, y1, NULL);
	for (int i = 0; i <= MAX_POLY_DEGREE; i++)
		mpz_init(coeffs[i]);
	int degree = -1;

	while (fgets(buf, MAX_LINE, fp)) {
		char *p = buf;
		while (*p == ' ' || *p == '\t') p++;
		if (*p == '#' || *p == '\n' || *p == '\0')
			continue;
		char tok[MAX_LINE];
		if (sscanf(p, "Y0: %s", tok) == 1) { mpz_set_str(y0, tok, 10); continue; }
		if (sscanf(p, "Y1: %s", tok) == 1) { mpz_set_str(y1, tok, 10); continue; }
		unsigned k;
		if (sscanf(p, "c%u: %s", &k, tok) == 2) {
			if ((int)k > MAX_POLY_DEGREE) continue;
			mpz_set_str(coeffs[k], tok, 10);
			if ((int)k > degree) degree = k;
			continue;
		}
	}
	fclose(fp);
	if (degree < 1) {
		fprintf(stderr, "no coefficients found\n");
		return 1;
	}

	poly_config_t config;
	poly_select_t s;
	poly_config_init(&config);
	memset(&s, 0, sizeof(s));
	mpz_poly_init(&s.rpoly);
	mpz_poly_init(&s.apoly);

	s.rpoly.degree = 1;
	mpz_set(s.rpoly.coeff[0], y0);
	mpz_set(s.rpoly.coeff[1], y1);
	s.apoly.degree = degree;
	for (int i = 0; i <= degree; i++)
		mpz_set(s.apoly.coeff[i], coeffs[i]);

	double fixed_skew = (argc > 2) ? atof(argv[2]) : 0.0;

	if (fixed_skew > 0.0) {
		s.skewness = fixed_skew;
	}
	else {
		/* golden-section maximization of Murphy E over log(skew) */
		double lo = log(0.01), hi = log(10000.0);
		const double gr = 0.6180339887498949;
		double c = hi - gr * (hi - lo);
		double d = lo + gr * (hi - lo);
		double fc, fd;
		for (int iter = 0; iter < 60; iter++) {
			s.skewness = exp(c);
			analyze_poly(&config, &s);
			fc = s.combined_score;
			s.skewness = exp(d);
			analyze_poly(&config, &s);
			fd = s.combined_score;
			if (fc < fd) { lo = c; c = d; d = lo + gr * (hi - lo); }
			else { hi = d; d = c; c = hi - gr * (hi - lo); }
			if (hi - lo < 1e-10) break;
		}
		s.skewness = exp((lo + hi) / 2);
	}
	analyze_poly(&config, &s);

	printf("skew %.6f, murphyE %.9e, size %.6e, alpha %.4f, "
	       "rroots %u\n",
	       s.skewness, s.combined_score, s.size_score,
	       s.root_score, s.num_real_roots);
	poly_config_free(&config);
	mpz_poly_free(&s.rpoly);
	mpz_poly_free(&s.apoly);
	mpz_clears(y0, y1, NULL);
	return 0;
}

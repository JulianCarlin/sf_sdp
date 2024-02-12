functions {
  real powlaw_lpdf(vector y, int sk, real c, real b) {
    real c1 = c + 1;
    return sk * (log(c1) - c1 * log(b)) + c * sum(log(b - y));
  }
}
data {
  int<lower=0> N;       // total number of samples
  int<lower=0> K;       // number of groups
  vector[N] y;          // all samples
  int s[K];             // list of group sizes
  vector[K] lower_bounds; // lower bounds on b for each group
}
parameters {
  real<lower=0> b_mu;            // hyperprior mean of b
  real<lower=0> b_sig;  // hyperior sd of b
  real<lower=1> c_mu;            // hyperior mean of c
  real<lower=0> c_sig;  // hyperprior sd of c
  /* vector<lower=0>[K] b_est;      // each b for the groups */
  vector<lower=lower_bounds>[K] b_est;      // each b for the groups
  vector<lower=1>[K] c_est;      // each c for the groups
}
transformed parameters {
  /* vector[K] bk_shifted = b_est + lower_bounds; */
}
model {
  // hyperpriors
  b_mu ~ normal(2, 1);
  b_sig ~ cauchy(0, 1);
  c_mu ~ normal(2, 1);
  c_sig ~ cauchy(0, 1);
  int pos = 1;          // initial index of 1 for slicing samples into groups
  for (k in 1:K) {
    // slice total samples into group yk
    vector[s[k]] yk = segment(y, pos, s[k]);
    pos = pos + s[k];        // move index up for next slice
    // priors for each group parameters
    /* bk_shifted[k] ~ normal(b_mu, b_sig); */
    b_est[k] ~ normal(b_mu, b_sig);
    c_est[k] ~ normal(c_mu, c_sig);
    // likelihood
    /* target += powlaw_lpdf(yk | s[k], c_est[k], bk_shifted[k]); */
    target += powlaw_lpdf(yk | s[k], c_est[k], b_est[k]);
  }
}

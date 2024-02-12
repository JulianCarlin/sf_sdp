functions {
  real pdelt_lpdf(vector delt, int ndelt, real length, real lam, real delta, real xcrs) {
    real index = lam * length * xcrs + delta;
    return ndelt * (log(index + 1) - log(length * xcrs)) + index * sum(log1m(delt / (length * xcrs)));
  }
}
data {
  int<lower=0> N;       // total number of samples
  int<lower=0> K;       // number of groups
  vector[N] delts;          // all samples
  int ndelts[K];             // list of group sizes
  vector[K] lengths;      // length associated with each group
  vector[K] lower_bounds; // lower bounds on b for each group
}
parameters {
  real<lower=0> lam_mu;             // hyperprior mean of lam0
  real<lower=0> lam_sig;            // hyperior sd of lam0
  real<lower=1> delta_mu;            // hyperior mean of delta
  real<lower=0> delta_sig;           // hyperprior sd of delta
  real<lower=0> xcrs_mu;         // hyperprior mean of xcrs
  real<lower=0> xcrs_sig;        // hyperior sd of xcrs
  vector<lower=0>[K] lam_est;       // each lam0 for the groups
  vector<lower=1>[K] delta_est;      // each delta for the groups
  vector<lower=lower_bounds>[K] xcrs_est;   // each xcrs for the groups
  /* vector[K] lam_est_raw;       // each lam0 for the groups */
  /* vector[K] delta_est_raw;      // each delta for the groups */
  /* vector[K] xcrs_est_raw;   // each xcrs for the groups */
}
transformed parameters {
  /* vector<lower=0>[K] lam_est;       // each lam0 for the groups */
  /* vector<lower=1>[K] delta_est;      // each delta for the groups */
  /* vector<lower=lower_bounds>[K] xcrs_est;   // each xcrs for the groups */
  /* lam_est = lam_mu + lam_sig * lam_est_raw; */
  /* delta_est = delta_mu + delta_sig * delta_est_raw; */
  /* xcrs_est = xcrs_mu + xcrs_sig * xcrs_est_raw; */
}
model {
  // hyperpriors
  lam_mu ~ normal(2, 1);
  /* lam_sig ~ cauchy(0, 1); */
  lam_sig ~ exponential(1);
  delta_mu ~ normal(2, 1);
  /* delta_sig ~ cauchy(0, 1); */
  delta_sig ~ exponential(1);
  xcrs_mu ~ normal(2, 1);
  /* xcrs_sig ~ cauchy(0, 1); */
  xcrs_sig ~ exponential(1);
  int pos = 1;          // initial index of 1 for slicing samples into groups
  for (k in 1:K) {
    // slice total samples into group yk
    vector[ndelts[k]] delt = segment(delts, pos, ndelts[k]);
    pos = pos + ndelts[k];        // move index up for next slice
    // priors for each group parameters
    xcrs_est[k] ~ normal(xcrs_mu, xcrs_sig);
    lam_est[k] ~ normal(lam_mu, lam_sig);
    delta_est[k] ~ normal(delta_mu, delta_sig);
    /* xcrs_est_raw[k] ~ std_normal(); */
    /* lam_est_raw[k] ~ std_normal(); */
    /* delta_est_raw[k] ~ std_normal(); */
    // likelihood
    target += pdelt_lpdf(delt | ndelts[k], lengths[k], lam_est[k], delta_est[k], xcrs_est[k]);
  }
}

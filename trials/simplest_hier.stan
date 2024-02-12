functions {
  real my_lpdf(vector delt, int nk, real tauk, real betak) {
    return nk * (log(betak + 1) - log(tauk)) + betak * sum(log1m(delt / tauk));
  }
}
data {
  int<lower=0> NT;            // total number of delt samples
  int<lower=0> K;             // number of groups
  vector[NT] delt;            // all samples of delt
  int st[K];                  // list of group sizes for delts
  vector[K] lower_bounds_tau; // lower bounds on tau for each group
}
parameters {
  real<lower=0> tau_mu;           // hyperprior mean of tau
  real<lower=0> tau_sig;          // hyperior sd of tau
  real<lower=0> beta_mu;           // hyperior mean of beta
  real<lower=0> beta_sig;          // hyperprior sd of beta
  vector<lower=0>[K] beta_est;     // each beta for the groups
  vector<lower=lower_bounds_tau>[K] tau_est;      // each tau for the groups
}
model {
  // hyperpriors
  tau_mu ~ normal(3, 1);
  tau_sig ~ exponential(1);
  beta_mu ~ normal(3, 1);
  beta_sig ~ exponential(1);
  int post = 1;          // initial index of 1 for slicing samples into groups
  for (k in 1:K) {
    // slice total samples into group yk
    vector[st[k]] deltk = segment(delt, post, st[k]);
    post = post + st[k];        // move index up for next slice
    // priors for each group parameters
    tau_est[k] ~ normal(tau_mu, tau_sig);
    beta_est[k] ~ normal(beta_mu, beta_sig);
    // likelihood
    target += my_lpdf(deltk | st[k], tau_est[k], beta_est[k]);
  }
}

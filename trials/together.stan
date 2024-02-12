functions {
  real comb2_lpdf(vector delt, vector dels, int nk, real tauk, real xik, real betak) {
    int nkm1 = nk - 1;
    return(nk + nkm1) * log(betak + 1) - nk * log(xik) - nkm1 * log(tauk) +
    betak * (sum(log1m(delt / tauk)) + sum(log1m(dels / xik)));
  }
}
data {
  int<lower=0> NT;            // total number of delt samples
  int<lower=0> NS;            // total number of dels samples
  int<lower=0> K;             // number of groups
  vector[NT] delt;            // all samples of delt
  vector[NS] dels;            // all samples of dels
  int st[K];                  // list of group sizes for delts
  int ss[K];                  // list of group sizes for delss
  vector[K] lower_bounds_tau; // lower bounds on tau for each group
  vector[K] lower_bounds_xi;  // lower bounds on tau for each group
}
parameters {
  real<lower=0> tau_mu;           // hyperprior mean of tau
  real<lower=0> tau_sig;          // hyperior sd of tau
  real<lower=0> xi_mu;            // hyperior mean of xi
  real<lower=0> xi_sig;           // hyperprior sd of xi
  real<lower=0> beta_mu;           // hyperior mean of lambda_0
  real<lower=0> beta_sig;          // hyperprior sd of lambda_0
  vector<lower=0>[K] beta_est;     // each lambda_0 for the groups
  vector<lower=lower_bounds_tau>[K] tau_est;      // each tau for the groups
  vector<lower=lower_bounds_xi>[K] xi_est;      // each xi for the groups
}
model {
  // hyperpriors
  tau_mu ~ normal(3, 1);
  tau_sig ~ exponential(1);
  xi_mu ~ normal(2, 1);
  xi_sig ~ exponential(1);
  beta_mu ~ normal(3, 1);
  beta_sig ~ exponential(1);
  int post = 1;          // initial index of 1 for slicing samples into groups
  int poss = 1;          // initial index of 1 for slicing samples into groups
  for (k in 1:K) {
    // slice total samples into group yk
    vector[st[k]] deltk = segment(delt, post, st[k]);
    vector[ss[k]] delsk = segment(dels, poss, ss[k]);
    post = post + st[k];        // move index up for next slice
    poss = poss + ss[k];        // move index up for next slice
    // priors for each group parameters
    tau_est[k] ~ normal(tau_mu, tau_sig);
    xi_est[k] ~ normal(xi_mu, xi_sig);
    beta_est[k] ~ normal(beta_mu, beta_sig);
    // likelihood
    target += comb2_lpdf(deltk | delsk, ss[k], tau_est[k], xi_est[k], beta_est[k]);
  }
}

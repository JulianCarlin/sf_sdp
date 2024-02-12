functions {
  real pdelt_lpdf(vector delt, int N, real lam, real bet) {
    real bl = bet * lam;
    vector[N] delt_on_beta = delt / bet;
    vector[N] log1m_delt_on_beta = log1m(delt_on_beta);
    return N * log(4.0 / (9.0 * bet)) + (1. + bl) * sum(log1m_delt_on_beta) - 2. * sum(log(2.0 / 3.0 - log1m_delt_on_beta)) + sum(log((1.5 * (2. + bl) * log1m_delt_on_beta - bl - 5.) ./ (1.5 * log1m_delt_on_beta - 1.)));
  }
}
data {
  int<lower=0> N;   
  vector[N] delt;    
  real lower_bound;    
}
parameters {
  real<lower=0> lam;       
  real<lower=lower_bound> bet;   
}
model {
  // priors
  bet ~ normal(2, 2);
  lam ~ normal(2, 2);
  // likelihood
  target += pdelt_lpdf(delt | N, lam, bet);
}

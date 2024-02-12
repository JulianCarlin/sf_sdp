functions {
  real pdelt_lpdf(vector delt, int ndelt, real lam, real delta, real bet) {
    real index = lam * bet + delta;
    return ndelt * (log(index + 1) - log(bet)) + index * sum(log1m(delt / bet));
  }
}
data {
  int<lower=0> N;   
  vector[N] delt;    
  real lower_bound;    
}
parameters {
  real<lower=0> lam;       
  real<lower=0> delta_raw;      
  real<lower=lower_bound> bet;   
}
transformed parameters {
  real<lower=1> delta = delta_raw + 1;
}
model {
  // priors
  bet ~ normal(2, 2);
  lam ~ normal(2, 2);
  delta_raw ~ gamma(2, 3/2);
  // likelihood
  target += pdelt_lpdf(delt | N, lam, delta, bet);
}

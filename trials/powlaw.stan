functions {
  real powlaw_lpdf(vector y, real c, real b) {
    real c1 = c + 1;
    return size(y) * (log(c1) - c1 * log(b)) + c * sum(log(b - y));
  }
}
data {
  int<lower=0> N;       // number of samples
  vector[N] y;            // samples
}
parameters {
  real b;               // power law offset
  real<lower=1> c;      // power law index
}
model {
  b ~ normal(2, 0.5);
  c ~ normal(1.5, 0.5);
  target += powlaw_lpdf(y | c, b);
}

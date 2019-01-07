//Original authors:
//Aki Vehtari, Tuomas Sivula

//Licence: CC-BY

// Comparison of k groups with common variance
data {
  int<lower=0> N; // number of data points
  vector[N] x; // year
  vector[N] y; // temperature
  real xpred; // predicted year
  real xpred2; // predicted year
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu;
  mu = alpha + beta*x;
}
model {
  y ~ normal(mu, sigma);
}
generated quantities {
    real ypred;
    real ypred2;
    vector[N] log_lik;
    ypred = normal_rng(alpha + beta*xpred, sigma);
    ypred2 = normal_rng(alpha + beta*xpred2, sigma);
    for (i in 1:N)
        log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
}
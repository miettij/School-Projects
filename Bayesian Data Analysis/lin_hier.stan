// Gaussian linear model with standardized data
data {
  int<lower=0> N; // number of data points #38
  vector[N] x; //years
  vector[N] y_HKI; // #38
  vector[N] y_JYV; // #38
  vector[N] y_HAI; // #38
  vector[N] y_OUL; // #38
  int<lower=0> M;
  vector[M] y_ALL;//
  
  real xpred; // input location for prediction
}
transformed data {
  vector[N] x_std;
  vector[N] y_HKIstd;
  vector[N] y_JYVstd;
  vector[N] y_HAIstd;
  vector[N] y_OULstd;
  real xpred_std;
  x_std = (x - mean(x)) / sd(x);
  y_HKIstd = (y_HKI - mean(y_HKI)) / sd(y_HKI);
  y_JYVstd = (y_JYV - mean(y_JYV)) / sd(y_JYV);
  y_HAIstd = (y_HAI - mean(y_HAI)) / sd(y_HAI);
  y_OULstd = (y_OUL - mean(y_OUL)) / sd(y_OUL);
  xpred_std = (xpred - mean(x)) / sd(x);
}
parameters {
  real alpha_HKI;
  real alpha_JYV;
  real alpha_HAI;
  real alpha_OUL;
  real beta_HKI;
  real beta_JYV;
  real beta_HAI;
  real beta_OUL;
  real<lower=0> sigma_std;
}
transformed parameters {
  vector[N] mu_HKIstd;
  vector[N] mu_JYVstd;
  vector[N] mu_HAIstd;
  vector[N] mu_OULstd;
  mu_HKIstd = alpha_HKI + beta_HKI*x_std;
  mu_JYVstd = alpha_JYV + beta_JYV*x_std;
  mu_HAIstd = alpha_HAI + beta_HAI*x_std;
  mu_OULstd = alpha_OUL + beta_OUL*x_std;
}
model {
  alpha_HKI ~ normal(0, 1);
  alpha_JYV ~ normal(0, 1);
  alpha_HAI ~ normal(0, 1);
  alpha_OUL ~ normal(0, 1);
  beta_HKI ~ normal(0, 1);
  beta_JYV ~ normal(0, 1);
  beta_HAI ~ normal(0, 1);
  beta_OUL ~ normal(0, 1);
  y_HKIstd ~ normal(mu_HKIstd, sigma_std);
  y_JYVstd ~ normal(mu_JYVstd, sigma_std);
  y_HAIstd ~ normal(mu_HAIstd, sigma_std);
  y_OULstd ~ normal(mu_OULstd, sigma_std);
}
generated quantities {
  vector[N] mu_HKI;
  vector[N] mu_JYV;
  vector[N] mu_HAI;
  vector[N] mu_OUL;
  vector[N] log_likHKI;
  vector[N] log_likJYV;
  vector[N] log_likHAI;
  vector[N] log_likOUL;
  real<lower=0> sigma;
  real ypred_HKI;
  real ypred_JYV;
  real ypred_HAI;
  real ypred_OUL;
  mu_HKI = mu_HKIstd*sd(y_HKI) + mean(y_HKI);
  mu_JYV = mu_JYVstd*sd(y_JYV) + mean(y_JYV);
  mu_HAI = mu_HAIstd*sd(y_HAI) + mean(y_HAI);
  mu_OUL = mu_OULstd*sd(y_OUL) + mean(y_OUL);
  sigma = sigma_std*sd(y_ALL);
  ypred_HKI = normal_rng((alpha_HKI + beta_HKI*xpred_std)*sd(y_HKI)+mean(y_HKI), sigma*sd(y_ALL));
  ypred_JYV = normal_rng((alpha_JYV + beta_JYV*xpred_std)*sd(y_JYV)+mean(y_JYV), sigma*sd(y_ALL));
  ypred_HAI = normal_rng((alpha_HAI + beta_HAI*xpred_std)*sd(y_HAI)+mean(y_HAI), sigma*sd(y_ALL));
  ypred_OUL = normal_rng((alpha_OUL + beta_OUL*xpred_std)*sd(y_OUL)+mean(y_OUL), sigma*sd(y_ALL));
  for (i in 1:N)
    log_likHKI[i] = normal_lpdf(y_HKI[i] | mu_HKI[i], sigma);
  for (i in 1:N)
    log_likJYV[i] = normal_lpdf(y_JYV[i] | mu_JYV[i], sigma);
  for (i in 1:N)
    log_likHAI[i] = normal_lpdf(y_HAI[i] | mu_HAI[i], sigma);
  for (i in 1:N)
    log_likOUL[i] = normal_lpdf(y_OUL[i] | mu_OUL[i], sigma);
}
// Poisson NMF
//
// Y ~ Poisson(Theta * B) with bells and whistles
//
// 18 April 2024

data {
  int<lower=0> U; // Number of users
  int<lower=0> I; // Number of items
  int<lower=0> K; // Latent space dimension
  array[U, I] int<lower=0> y; // Data matrix
}

parameters {
  real<lower=0> tau; // Global shrinkage hyperparameter
  vector<lower=0>[K] b; // Local shrinkage hyperparameters
  array[U] vector<lower=0>[K] theta; // Latent space basis
  // array[I] positive_ordered[K] beta; // Coefficients
  array[I] vector<lower=0>[K] beta; // Coefficients
}

transformed parameters{
  // Poisson intensities
  array[U, I] real<lower=0> lambda;
  
  for (u in 1:U){
    for (i in 1:I){
      lambda[u, i] = dot_product(theta[u], beta[i]);
    }
  }
}

model {
  // Priors on global hyperparameter
  tau ~ student_t(3, 0, 5);
  // tau ~ cauchy(0, 1);
  
  // Priors on local hyperparameters
  b ~ student_t(3, 0, 5);
  // b ~ cauchy(0, 1);

  // Prior on theta
  for (k in 1:K){
    for (u in 1:U){
    theta[u, k] ~ normal(0, tau * b[k]);
    }
  }
  
  // Prior on beta
  for (i in 1:I){
    beta[i] ~ exponential(1);
  }
  
  // Likelihood
  for (u in 1:U){
    for (i in 1:I){
      y[u, i] ~ poisson(lambda[u, i]);
    }
  }
}

generated quantities{
  // Samples from posterior predictive distribution
  array[U, I] int<lower=0> y_rep;
  
  for (u in 1:U){
    for (i in 1:I){
      y_rep[u, i] = poisson_rng(lambda[u, i]);
    }
  }
}


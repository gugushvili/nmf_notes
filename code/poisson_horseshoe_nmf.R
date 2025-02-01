####################
#                  #
# Bayesian NMF     #
#                  #
# Shota Gugushvili #
#                  #
# 11 December 2024 #
#                  #
####################

################################################################################

## Preliminaries

# Packages

library(cmdstanr) # Stan
library(tidybayes) # Manipulate Stan output

library(ggplot2) # Plotting
library(hrbrthemes) # Plotting themes
library(ggdendro) # Dendrograms with ggplot2
library(ggfortify) # Plot prcomp with ggplot2
library(plotrix) # Hinton diagram
library(viridis) # Colour schemes

library(dplyr) # Wrangling

library(fossil) # Rand and adjusted Rand scores

library(ASRgenomics) # Apple clone data and PCA for marker data

## Plotting options

# hrbrthemes requires these fonts:
# https://github.com/hrbrmstr/hrbrthemes/tree/master/inst/fonts/roboto-condensed

# theme_set(hrbrthemes::theme_ipsum_rc(grid = "XY"))

# Working directory: modify path as needed

setwd("~/Library/CloudStorage/Dropbox/NMF")

################################################################################

## Data

# theta matrix is UxI, beta matrix is IxK
# theta matrix has small positive entries everywhere except blocks along the diagonal,
# where the entries are large
# Construction for beta is similar

# Construct data list: check Stan file for names

U <- 60
I <- 900
K <- 3

# Set seed for reproducibility

set.seed(4556789)

# theta matrix

theta <- matrix(runif(n = U * K, min = 0.01, max = 0.02), nrow = U)
theta[1:(U/3), 1] <- theta[1:(U/3), 1] + runif(n = U/3, min = 1, max = 2)
theta[((U/3) + 1):(2*U/3), 2] <- theta[((U/3) + 1):(2*U/3), 2] + runif(n = U/3, min = 1, max = 2)
theta[((2*U/3) + 1):U, 3] <- theta[((2*U/3) + 1):U, 3] + runif(n = U/3, min = 1, max = 2)
theta

# beta matrix

beta <- matrix(runif(n = K * I, min = 0.01, max = 0.02), nrow = K)
beta[1, 1:(I/3)] <- beta[1, 1:(I/3)] + runif(n = I/3, min = 1, max = 2)
beta[2, ((I/3) + 1):(2 * I / 3)] <- beta[2, ((I/3) + 1):(2 * I / 3)] + runif(n = I/3, min = 1, max = 2)
beta[3, ((2*I/3) + 1):I] <- beta[3, ((2*I/3) + 1):I] + runif(n = I/3, min = 1, max = 2)

# Poisson intensities: lambda = theta * beta

lambda <- theta %*% beta

# Data

y <- matrix(data = NA, nrow = U, ncol = I)

for (u in 1:U){
  for (i in 1:I){
    y[u, i] <- rpois(n = 1, lambda = lambda[u, i])
  }
}

sum(is.na(y)) # Sanity check

# Check frequencies

table(y)

# Some plots

color2D.matplot(x = theta, Hinton = TRUE, xlab = "Meta-Variables", ylab = "Subjects", axes = FALSE)

color2D.matplot(x = beta, Hinton = TRUE, xlab = "Variables", ylab = "Meta-variables", axes = FALSE)

color2D.matplot(x = lambda, Hinton = TRUE, xlab = "Variables", ylab = "Subjects", axes = FALSE)

color2D.matplot(x = y, Hinton = TRUE, xlab = "Variables", ylab = "Subjects", axes = FALSE)

################################################################################

## Horseshoe-like hierarchical construction

# https://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf

## Fit model with variational (cmdstanr)

# https://mc-stan.org/cmdstanr/articles/cmdstanr.html

file5 <- file.path("poisson_horseshoe_nmf.stan") # Load Stan file with model
mod5 <- cmdstan_model(stan_file = file5) # Compile model

mod5$print() # Check code

# Prepare data for cmdstanr

nmf_data5 <- list(
  U = U,
  I = I,
  K = 5,
  y = y
  )

fit_variational5 <- mod5$variational(data = nmf_data5,
                                     seed = 4567812,
                                     algorithm = "meanfield", # "fullrank",
                                     iter = 100000, # Default 1e4 might not suffice for small tol_rel_obj
                                     grad_samples = 1,
                                     elbo_samples = 100,
                                     tol_rel_obj = 0.001, # Default 0.01 seems too crude
                                     eval_elbo = 100,
                                     draws = 1000, # > 1000 is slow and requires more storage
                                     adapt_engaged = TRUE
                                     )

# Extract theta matrix

theta_hat5 <- matrix(data = pull(fit_variational5$summary(variables = c("theta"), "median"), name = "median"),
                     byrow = FALSE,
                     ncol = 5
                     )

# Summarise shrinkage parameters

fit_variational5$summary(variables = c("b"), "median")
fit_variational5$summary(variables = c("tau"), "median")

draws_df_b <- fit_variational5$draws(format = "df", variables = "b")
str(draws_df_b)

summarise_draws(tidy_draws(draws_df_b))

draws_df_b %>%
  spread_rvars(b[metavariable]) %>%
  median_qi(b)

draws_df_b %>%
  spread_rvars(b[var_index]) %>%
  ggplot(aes(y = var_index, xdist = b)) +
  stat_pointinterval() +
  xlab("beta") +
  ylab("meta-variable index")

# Plot latent variables

color2D.matplot(x = theta_hat5, Hinton = TRUE, xlab = "Meta-variables", ylab = "Subjects") # NMF results

color2D.matplot(x = theta, Hinton = TRUE, xlab = "Meta-variables", ylab = "Subjects") # Truth

# Plot latent variables with columns reordered according to importance

df_reorder <- data.frame(b = pull(fit_variational5$summary(variables = c("b"), "median"), name = "median"),
                         index = 1:5) 

str(df_reorder)

df_reorder <- df_reorder %>%
  arrange(desc(b))

df_reorder

color2D.matplot(x = theta_hat5[, df_reorder[, "index"]], Hinton = TRUE, xlab = "Meta-variables", ylab = "Subjects")

true_clusters5 <- rep(c(1, 2, 3), times = c(20, 20, 20))

data.frame(Variable1 = theta_hat5[, df_reorder[1, "index"]],
           Variable2 = theta_hat5[, df_reorder[2, "index"]],
           cluster = as.factor(true_clusters5)
           ) %>% 
  ggplot() +
  geom_point(aes(x = Variable1, y = Variable2, colour = cluster)) +
  scale_colour_viridis_d(option = "viridis")

# Clustering via K-means

cluster_kmeans5 <- kmeans(x = theta_hat5, centers = 3, nstart = 10)$cluster

rand.index(true_clusters5, cluster_kmeans5)
adj.rand.index(true_clusters5, cluster_kmeans5)

# Comparison to PCA

pca5 <- prcomp(y, center = TRUE, scale. = TRUE, rank. = 5)

summary(pca5)

biplot(pca5, pc.biplot = TRUE)

color2D.matplot(x = pca5$x, Hinton = TRUE)

autoplot(pca5, data = data.frame(cluster = as.factor(true_clusters5)), colour = 'cluster') +
  scale_colour_viridis_d(option = "viridis")

################################################################################

## Minimal implementation without explicit lambda

# This is faster

file6 <- file.path("poisson_horseshoe_nmf_minimal.stan")
mod6 <- cmdstan_model(stan_file = file6)

mod6$print()

nmf_data6 <- list(
  U = U,
  I = I,
  K = 5,
  y = y
  )

fit_variational6 <- mod6$variational(data = nmf_data6,
                                     seed = 4567812,
                                     algorithm = "meanfield", # "fullrank",
                                     iter = 100000, # Default 1e4 might not suffice for small tol_rel_obj
                                     grad_samples = 1,
                                     elbo_samples = 100,
                                     tol_rel_obj = 0.001, # Default 0.01 seems too crude
                                     eval_elbo = 100,
                                     draws = 1000, # > 1000 is slow and requires more storage
                                     adapt_engaged = TRUE
                                     )

theta_hat6 <- matrix(data = pull(fit_variational6$summary(variables = c("theta"), "median"), name = "median"),
                     byrow = FALSE,
                     ncol = 5
                     )

fit_variational6$summary(variables = c("b"), "median")
fit_variational6$summary(variables = c("tau"), "median")

# Plot latent variables

color2D.matplot(x = theta_hat6, Hinton = TRUE) # NMF results

color2D.matplot(x = theta, Hinton = TRUE) # Truth

# Plot reordered columns

reordering <- fit_variational6$summary(variables = c("b"), "median") %>%
  mutate(reordering = 1:5)  %>%
  arrange(desc(median)) %>%
  pull(name = reordering)

color2D.matplot(x = theta_hat6[, reordering], Hinton = TRUE) # NMF results

# Clustering by taking argmax across rows 

cluster_nmf6 <- max.col(theta_hat6)

# True classes

true_clusters6 <- max.col(theta)

# Accuracy measures

rand.index(true_clusters6, cluster_nmf6)
adj.rand.index(true_clusters6, cluster_nmf6)

# Clustering via K-means

cluster_kmeans6 <- kmeans(x = theta_hat6, centers = 3, nstart = 10)$cluster

rand.index(true_clusters6, cluster_kmeans6)
adj.rand.index(true_clusters6, cluster_kmeans6)

################################################################################

## Apple clone data

# Load data

data("geno.apple")

# Checks

glimpse(geno.apple)

str(geno.apple)

# Bayesian NMF

file_apple <- file.path("poisson_horseshoe_nmf_minimal.stan")
mod_apple <- cmdstan_model(stan_file = file_apple)

mod_apple$print()

nmf_apple <- list(
  U = nrow(geno.apple),
  I = ncol(geno.apple),
  K = 10,
  y = geno.apple
  )

fit_apple <- mod_apple$variational(data = nmf_apple,
                                   seed = 4567812,
                                   algorithm = "meanfield", # "fullrank",
                                   iter = 100000, # Default 1e4 might not suffice for small tol_rel_obj
                                   grad_samples = 1,
                                   elbo_samples = 100,
                                   tol_rel_obj = 0.001, # Default 0.01 seems too crude
                                   eval_elbo = 100,
                                   draws = 1000, # > 1000 is slow and requires more storage
                                   adapt_engaged = TRUE
                                   )

theta_hat_apple <- matrix(data = pull(fit_apple$summary(variables = c("theta"), "median"), name = "median"),
                          byrow = FALSE,
                          ncol = 10
                          )

fit_apple$summary(variables = c("b"), "median")
fit_apple$summary(variables = c("tau"), "median")

# Plot reordered columns

reordering_apple <- fit_apple$summary(variables = c("b"), "median") %>%
  mutate(reordering = 1:10)  %>%
  arrange(desc(median)) %>%
  pull(name = reordering)

attr(theta_hat_apple, "dimnames")[[1]] <- attr(geno.apple, "dimnames")[[1]]

heatmap(theta_hat_apple[, reordering_apple], Colv = NA, Rowv = NA, scale = "none")

# Plot first two latent variables

data.frame(Variable1 = theta_hat_apple[, reordering_apple[[1]]],
           Variable2 = theta_hat_apple[, reordering_apple[[2]]],
           Family = as.factor(pheno.apple$Family)
           ) %>% 
  ggplot() +
  geom_point(aes(x = Variable1, y = Variable2, colour = Family))  +
  scale_colour_viridis_d(option = "magma")

# Clustering via K-means

set.seed(12345)

cluster_kmeans_apple <- kmeans(x = theta_hat_apple, centers = 17, nstart = 10)$cluster

apple_families <- as.numeric(as.factor(pheno.apple$Family))

rand.index(apple_families, cluster_kmeans_apple)

# Hierarchical clustering

hc_apple <- hclust(dist(theta_hat_apple), "ave")

# Cut dendrogram

# https://atrebas.github.io/post/2019-06-08-lightweight-dendrograms/

dendro_data_k <- function(hc, k) {
  
  hcdata    <-  ggdendro::dendro_data(hc, type = "rectangle")
  seg       <-  hcdata$segments
  labclust  <-  cutree(hc, k)[hc$order]
  segclust  <-  rep(0L, nrow(seg))
  heights   <-  sort(hc$height, decreasing = TRUE)
  height    <-  mean(c(heights[k], heights[k - 1L]), na.rm = TRUE)
  
  for (i in 1:k) {
    xi      <-  hcdata$labels$x[labclust == i]
    idx1    <-  seg$x    >= min(xi) & seg$x    <= max(xi)
    idx2    <-  seg$xend >= min(xi) & seg$xend <= max(xi)
    idx3    <-  seg$yend < height
    idx     <-  idx1 & idx2 & idx3
    segclust[idx] <- i
  }
  
  idx                    <-  which(segclust == 0L)
  segclust[idx]          <-  segclust[idx + 1L]
  hcdata$segments$clust  <-  segclust
  hcdata$segments$line   <-  as.integer(segclust < 1L)
  hcdata$labels$clust    <-  labclust
  
  hcdata
}

set_labels_params <- function(nbLabels,
                              direction = c("tb", "bt", "lr", "rl"),
                              fan       = FALSE) {
  if (fan) {
    angle       <-  360 / nbLabels * 1:nbLabels + 90
    idx         <-  angle >= 90 & angle <= 270
    angle[idx]  <-  angle[idx] + 180
    hjust       <-  rep(0, nbLabels)
    hjust[idx]  <-  1
  } else {
    angle       <-  rep(0, nbLabels)
    hjust       <-  0
    if (direction %in% c("tb", "bt")) { angle <- angle + 45 }
    if (direction %in% c("tb", "rl")) { hjust <- 1 }
  }
  list(angle = angle, hjust = hjust, vjust = 0.5)
}

plot_ggdendro <- function(hcdata,
                          direction   = c("lr", "rl", "tb", "bt"),
                          fan         = FALSE,
                          scale.color = NULL,
                          branch.size = 1,
                          label.size  = 3,
                          nudge.label = 0.01,
                          expand.y    = 0.1) {
  
  direction <- match.arg(direction) # if fan = FALSE
  ybreaks   <- pretty(segment(hcdata)$y, n = 5)
  ymax      <- max(segment(hcdata)$y)
  
  ## branches
  p <- ggplot() +
    geom_segment(data         =  segment(hcdata),
                 aes(x        =  x,
                     y        =  y,
                     xend     =  xend,
                     yend     =  yend,
                     linetype =  factor(line),
                     colour   =  factor(clust)),
                 lineend      =  "round",
                 show.legend  =  FALSE,
                 linewidth    =  branch.size)
  
  ## orientation
  if (fan) {
    p <- p +
      coord_polar(direction = -1) +
      scale_x_continuous(breaks = NULL,
                         limits = c(0, nrow(label(hcdata)))) +
      scale_y_reverse(breaks = ybreaks)
  } else {
    p <- p + scale_x_continuous(breaks = NULL)
    if (direction %in% c("rl", "lr")) {
      p <- p + coord_flip()
    }
    if (direction %in% c("bt", "lr")) {
      p <- p + scale_y_reverse(breaks = ybreaks)
    } else {
      p <- p + scale_y_continuous(breaks = ybreaks)
      nudge.label <- -(nudge.label)
    }
  }
  
  # labels
  labelParams <- set_labels_params(nrow(hcdata$labels), direction, fan)
  hcdata$labels$angle <- labelParams$angle
  
  p <- p +
    geom_text(data        =  label(hcdata),
              aes(x       =  x,
                  y       =  y,
                  label   =  label,
                  colour  =  factor(clust),
                  angle   =  angle),
              vjust       =  labelParams$vjust,
              hjust       =  labelParams$hjust,
              nudge_y     =  ymax * nudge.label,
              size        =  label.size,
              show.legend =  FALSE)
  
  # colors and limits
  if (!is.null(scale.color)) {
    p <- p + scale_color_manual(values = scale.color)
  }
  
  ylim <- -round(ymax * expand.y, 1)
  p    <- p + expand_limits(y = ylim)
  
  p
}

ggdendrogram(hc_apple, rotate = TRUE, size = 2) +
  theme_ipsum_rc(grid = FALSE) +
  xlab("genotype") +
  ylab("cluster distance")

plot_ggdendro(dendro_data_k(hc_apple, k = 18),
              direction   = "tb",
              # expand.y    = -0.2,
              branch.size = 1) +
  theme_ipsum_rc(grid = FALSE) +
  xlab("genotype") +
  ylab("cluster distance") +
  scale_colour_viridis_d(option = "plasma")

plot_ggdendro(dendro_data_k(hc_apple, k = 18),
              direction   = "tb",
              # expand.y    = -0.2,
              fan         = TRUE,
              branch.size = 1) +
  theme_ipsum_rc(grid = FALSE) +
  scale_colour_viridis_d(option = "plasma") +
  theme_void()

# Compare to PCA with 10 principal components

SNP_pca <- snp.pca(M = geno.apple, ncp = 10)
SNP_pca$eigenvalues
head(SNP_pca$pca.scores)
SNP_pca$plot.scree
SNP_pca$plot.pca

heatmap(SNP_pca$pca.scores, Colv = NA, Rowv = NA, scale = "none")

grp_apple <- as.factor(pheno.apple$Family)
SNP_pca_grp <- snp.pca(M = geno.apple, groups = grp_apple, label = FALSE, ellipses = TRUE)
SNP_pca_grp$plot.pca

SNP_pca_grp <- snp.pca(M = geno.apple, groups = grp_apple, label = FALSE)
SNP_pca_grp$plot.pca

set.seed(12345)

cluster_kmeans_pca <- kmeans(x = SNP_pca$pca.scores, centers = 17, nstart = 10)$cluster

rand.index(apple_families, cluster_kmeans_pca)

################################################################################

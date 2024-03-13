
.libPaths("~/Rlib/")
setwd("~/Code/vi_variance/script")

library(ggplot2)
source("fgvi_solve.r")
library(latex2exp)
library("RColorBrewer")

brewer.pal(n = 6, name = "RdBu")
# "#FED976" "#FEB24C" "#FD8D3C" "#F03B20" "#BD0026"
options(ggplot2.discrete.colour=
          c("#B2182B","#EF8A62","#FDDBC7","#D1E5F0","#67A9CF","#2166AC"))


n_divergences <- 5
n <- 2
eps <- c(seq(from = 0, to = .99, by = 0.01), 0.995, 0.999)
alpha <- 0.5

y0 <- rep(1, n)

var_estimate <- array(NA, dim = c(n_divergences, length(eps), n))
entropy_estimate <- array(NA, dim = c(n_divergences, length(eps)))
C_precision <- array(NA, dim = c(n, length(eps)))

for (i in 1:length(eps)) {
  C <- matrix(eps[i], nrow = n, ncol = n)
  diag(C) <- 1
  C_inv <- solve(C)
  
  var_estimate[1, i, ] <- psi_kl_rev(C)
  var_estimate[2, i, ] <- psi_kl_fwd(C)
  var_estimate[3, i, ] <- psi_sm_fwd(C)
  var_estimate[4, i, ] <- psi_sm_rev(C)
  var_estimate[5, i, ] <- psi_alpha(C, alpha, y0)
  
  for (j in 1:n_divergences) {
    entropy_estimate[j, i] <- 0.5 * sum(log(var_estimate[j, i, ]))
  }

  C_precision[, i] <- diag(C_inv)

  # update initial guests (for speed and stability)
  y0 <- var_estimate[5, i, ]
}

variational_objectives <- c("KL(q||p)", "KL(p||q)", "Score(p||q)", "Score(q||p)",
                            TeX("$\\alpha$"))

vo = factor(rep(variational_objectives, length(eps)),
            levels = c("Score(p||q)", "KL(p||q)", "alpha", "KL(q||p)",
                       "Score(q||p)"))

detC <- 1 - eps^2
entropyC <- rep(0.5 * log(detC), each = n_divergences)

plot.data <- data.frame(var = c(var_estimate[, , 1]),
                      divergence = vo,
                      eps = rep(eps, each = length(variational_objectives)),
                      detC = rep(detC, each = length(variational_objectives)),
                      entropy = c(entropy_estimate) - entropyC)


# Variance
p <- ggplot(data = plot.data, aes(x = detC, y = var, color = divergence)) +
  geom_line(linewidth = 1.5) + theme_bw() + scale_x_reverse() +
  theme(legend.background = element_rect(linewidth = 0.5, linetype="solid", 
                                         color = "black")) +
  xlab("|C|") + ylab("Relative variance") + # ylab(TeX("$\\R_{ii}$")) +
  scale_color_discrete(labels = unname(TeX(c("$D_F(p||q)$", "KL(p||q)", "$\\alpha = 0.5$", "KL(q||p)",
                                             "Score(q||p)")))) +
  theme(text = element_text(size = 20)) + theme(legend.position = "none")
p


# Entropy
p <- ggplot(data = plot.data, aes(x = detC, y = entropy, color = divergence)) +
  geom_line(linewidth = 1.5) + theme_bw() + scale_x_reverse() +
  theme(legend.background = element_rect(linewidth = 0.5, linetype="solid", 
                                         color = "black")) +
  xlab("|C|") + ylab("Entropy gap") + # ylab(TeX("$H(q) - H(p)$")) +
  scale_color_discrete(labels = 
                         unname(TeX(c("$SM(p||q)$", "KL(p||q)", "$R_{\\alpha}(p||q)$", "KL(q||p)",
                                      "$SM(q||p)$")))) +
  theme(text = element_text(size = 20))
p


# Precision
p <- ggplot(data = plot.data, aes(x = detC, 
                y = 1 / (var * rep(C_precision[1, ], each = n_divergences)), 
                                                     color = divergence)) +
  geom_line(linewidth = 1.5) + theme_bw() + scale_x_reverse() +
  theme(legend.background = element_rect(size = 0.5, linetype="solid", 
                                         color = "black")) +
  xlab("|C|") + ylab("Relative precision") + # ylab(TeX("$\\tilde{R}_{ii}$")) +
  scale_color_discrete(labels = 
                         unname(TeX(c("$D_F(p||q)$", "KL(p||q)",
                                      "$\\alpha = 0.5$", "KL(q||p)",
                                      "D_F(q||p)")))) +
  theme(text = element_text(size = 20)) + theme(legend.position = "none")
p



###############################################################################
## Ellipse plots

options(ggplot2.discrete.colour=
          c("black", "#67A9CF", "#D1E5F0", "#FDDBC7", "#EF8A62", "#B2182B"))
          # c("black", "#B2182B","#EF8A62","#FDDBC7","#D1E5F0","#67A9CF","#2166AC"))

index_eps = match(0.9, eps)

# generate data
n_sim = 1000
z <- array(NA, c(n_sim, n))
x <- array(NA, c(6 * n_sim, n))

C <- matrix(eps[index_eps], nrow = n, ncol = n)
diag(C) <- 1
L <- chol(C)

VI_var <- plot.data[plot.data$eps == 0.75, ]$var

for (i in 1:n_sim) {
  z[i,] <- rnorm(n)
  x[i,] <- t(L) %*% z[i,]
  x[n_sim + i, ] <- VI_var[1] * z[i,]       # KL(q||p)
  x[n_sim * 2 + i, ] <- VI_var[2] * z[i, ]  # KL(p||q)
  x[n_sim * 3 + i, ] <- VI_var[3] * z[i, ]  # D_s(p||q)
  x[n_sim * 4 + i, ] <- VI_var[4] * z[i, ]  # D_s(q||p)
  x[n_sim * 5 + i, ] <- VI_var[5] * z[i, ]  # D_alpha(p||q)
}

approx = factor(c("target", "KL(q||p)", "KL(p||q)", "D_s(p||q)", "D_s(q||p)", "D_alpha"),
                levels = c("target", "D_s(q||p)", "KL(q||p)", "D_alpha", "KL(p||q)", "D_s(p||q)"))


ellipse_data <- data.frame(z1 = x[,1], z2 = x[,2],
                           divergence = rep(approx, each=n_sim))

p <- ggplot(data = ellipse_data, aes(x=z1, y=z2, color=divergence)) +
  stat_ellipse(linewidth=2) + theme_bw() +
  scale_color_discrete(labels =
                         unname(TeX(c("target", "$D_F(p||q)$", "KL(p||q)",
                                      "$\\alpha = 0.5$", "KL(q||p)",
                                      "$D_F(q||p)$")))) +
  xlab(TeX("$z_1$")) + ylab(TeX("$z_2")) + theme(text = element_text(size = 20)) +
  theme(legend.position = "none")
p

p + theme(
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks = element_blank())
  



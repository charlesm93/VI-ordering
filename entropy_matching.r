
.libPaths("~/Rlib/")
setwd("~/Code/vi_variance/script")

library(rootSolve)
library(ggplot2)
source("fgvi_solve.r")
library(latex2exp)


n <- seq(from = 2, to = 10, by = 2)
eps <- seq(from = 0.05, to = 0.95, by = 0.05)
alpha_match <- array(NA, dim = c(length(n), length(eps)))

for (k in 1:length(n)) {
  for (j in 1:length(eps)) {
    # calculate entropy of target Gaussian
    C <- matrix(eps[j], nrow = n[k], ncol = n[k])
    diag(C) <- 1
    logDetC <- log(det(C))
    
    alpha <- seq(from = 0.01, to = 0.99, by = 0.005)
    logDetPsi <- rep(NA, length(alpha))
    psi <- rep(0.5, n[k])
    for (i in 1:length(alpha)) {
      psi <- psi_alpha(C, alpha[i], psi)
      
      logDetPsi[i] <- sum(log(psi))
    }
    
    alpha_match[k, j] <- alpha[which.min(abs(logDetPsi - logDetC))]
  }
}


plot.data <- data.frame(alpha = c(alpha_match),
                        eps = rep(eps, each = length(n)),
                        n = factor(rep(n, length(eps)),
                                      levels = c("10", "9", "8", "7", "6",
                                                 "5", "4", "3", "2", "1"))
                        )

p <- ggplot(data = plot.data,
            aes(x = eps, y = alpha, color = n)) + theme_bw() +
  geom_point(size = 1) + geom_line(linewidth=1.5) +
  xlab(TeX("$\\epsilon$")) + ylab(TeX("Entropy-matching $\\alpha")) +
  theme(text = element_text(size = 20))
p


# entropy_vi <- function(alpha) {
#   fun <- function(a) {
#     a^3 + alpha * (n * eps - 1 + eps) * a^2 - 
#       (n * alpha^2 * eps * (1 - eps) + (1 - alpha)) * a -
#       (n - 1) * (1 - alpha) * alpha * eps
#   }
#   
#   uni <- uniroot(fun, c(0, 1))$root
#   
#   # calculate entropy of approximation
#   uni^n
# }
# 
# alpha <- seq(from = 0.1, to = 0.9, by = 0.01)
# entropy_alpha <- rep(NA, length(alpha))
# 
# for (i in 1:length(alpha)) {
#   entropy_alpha[i] <- entropy_vi(alpha[i])  
# }




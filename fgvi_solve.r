
# A collection of functions to calculate the FG-VI solution when approximating
# a Gaussian target. Takes in as input the correlation matrix C.

.libPaths("~/Rlib/")
library(rootSolve)
library(CVXR)

psi_kl_fwd <- function(C) {
  ## Minimize KL(p||q)
  diag(C)
}

psi_kl_rev <- function(C) {
  # Minimize KL(q||p)
  1 / diag(solve(C))
}

psi_alpha <- function(C, alpha, y0) {
  ## Minimize the Reyni alpha divergence by solving a root-finding problem.
  ## (Specifically the alpha-precision matching equation)
  ## This requires a (good) initial guess. When computing multiple
  ## solutions sequentially, the previous solutions should be used as an
  ## initial guess.
  ## f.root can be examine to check if the solver finds a stationary point.
  
  algebra_equation <- function(psi) {
    ndim = length(psi)
    Psi = matrix(0, nrow = ndim, ncol = ndim)
    diag(Psi) = psi
    
    Psi_inv = solve(Psi)
    C_inv = solve(C)
    Sigma_tilde = solve(alpha * C_inv + (1 - alpha) * Psi_inv)
    
    psi - diag(Sigma_tilde)
  }

  psi <- multiroot(algebra_equation, start = y0)
  
  psi$root
  # list(root = psi$root, froot = psi$f.root)
}

psi_alpha2 <- function(C, alpha, y0) {
  ## Same as above, but solves the alpha-variance matching equation.
  
  algebra_equation <- function(psi) {
    ndim = length(psi)
    Psi = matrix(0, nrow = ndim, ncol = ndim)
    diag(Psi) = psi
    Sigma_tilde = solve(alpha * Psi + (1 - alpha) * C)
    
    1 / psi - diag(Sigma_tilde)
  }
  
  psi <- multiroot(algebra_equation, start = y0)

  psi$root
}

psi_sm_fwd <- function(C) {
  ## Minimize D_SM(p||q) by solving a KKT problem solved by
  ## s = Psi_{ii}^{-1} Sigma_{ii}
  n <- dim(C)[1]
  
  H <- matrix(NA, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:i) {
      H[i, j] = C[i, j]^2 / (C[i, i] * C[j, j])
      H[j, i] = H[i, j]
    }
  }
  
  s <- Variable(n)
  objective <- Minimize(0.5 * quad_form(s, H) - sum(s))
  constraint <- list(s >= 0)
  
  problem <- Problem(objective, constraint)
  result <- solve(problem)
  
  # The abs is to deal with 0 value that get reported as (small)
  # negative values.
  1 / (c(abs(result$getValue(s))) / diag(C))
}

psi_sm_rev <- function(C) {
  ## Minimize D_SM(q||p) by solving a KKT problem solved by 
  ## s = Psi_{ii} Sigma^{-1}_{ii}.
  C_inv <- solve(C)
  n <- dim(C)[1]
  
  H <- matrix(NA, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:i) {
      H[i, j] = C_inv[i, j]^2 / (C_inv[i, i] * C_inv[j, j])
      H[j, i] = H[i, j]
    }
  }
  
  s <- Variable(n)
  objective <- Minimize(0.5 * quad_form(s, H) - sum(s))
  constraint <- list(s >= 0)
  
  problem <- Problem(objective, constraint)
  result <- solve(problem)
  
  c(result$getValue(s)) / diag(C_inv)
}

#####################################################################
## Test

# This matrix produces zero-variance and zero-precision with the
# score divergences. Need to make sure the variances are oder

if (FALSE) {
  C <- matrix(c(2.75, 0.71, -2.46,
                0.71, 1.51, -1.79,
                -2.46, -1.79, 3.59), nrow = 3)

  psi_kl_fwd(C)
  # 2.75 1.51 3.59
  
  psi_kl_rev(C)
  # 0.6321693 0.3667704 0.3841117
  
  psi = psi_alpha(C, alpha = 0.5, y0 = diag(C))
  psi
  # 1.4954419 0.8450198 1.3147071
  
  alpha = 0.5
  Psi = matrix(0, ncol = ncol(C), nrow = nrow(C))
  diag(Psi) = psi
  1 / diag(solve(alpha * Psi + (1 - alpha) * C))
  
  
  psi_alpha2(C, alpha = 0.5, y0 = diag(C))
  
  psi_sm_fwd(C, y0 = 1 / diag(C))
  # 3.083841e+00  1.693309e+00 1.917991e+23
  
  
  psi_sm_rev(C)
  # 4.496120e-01  2.608547e-01 -1.263594e-23
}

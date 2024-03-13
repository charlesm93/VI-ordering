import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jax import jit, grad, random
import numpyro.distributions as dist
import optax

class FGVI():
    def __init__(self, D, lp, lp_g=None, alpha=None):
        """
        Inputs:
          D: (int) Dimension of parameters.
          lp: Function to evaluate log_density whose gradient can be
              evaluated with jax.grad(lp)
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g
        self.alpha = alpha

    def minimize_loss(self, loss_function, key, opt, mean=None, cov_diag=None,
                      batch_size=8, niter=1000, nprint=10, monitor=None,
                      scheduler=None):
        """
        Main function to fit a factorized Gaussian distribution
        """
        lossf = loss_function # jit(loss_function, static_argnums=(2))

        # @jit
        def opt_step(params, opt_state, key):
            loss, grads =\
              jax.value_and_grad(lossf, argnums=0)(params, key, batch_size=batch_size)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if mean is None:
            mean = jnp.zeros(self.D)
        if cov_diag is None:
            cov_diag = jnp.ones(self.D)

        # Optimization is done on the unconstrained scale (using a log transformation)
        log_cov_diag = np.log(cov_diag)
        params = (mean, log_cov_diag)

        # Run optimization
        opt_state = opt.init(params)
        losses = []
        nevals = 1

        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {i} of {niter}')
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    # To be tested
                    mean = params[0]
                    cov_diag_log = params[1]
                    cov_diag = np.exp(cov_diag_log)

                    monitor(i, [mean, cov_diag], self.lp, key, nevals=nevals)
                    nevals = 0

            params, opt_state, loss = opt_step(params, opt_state, key)
            if scheduler is not None:
                optax.scale_by_schedule(scheduler)

            key, _ = random.split(key)
            losses.append(loss)
            nevals += batch_size

        # Convert back to mean and diagonal of covariance matrix
        mean = params[0]
        cov_diag_log = params[1]
        cov_diag = np.exp(cov_diag_log)
        # if monitor is not None:
        #    ...

        return mean, cov_diag, losses

###############################################################################

class FG_ADVI(FGVI):
    """
    Class to fit a factorized Gaussian by maximizing the ELBO, equivalently
    minimizing the Kullback-Leibler divergence, KL(q||p).
    """

    def __init__(self, D, lp):
        """
        Inputs:
          D: Dimension of parameters.
          lp: Function to evaluate log_density whose gradient can be evaluated
              with jax.grad(lp)
        """
        super().__init__(D=D, lp=lp)

    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate negative-ELBO.
        """
        loc, log_cov_diag = params

        # Reparameterization trick
        eps = dist.Normal(0, 1).expand([self.D]).sample(key, (batch_size, ))
        scale = jnp.exp(0.5 * log_cov_diag)
        samples = loc + jnp.multiply(scale, eps)

        # q = dist.Independent(dist.Normal(loc=loc, scale=scale), 1)
        q = dist.Normal(loc=loc, scale=scale)
        logl = jnp.sum(self.lp(samples))
        logq = jnp.sum(q.log_prob(samples))
        negelbo = logq - logl

        return negelbo

    def fit(self, key, opt, mean=None, cov_diag=None, batch_size=8, niter=1000, nprint=10, monitor=None):
        """
        Main function to fit factorized Gaussian to target.
        ToDo: add description of arguments...
        """
        return self.minimize_loss(self.loss_function, key, opt, mean=mean,
                                  cov_diag=cov_diag, batch_size=batch_size,
                                  niter=niter, nprint=nprint, monitor=monitor)


class FG_alpha(FGVI):
    """
    Class to fit factorized Gaussian by targeting alpha divergence.
    """

    def __init__(self, D, lp, alpha):
        """
        Inputs:
            D: Dimension of parameters.
            lp: Function to evaluate log_density whose gradient can be evaluated
                with jax.grad(lp)
            alpha: Parameter for alpha-divergence. For now assumed to between
                   0 and 1.
        """
        super().__init__(D=D, lp=lp, alpha=alpha)

    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate alpha-divergence.
        """
        loc, log_cov_diag = params

        # Reparameterization trick
        eps = dist.Normal(0, 1).expand([self.D]).sample(key, (batch_size, ))
        scale = jnp.exp(0.5 * log_cov_diag)
        samples = loc + jnp.multiply(scale, eps)

        q = dist.Normal(loc=loc, scale=scale)

        logl = self.lp(samples)
        logq = jnp.sum(q.log_prob(samples), axis=1)
        logw =  logl - logq

        # NOTE: might not need the weight trick if using logsumexp.
        # c = jnp.max(logw)
        # logw -= c
        
        return - jnp.sum(jnp.exp(self.alpha * logw))
        # return -jss.logsumexp(logw, b=self.alpha)

    
    def fit(self, key, opt, mean=None, cov_diag=None, batch_size=8, niter=1000,
            nprint=10, monitor=None, scheduler=None):
        """
        Main function to fit factorized Gaussian to target.
        ToDo: add description of arguments...
        """
        return self.minimize_loss(self.loss_function, key, opt, mean=mean,
                                  cov_diag=cov_diag, batch_size=batch_size,
                                  niter=niter, nprint=nprint, monitor=monitor,
                                  scheduler=scheduler)

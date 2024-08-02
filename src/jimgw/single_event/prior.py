from dataclasses import field

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from jimgw.prior import Prior, CombinePrior, UniformPrior, PowerLawPrior, SinePrior, CosinePrior

@jaxtyped(typechecker=typechecker)
class UniformSpherePrior(CombinePrior):

    def __repr__(self):
        return f"UniformSpherePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "UniformSpherePrior only takes the name of the vector"
        self.parameter_names = [
            f"{self.parameter_names[0]}_mag",
            f"{self.parameter_names[0]}_theta",
            f"{self.parameter_names[0]}_phi",
        ]
        super().__init__(
            [
                UniformPrior(0.0, 1.0, [self.parameter_names[0]]),
                SinePrior([self.parameter_names[1]]),
                UniformPrior(0.0, 2 * jnp.pi, [self.parameter_names[2]]),
            ]
        )

@jaxtyped(typechecker=typechecker)
class UniformComponentMassPrior(CombinePrior):
    """
    A prior in the range [xmin, xmax) for component masses which assumes the
    component masses to be uniformly distributed.
    """

    def __repr__(self):
        return f"UniformComponentMassPrior(xmin={self.xmin}, xmax={self.xmax}, naming={self.parameter_names})"

    def __init__(self, xmin: float, xmax: float):
        self.parameter_names = ["m_1", "m_2"]
        super().__init__(
            [
                UniformPrior(xmin, xmax, ["m_1"]),
                UniformPrior(xmin, xmax, ["m_2"]),
            ]
        )

    def log_prob(self, z: dict[str, Float]) -> Float:
        output = super().log_prob(z)
        output += jnp.log(2.)

@jaxtyped(typechecker=typechecker)
class UniformComponentChirpMassPrior(PowerLawPrior):
    """
    A prior in the range [xmin, xmax) for chirp mass which assumes the
    component masses to be uniformly distributed.

    p(M_c) ~ M_c
    """

    def __repr__(self):
        return f"UniformInComponentsChirpMassPrior(xmin={self.xmin}, xmax={self.xmax}, naming={self.parameter_names})"

    def __init__(self, xmin: float, xmax: float):
        super().__init__(xmin, xmax, 1.0, ["M_c"])


def trace_prior_parent(prior: Prior, output: list[Prior] = []) -> list[Prior]:
    if prior.composite:
        if isinstance(prior.base_prior, list):
            for subprior in prior.base_prior:
                output = trace_prior_parent(subprior, output)
        elif isinstance(prior.base_prior, Prior):
            output = trace_prior_parent(prior.base_prior, output)
    else:
        output.append(prior)

    return output


# ====================== Things below may need rework ======================


# @jaxtyped(typechecker=typechecker)
# class AlignedSpin(Prior):
#     """
#     Prior distribution for the aligned (z) component of the spin.

#     This assume the prior distribution on the spin magnitude to be uniform in [0, amax]
#     with its orientation uniform on a sphere

#     p(chi) = -log(|chi| / amax) / 2 / amax

#     This is useful when comparing results between an aligned-spin run and
#     a precessing spin run.

#     See (A7) of https://arxiv.org/abs/1805.10457.
#     """

#     amax: Float = 0.99
#     chi_axis: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))
#     cdf_vals: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))

#     def __repr__(self):
#         return f"Alignedspin(amax={self.amax}, naming={self.naming})"

#     def __init__(
#         self,
#         amax: Float,
#         naming: list[str],
#         transforms: dict[str, tuple[str, Callable]] = {},
#         **kwargs,
#     ):
#         super().__init__(naming, transforms)
#         assert self.n_dim == 1, "Alignedspin needs to be 1D distributions"
#         self.amax = amax

#         # build the interpolation table for the ppf of the one-sided distribution
#         chi_axis = jnp.linspace(1e-31, self.amax, num=1000)
#         cdf_vals = -chi_axis * (jnp.log(chi_axis / self.amax) - 1.0) / self.amax
#         self.chi_axis = chi_axis
#         self.cdf_vals = cdf_vals

#     @property
#     def xmin(self):
#         return -self.amax

#     @property
#     def xmax(self):
#         return self.amax

#     def sample(
#         self, rng_key: PRNGKeyArray, n_samples: int
#     ) -> dict[str, Float[Array, " n_samples"]]:
#         """
#         Sample from the Alignedspin distribution.

#         for chi > 0;
#         p(chi) = -log(chi / amax) / amax  # halved normalization constant
#         cdf(chi) = -chi * (log(chi / amax) - 1) / amax

#         Since there is a pole at chi=0, we will sample with the following steps
#         1. Map the samples with quantile > 0.5 to positive chi and negative otherwise
#         2a. For negative chi, map the quantile back to [0, 1] via q -> 2(0.5 - q)
#         2b. For positive chi, map the quantile back to [0, 1] via q -> 2(q - 0.5)
#         3. Map the quantile to chi via the ppf by checking against the table
#            built during the initialization
#         4. add back the sign

#         Parameters
#         ----------
#         rng_key : PRNGKeyArray
#             A random key to use for sampling.
#         n_samples : int
#             The number of samples to draw.

#         Returns
#         -------
#         samples : dict
#             Samples from the distribution. The keys are the names of the parameters.

#         """
#         q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
#         # 1. calculate the sign of chi from the q_samples
#         sign_samples = jnp.where(
#             q_samples >= 0.5,
#             jnp.zeros_like(q_samples) + 1.0,
#             jnp.zeros_like(q_samples) - 1.0,
#         )
#         # 2. remap q_samples
#         q_samples = jnp.where(
#             q_samples >= 0.5,
#             2 * (q_samples - 0.5),
#             2 * (0.5 - q_samples),
#         )
#         # 3. map the quantile to chi via interpolation
#         samples = jnp.interp(
#             q_samples,
#             self.cdf_vals,
#             self.chi_axis,
#         )
#         # 4. add back the sign
#         samples *= sign_samples

#         return self.add_name(samples[None])

#     def log_prob(self, x: dict[str, Float]) -> Float:
#         variable = x[self.naming[0]]
#         log_p = jnp.where(
#             (variable >= self.amax) | (variable <= -self.amax),
#             jnp.zeros_like(variable) - jnp.inf,
#             jnp.log(-jnp.log(jnp.absolute(variable) / self.amax) / 2.0 / self.amax),
#         )
#         return log_p

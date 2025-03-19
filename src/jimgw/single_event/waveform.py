from jaxtyping import Array, Float
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from ripple.waveforms.TaylorF2 import gen_TaylorF2_hphc
from ripple.waveforms.TaylorF2QM import gen_TaylorF2_hphc as gen_TaylorF2QM_hphc
from ripple.waveforms.X_NRTidalv2 import gen_NRTidalv2_hphc
from ripple.waveforms.X_NRTidalv2_no_taper import gen_NRTidalv2_no_taper_hphc
from ripple.waveforms.sine_gaussian import td_sineGaussian, fd_sineGaussian, sum_fd_sineGaussian, sum_td_sineGaussian
from abc import ABC
import jax.numpy as jnp


class Waveform(ABC):
    required_keys: list[str]
    def __init__(self):
        return NotImplemented

    def __call__(
        self, axis: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        return NotImplemented


class RippleIMRPhenomD(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref
        self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'd_L', 'phase_c', 'iota']

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleIMRPhenomD(f_ref={self.f_ref})"


class RippleIMRPhenomPv2(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref
        self.required_keys = ['M_c', 'eta', 's1_x', 's1_y', 's1_z', 's2_x', 's2_y', 's2_z', 'd_L', 'phase_c', 'iota']

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleIMRPhenomPv2(f_ref={self.f_ref})"


waveform_preset = {
    "RippleIMRPhenomD": RippleIMRPhenomD,
    "RippleIMRPhenomPv2": RippleIMRPhenomPv2,
}

class RippleTaylorF2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        if self.use_lambda_tildes:
            self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'lambda_tilde', 'delta_lambda_tilde', 'd_L', 'phase_c', 'iota']
        else:
            self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 'phase_c', 'iota']

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_param,
            second_lambda_param,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_TaylorF2_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes)
        output["p"] = hp
        output["c"] = hc
        return output
    
class RippleIMRPhenomD_NRTidalv2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        if self.use_lambda_tildes:
            self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'lambda_tilde', 'delta_lambda_tilde', 'd_L', 'phase_c', 'iota']
        else:
            self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 'phase_c', 'iota']

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_param,
            second_lambda_param,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        
        hp, hc = gen_NRTidalv2_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes)
        output["p"] = hp
        output["c"] = hc
        return output

class RippleSineGaussian_td(Waveform):

    def __init__(self):
        self.required_keys = ['t_0', 'f_0', 'Q', 'log10_A', 'phi_0']
    def __call__(
        self, time: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["t_0"],
                params["f_0"],
                params["Q"],
                params["log10_A"],
                params["phi_0"],
            ]
        )
        output["h_0"] = td_sineGaussian(time, theta)
        return output

    def __repr__(self):
        return f"RippleSineGaussian_td()"
    
class RippleSineGaussian_fd(Waveform):

    n: int

    def __init__(self, n):
        self.n = n
        
        self.required_keys = ['t_0', 'f_0', 'Q', 'log10_A', 'phi_0']
    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["t_0"],
                params["f_0"],
                params["Q"],
                params["log10_A"],
                params["phi_0"],
            ]
        )
        output["h_0"] = fd_sineGaussian(frequency, theta)
        return output

    def __repr__(self):
        return f"RippleSineGaussian_fd()"

class RippleSineGaussian_fd_stack(Waveform):
    n: int

    def __init__(self, n):
        self.n = n
        self.required_keys = [f"{param}_{i}" for param in ['t_0', 'f_0', 'Q', 'log10_A', 'phi_0'] for i in range(n+1)]
    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, list]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {"h_0": 0}
        for i in range(self.n):
            theta = jnp.array([
                                params[f"t_0_{i}"],
                                params[f"f_0_{i}"],
                                params[f"Q_{i}"],
                                params[f"log10_A_{i}"],
                                params[f"phi_0_{i}"],
                        ])
            output["h_0"] += fd_sineGaussian(frequency, theta)
        return output
    def __repr__(self):
        return f"RippleSineGaussian_fd_stack()"

class RippleSineGaussian_td_stack(Waveform):
    def __init__(self):
        self.required_keys = ['t_0', 'f_0', 'Q', 'log10_A', 'phi_0']
    def __call__(
        self, time: Float[Array, " n_dim"], params: dict[str, list]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["t_0"],
                params["f_0"],
                params["Q"],
                params["log10_A"],
                params["phi_0"],
            ]
        )
        output["h_0"] = sum_td_sineGaussian(time, theta)
        return output

class RippleIMRPhenomD_NRTidalv2_no_taper(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        if self.use_lambda_tildes:
            self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'lambda_tilde', 'delta_lambda_tilde', 'd_L', 'phase_c', 'iota']
        else:
            self.required_keys = ['M_c', 'eta', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 'phase_c', 'iota']

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_param,
            second_lambda_param,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        
        hp, hc = gen_NRTidalv2_no_taper_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes)
        output["p"] = hp
        output["c"] = hc
        return output    
# FIXME: UNUSED FOR NOW
# class RippleTaylorF2QM(Waveform):
    
#     # TODO: add the possibility to sample over the QM parameter here, add it to the params dict!

#     f_ref: float

#     def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
#         self.f_ref = f_ref
#         self.use_lambda_tildes = use_lambda_tildes

#     def __call__(self, frequency: Array, params: dict) -> dict:
#         output = {}
#         ra = params["ra"]
#         dec = params["dec"]
        
#         if self.use_lambda_tildes:
#             first_lambda_param = params["lambda_tilde"]
#             second_lambda_param = params["delta_lambda_tilde"]
#         else:
#             first_lambda_param = params["lambda_1"]
#             second_lambda_param = params["lambda_2"]
        
#         theta = [
#             params["M_c"],
#             params["eta"],
#             params["s1_z"],
#             params["s2_z"],
#             first_lambda_param,
#             second_lambda_param,
#             params["d_L"],
#             0,
#             params["phase_c"],
#             params["iota"],
#         ]
#         hp, hc = gen_TaylorF2QM_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes)
#         output["p"] = hp
#         output["c"] = hc
#         return output
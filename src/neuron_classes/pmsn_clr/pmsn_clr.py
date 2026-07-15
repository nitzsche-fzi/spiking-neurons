import torch
import hashlib
import json
from ...polynomial_compiler import get_program, get_stacked_program_code, create_polynomials
from ...polynomial_compiler.simp import simplify
from ..base import BaseNeuronClass
from .pmsn_clr_params import PMSN_CLR_Params
from ...surrogate_gradient.Heaviside import get_heaviside
import re


def replace_var_names(equation, var_map):
    """
    Replace variable names in the equation based on the provided mapping.
    """
    for old_var, new_var in var_map.items():
        equation = equation.replace(old_var, new_var)
    return equation

def cast_float_constants(code):
    # wrap float with data_t: 0.02038922 -> data_t(0.02038922)
    pattern = re.compile(r"(?<!\w)(-?\d*\.\d+f?)(?!\w)")
    return pattern.sub(lambda m: f"data_t({m.group(0)})", code)

class PMSN_CLR(BaseNeuronClass):
    """
    PMSN_CLR neuron model: Polynomial Mmlti-state neuron with constant threshold and linear reset.
    """
    @staticmethod
    def load(path):
        params = PMSN_CLR_Params.load(path)
        return PMSN_CLR(params)

    def __init__(self, params):
        self.params = params
        self.n_states = params.n_states
        self.degree = params.degree

        energy_idle = float(getattr(params, "energy_idle", 0.0))
        energy_spike = float(getattr(params, "energy_spike", 0.0))
        super().__init__(energy_idle=energy_idle, energy_spike=energy_spike)

        self.polynomial_code, pcn = get_stacked_program_code(params.polynomial_coeffs, params.degree)
        self.reset_code, rcn = get_stacked_program_code(params.reset_coeffs, 1)
        self.polynomial_fn = get_program(self.polynomial_code, pcn)
        self.reset_fn = get_program(self.reset_code, rcn)
        self.threshold_fn = get_heaviside(params.surrogate_method, params.surrogate_alpha)

        self.state_delta = params.state_delta
        self.surrogate_alpha = params.surrogate_alpha
        self.threshold = float(params.threshold)

        #self.poly_coeffs = params.dense_polynomial_coeffs
        #self.reset_coeffs = params.dense_reset_coeffs

        # Register as buffer so it moves with .to(device)
        self.register_buffer("resting_state", params.resting_state.clone())
        #self.register_buffer("polynomial_coeffs", params.polynomial_coeffs.clone())
        #self.register_buffer("reset_coeffs", params.reset_coeffs.clone())
        #self.register_buffer("threshold", torch.tensor(self.threshold))

    def init_state(self, batch_size: int, n_neurons: int) -> torch.Tensor:
        """
        Initialize the state of the neuron based on the shape.
        Resulting state should have shape [n_states, batch_size, n_neurons]
        """
        return self.init_state_like(
            torch.empty(
                batch_size,
                n_neurons,
                device=self.resting_state.device,
                dtype=self.resting_state.dtype,
            )
        )

    def init_state_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize a state tensor with shape [n_states, *x.shape].
        """
        state_shape = (self.n_states, *x.shape)
        view_shape = (self.n_states, *([1] * x.dim()))
        return self.resting_state.view(view_shape).expand(state_shape).clone()

    def forward(self, x, state=None):
        if state is None:
            state = self.init_state_like(x)
            state = state.to(device=x.device, dtype=x.dtype)
        new_state = state + self.polynomial_fn(x, state) * self.state_delta
        spikes = self.threshold_fn(new_state[0] - self.threshold)
        reset = self.reset_fn(x, new_state)
        if self.params.detach_spikes:
            detached_spikes = spikes.detach()
        else:
            detached_spikes = spikes
        new_state += (reset - new_state) * detached_spikes
        return spikes, new_state
    
    def get_forward_code(self):
        return "State Polynomial:\n" + self.polynomial_code + "\n\nReset Polynomial:\n" + self.reset_code

    # TODO: Remove from neuron class. Maybe make a helper or move to GA repo
    def get_c_code(self):
        """
        Generate optimized C code for the forward pass and state initialization.
        Right now this is only used by the genetic algorithm for hardware synthesis.
        """
        # preprocess coefficients
        coeffs = self.params.polynomial_coeffs.clone()
        coeffs *= self.state_delta
        for i in range(self.n_states):
            coeffs[i, i+2] += 1.0  # inject identity term for state[i]

        # create and simplify polynomials
        polynomials = create_polynomials(coeffs, self.degree)
        reset_polynomials = create_polynomials(self.params.reset_coeffs, 1)
        simplified_polynomials = simplify(polynomials, min_advantage=1)
        simplified_reset_polynomials = simplify(reset_polynomials, min_advantage=1)

        # replacement map (pointer-per-state interface)
        replace_map = {f"x{i}": f"*s{i-1}" for i in range(1, self.n_states+1)}
        replace_map["x0"] = "x"

        # generate forward code
        params = ['data_t x'] + [f'data_t *s{i}' for i in range(self.n_states)]
        forward_signature = f"bool forward({', '.join(params)})"
        forward_code = forward_signature + " {\n"

        # Temporaries from state-update polynomials
        for temp_vars in simplified_polynomials[0][0]:
            temp_name, expr = temp_vars.split("=")
            expr = replace_var_names(expr.strip(), replace_map)
            forward_code += f"    data_t {temp_name.strip()} = {expr};\n"
        if simplified_polynomials[0][0]:
            forward_code += "\n"

        # Compute candidate new states for all i (including 0)
        for i in range(self.n_states):
            expri = replace_var_names(simplified_polynomials[0][1][i], replace_map)
            forward_code += f"    data_t new_state_{i} = {expri};\n"
        forward_code += "\n"

        # Threshold check on new_state_0
        # (use self.threshold to match coworker's code; no float suffix to respect data_t)
        forward_code += f"    bool spike = (new_state_0 > {self.threshold});\n\n"

        # If spike: compute reset using candidate new_state_* (i.e., *s{j} -> new_state_j)
        forward_code += "    if (spike) {\n"

        # Optional: temporaries for reset (if simplify() added any).
        # Evaluate inside spike branch. Make them depend on candidate new_state_* as well.
        for temp_vars in simplified_reset_polynomials[0][0]:
            temp_name, expr = temp_vars.split("=")
            expr = replace_var_names(expr.strip(), replace_map)
            # Substitute *s{j} -> new_state_j so reset temps can depend on candidate states
            for j in range(self.n_states):
                expr = expr.replace(f"*s{j}", f"new_state_{j}")
            forward_code += f"        data_t {temp_name.strip()} = {expr};\n"
        if simplified_reset_polynomials[0][0]:
            forward_code += "\n"

        # Assign reset values; allow reset exprs to depend on candidate new_state_j
        for i in range(self.n_states):
            reset_expr = replace_var_names(simplified_reset_polynomials[0][1][i], replace_map)
            reset_expr = reset_expr or "0.0"
            for j in range(self.n_states):
                reset_expr = reset_expr.replace(f"*s{j}", f"new_state_{j}")
            forward_code += f"        *s{i} = {reset_expr};\n"

        forward_code += "        return true;\n"
        forward_code += "    }\n\n"

        # Else: commit candidate new states
        for i in range(self.n_states):
            forward_code += f"    *s{i} = new_state_{i};\n"

        forward_code += "    return false;\n}\n"

        # Cast all float constants to data_t-compatible literals if you rely on it elsewhere
        forward_code = cast_float_constants(forward_code)

        # state init code
        params = [f'data_t *s{i}' for i in range(self.n_states)]
        state_init_signature = f"void init_state({', '.join(params)})"
        state_init_code = state_init_signature + " {\n"
        for i in range(self.n_states):
            state_init_code += f"    *s{i} = {self.params.resting_state[i]};\n"
        state_init_code += "}\n"

        return forward_code, state_init_code, forward_signature, state_init_signature



    def get_identifier(self) -> str:
        """Generate a unique and deterministic identifier for this neuron."""

        param_hash = hashlib.md5()
        # Include the most important parameters that define the neuron behavior
        key_params = {
            'n_states': self.n_states,
            'degree': self.degree,
            'threshold': self.threshold,
            'state_delta': self.state_delta,
            'surrogate_alpha': self.surrogate_alpha,
            'polynomial_coeffs': self.params.polynomial_coeffs.flatten().tolist(),
            'reset_coeffs': self.params.reset_coeffs.flatten().tolist(),
            'resting_state': self.params.resting_state.tolist()
        }
        param_str = json.dumps(key_params, sort_keys=True)
        param_hash.update(param_str.encode())
        
        # Format: n{n_states}_d{degree}_{hex_hash}
        return f"n{self.n_states}d{self.degree}_{param_hash.hexdigest()[:8]}"

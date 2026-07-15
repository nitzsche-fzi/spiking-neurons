import itertools
from ..polynomial_compiler.simp import simplify, simplify_just_gather

def create_polynomials(coeffs, degree, pre_combinations=[]):
    # coeffs is a torch tensor of shape [n_polynomials, n_monomials]
    n_polynomials = coeffs.shape[0]
    n_inputs = n_polynomials + 1
    if len(pre_combinations) == 0:
        pre_combinations = list(itertools.chain.from_iterable(
            itertools.combinations_with_replacement(range(n_inputs), d)
            for d in range(degree + 1)
        ))
    # pre_combinations is of this shape: [[],[0],[1],[0,0],[0,1],[1,1], etc]
    polynomials = []
    for i in range(n_polynomials):
        polynomial = []
        for j,comb in enumerate(pre_combinations):
            monomial = []
            for k in comb:
                monomial.append(f"x{k}")
            monomial = f"{coeffs[i,j]:.8f}*{'*'.join(monomial)}"
            if monomial[-1] == "*":
                monomial = monomial[:-1]
            if coeffs[i,j] != 0.0:
                polynomial.append(monomial)
        polynomials.append("+".join(polynomial))
    return polynomials

def get_stacked_program_code(coeffs, degree):
    polynomials = create_polynomials(coeffs, degree)
    so = [
        simplify(polynomials.copy(), min_advantage=1),
        simplify_just_gather(polynomials.copy())
    ]
    so = [(s, o, i) for i, (s, o) in enumerate(so)]
    best_simp, best_ops, idx = min(so, key=lambda x: x[1][1][0]) # get the one with the least multiplications

    program = "def PROGRAM_NAME(x,state):\n"

    if idx != (len(so) - 1):
        for term in best_simp[0]:
            program += f"    {term}\n"
        best_simp = best_simp[1]

    program += "    out = torch.empty_like(state, device=state.device)\n"
    for i, term in enumerate(best_simp):
        if len(term) > 0:
            program += f"    out[{i}] = {term}\n"
        else:
            program += f"    out[{i}] = 0.0\n"

    program += "    return out\n"
    program = program.replace("x0", "x")
    program = program.replace("+-", "-")
    program.replace("1.00*", "")
    program.replace("1.0*", "")
    program.replace("1*", "")
    vars = [f"x{i}" for i in range(len(polynomials) + 1)]
    for var in vars:
        if var != "x":
            i = int(var[1:])
        program = program.replace(var, f"state[{i-1}]")
    program_hash = hash(program)
    # hash must be converted to positive hex
    if program_hash < 0:
        program_hash = -program_hash
    program_hash = str(hex(program_hash))[2:]
    program_name = f"polynomial_{program_hash}"
    program = program.replace("PROGRAM_NAME", program_name)
    return program, program_name

def get_program(program, program_name):
    local_scope = {}
    exec(program, globals(), local_scope)
    return local_scope[program_name]

# use jit trace in order to make the program faster

def get_stacked_program(coeffs, degree):
    program, program_name = get_stacked_program_code(coeffs,degree)
    local_scope = {}
    exec(program, globals(), local_scope)
    raw_fun = local_scope[program_name]
    return raw_fun

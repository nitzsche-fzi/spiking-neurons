import numpy as np
import itertools as it

def get_vars(term) -> list:
    if term == "":
        ret = []
    else:
        ret = term.split('*')
        if is_float(ret[0]):
            ret = ret[1:]
    return ret

def get_constant(term) -> str:
    vars = term.split('*')
    if is_float(vars[0]):
        ret = vars[0]
    else:
        ret = "1.0"
    return ret

def get_subexpr_benefit(key,val):
    # key is a string of the form "x*y*z" 
    # val is a list of lists of the form [[i,j],[k,l],...]
    # this function returns a positive integer of what it would save to introduce a new variable for the subexpression
    # e.g. if len(val) = 1, it wouldnt save anything to introduce a new variable, but if len(val) = 2, it would save n multiplications, 
    # where n is the number of multiplications in key
    n_mults = len(key.split('*'))
    n_occurrences = len(val)
    return n_mults * (n_occurrences - 1)

def count_subexpr_occurences(subexpr : str, term : str) -> int:
    # subexpr is a string of the form "x*y"
    # term is a string of the form "0.2*x*y*z", or "0.2*x*y*y*x"
    # for the first one, it should return 1, for the second one, it should return 2
    # -> this function counts how many times the subexpression occurs in the term
    subexpr_vars = get_vars(subexpr)
    term_vars = get_vars(term)
    n_occurrences = 0
    while True:
        try:
            #remove all subexpr_vars from term_vars
            for var in subexpr_vars:
                term_vars.remove(var)
            n_occurrences += 1
        except ValueError:
            break
    return n_occurrences

def clean_polynomial(dirty_polynomial):
    polynomial = dirty_polynomial.strip()
    if polynomial == "":
        return ""
    if "(" in polynomial:
        raise ValueError("Polynomial contains parentheses, which is not supported")
    polynomial = polynomial.replace("-", "+-")
    polynomial = polynomial.replace("++", "+")
    if polynomial[0] == "+":
        polynomial = polynomial[1:]
    
    cleaned_polynomial = [term for term in polynomial.split('+') if float(get_constant(term)) != 0.0]
    # now we have a list of terms of the form [constant] * [product of variables]
    # we first sort the product of variables in each term 
    for j, term in enumerate(cleaned_polynomial):
        factors = term.strip().split('*')
        if len(factors) == 1: # if there is only one factor, just return that
            cleaned_polynomial[j] = factors[0]
        else:
            sorted_vars = sorted(get_vars(term)) 
            constant = get_constant(term)
            ret = f"{'*'.join(sorted_vars)}"
            if float(constant) != 1.0:
                ret = f"{constant}*{ret}"
            cleaned_polynomial[j] = ret
    ret = "+".join(cleaned_polynomial)
    return ret

def advanced_cse(dirty_polynomials, min_advantage, var_it = 0):
    assert min_advantage >= 1.0, "min_advantage must be at least 1: if it's <1, the program would run forever"
    subexpressions = {} # dict of subexpressions and where they are used
    cleaned_polynomials = []
    for i,polynomial in enumerate(dirty_polynomials):
        polynomial = polynomial.strip()
        if polynomial == "": # if the polynomial is empty, we skip it
            cleaned_polynomials.append("")
            continue
        # first we remove all terms where the coefficient is zero
        cleaned_polynomial = clean_polynomial(polynomial)
        cleaned_polynomials.append(cleaned_polynomial)
        # now we have a list of terms where the variables are sorted. We now find all subexpressions with it.
        for j,term in enumerate(cleaned_polynomial.split('+')):
            variables = get_vars(term)
            combs = set()
            for k in range(2,len(variables)+1):
                c = list(it.combinations(variables, k))
                for t in c:
                    combs.add('*'.join(t))
            for comb in combs:
                n_occurences = count_subexpr_occurences(comb, term)
                for _ in range(n_occurences):
                    if comb in subexpressions:
                        subexpressions[comb].append([i,j]) # append polynomial index and term index
                    else:
                        subexpressions[comb] = [[i,j]]
    subexpressions = [(key, val) for key, val in subexpressions.items()] # now it is a list of elements of shape (subexpression, [[poly_index, term_index],...])
    subexpressions = [(key, val, get_subexpr_benefit(key,val)) for key, val in subexpressions]
    subexpressions = sorted(subexpressions, key = lambda x: x[2], reverse = True)
    if len(subexpressions) == 0: # if there are no subexpressions
        return cleaned_polynomials, var_it
    to_replace = subexpressions[0] # to replace is a tuple of the form (subexpression, [[poly_index, term_index],...], benefit) with the highest benefit
    if to_replace[2] < min_advantage:
        return cleaned_polynomials, var_it # if there is no benefit to introducing a new variable, we return the polynomials as they are
    new_var_name = f"tmp{var_it}"
    for occurence in to_replace[1]:
        terms = cleaned_polynomials[occurence[0]].split('+')
        term = terms[occurence[1]]
        term = replace_in_term(term, to_replace[0], new_var_name)
        terms[occurence[1]] = term
        cleaned_polynomials[occurence[0]] = '+'.join(terms)
    new_polynomial = f"{to_replace[0]}"
    cleaned_polynomials.append(new_polynomial)
    return advanced_cse(cleaned_polynomials, min_advantage, var_it + 1)

def remove_var(term, var):
    vars = get_vars(term)
    constant = get_constant(term)
    vars.remove(var)
    ret = "*".join(vars)
    if ret == "":
        ret = constant
    elif float(constant) != 1.0:
        ret = f"{constant}*{ret}"
    return ret

def replace_in_term(term, to_replace, new_var_name):
    constant = get_constant(term)
    vars = get_vars(term)
    to_remove_vars = get_vars(to_replace)
    for var in to_remove_vars:
        vars.remove(var)
    vars = sorted(vars + [new_var_name])
    ret = "*".join(vars)
    if ret == "":
        ret = constant
    elif float(constant) != 1.0:
        ret = f"{constant}*{ret}"
    return ret
    
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def gather_terms(dirty_polynomial):
    cleaned_polynomial = clean_polynomial(dirty_polynomial)
    terms = cleaned_polynomial.split('+')
    variables = {} # this is a map from variable name to the indices of the terms where it occurs
    for i,term in enumerate(terms):
        vars = get_vars(term)
        for var in vars:
            if var in variables:
                variables[var].add(i)
            else:
                variables[var] = set([i])
    if len(variables) == 0:
        return cleaned_polynomial
    variables = [(key, val) for key, val in variables.items()]
    variables = sorted(variables, key = lambda x: len(x[1]), reverse = True)
    most_common_var = variables[0]
    n_occurrences = len(most_common_var[1])
    if n_occurrences == 1: 
        return cleaned_polynomial 
    p1 = [] 
    p2 = [] 
    for i,term in enumerate(terms):
        if i in most_common_var[1]:
            to_p2 = remove_var(term, most_common_var[0])
            p2.append(to_p2)
        else:
            p1.append(term)
    p1 = '+'.join(p1)
    p2 = '+'.join(p2)
    p1 = gather_terms(p1)
    p2 = gather_terms(p2)
    ret = p1 + "+" + most_common_var[0] + "*(" + p2 + ")"
    if ret[0] == "+":
        ret = ret[1:]
    if ret[-1] == "+":
        ret = ret[:-1]
    return ret

def post_process_temp_vars(new_terms, gathered_polynomials):
    for i in range(len(new_terms)):
        count = 0
        temp_var = f"tmp{i}"
        for polynomial in gathered_polynomials:
            count += polynomial.count(temp_var)
        for term in new_terms:
            count += term.count(temp_var)
        if count < 2:
            for j in range(len(gathered_polynomials)):
                gathered_polynomials[j] = gathered_polynomials[j].replace(temp_var, new_terms[i])
            for j in range(len(new_terms)):
                if i != j:
                    new_terms[j] = new_terms[j].replace(temp_var, new_terms[i])
            new_terms[i] = ""
        else:
            new_terms[i] = temp_var + "=" + new_terms[i]
    # remove empty strings from new_terms
    new_terms = [term for term in new_terms if term != ""]
    new_terms = reorder_temp_vars(new_terms)
    return new_terms, gathered_polynomials

def reorder_temp_vars(new_terms):
    # Extract variable names and their corresponding expressions
    temp_var_names = [term.split("=")[0].strip() for term in new_terms]
    temp_eqs = [term.split("=")[1].strip() for term in new_terms]

    # Create a dependency graph based on variable usage
    dependency_graph = {var: set() for var in temp_var_names}
    
    for var, expr in zip(temp_var_names, temp_eqs):
        for dep_var in temp_var_names:
            if dep_var in expr:  # Check if the current var depends on another temp
                dependency_graph[var].add(dep_var)

    # Perform a topological sort to determine the correct order
    sorted_vars = []
    visited = set()

    def visit(var):
        if var not in visited:
            visited.add(var)
            for dep in dependency_graph[var]:
                visit(dep)
            sorted_vars.append(var)

    for var in temp_var_names:
        visit(var)
    # Reorder the expressions based on the sorted variable order
    sorted_terms = [f"{var} = {temp_eqs[temp_var_names.index(var)]}" for var in sorted_vars]
    return sorted_terms
    
def count_operations(polynomials):
    n_mults = 0
    n_adds = 0
    for polynomial in polynomials:
        terms = polynomial.split('+')
        for term in terms:
            n_mults += term.count("*")
        n_adds = polynomial.count("+")
    return np.array([n_mults, n_adds])

def simplify(polynomials, min_advantage = 1.0):
    old_ops = count_operations(polynomials)
    n_original_terms = len(polynomials)
    new_polynomials,n_temp_vars = advanced_cse(polynomials, min_advantage)

    altered_originals = new_polynomials[:n_original_terms]
    new_terms = new_polynomials[n_original_terms:]
    new_terms, altered_originals = post_process_temp_vars(new_terms, altered_originals)
    gathered_polynomials = []
    for polynomial in altered_originals:
        gathered_polynomials.append(gather_terms(polynomial).replace("1.0*", ""))
    new_ops = count_operations(gathered_polynomials) + count_operations(new_terms)

    return (new_terms, gathered_polynomials), (old_ops, new_ops)

def simplify_just_gather(polynomials):
    old_ops = count_operations(polynomials)
    gathered_polynomials = []
    for polynomial in polynomials:
        gathered_polynomials.append(gather_terms(polynomial).replace("1.0*", ""))
    new_ops = count_operations(gathered_polynomials)
    return gathered_polynomials, (old_ops, new_ops)

from math import factorial
import itertools
import torch

def combination(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def count_monomials(polynomial_dimension, degree):
    """
    Calculates the number of monomials for a given number of state variables and degree.
    :param polynomial_dimension:
    :param degree:
    :return:
    """
    return combination(polynomial_dimension + 1 + degree, degree)

def get_monomial_combinations(n_polynomials, degree):
    combinations = list(itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(n_polynomials + 1), d)
        for d in range(degree + 1)
    ))
    ret = [sorted(list(c)) for c in combinations]
    return ret

def get_coefficients(coeffs, n_polynomials, degree):
    n_monomials = count_monomials(n_polynomials, degree)
    combinations = get_monomial_combinations(n_polynomials, degree)
    coefficients = torch.zeros(n_polynomials, n_monomials)
    for i, polynomial_coeffs in enumerate(coeffs):
        for indices, coefficient in polynomial_coeffs:
            monomial_index = combinations.index(sorted(indices))
            coefficients[i, monomial_index] = coefficient
    return coefficients
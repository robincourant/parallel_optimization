using Distributions
using LinearAlgebra

function initialize_abundance(p, equality, positivity)
    """Initialize abundance vector of size `p` with a uniform distribution.

    :param equality: add the equality constraint, ie: sum of the component equals to 1.
    TODO :param positive: add the posity constraint, ie: every component is positive.
    :return: the initialized abundance vector with wanted constraint.
    """
    uniform_distribution = Uniform(0, 1)
    A = rand(uniform_distribution, p, 1)

    if equality
        A /= sum(A)
    end
    return A
end


function is_feasible(abundance)
    """Indicates if 'abundance' verifies constraints."""
    P = 4
    feasible = false
    if (sum(abundance) - 1) <= 1e-3
        feasible = true
    end
    for i = 1:P
        if abundance[i] < 0
            feasible = false
        end
    end
    return feasible
end


function get_distance(a, b)
    """Compute the norm L2 of a and b."""
    return norm(a - b)^2
end


function get_function(x, S, a)
    """Compute the function to minimize: ||X - SA||."""
    return 0.5 * get_distance(x, S * a)
end


function get_gradient(x, S, a)
    """Compute the gradient of the function to minimize: ||X - SA||."""
    return -S' * (x - S * a)
end


function get_hessian(x, S, a)
    """Compute the hessian of the function to minimize: ||X - SA||."""
    return S' * S
end
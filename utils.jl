using Distributions
using LinearAlgebra

function get_feasible_point()
    """
    Initialize a random vector with respect to the equality
    and positivity constraints.
    """
    uniform_distribution = Uniform(4, 5)
    A = rand(uniform_distribution, 4, 1)
    A /= sum(A)
    return A
end

function get_feasible_point_matrix(p, n)
    """
    Initialize a random vector with respect to the equality
    and positivity constraints.
    """
    uniform_distribution = Uniform(0, 1)
    X = rand(uniform_distribution, p, n)
    for i = 1:n
        X[:, i] /= sum(X[:, i])
    end
    return X
end


function is_feasible(a)
    """Check if a point `a` is feasible (respects equality and positivity constraints)."""
    feasible = false
    if (sum(a) - 1) <= 1e-3
        feasible = true
    end
    for i = 1:4
        if a[i] < 0
            feasible = false
        end
    end
    return feasible
end


function get_function(X, S, A)
    """Compute the function to minimize: ||X - SA||."""
    return 0.5 * norm(X - S * A)^2
end


function get_gradient(x, S, a)
    """Compute the gradient of the function to minimize: ||X - SA||."""
    return -S' * (x - S * a)
end


function get_hessian(x, S, a)
    """Compute the hessian of the function to minimize: ||X - SA||."""
    return S' * S
end


function get_gradient_vector(X, S, A0, U, Z)
    p = size(S)[2]
    n = size(X)[2]
    ∇ = Z'S' * (S * (A0 + Z * U) - X)

    return reshape(∇, (p - 1) * n)
end


function get_hessian_vector(X, S, Z)
    n = size(X)[2]
    H = kron(Matrix(I, n, n), S * Z)
    return H' * H
end

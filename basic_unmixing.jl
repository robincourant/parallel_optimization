using LinearAlgebra

function get_distance(a, b)
    """Compute the norm L2 of a and b."""
    return norm(a - b)^2
end


function get_projection(x0, a, b)
    """
    Compute the value of the projection of `x0` with equality constraints.
    ie: min ||x0 - p|| with p'a = b.
    """
    γ = (x0'*a-b)[1] / norm(c)^2
    p = x0 - γ * c
    return p
end


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


function projected_gradient_vector(x, S, max_iter = 500, min_precision = 1e-8)
    """
    Projected gradient method to minimize `||x - Sa||` where `x` and `a` are vectors.
    It works with the constraint of equality: TODO add the inequality constraint.

    :param x: pixel vector of shape (1, 255).
    :param S: source matrix of shape (255, 4).
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated abundance vector.
    """
    # Initialize y and x with constrained random sample
    a = initialize_abundance(4, true, false)
    p = initialize_abundance(4, true, false)
    # Initialize the stepsize with twice the reciprocal of the smallest
    # Lipschitz constant of the gradient
    t = 1 / opnorm(S' * S)
    ∇f(y) = S' * (S * y - x)

    n_iter = 0
    round_loss = get_distance(x, S * a)
    loss = [round_loss]
    while n_iter < max_iter
        # Update the abundance vector
        a = p - t * ∇f(p)
        # Compute its projection in the equality constraint set
        p = get_projection(a, ones(4, 1), ones(1, 1))

        n_iter += 1
        round_loss = get_distance(x, S * a)
        loss = append!(loss, [round_loss])
        if round_loss < min_precision
            break
        end
    end

    return a, loss
end


function projected_gradient(X, S, max_iter = 500, min_precision = 1e-8)
    """
    Estimate the abundance matrix for each pixel of the image X, by estimating
    the abundance vector of each pixel.

    :param X: image matrix of shape (n_pixels, 255).
    :param S: source matrix of shape (255, 4).
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated abundance vector.
    """
    l, n = size(X)
    A = zeros(4, 0)
    # Iterate over each pixel vector
    for j = 1:n
        x = reshape(X[:, j], l, 1)
        # Estimate its abundance vector
        a, _ = projected_gradient_vector(x, S, max_iter, min_precision)
        A = [A a]
    end
    return A
end

using Base.Iterators

include("./gradient_projection.jl")
include("./log_barrier.jl")


function estimate_abundance(X, S, method, max_iter = 500, min_precision = 1e-8)
    """
    Estimate the abundance matrix for each pixel of the image X, by estimating
    the abundance vector of each pixel.

    :param X: image matrix of shape (n_pixels, 255).
    :param S: source matrix of shape (255, 4).
    :param method: which method to use:
        - `projected_gradient_1c`: projected-gradient_1c with 1 constraints;
        - `projected_gradient_2c`: projected-gradient_2c with 2 constraints;
        - `log_barrier`: log-barrier with 2 constraints.
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated image and the estimated abundance vector.
    """
    if method == "projected_gradient_1c"
        optimizer = projected_gradient_1c_vector
    elseif method == "projected_gradient_2c"
        optimizer = projected_gradient_2c_vector
    elseif method == "log_barrier"
        optimizer = log_barrier_vector
    end

    n1, n2, l = size(X)
    new_X = zeros(n1, n2, l)
    A = zeros(n1, n2, 4)
    # Iterate over each pixel vector
    for (i, j) in product(1:n1, 1:n2)
        x = X[i, j, :]
        # Estimate its abundance vector
        a, _ = optimizer(x, S, max_iter, min_precision)
        new_X[i, j, :] = S * a
        A[i, j, :] = a
    end
    return new_X, A
end
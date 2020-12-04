using Base.Iterators
using Statistics

include("./gradient_projection.jl")
include("./interior_point_least_square.jl")
include("./log_barrier.jl")
include("./utils.jl")


function get_optimizer_from_string(optimizer_name)
    """Return a callable optimizer from strings.

    :param optimizer_name: which method to use:
        - `projected_gradient_1c`: projected-gradient_1c with 1 constraints;
        - `projected_gradient_2c`: projected-gradient_2c with 2 constraints;
        - `log_barrier`: log-barrier with 2 constraints.
    :return: a callable optimizer.
    """
    if optimizer_name == "projected_gradient_1c"
        return projected_gradient_1c_vector
    elseif optimizer_name == "projected_gradient_2c"
        return projected_gradient_2c_vector
    elseif optimizer_name == "log_barrier"
        return log_barrier_vector
    end
end


function get_estimator_from_string(estimator_name)
    """Return a callable estimator from strings.

    :param estimator_name: which method to use:
        - `serial_pixel`: serial pixel-based approach;
        - `serial_image`: serial image-based approach;
    :return: a callable estimator.
    """
    if estimator_name == "serial_pixel"
        return estimate_abundance_pixel
    elseif estimator_name == "serial_image"
        return estimate_abundance_image
    end
end

function estimate_abundance_pixel(X, S, optimizer_name, max_iter = 500, min_precision = 1e-8)
    """
    Estimate the abundance matrix for each pixel of the image X, by estimating
    the abundance vector of each pixel.

    :param X: image matrix of shape (n_pixels, 255).
    :param S: source matrix of shape (255, 4).
    :param optimizer_name: which method to use:
        - `projected_gradient_1c`: projected-gradient_1c with 1 constraints;
        - `projected_gradient_2c`: projected-gradient_2c with 2 constraints;
        - `log_barrier`: log-barrier with 2 constraints.
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated image, the estimated abundance vector and the mean loss over
        pixel for each iteration.
    """
    optimizer = get_optimizer_from_string(optimizer_name)

    n1, n2, l = size(X)
    new_X = zeros(n1, n2, l)
    A = zeros(n1, n2, 4)
    loss = Array{Float64}[]
    # Iterate over each pixel vector
    for (i, j) in product(1:n1, 1:n2)
        x = X[i, j, :]
        # Estimate its abundance vector
        a, pixel_loss = optimizer(x, S, max_iter, min_precision)
        new_X[i, j, :] = S * a
        A[i, j, :] = a
        push!(loss, pixel_loss)
    end
    mean_loss = mean(loss, dims = 1)[1]
    # mean_loss = [mean(loss[k][loss[k].>0]) for k = 1:size(loss)[1]]

    return new_X, A, mean_loss
end


function estimate_abundance_image(
    X,
    S,
    optimizer_name = "ipls",
    max_iter = 500,
    min_precision = 1e-8,
)
    """
    Estimate the abundance matrix for each pixel of the image X, with a
    image-based approach.

    :param X: image matrix of shape (n_pixels, 255).
    :param S: source matrix of shape (255, 4).
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated image, the estimated abundance vector and the loss for each iteration.
    """
    l, p = size(S)
    n1, n2, _ = size(X)

    a, loss = interior_point_least_square(X, S, max_iter, min_precision)
    new_X, A = vector_to_image(S, a, l, p, n1 * n2)

    return new_X, A, loss
end

function get_runtime_series(X, S, optimizer_name, estimator_name, n_points = 20)
    """Compute the runtime series depending on the number of pixel"""
    estimator = get_estimator_from_string(estimator_name)

    n1, n2, _ = size(X)
    n_pixels = [floor(Int, k) for k in LinRange(1, n1, n_points)]
    runtime_series = []
    for k_pixel in n_pixels
        Y = X[1:k_pixel, 1:k_pixel, :]
        profiler = @timed estimator(Y, S, optimizer_name)
        append!(runtime_series, profiler.time)
    end
    return n_pixels, runtime_series
end

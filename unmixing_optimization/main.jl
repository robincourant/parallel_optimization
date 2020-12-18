using Base.Iterators
using Statistics
using Distributed
using DistributedArrays

include("./unmixing_methods/gradient_projection.jl")
include("./unmixing_methods/interior_point_least_square.jl")
include("./unmixing_methods/log_barrier.jl")
include("./unmixing_methods/primal_dual_interior_point.jl")
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
    elseif optimizer_name == "ipls"
        return interior_point_least_square
    elseif optimizer_name == "primal_dual"
        return primal_dual_interior_point
    end
end


function get_estimator_from_string(estimator_name)
    """Return a callable estimator from strings.

    :param estimator_name: which method to use:
        - `serial_pixel`: serial pixel-based approach;
        - `parallel_pixel`: serial pixel-based approach;
        - `serial_image`: serial image-based approach;
    :return: a callable estimator.
    """
    if estimator_name == "serial_pixel"
        return estimate_abundance_pixel
    elseif estimator_name == "parallel_pixel"
        return estimate_abundance_parallel
    elseif estimator_name == "serial_image"
        return estimate_abundance_image
    end
end


function estimate_abundance(X, S, optimizer_name, estimator_name, max_iter=20)
    """Compute the runtime series depending on the number of pixel"""
    estimator = get_estimator_from_string(estimator_name)
    new_X, A, mean_loss = estimator(X, S, optimizer_name, max_iter)

    return new_X, A, mean_loss
end


function estimate_abundance_pixel(X, S, optimizer_name, max_iter=500, min_precision=1e-8)
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
    mean_loss = mean(loss, dims=1)[1]

    return new_X, A, mean_loss
end


function estimate_abundance_image(X, S, optimizer_name, max_iter=50, min_precision=1e-8)
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
    flat_image = reshape(X, n1 * n2, l)'
    optimizer = get_optimizer_from_string(optimizer_name)

    a, loss = optimizer(flat_image, S, max_iter, min_precision)
    new_X, A = vector_to_image(S, a, l, p, n1 * n2)

    return new_X, A, loss
end


function estimate_abundance_parallel(X, S, optimizer_name, max_iter=500, min_precision=1e-8)
    """
    Estimate the new matrix X by computing in parallel on all available processors.

    :param X: image matrix of shape (n_pixels_1, n_pixels_2, 255).
    :param S: source matrix of shape (255, 4).
    :param optimizer_name: which method to use:
        - `projected_gradient_1c`: projected-gradient_1c with 1 constraints;
        - `projected_gradient_2c`: projected-gradient_2c with 2 constraints;
        - `log_barrier`: log-barrier with 2 constraints.
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated image
    """
    n_column = size(X)[2]
    X_parallel = [X[:,i,:] for i = 1:n_column]

    # Need to use pmap for each parameter to return -> not optimal as it makes code run several times
    new_X_divided = pmap(
        x -> estimate_abundance_pixel(
            reshape(x, n_column, 1, 255), S, optimizer_name, max_iter
        )[1],
        X_parallel
    )

    """
    For the parallel part, we choose to look only at the new_X matrix
    We could get other outputs just as in the serial implementation but for that
    we would have to uncomment the following lines and it would make the function run several 
    times
    """
    
    """
    @time  new_A_divided = pmap(x -> estimate_abundance(reshape(x, n_column, 1, 255), S, optimizer_name, max_iter, min_precision)[2], X_parallel)
    @time mean_loss_divided = pmap(x -> estimate_abundance(reshape(x, n_column, 1, 255), S, optimizer_name, max_iter, min_precision)[3], X_parallel)
    @time loss_divided = pmap(x -> estimate_abundance(reshape(x, n_column, 1, 255), S, optimizer_name, max_iter, min_precision)[4], X_parallel)
    """

    # Reconstructing X matrix
    new_X = zeros(n_column, n_column, 255)
    for i = 1:n_column
        new_X[:,i,:] = new_X_divided[i]
    end
    return new_X, 0, 0
end


function get_runtime_series(X, S, optimizer_name, estimator_name, n_points=20)
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
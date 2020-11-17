using LinearAlgebra

function get_distance(a, b)
    return 0.5 * norm(a - b)^2
end

function initialize_abundance(n, p, n_constraints)
    """
    Initialize the abundance matrix `A` with random coefficients (uniformly
    distributed). If the two constraints should be respected `n_constraint = 2`
    these coefficients are normalized in order that the sum over rows is 1.
    """
    A = ones(p, n)
    uniform_distribution = Uniform(0, 1)
    for j = 1:p
        A[j, :] = rand(uniform_distribution, n)
        if n_constraints == 2
            sum_row = sum(A[j, :])
            A[j, :] = map(x -> x / sum_row, A[j, :])
        end
    end
    return A
end

function abundance_estimation(X, S, nb_constraints)
    max_iter = 10e2
    precision = 10e-8
    # The main stepsize is set to twice the reciprocal of the smallest
    # Lipschitz constant of the gradient
    main_stepsize = LinearAlgebra.opnorm(S' * S)
    projection_stepsize = 0.01
    # Initialize an abundance matrix
    n, p = size(X)[2], size(S)[2]
    main_A = initialize_abundance(n, p, nb_constraints)

    loss = [get_distance(X, S * main_A)]
    main_iter = 0
    projection_iter = 0
    while main_iter < max_iter
        main_∇f = S' * (S * main_A - X)
        main_A -= main_stepsize * main_∇f

        projection_A = initialize_abundance(n, p, nb_constraints)
        while projection_iter < max_iter
            projection_∇f = projection_A - main_A
            projection_A -= projection_stepsize * projection_∇f
            projection_iter += 1
        end

        main_A = projection_A
        loss = append!(loss, [get_distance(X, S * main_A)])
        main_iter += 1
    end
    return main_A, loss
end

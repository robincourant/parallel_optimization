using LinearAlgebra

include("./utils.jl")


function get_projection(x0, a, b)
    """
    Compute the value of the projection of `x0` with equality constraints.
    ie: min ||x0 - p|| with p'a = b.
    """
    γ = (x0'*a-b)[1] / norm(a)^2
    p = x0 - γ * a
    return p
end


function projected_gradient_1c_vector(x, S, max_iter = 500, min_precision = 1e-8)
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
    round_loss = get_function(x, S, a)
    loss = [round_loss]
    while n_iter < max_iter
        # Update the abundance vector
        a = p - t * ∇f(p)
        # Compute its projection in the equality constraint set
        p = get_projection(a, ones(4, 1), ones(1, 1))

        n_iter += 1
        round_loss = get_function(x, S, a)
        loss = append!(loss, [round_loss])
        if round_loss < min_precision
            break
        end
    end

    return a, loss
end


function basic_backtracking(x, S, p, f, ∇f, d, t)
    """Bactracking method to find optimal `step_size`.

    :param x:pixel vector, shape (255,1).
    :param S: source matrix, shape (255,4).
    :param p: initial point.
    :param f, ∇f: function to be minimized and its gradient.
    :param d: search direction.
    :return: optimal_stepsize and the number of loops.
    """
    n_loops = 0
    # backtracking parameters
    alpha = 0.25
    beta = 0.7

    while f(x, S, p + t * d) > f(x, S, p) + sum(t * alpha * ∇f(x, S, p)' * d)
        t = beta * t
        n_loops += 1
    end
    return (t, n_loops)
end


function projected_gradient_2c_vector(x, S, max_iter = 500, min_precision = 1e-8)
    """
    Projected gradient method to minimize `||x - Sa||` where `x` and `a` are vectors.
    Using the the gradient projection for linear constraints method from Mr.Chonavel lesson.

    :param x: pixel vector of shape (255,1).
    :param S: source matrix of shape (255, 4).
    :param max_iter: maximum number of iterations.
    :return: estimated abundance vector and the evolution of cost.
    """
    # Initializing abundance vector
    P = 4
    abundance = initialize_abundance(4, true, true)
    round_loss = get_function(x, S, abundance)
    loss = [round_loss]

    KKT = false
    iter = 0
    while KKT == false && iter < max_iter
        # Find active constraints and define the contraint matrix
        constraint_mat = [1 for i = 1:P]' # sum to one constraint always active

        for i = 1:P
            if abundance[i] <= 0
                new_line = [0 for k = 1:P]'
                new_line[i] = -1
                constraint_mat = vcat(constraint_mat, new_line)
            end
        end
        constraint_mat = constraint_mat'

        # Calculate projection matrix and search direction
        projection_mat =
            Matrix(I, P, P) -
            constraint_mat * inv(constraint_mat' * constraint_mat) * constraint_mat'
        gradient = get_gradient(x, S, abundance)
        d = -projection_mat * gradient

        # If d!=0
        if d != [0; 0; 0; 0]
            α = 0
            dα = (1 / opnorm(transpose(S) * S))
            n_iter_1 = 0

            # Looking for αmax such that (abundance+αmax*d) is feasible
            while is_feasible(abundance + (α + dα) * d) == true && n_iter_1 < max_iter
                α += dα
                n_iter_1 += 1
            end

            # Looking for the optimal stepsize
            α, n_loops =
                basic_backtracking(x, S, abundance, get_function, get_gradient, d, α)
            abundance = abundance + α * d
            round_loss = get_function(x, S, abundance)
            loss = append!(loss, [round_loss])
            round_loss = get_function(x, S, abundance)
            loss = append!(loss, [round_loss])
            # Return to step 2

            # If d=0
        else
            # Set Lagrangian vector
            λ = -1 * inv(constraint_mat' * constraint_mat) * constraint_mat' * gradient
            # It nearly never happen so we don't solve the problem when it occurs
            for i = 1:length(λ)
                if i >= 2 && λ[i] < 0
                    print("wrong lambda")
                end
            end
            KKT = true
        end
        iter += 1

    end
    return abundance, loss
end






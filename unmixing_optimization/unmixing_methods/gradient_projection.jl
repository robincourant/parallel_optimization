using LinearAlgebra

include("../utils.jl")


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
    a = get_feasible_point()
    p = get_feasible_point()
    # Initialize the stepsize with twice the reciprocal of the smallest
    # Lipschitz constant of the gradient
    t = 1 / opnorm(S' * S)
    ∇f(y) = S' * (S * y - x)

    n_iter = 1
    loss = zeros(max_iter)
    loss[1] = get_function(x, S, a)
    while n_iter < max_iter
        # Update the abundance vector
        a = p - t * ∇f(p)
        # Compute its projection in the equality constraint set
        p = get_projection(a, ones(4, 1), ones(1, 1))

        n_iter += 1
        round_loss = get_function(x, S, a)
        loss[n_iter] = round_loss
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
    # Backtracking parameters
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
    a = get_feasible_point()
    loss = zeros(max_iter)
    loss[1] = get_function(x, S, a)

    KKT = false
    n_iter = 1
    while KKT == false && n_iter < max_iter
        # Find active constraints and define the contraint matrix
        constraint_mat = [1 for i = 1:4]' # sum to one constraint always active

        for i = 1:4
            if a[i] <= 0
                new_line = [0 for k = 1:4]'
                new_line[i] = -1
                constraint_mat = vcat(constraint_mat, new_line)
            end
        end
        constraint_mat = constraint_mat'

        # Calculate projection matrix and search direction
        projection_mat =
            Matrix(I, 4, 4) -
            constraint_mat * inv(constraint_mat' * constraint_mat) * constraint_mat'
        gradient = get_gradient(x, S, a)
        d = -projection_mat * gradient

        # If d!=0
        if d != [0; 0; 0; 0]
            α = 0
            dα = (1 / opnorm(transpose(S) * S))
            k_iter = 0

            # Looking for αmax such that (a+αmax*d) is feasible
            while is_feasible(a + (α + dα) * d) == true && k_iter < max_iter
                α += dα
                k_iter += 1
            end

            # Looking for the optimal stepsize
            α, n_loops = basic_backtracking(x, S, a, get_function, get_gradient, d, α)
            a = a + α * d

            loss[n_iter+1] = get_function(x, S, a)
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
        n_iter += 1

    end
    return a, loss
end






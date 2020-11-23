using LinearAlgebra
using Base.Iterators
using Distributions

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

function cost(X, S, A)
    """Compute cost.

    :param X: data matrix, size : L*N
    :param S: endmember matrix, size : L*P
    :param A: abundance matrix, size: P*N
    :return cost: distance between X and S*A
    """
    cost = (1 / 2) * (norm(X - S * A))^2
    return(cost)
end

function gradient_cost(x, S, A)
    gradient = -S' * (x - S * A)
    return gradient
end
function is_feasible(abundance)
    """Indicates if 'abundance' verifies constraints

    :param abundance: vector of size(P,1)
    :return feasible: bool
    """
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

function backtracking(x, S, p, f, ∇f, d, t)
    """Bactracking method to set optimal step_size.

    :param x:pixel vector, shape (255,1)
    :param S: source matrix, shape (255,4)
    :param p: initial point
    :param f, ∇f: function to be minimized and its gradient
    :param d: search direction
    :return t: optimal_stepsize
    :return n: number of loops
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

function projected_gradient_vector_antoine(x, S, max_iter=1000)
    """Projected gradient method to minimize `||x - Sa||` where `x` and `a` are vectors.
    Using the the gradient projection for linear constraints method from Mr.Chonavel lesson.

    :param x: pixel vector of shape (255,1).
    :param S: source matrix of shape (255, 4).
    :param max_iter: maximum number of iterations.
    :return abundance: estimated abundance vector.
    :return cost_evol: evolution of cost
    """
    # 1) Initializing abundance vector
    P = 4
    abundance = initialize_abundance(4, true, true)
    cost_evol = [cost(x, S, abundance)]


    KKT = false
    iter = 0
    while KKT == false && iter < max_iter

        # 2) Find active constraints and define the contraint matrix
        constraint_mat = [1 for i = 1:P ]' # sum to one constraint always active

        for i = 1:P
            if abundance[i] <= 0
                new_line = [ 0 for k = 1:P]'
                new_line[i] = -1
                constraint_mat = vcat(constraint_mat, new_line)
            end
        end
        constraint_mat = constraint_mat'

        # 3) calculate projection matrix and search direction

        projection_mat = Matrix(I, P, P) - constraint_mat * inv(constraint_mat' * constraint_mat) * constraint_mat'
        gradient = gradient_cost(x, S, abundance)
        d = -projection_mat * gradient

        # 4) if d!=0
        if d != [0;0;0;0]
            α = 0
            dα = (1 / opnorm(transpose(S) * S))
            n_iter_1 = 0

            # Looking for αmax such that (abundance+αmax*d) is feasible
            while is_feasible(abundance + (α + dα) * d) == true && n_iter_1 < max_iter
                α += dα
                n_iter_1 += 1
            end

            # Looking for the optimal stepsize
            α, n_loops = backtracking(x, S, abundance, cost, gradient_cost, d, α)
            abundance = abundance + α * d
            append!(cost_evol, cost(x, S, abundance))
            # Return to step 2

        # 5) if d=0
        else
            # Set Lagrangian vector
            λ = -1 * inv(constraint_mat' * constraint_mat) * constraint_mat' * gradient

            # If λ[i] for an inequality constraints <0 : KKT are not verified.
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
    return abundance, cost_evol
end

function projected_gradient_antoine(X, S, max_iter=100)
    """Estimate the abundance matrix for each pixel of the image X, by estimating
    the abundance vector of each pixel.

    :param X: image matrix of shape (n_pixels, 255).
    :param S: source matrix of shape (255, 4).
    :param max_iter: maximum number of iterations.
    :param min_precision: minimum precision value to stop the algorithm.
    :return: the estimated image and the estimated abundance vector.
    """
    n1, n2, l = size(X)
    new_X = zeros(n1, n2, l)
    A = zeros(n1, n2, 4)
    # Iterate over each pixel vector
    for (i, j) in product(1:n1, 1:n2)
        x = X[i, j, :]
        # Estimate its abundance vector
        a, _ = projected_gradient_vector_antoine(x, S)
        new_X[i, j, :] = S * a
        A[i, j, :] = a
    end
    return new_X, A
end


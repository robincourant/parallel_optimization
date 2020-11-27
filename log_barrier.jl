using LinearAlgebra

include("./utils.jl")


function ϕ(a)
    """Compute the logarithmic barrier function."""
    ϕ = 0
    for i = 1:4
        ϕ += -1 * log(a[i])
    end
    return ϕ
end

function ∇ϕ(a)
    """Compute the gradient of the logarithmic barrier function."""
    ∇ϕ = -1 * [(1 / a[i]) for i = 1:4]
    return ∇ϕ
end

function ∇2ϕ(a)
    """Compute the hessian of the logarithmic barrier function."""
    ∇2ϕ = zeros(4, 4)
    for i = 1:4
        ∇2ϕ[i, i] = 1 / a[i]^2
    end
    return ∇2ϕ
end

function backtracking_log_barrier(x, S, pt, f, ∇f, ϕ, ∇ϕ, d, t)
    """Bactracking method to find the optimal `step_size`.

    :param pt: initial point
    :param f, ∇f: function to be minimized and its gradient
    :param ϕ, ∇ϕ: logarithmic barrier function and its gradient
    :param d: search direction
    :param t:log-barrier coef
    :return: the optimal step_size and the number of loops
    """
    # backtracking parameters
    α, β = 0.25, 0.7
    n = 0
    ρ = 1

    # `ϕ_defined` is used to avoid definition issue of the `ϕfunction`
    # at beginning of while loop
    ϕ_defined = true
    for i = 1:4
        if (pt+ρ*d)[i] < 0
            ϕ_defined = false
        end
    end

    while ϕ_defined == false ||
        (t * f(x, S, pt + ρ * d) + ϕ(pt + ρ * d)) >
        t * f(x, S, pt) + ϕ(pt) + sum(ρ * α * (t * ∇f(x, S, pt) + ∇ϕ(pt))' * d)
        if ϕ_defined == false
            ϕ_defined = true
            for i = 1:4
                if (pt+ρ*d)[i] < 0
                    ϕ_defined = false
                end
            end
        end
        ρ *= β
        n += 1
    end
    return ρ, n
end

function log_barrier_vector(x, S, max_iter = 100, min_precision = 1e-8)
    """Log-barrier method to minimize `||x - S * a||^2` where 'x' and 'a' are vectors.

    :param x: pixel vector, shape(255,1).
    :param S: source matrix, shape (255,P).
    :return: the estimated abundance vector and evolution of cost.
    """
    # Initialization
    a = get_feasible_point()
    loss = zeros(max_iter)
    loss[1] = get_function(x, S, a)

    # log_barrier parameters (to optimize?)
    t = 1
    α = 1.5
    μ = 15

    # Equality constraint
    constraint = [1 1 1 1]

    prec = 1.e-20
    n_iter = 1

    # Number of backtracking loops
    nb_loops_IP = 0

    # Initialize d otherwise d not define out of for loop
    d = 0
    while true && n_iter < max_iter
        for k = 1:μ
            # Computing the search direction
            mat_to_inv = vcat(
                hcat(t * get_hessian(x, S, a) + ∇2ϕ(a), constraint'),
                hcat(constraint, 0),
            )
            d = (-1*inv(mat_to_inv)*vcat(t * get_gradient(x, S, a) + ∇ϕ(a), 0))[1:4]
            # Computing optimal stepsize
            step_size, n_loops =
                backtracking_log_barrier(x, S, a, get_function, get_gradient, ϕ, ∇ϕ, d, t)
            a += step_size * d
            nb_loops_IP += n_loops

        end
        # Updating `t`
        t *= α

        n_iter += 1
        loss[n_iter] = get_function(x, S, a)
        # Stop condition
        if sum(-1 * (t * get_gradient(x, S, a) + ∇ϕ(a))' * d / 2) < prec
            break
        end
    end
    return a, loss
end
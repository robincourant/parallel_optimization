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

function ∇cost(x, S, Abundance)
    gradient = -S' * (x - S * Abundance)
    return gradient
end

function ∇2cost(x, S, Abundance)
    ∇2cost = S' * S
    return ∇2cost
end

function ϕ(a)
    """compute ϕ(a) for log-barrier method
    :param a: vector of size (P,1)
    :return ϕ: ϕ(a), float
    """
    P = 4
    ϕ = 0
    for i = 1:P
        ϕ += -1 * log(a[i])
    end
    return ϕ
end

function ∇ϕ(a)
    """compute ∇ϕ(a)
    :param a: vector of size (P,1)
    return ∇ϕ(a): size (P,1)
    """
    P = 4
    ∇ϕ = -1 * [(1 / a[i]) for i = 1:P]
    
    return ∇ϕ
end

function ∇2ϕ(a)
    """Compute ∇2ϕ(a)
    :param a: vector of size (P,1)
    :return ∇2ϕ: matrix of size (P,P)
    """
    P = 4
    ∇2ϕ = zeros(P, P)
    for i = 1:P
        ∇2ϕ[i,i] = 1 / a[i]^2
    end
    return ∇2ϕ
end

function backtracking_log_barrier(x, S, pt, f, ∇f, ϕ, ∇ϕ, d, t)
    """ bactracking to find the optimal step_size.

    :param pt: initial point
    :param f, ∇f: function to be minimized and its gradient
    :param ϕ, ∇ϕ: logarithmic barrier function and its gradient
    :param d: search direction
    :param t:log-barrier coef
    :return ρ: optimal step_size
    :return n: number of loops
    """
    P = 4
    # backtracking parameters
    α, β = 0.25, 0.1 
    n = 0 
    ρ = 0.5
    
    # ϕdefined is used to avoid definition issue of the ϕfunction at beginning of while loop
    ϕdefined = true
    for i = 1:P
        if (pt + ρ * d)[i] < 0
            ϕdefined = false
        end
    end
    
    while ϕdefined == false || (t * f(x, S, pt + ρ * d) + ϕ(pt + ρ * d)) > t * f(x, S, pt) + ϕ(pt) + sum(ρ * α * (t * ∇f(x, S, pt) + ∇ϕ(pt))' * d)
        if ϕdefined == false
            ϕdefined = true
            for i = 1:P
                if (pt + ρ * d)[i] < 0
                    ϕdefined = false
                end
            end
        end
        ρ *= β
        n += 1
    end
    return ρ, n
end

function log_barrier_vector(x, S, max_iter=100)
    """log barrier method to minimize ||x-S*a||^2 where 'x' and 'a' vectors
    
    :param x: pixel vector, shape(255,1)
    :param S: source matrix, shape (255,P)
    :return abundance: estimated abundance vector
    :return cost_evol: evolution of cost
    """
    P = 4
    # Initialization
    abundance = initialize_abundance(4, true, true)
    cost_evol = [cost(x, S, abundance)]
    
    # log_barrier parameters (to optimize?)
    t = 1
    α = 1.5
    μ = 15
    
    # Equality constraint
    constraint = [1 1 1 1] 
    

    prec        = 1.e-20
    iter = 0  
    
    nb_loops_IP = 0  # number of backtracking loops

    # initialize d otherwise d not define out of for loop (we love julia)
    d = 0
    while true && iter < max_iter
        iter += 1
        for k = 1:μ
            
            # Computing the search direction
            mat_to_inv = vcat(hcat(t * ∇2cost(x, S, abundance) + ∇2ϕ(abundance), constraint'), hcat(constraint, 0))
            d = (-1 * inv(mat_to_inv) * vcat(t * ∇cost(x, S, abundance) + ∇ϕ(abundance), 0))[1:P]
            # Computing optimal stepsize
            step_size, n_loops = backtracking_log_barrier(x, S, abundance, cost, ∇cost, ϕ, ∇ϕ, d, t)
            abundance += step_size * d
            nb_loops_IP += n_loops
            append!(cost_evol, cost(x, S, abundance))
            
        end
        
        # Updating t
        t *= α

        # Stop condition
        if sum(-1 * (t * ∇cost(x, S, abundance) + ∇ϕ(abundance))' * d / 2) < prec 
            break
        end
    end
    return abundance, cost_evol
end

function log_barrier(X, S, max_iter=100)
    """Estimate the abundance matrix for each pixel of the image X, by estimating
    the abundance vector of each pixel.
    Using the log_barrier method

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
        a, cost_evol = log_barrier_vector(x, S, 100)
        A = [A a]
    end
    return A
end
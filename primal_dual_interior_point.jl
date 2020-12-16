include("./utils.jl")

function update_values(x, λ, ν, Δy, stepsize, b, A, μ, H, flat_image, S)
    p = size(S)[2]
    l, n = size(flat_image)
    pn = p * n

    # Compute primal-dual update direction
    Δx = Δy[1:pn]
    Δλ = Δy[pn+1:2*pn]
    Δν = Δy[2*pn+1:end]

    # Update variables
    x += stepsize * Δx
    X = Diagonal(x)
    λ += stepsize * Δλ
    Λ = Diagonal(λ)
    ν += stepsize * Δν

    Df = -Matrix(I, pn, pn)
    ∇f0 = reshape(get_gradient(flat_image, S, reshape(x, p, n)), pn)

    η = x' * λ
    t = μ * pn / η

    r_cent = ∇f0 + (Df' * λ) + (A' * ν)
    r_dual = Λ * x - (1 / t) * ones(pn)
    r_prim = A * x - b

    M = [
        H Df' A'
        -Λ*Df -X zeros(pn, n)
        A zeros(n, pn + n)
    ]

    return x, λ, ν, r_cent, r_dual, r_prim, M
end

function backtracking_pdip(x, λ, ν, r, Δy, b, A, μ, H, flat_image, S)
    pn, = size(λ)

    # Compute the largest positive step length that give `new_λ` > 0
    k_step = 0.01
    s_max = k_step
    while s_max <= 1
        new_λ = λ + (s_max + k_step) * Δy[pn+1:2*pn]
        if any(x -> x <= 0, new_λ)
            break
        end
        s_max += k_step
    end
    new_λ = λ + s_max * Δy[pn+1:2*pn]

    stepsize = 0.99 * s_max
    α, β = 0.1, 0.3
    new_x, _, _, new_r_cent, new_r_dual, new_r_prim, _ =
        update_values(x, λ, ν, Δy, stepsize, b, A, μ, H, flat_image, S)
    new_r = [new_r_cent; new_r_dual; new_r_prim]
    # Inequality constraint condition
    while any(x -> x <= 0, new_x)
        stepsize *= β
        new_x, _, _, new_r_cent, new_r_dual, new_r_prim, _ =
            update_values(x, λ, ν, Δy, stepsize, b, A, μ, H, flat_image, S)
        new_r = [new_r_cent; new_r_dual; new_r_prim]
    end

    while (norm(new_r) > (1 - α * stepsize) * norm(r))
        stepsize *= β
        _, _, _, new_r_cent, new_r_dual, new_r_prim, _ =
            update_values(x, λ, ν, Δy, stepsize, b, A, μ, H, flat_image, S)
        new_r = [new_r_cent; new_r_dual; new_r_prim]
    end
    return stepsize
end

function primal_dual_interior_point(flat_image, S, max_iter)
    p = size(S)[2]
    l, n = size(flat_image)
    pn = p * n

    x = reshape(get_feasible_point_matrix(p, n), pn)
    X = Diagonal(x)
    λ = 1 ./ x
    Λ = Diagonal(λ)
    ν = ones(n)

    ∇f0 = reshape(get_gradient(flat_image, S, reshape(x, p, n)), pn)
    Df = -Matrix(I, pn, pn)
    A = get_constrain_matrix(n, p)
    b = ones(n)

    μ = 10
    η = x' * λ
    t = μ * pn / η

    r_cent = ∇f0 + (Df' * λ) + (A' * ν)
    r_dual = Λ * x - (1 / t) * ones(pn)
    r_prim = A * x - b
    r = [r_cent; r_dual; r_prim]

    S_ = kron(Matrix(I, n, n), S)
    H = S_' * S_

    M = [
        H Df' A'
        -Λ*Df -X zeros(pn, n)
        A zeros(n, pn + n)
    ]

    n_iter = 0
    ϵ_fea = 1e-8
    ϵ = 1e-8
    loss = zeros(max_iter)
    loss[1] = get_function(flat_image, S, reshape(x, p, n))
    while n_iter < max_iter
        # Compute primal-dual update direction
        Δy = -inv(M) * r

        # Compute the optimal stepsize
        stepsize = backtracking_pdip(x, λ, ν, r, Δy, b, A, μ, H, flat_image, S)
        # println(stepsize)

        # Update values
        x, λ, ν, r_cent, r_dual, r_prim, M =
            update_values(x, λ, ν, Δy, stepsize, b, A, μ, H, flat_image, S)
        r = [r_cent; r_dual; r_prim]
        η = x' * λ

        # Stopping criterion
        if (norm(r_prim) < ϵ_fea) && (norm(r_dual) < ϵ_fea) && (η < ϵ)
            break
        end

        n_iter += 1
        # Compute loss
        round_loss = get_function(flat_image, S, reshape(x, p, n))
        loss[n_iter] = round_loss
    end
    return x, loss
end
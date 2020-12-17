include("../utils.jl")

function vector_to_image(S, a, l, p, n)
    """Transform vector to matrix and image."""
    k = floor(Int, sqrt(n))
    A = reshape(a, p, n)
    new_X = permutedims(reshape(S * reshape(a, p, n), l, k, k), [2, 3, 1])

    return new_X, A
end


function get_sumtoone_null(p)
    """
    Initialize = the sum-to-one constraint, a null space matrix of size (p, p-1)
    is given by:
                        Zij =
                           | - 1 if i = j,
                           | - −1 if i = j + 1,
                           | - 0 otherwise.
    """
    z = zeros(p - 1)
    z[p-1] = -1
    Z = hcat(Bidiagonal(ones(p - 1), -ones(p - 2), :U), z)'
    return Z
end


function update_directions(u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p)
    """Update search directions."""
    # Update directions
    d_u = inv(∇2ϕ + T' * inv_A_diag * Λ * T) * (-∇ϕ + T' * inv_A_diag * Mu)
    d_λ = inv_A_diag * (Mu - Λ * (T * (u + d_u) + a0))

    # Update variables
    u += α * d_u
    U = reshape(u, p - 1, n)
    A = T * u + a0
    inv_A_diag = inv(Diagonal(A))
    ∇ϕ = get_gradient_vector(X, S, A0, U, Z)

    λ += α * d_λ
    Λ = Diagonal(λ)

    δ = A' * λ

    return u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ
end

function Ψ(ϕu, A, λ, μ)
    s1 = sum(log.(A))
    s2 = sum(log.(λ .* A))

    return ϕu - μ * (s1 + s2) + (λ' * A)
end


function ∇Ψ(∇ϕ, T, A, inv_A_diag, λ, Λ, μ)
    pn = size(T)[1]

    ∇Ψ_λ = A - (μ ./ λ)
    ∇Ψ_u = ∇ϕ + (ones(1, pn) * (Λ - (2 * μ * inv_A_diag)) * T)'

    return [∇Ψ_λ; ∇Ψ_u]
end


function backtracking_ipls(λ, Λ, d_λ, u, U, d_u, Z, X, S, a0, A, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, μ, Mu, θ)
    pn, = size(λ)
    p, n = size(A0)

    # Compute the largest positive step length that give `new_λ` > 0
    k_step = 0.01
    s_max = k_step
    while s_max <= 1
        new_λ = λ + (s_max + k_step) * d_λ
        if any(x -> x <= 0, new_λ)
            break
        end
        s_max += k_step
    end
    α = 0.99 * s_max
    σ, τ = 0.01, 0.3

    d = [d_λ; d_u]
    Ψ0 = Ψ(get_function(X, S, A0 + Z * U), A, λ, μ)
    ∇Ψ0 = ∇Ψ(∇ϕ, T, A, inv_A_diag, λ, Λ, μ)


    _, new_U, new_A, _, _, new_λ, _, new_δ =
        update_directions(u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p)
    new_μ, _ = update_pertubation(θ, new_δ, p, n)
    new_Ψ = Ψ(get_function(X, S, A0 + Z * new_U), new_A, new_λ, new_μ)

    while new_Ψ > Ψ0 + (σ * α * d' * ∇Ψ0)[1]
        α *= τ
        _, new_U, new_A, _, _, new_λ, _, new_δ =
            update_directions(u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p)
        new_μ, _ = update_pertubation(θ, new_δ, p, n)
        new_Ψ = Ψ(get_function(X, S, A0 + Z * new_U), new_A, new_λ, new_μ)
    end
    return α
end

function update_pertubation(θ, δ, p, n)
    """Update pertubation parameter."""
    μ = (θ * δ) / (p * n)
    Mu = μ * ones(p * n)
    return μ, Mu
end

function interior_point_least_square(X, S, max_iter, min_precision)
    """"
    Image-based unmixing method with interior point least square method.
    cf: https://hal.archives-ouvertes.fr/hal-00828013/document
    """
    p = size(S)[2]
    l, n = size(X)

    # Variable to minimize with the reparametrization:
    # A = A0 - ZU
    U = zeros(p - 1, n)
    u = reshape(U, (p - 1) * n)

    Z = get_sumtoone_null(p)
    T = kron(Matrix(I, n, n), Z)

    A0 = get_feasible_point_matrix(p, n)
    a0 = reshape(A0, p * n)

    A = T * u + a0
    inv_A_diag = inv(Diagonal(A))

    ∇ϕ = get_gradient_vector(X, S, A0, U, Z)
    ∇2ϕ = get_hessian_vector(X, S, Z)

    λ = reshape(get_feasible_point_matrix(p, n), p * n)
    Λ = Diagonal(λ)

    # Parameter initialization
    θ = 0.5
    δ = A' * λ
    μ = (θ * δ) / (p * n)
    Mu = μ * ones(p * n)
    μ_min = 1e-9  # > 0
    # Primal Newton direction
    r_prim = ∇ϕ - T' * λ
    η_prim = 5  # > 0
    ϵ_prim = η_prim * μ
    # Dual Newton direction
    r_dual = δ / (p * n)
    η_dual = 1.9  # In [1, 1/θ]
    ϵ_dual = η_dual * μ

    # Initial directions
    d_u = inv(∇2ϕ + T' * inv_A_diag * Λ * T) * (-∇ϕ + T' * inv_A_diag * Mu)
    d_λ = inv_A_diag * (Mu - Λ * (T * (u + d_u) + a0))

    n_iter_1 = 0
    # Initialize loss
    loss = zeros(max_iter)
    loss[1] = get_function(X, S, A0 + Z * U)
    while (μ > μ_min) && (n_iter_1 < max_iter)

        n_iter_2 = 0
        while (maximum(r_prim) > ϵ_prim) && (r_dual > ϵ_dual) && (n_iter_2 < max_iter)
            # Line search
            α = backtracking_ipls(
                λ, Λ, d_λ, u, U, d_u, Z, X, S, a0, A, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, μ, Mu, θ
            )
            # Update search directions and parameters
            u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ = update_directions(
                u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p
            )
            # Update primal Newton direction
            r_prim = ∇ϕ - T' * λ

            n_iter_2 += 1
        end
        # Update perturbation parameters
        μ, Mu = update_pertubation(θ, δ, p, n)
        # Line search
        α = backtracking_ipls(
            λ, Λ, d_λ, u, U, d_u, Z, X, S, a0, A, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, μ, Mu, θ
        )
        # Update search directions and parameters
        u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ = update_directions(
            u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p
        )

        # Update primal and dual Newton directions
        r_prim = ∇ϕ - T' * λ
        r_dual = δ / (p * n)

        n_iter_1 += 1
        # Compute loss
        round_loss = get_function(X, S, A0 + Z * U)
        loss[n_iter_1] = round_loss
    end

    return (A0 + Z * U), loss
end
include("./utils.jl")

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


function backtracking_ipls(α, d_λ, d_u, u, U, T, a0, X, S, A0, A, inv_A_diag, Z, ∇ϕ, λ, Λ, μ)
    """Bactracking method to find optimal `step_size`."""
    n_loops = 0
    # Backtracking parameters
    τ = 0.8
    σ = 0.5

    d = [d_λ; d_u]
    Ψ0 = Ψ(get_function(X, S, A0 + Z * U), A, λ, μ)
    ∇Ψ0 = ∇Ψ(∇ϕ, T, A, inv_A_diag, λ, Λ, μ)

    u1, U1, A1, inv_A_diag1, λ1, Λ1 = update_variables(α, d_λ, d_u, u, T, a0, X, S, A0, Z, λ)
    Ψα = Ψ(get_function(X, S, A0 + Z * U1), A1, λ1, μ)
    while Ψα > (Ψ0 + σ * α * (d'*∇Ψ0)[1])
        # while Ψα > (Ψ0 + σ * α * sum(∇Ψ0))
        α = τ * α
        u1, U1, A1, inv_A_diag1, λ1, Λ1 = update_variables(α, d_λ, d_u, u, T, a0, X, S, A0, Z, λ)
        Ψα = Ψ(get_function(X, S, A0 + Z * U1), A1, λ1, μ)
        n_loops += 1
    end
    println(n_loops)
    return α, n_loops
end


function update_variables(α, d_λ, d_u, u, T, a0, X, S, A0, Z, λ)
    u += α * d_u
    U = reshape(u, p - 1, n)
    A = T * u + a0
    inv_A_diag = inv(Diagonal(A))
    ∇ϕ = get_gradient_vector(X, S, A0, U, Z)

    λ += α * d_λ
    Λ = Diagonal(λ)

    return u, U, A, inv_A_diag, λ, Λ
end

function get_new_step(u, U, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, A, inv_A_diag, λ, Λ, μ, Mu)
    """Update search directions. step size and variables."""
    # Update directions
    d_u = inv(∇2ϕ + T' * inv_A_diag * Λ * T) * (-∇ϕ + T' * inv_A_diag * Mu)
    # d_λ = inv_A_diag * (Mu - Λ * (T * (u + d_u) + a0))
    d_λ = inv_A_diag * (Mu - Λ * (T * u + a0) - Λ * T * d_u)
    # println(d_λ == inv_A_diag * (Mu - Λ * (T * (u + d_u) + a0)))
    # Find an initil step size
    α_plus = 0.01
    while α_plus < 1
        if any(x -> x .<= 0, λ + (α_plus + 0.01) * d_λ) ||
           any(x -> x .<= 0, A + (α_plus + 0.01) * T * d_u)
            break
        end
        α_plus += 0.01
    end
    α = min(1, 0.99 * α_plus)
    println("α1 ", α)
    # Update step_size
    α, _ = backtracking_ipls(α, d_λ, d_u, u, U, T, a0, X, S, A0, A, inv_A_diag, Z, ∇ϕ, λ, Λ, μ)
    println("α2 ", α, "\n")
    # Update variables
    u, U, A, inv_A_diag, λ, Λ = update_variables(α, d_λ, d_u, u, T, a0, X, S, A0, Z, λ)
    δ = A' * λ

    return u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ
end


function update_pertubation(θ, δ, p, n)
    """Update pertubation parameter."""
    μ = (θ * δ) / (p * n)
    Mu = μ * ones(p * n)
    return μ, Mu
end



function interior_point_least_square(X, S, max_iter = 100, min_precision = 1e-8)
    """"
    Image-based unmixing method with interior point least square method.
    cf: https://hal.archives-ouvertes.fr/hal-00828013/document
    """
    p = size(S)[2]
    n1, n2, _ = size(X)
    n = n1 * n2
    X = reshape(X, n1 * n2, l)'

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
    # λ = 2 * ones(p * n) # ./ (p * n)
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

    n_iter_1 = 0
    # Initialize loss
    loss = zeros(max_iter)
    loss[1] = get_function(X, S, A0 + Z * U)
    while (μ > μ_min) && (n_iter_1 < max_iter)

        n_iter_2 = 0
        while (maximum(r_prim) > ϵ_prim) && (r_dual > ϵ_dual) && (n_iter_2 < max_iter)
            # Update search directions, step size and variables
            u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ =
                get_new_step(u, U, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, A, inv_A_diag, λ, Λ, μ, Mu)

            # Update primal Newton direction
            r_prim = ∇ϕ - T' * λ
            # r_dual = δ / (p * n)
            n_iter_2 += 1
        end
        # Update perturbation parameters
        μ, Mu = update_pertubation(θ, δ, p, n)
        # Update search directions, step size and variables
        u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ =
            get_new_step(u, U, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, A, inv_A_diag, λ, Λ, μ, Mu)

        # Update primal and dual Newton directions
        r_prim = ∇ϕ - T' * λ
        r_dual = δ / (p * n)

        n_iter_1 += 1
        # Compute loss
        round_loss = get_function(X, S, A0 + Z * U)
        loss[n_iter_1] = round_loss
        if round_loss < min_precision
            break
        end
    end
    return (A0 + Z * U), loss
end
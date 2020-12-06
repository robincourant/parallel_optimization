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


function update_pertubation(θ, δ, p, n)
    """Update pertubation parameter."""
    μ = (θ * δ) / (p * n)
    Mu = μ * ones(p * n)
    return μ, Mu
end


function Ψ(ϕu, A, λ, μ)
    s1 = sum(log.(A))
    s2 = sum(log.(λ .* A))

    return ϕu - μ * (s1 + s2) + (λ' * A)
end


function ∇Ψ(∇ϕu, T, A, λ, μ)
    n0 = size(T)[1]

    ∇Ψ_λ = A - (μ ./ λ)
    ∇Ψ_u = ∇ϕu + (λ' * T)' - 2 * μ * (ones(n0)' * (T ./ A))'

    return [∇Ψ_λ; ∇Ψ_u]
end


function backtracking_ipls(Ψα, Ψ0, ∇Ψ0, α)
    """Bactracking method to find optimal `step_size`."""
    n_loops = 0
    # Backtracking parameters
    τ = 0.25
    σ = 0.7

    println(sum(∇Ψ0), "\n")
    println(Ψα, "\n")
    println(Ψ0, "\n")
    println(Ψα - Ψ0, "\n")
    println(σ * α * sum(∇Ψ0), "\n")
    println("\n")
    while (Ψα - Ψ0) > (σ * α * sum(∇Ψ0))
        α = τ * α
        n_loops += 1
    end
    return α, n_loops
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
    # Step size
    α = 0.1

    n_iter_1 = 0
    # Initialize loss
    loss = zeros(max_iter)
    loss[1] = get_function(X, S, A0 + Z * U)
    while (μ > μ_min) && (n_iter_1 < max_iter)

        n_iter_2 = 0
        while (maximum(r_prim) > ϵ_prim) && (r_dual > ϵ_dual) && (n_iter_2 < max_iter)
            Ψ0 = Ψ(get_function(X, S, A0 + Z * U), A, λ, μ)
            ∇Ψ0 = ∇Ψ(∇ϕ, T, A, λ, μ)

            # Update search directions and parameters
            u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ =
                update_directions(u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p)

            Ψα = Ψ(get_function(X, S, A0 + Z * U), A, λ, μ)
            # Update stepsize
            α, _ = backtracking_ipls(Ψα, Ψ0, ∇Ψ0, α)
            println(n_iter_2, α)
            # Update primal Newton direction
            r_prim = ∇ϕ - T' * λ

            n_iter_2 += 1
        end
        # Update perturbation parameters
        μ, Mu = update_pertubation(θ, δ, p, n)
        # Update search directions and parameters
        u, U, A, inv_A_diag, ∇ϕ, λ, Λ, δ =
            update_directions(u, Z, X, S, a0, A0, ∇ϕ, ∇2ϕ, T, inv_A_diag, α, λ, Λ, Mu, n, p)

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
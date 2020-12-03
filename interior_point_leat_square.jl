include("./utils.jl")

function get_sumtoone_null(p)
    z = zeros(p - 1)
    z[p-1] = -1
    Z = hcat(Bidiagonal(ones(p - 1), -ones(p - 2), :U), z)'
    return Z
end


function interior_point_least_square(X, S, max_iter = 100)
    p = size(S)[2]
    n = size(X)[2]

    # Variable initialization
    U = zeros(p - 1, n) # get_feasible_point_matrix(p - 1, n)
    u = reshape(U, (p - 1) * n)
    Z = get_sumtoone_null(p)
    T = kron(Matrix(I, n, n), Z)
    A0 = get_feasible_point_matrix(p, n)
    a0 = reshape(A0, p * n)
    A = T * u + a0
    A_diag = Diagonal(A)
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

    r_prim = ∇ϕ - T' * λ
    η_prim = 5  # > 0
    ϵ_prim = η_prim * μ

    r_dual = δ / (p * n)
    η_dual = 1.9  # In [1, 1/θ]
    ϵ_dual = η_dual * μ

    α = 0.1

    loss = zeros(max_iter)
    loss[1] = get_function(X, S, A0 + Z * U)

    # First condition
    n_iter_1 = 0
    while (μ > μ_min) && (n_iter_1 < max_iter)
        # Second condition
        n_iter_2 = 0
        # while (maximum(r_prim) > ϵ_prim) && (r_dual > ϵ_dual) && (n_iter_2 < max_iter)
        while (maximum(∇ϕ - T' * λ) > ϵ_prim) &&
                  ((δ / (p * n)) > ϵ_dual) &&
                  (n_iter_2 < max_iter)
            # Update directions
            d_u = inv(∇2ϕ + T' * inv(A_diag) * Λ * T) * (-∇ϕ + T' * inv(A_diag) * Mu)
            d_λ = inv(A_diag) * (Mu - Λ * (T * u + a0) - (Λ * T * d_u))

            # Update variables
            u += α * d_u
            U = reshape(u, p - 1, n)
            A = T * u + a0
            A_diag = Diagonal(A)
            ∇ϕ = get_gradient_vector(X, S, A0, U, Z)

            λ += α * d_λ
            Λ = Diagonal(λ)

            r_prim = ∇ϕ - T' * λ
            n_iter_2 += 1
        end

        δ = A' * λ
        r_dual = δ / (p * n)

        μ = (θ * δ) / (p * n)
        Mu = μ * ones(p * n)

        # Update directions
        d_u = inv(∇2ϕ + T' * inv(A_diag) * Λ * T) * (-∇ϕ + T' * inv(A_diag) * Mu)
        d_λ = inv(A_diag) * (Mu - Λ * (T * u + a0) - (Λ * T * d_u))

        # Update variables
        u += α * d_u
        U = reshape(u, p - 1, n)
        A = T * u + a0
        A_diag = Diagonal(A)
        ∇ϕ = get_gradient_vector(X, S, A0, U, Z)

        λ += α * d_λ
        Λ = Diagonal(λ)

        r_prim = ∇ϕ - T' * λ
        n_iter_2 += 1

        n_iter_1 += 1
        round_loss = get_function(X, S, A0 + Z * U)
        loss[n_iter_1] = round_loss
    end
    return (A0 + Z * U), loss
end
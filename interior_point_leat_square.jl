include("./utils.jl")


function interior_point_least_square(X, S, max_iter = 100)
    p = size(S)[2]
    n = size(X)[2]

    # Initialization
    A0 = get_feasible_point_matrix(p, n)
    U = zeros(p - 1, n) # get_feasible_point_matrix(p - 1, n)
    z = zeros(p - 1)
    z[p-1] = -1
    Z = hcat(Bidiagonal(ones(p - 1), -ones(p - 2), :U), z)'
    ZU = Z * U
    A = A0 + Z * U

    # Vectorize A and U
    a = reshape(A, n * p)

    λ = reshape(get_feasible_point_matrix(p, n), p * n)
    Λ = Diagonal(λ)
    A_diag = Diagonal(a)

    θ = 0.5
    δ = a' * λ
    μ = (θ * δ) / (p * n)
    Mu = μ * ones(p * n)
    μ_min = 1e-9  # > 0

    ∇ϕ = -get_gradient_vector(X, S, ZU)
    ∇2ϕ = get_hessian_vector(X, S, U)
    r_prim = ∇ϕ - λ
    η_prim = 100  # > 0
    ϵ_prim = η_prim * μ

    η_dual = 1.9  # In [1, 1/θ]
    ϵ_dual = η_dual * μ

    α = 0.1

    loss = zeros(max_iter)
    loss[1] = get_function(X, S, A)

    println("max ", maximum(r_prim))
    println("ϵ_prim ", ϵ_prim)
    println("delta ", δ / (p * n))
    println("ϵ_dual ", ϵ_dual)
    # First condition
    n_iter_1 = 0
    while (μ > μ_min) && (n_iter_1 < max_iter)
        # Second condition
        n_iter_2 = 0
        while (maximum(r_prim) > ϵ_prim) && (δ / (p * n) > ϵ_dual) && (n_iter_2 < max_iter)
            ∇ϕ = -get_gradient_vector(X, S, ZU)
            d_a = inv(∇2ϕ + inv(A_diag) * Λ) * (-∇ϕ + inv(A_diag) * Mu)
            d_λ = inv(A_diag) * (Mu - Λ * (a + d_a))

            a += α * d_a
            A_diag = Diagonal(a)
            ZU += reshape(α * d_a, p, n)
            λ += α * d_λ
            Λ = Diagonal(λ)

            n_iter_2 += 1
            println(d_a)
            println("#########")
        end

        δ = a' * λ
        μ = (θ * δ) / (p * n)
        Mu = μ * ones(p * n)

        n_iter_1 += 1
        round_loss = get_function(X, S, reshape(a, p, n))
        loss[n_iter_1] = round_loss
    end
    return a, loss
end
using SpecializingFactorizations
using LinearAlgebra
using BenchmarkTools

const SUITE = BenchmarkGroup()

const N = 512

# A representative dense `Matrix` per detected form (well-conditioned, nonsingular).
function _matrix(form::Symbol, n::Int)
    if form === :diagonal
        return diagm(0 => 2.0 .+ (1:n) ./ n)
    elseif form === :tridiagonal
        return diagm(0 => fill(4.0, n), 1 => fill(-0.5, n - 1), -1 => fill(0.5, n - 1))
    elseif form === :banded
        return diagm(
            0 => fill(8.0, n), 1 => fill(0.5, n - 1), -1 => fill(0.5, n - 1),
            2 => fill(0.25, n - 2), -2 => fill(0.25, n - 2),
        )
    elseif form === :uppertriangular
        U = [i <= j ? 0.3 * sin(i + 2j) : 0.0 for i in 1:n, j in 1:n]
        for k in 1:n
            U[k, k] = 3.0 + k / n
        end
        return U
    elseif form === :spd
        M = [0.2 * cos(i * 0.5 + j) for i in 1:n, j in 1:n]
        return Matrix(M * M' + n * I)
    elseif form === :symindef
        return [i == j ? ((-1)^i) * float(n) : 0.3 * cos(i + j) for i in 1:n, j in 1:n]
    else # :general
        return [i == j ? float(n) : 0.3 * sin(i + 2j) for i in 1:n, j in 1:n]
    end
end

const FORMS = (:diagonal, :tridiagonal, :banded, :uppertriangular, :spd, :symindef, :general)

SUITE["factorize"] = BenchmarkGroup()   # detect + specialized factorization (specializinglu)
SUITE["solve"] = BenchmarkGroup()       # warm in-place solve (ldiv!) — must stay 0-alloc
SUITE["full"] = BenchmarkGroup()        # detect + factor + solve, from scratch (\)

for form in FORMS
    A = _matrix(form, N)
    b = ones(N)
    x = similar(b)
    F = specializinglu(A)               # pre-factored workspace for the warm-solve benchmark
    SUITE["factorize"][string(form)] = @benchmarkable specializinglu($A)
    SUITE["solve"][string(form)] = @benchmarkable ldiv!($x, $F, $b)
    SUITE["full"][string(form)] = @benchmarkable specializinglu($A) \ $b
end

# the O(n²) structure-detection scan (early-exit vs full symmetric scan)
SUITE["detect"] = BenchmarkGroup()
let Agen = _matrix(:general, N), Asym = _matrix(:spd, N)
    SUITE["detect"]["general"] = @benchmarkable detect_form($Agen)
    SUITE["detect"]["symmetric"] = @benchmarkable detect_form($Asym)
end

# Rank-revealing QR: full-rank overdetermined / underdetermined and a
# rank-deficient case, across factorization, warm solve (0-alloc), and full.
SUITE["qr_factorize"] = BenchmarkGroup()
SUITE["qr_solve"] = BenchmarkGroup()       # warm in-place ldiv! — must stay 0-alloc
SUITE["qr_full"] = BenchmarkGroup()

let cases = (
        "over_full" => randn(N, N ÷ 2),
        "under_full" => randn(N ÷ 2, N),
        "deficient" => randn(N, 8) * randn(8, N),   # N×N, rank 8
    )
    for (name, A) in cases
        m, n = size(A)
        b = ones(m)
        x = Vector{Float64}(undef, n)
        F = specializingqr(A)
        SUITE["qr_factorize"][name] = @benchmarkable specializingqr($A)
        SUITE["qr_solve"][name] = @benchmarkable ldiv!($x, $F, $b)
        SUITE["qr_full"][name] = @benchmarkable specializingqr($A) \ $b
    end
end

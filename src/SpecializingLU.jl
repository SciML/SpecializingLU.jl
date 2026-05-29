module SpecializingLU

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasInt
using LinearAlgebra.LAPACK: gttrf!, gttrs!, gbtrf!, gbtrs!,
    potrf!, potrs!, sytrf!, sytrs!, hetrf!, hetrs!, getrf!, getrs!

export MatrixForm,
    GENERAL, DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR,
    LOWER_BIDIAGONAL, UPPER_BIDIAGONAL, TRIDIAGONAL, BANDED,
    SYMMETRIC_POSITIVE_DEFINITE, SYMMETRIC_INDEFINITE, HERMITIAN_INDEFINITE
export SpecializedLU, specializinglu, specializinglu!, detect_form, DetectionResult,
    matrixform, issuccess, isfactored

# ---------------------------------------------------------------------------
# Matrix form taxonomy
# ---------------------------------------------------------------------------

"""
    MatrixForm

Enum tagging the structural form of a dense matrix that determines which
specialized factorization/solve is used. Using an enum (a runtime `Int8`
value) rather than distinct Julia *types* is what keeps the whole pipeline
type-stable: [`specializinglu`](@ref) always returns the same concrete
[`SpecializedLU`](@ref) type regardless of the detected form, and the solve
branches on this enum at runtime while every branch remains inferable.

The structural forms (`DIAGONAL`, `*_TRIANGULAR`, `*_BIDIAGONAL`,
`TRIDIAGONAL`, `BANDED`, `GENERAL`) are produced by the cheap O(n)/O(n²)
[`detect_form`](@ref) scan. The symmetric forms
(`SYMMETRIC_POSITIVE_DEFINITE`, `SYMMETRIC_INDEFINITE`,
`HERMITIAN_INDEFINITE`) are *resolved during factorization* (a Cholesky
attempt decides definiteness) and only for `BlasFloat` element types.
"""
@enum MatrixForm::Int8 begin
    GENERAL = 0
    DIAGONAL = 1
    LOWER_TRIANGULAR = 2
    UPPER_TRIANGULAR = 3
    LOWER_BIDIAGONAL = 4
    UPPER_BIDIAGONAL = 5
    TRIDIAGONAL = 6
    BANDED = 7
    SYMMETRIC_POSITIVE_DEFINITE = 8
    SYMMETRIC_INDEFINITE = 9
    HERMITIAN_INDEFINITE = 10
end

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

"""
    DetectionResult

The cheap structural classification of a matrix, returned by
[`detect_form`](@ref). `form` is the structural [`MatrixForm`](@ref) (never
one of the resolved symmetric forms — those require the factorization).
`kl`/`ku` are the detected lower/upper bandwidths, and `issym`/`isherm`
flag exact symmetry/Hermitian-ness (relevant only when `form == GENERAL`,
signalling that a Cholesky/Bunch-Kaufman specialization may apply).
"""
struct DetectionResult
    form::MatrixForm
    kl::Int
    ku::Int
    issym::Bool
    isherm::Bool
end

default_bandwidth_cutoff(n::Int) = max(16, n ÷ 4)

"""
    detect_form(A; bandwidth_cutoff = max(16, n ÷ 4)) -> DetectionResult

Classify the structural form of the square dense matrix `A` in a single
O(n²) column-major pass. Detection is
*structural*: a position counts as a band entry iff it is exactly nonzero
(`!iszero`). This mirrors the semantics of structured array types in
`LinearAlgebra` and `SparseMatrixIdentification.jl`, but returns a runtime
enum instead of a type.

The same pass computes the lower/upper bandwidths and whether `A` is
symmetric / Hermitian, so a dense symmetric matrix can later be routed to
Cholesky or Bunch–Kaufman. A matrix is classified `BANDED` only when its
band is genuinely narrow (`kl + ku + 1 <= bandwidth_cutoff`); wider bands
fall through to the symmetric/general dense path.
"""
function detect_form(A::AbstractMatrix; bandwidth_cutoff::Integer = -1)
    m, n = size(A)
    m == n || throw(DimensionMismatch("detect_form requires a square matrix, got $(m)×$(n)"))
    cutoff = bandwidth_cutoff < 0 ? default_bandwidth_cutoff(n) : Int(bandwidth_cutoff)

    kl = 0
    ku = 0
    issym = true
    isherm = true

    # Single column-major pass: lower/upper bandwidth from the nonzero
    # pattern, plus exact symmetry / Hermitian-ness from the i<j pairs.
    @inbounds for j in 1:n
        for i in 1:n
            aij = A[i, j]
            if !iszero(aij)
                d = i - j
                if d > 0
                    d > kl && (kl = d)
                elseif d < 0
                    -d > ku && (ku = -d)
                end
            end
            if i < j
                aji = A[j, i]
                issym &= (aij == aji)
                isherm &= (aij == conj(aji))
            elseif i == j
                # A Hermitian matrix must have a real diagonal; a complex
                # symmetric matrix (issym) imposes no diagonal constraint.
                isherm &= isreal(aij)
            end
        end
        # Early-out for the common unstructured case: once a nonzero band on
        # both sides rules out triangular, the band is wider than `cutoff`
        # (so not banded), and both symmetry flags have already failed, the
        # only remaining classification is GENERAL — no need to finish the
        # scan. These conditions are monotonic, so this is safe.
        if kl > 0 && ku > 0 && (kl + ku + 1) > cutoff && !issym && !isherm
            return DetectionResult(GENERAL, kl, ku, false, false)
        end
    end

    form = _classify(kl, ku, n, cutoff)
    return DetectionResult(form, kl, ku, issym, isherm)
end

function _classify(kl::Int, ku::Int, n::Int, cutoff::Int)
    if kl == 0 && ku == 0
        return DIAGONAL
    elseif ku == 0
        return kl == 1 ? LOWER_BIDIAGONAL : LOWER_TRIANGULAR
    elseif kl == 0
        return ku == 1 ? UPPER_BIDIAGONAL : UPPER_TRIANGULAR
    elseif kl == 1 && ku == 1
        return TRIDIAGONAL
    elseif (kl + ku + 1) <= cutoff && (kl + ku + 1) < n
        return BANDED
    else
        # Wide band / full. The symmetric refinement (Cholesky vs
        # Bunch-Kaufman) happens at factorization time and only for
        # BlasFloat; here we just report GENERAL plus the symmetry flags.
        return GENERAL
    end
end

# ---------------------------------------------------------------------------
# The single, type-stable workspace
# ---------------------------------------------------------------------------

"""
    SpecializedLU{T,R}

A single, concrete, reusable workspace holding a specialized factorization of
a dense matrix. Its Julia type is fixed (it does not depend on the detected
[`MatrixForm`](@ref)), so constructing and solving with it is type-stable.
The active `form` field selects, at runtime, which subset of the buffers is
meaningful and which specialized solve is dispatched.

Buffers are sized lazily: a `DIAGONAL` or `TRIDIAGONAL` matrix never
allocates the O(n²) `fact` buffer, and a reusable workspace only grows a
buffer when a new matrix needs more space than the last.

`T` is the element type; `R = real(T)` is used for the real diagonal that the
positive-definite tridiagonal routines require.
"""
mutable struct SpecializedLU{T, R}
    form::MatrixForm
    n::Int
    kl::Int
    ku::Int
    uplo::Char           # 'U'/'L' for Cholesky / Bunch-Kaufman / triangular
    issym::Bool
    isherm::Bool
    info::Int            # nonzero ⇒ singular (LU) ; resolved Cholesky failure is not an error
    factored::Bool       # false ⇒ GENERAL was detected but left for the host (fallback_lu=false)
    # dense factored storage (n×n): triangular / Cholesky / Bunch-Kaufman / LU
    fact::Matrix{T}
    ipiv::Vector{Int}
    # vector storage: diagonal, bidiagonal off-diagonals, tridiagonal (gttrf)
    dvec::Vector{T}      # main diagonal (DIAGONAL / BIDIAGONAL) or `d` for TRIDIAGONAL
    dl::Vector{T}        # sub-diagonal
    du::Vector{T}        # super-diagonal
    du2::Vector{T}       # 2nd super-diagonal fill from gttrf!
    # banded AB storage (2kl+ku+1) × n for gbtrf
    band::Matrix{T}
end

function SpecializedLU{T}() where {T}
    R = real(T)
    return SpecializedLU{T, R}(
        GENERAL, 0, 0, 0, 'U', false, false, 0, false,
        Matrix{T}(undef, 0, 0), Int[],
        T[], T[], T[], T[],
        Matrix{T}(undef, 0, 0)
    )
end

"""
    SpecializedLU{T}(n::Integer)

Construct a workspace with buffers pre-sized for an `n×n` matrix of element
type `T`. Useful for hosts (e.g. `LinearSolve.jl`) that want an
allocation-free reuse path for the common dense/LU/symmetric case; the band
buffer is still grown lazily when a narrow-banded matrix is first seen.
"""
function SpecializedLU{T}(n::Integer) where {T}
    R = real(T)
    nn = Int(n)
    return SpecializedLU{T, R}(
        GENERAL, 0, 0, 0, 'U', false, false, 0, false,
        Matrix{T}(undef, nn, nn), Vector{Int}(undef, nn),
        Vector{T}(undef, nn), Vector{T}(undef, max(nn - 1, 0)),
        Vector{T}(undef, max(nn - 1, 0)), Vector{T}(undef, max(nn - 2, 0)),
        Matrix{T}(undef, 0, 0)
    )
end

matrixform(F::SpecializedLU) = F.form
LinearAlgebra.issuccess(F::SpecializedLU) = F.factored && F.info == 0

"""
    isfactored(F::SpecializedLU) -> Bool

Whether `F` actually holds a usable factorization. `false` means a `GENERAL`
(unstructured) matrix was detected but deliberately not factored because
`specializinglu` was called with `fallback_lu = false`; the host is expected
to inspect [`matrixform`](@ref) and supply its own LU for that case.
"""
isfactored(F::SpecializedLU) = F.factored
Base.size(F::SpecializedLU) = (F.n, F.n)
Base.size(F::SpecializedLU, i::Integer) = i <= 2 ? F.n : 1
Base.eltype(::SpecializedLU{T}) where {T} = T

function Base.show(io::IO, F::SpecializedLU{T}) where {T}
    print(io, "SpecializedLU{$T} of size $(F.n)×$(F.n), form = $(F.form)")
    return F.info == 0 || print(io, " (singular, info=$(F.info))")
end

# small helpers -------------------------------------------------------------

@inline _resize!(v::Vector, k::Int) = (length(v) == k || resize!(v, k); v)

@inline function _identity_pivots!(p::Vector{Int})
    @inbounds for k in eachindex(p)
        p[k] = k
    end
    return p
end

# Index of the first zero diagonal entry (a zero pivot ⇒ singular), else 0.
@inline function _first_zero(d::AbstractVector, n::Int)
    @inbounds for k in 1:n
        iszero(d[k]) && return k
    end
    return 0
end

@inline function _first_zero_diag(A::AbstractMatrix, n::Int)
    @inbounds for k in 1:n
        iszero(A[k, k]) && return k
    end
    return 0
end

function _ensure_fact!(F::SpecializedLU{T}, n::Int) where {T}
    if size(F.fact, 1) != n || size(F.fact, 2) != n
        F.fact = Matrix{T}(undef, n, n)
    end
    return F.fact
end

function _ensure_band!(F::SpecializedLU{T}, rows::Int, n::Int) where {T}
    if size(F.band, 1) != rows || size(F.band, 2) != n
        F.band = Matrix{T}(undef, rows, n)
    end
    return F.band
end

# ---------------------------------------------------------------------------
# Public construction / factorization
# ---------------------------------------------------------------------------

"""
    specializinglu(A; bandwidth_cutoff = max(16, n ÷ 4)) -> SpecializedLU

Detect the structural form of the square dense matrix `A` and build the
corresponding specialized factorization, returning a [`SpecializedLU`](@ref)
workspace. Always returns the same concrete type regardless of `A`'s
structure (type-stable).

The specialized factorization is then as cheap as the structure allows (e.g.
O(n) for tridiagonal via LAPACK `gttrf`, Cholesky for symmetric positive
definite, Bunch–Kaufman for symmetric indefinite, banded LU for narrow bands,
and a triangular/diagonal solve needs no factorization at all). Element types
that are not `BlasFloat` use the structural fast paths plus a generic dense LU
fallback for the remaining forms.

Detection cost: structurally *certifying* a form on dense storage is Θ(n²)
worst case (you must read the entries that have to vanish — there is no way to
prove a dense matrix is e.g. diagonal in O(n)). It early-exits to ≈O(n) for
unstructured matrices, and is always far below the O(n³) factorization it
replaces. The "O(n)" wins are the specialized *solves*.

# The `GENERAL` (unstructured) branch and `fallback_lu`

A `GENERAL` matrix has no specialized factorization. By default
(`fallback_lu = true`) the package factors it with a plain `lu!` so that it is
a complete standalone solver. **A host that wants its own CPU-dependent LU
(e.g. `LinearSolve.jl` choosing `RFLUFactorization` / MKL / AppleAccelerate by
value) should pass `fallback_lu = false`.** Then a `GENERAL` matrix is detected
but *not factored*: `matrixform(F) == GENERAL`, `isfactored(F) == false`, and
the host supplies the LU. (`lu!` alone does **not** reproduce LinearSolve's
value-level choice, which is why that choice must stay on the host side.)
"""
function specializinglu(A::AbstractMatrix{T}; kwargs...) where {T}
    # Factorizing requires a field; promote integer/rational element types to
    # their float, matching `LinearAlgebra.lu` / `\`. `float(T)` is resolved at
    # compile time, so this stays type-stable and copy-free for float inputs.
    S = float(T)
    F = SpecializedLU{S}()
    Af = S === T ? A : convert(AbstractMatrix{S}, A)
    return specializinglu!(F, Af; kwargs...)
end

"""
    specializinglu!(F::SpecializedLU, A; bandwidth_cutoff = -1, fallback_lu = true) -> F
    specializinglu!(F::SpecializedLU, A, d::DetectionResult; fallback_lu = true) -> F

Re-detect and re-factor `A` into the existing workspace `F`, reusing (and
growing only as needed) `F`'s buffers. Useful in hot loops where many matrices
of the same size and element type are solved in sequence.

Pass a precomputed `d` (from [`detect_form`](@ref)) to skip the detection
scan — handy for a host that already inspected the form to decide whether to
delegate the `GENERAL` case. See [`specializinglu`](@ref) for `fallback_lu`.
"""
function specializinglu!(
        F::SpecializedLU{T}, A::AbstractMatrix{T};
        bandwidth_cutoff::Integer = -1, fallback_lu::Bool = true
    ) where {T}
    d = detect_form(A; bandwidth_cutoff = bandwidth_cutoff)
    return specializinglu!(F, A, d; fallback_lu = fallback_lu)
end

function specializinglu!(
        F::SpecializedLU{T}, A::AbstractMatrix{T},
        d::DetectionResult; fallback_lu::Bool = true
    ) where {T}
    F.n = size(A, 1)
    F.kl = d.kl
    F.ku = d.ku
    F.issym = d.issym
    F.isherm = d.isherm
    F.info = 0
    F.factored = true
    _factorize!(F, A, d.form, fallback_lu)
    return F
end

# ---------------------------------------------------------------------------
# Factorization dispatch (fills the workspace, sets the resolved form)
# ---------------------------------------------------------------------------

function _factorize!(
        F::SpecializedLU{T}, A::AbstractMatrix{T}, form::MatrixForm,
        fallback_lu::Bool
    ) where {T}
    n = F.n
    if form == DIAGONAL
        d = _resize!(F.dvec, n)
        @inbounds for k in 1:n
            d[k] = A[k, k]
        end
        F.info = _first_zero(d, n)
        F.form = DIAGONAL
    elseif form == LOWER_BIDIAGONAL
        d = _resize!(F.dvec, n)
        dl = _resize!(F.dl, n - 1)
        @inbounds for k in 1:n
            d[k] = A[k, k]
        end
        @inbounds for k in 1:(n - 1)
            dl[k] = A[k + 1, k]
        end
        F.info = _first_zero(d, n)
        F.form = LOWER_BIDIAGONAL
    elseif form == UPPER_BIDIAGONAL
        d = _resize!(F.dvec, n)
        du = _resize!(F.du, n - 1)
        @inbounds for k in 1:n
            d[k] = A[k, k]
        end
        @inbounds for k in 1:(n - 1)
            du[k] = A[k, k + 1]
        end
        F.info = _first_zero(d, n)
        F.form = UPPER_BIDIAGONAL
    elseif form == LOWER_TRIANGULAR
        copyto!(_ensure_fact!(F, n), A)
        F.info = _first_zero_diag(F.fact, n)
        F.form = LOWER_TRIANGULAR
    elseif form == UPPER_TRIANGULAR
        copyto!(_ensure_fact!(F, n), A)
        F.info = _first_zero_diag(F.fact, n)
        F.form = UPPER_TRIANGULAR
    elseif form == TRIDIAGONAL
        _factorize_tridiagonal!(F, A)
    elseif form == BANDED
        _factorize_banded!(F, A)
    else # GENERAL — possibly refine to a symmetric specialization
        _factorize_general!(F, A, fallback_lu)
    end
    return F
end

# --- tridiagonal -----------------------------------------------------------

function _factorize_tridiagonal!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    n = F.n
    dl = _resize!(F.dl, n - 1)
    d = _resize!(F.dvec, n)
    du = _resize!(F.du, n - 1)
    @inbounds for k in 1:n
        d[k] = A[k, k]
    end
    @inbounds for k in 1:(n - 1)
        dl[k] = A[k + 1, k]
        du[k] = A[k, k + 1]
    end
    # gttrf! throws LAPACKException (info>0) on an exactly-singular factor,
    # unlike getrf!/potrf!. Normalize to the `info` contract so `issuccess`
    # is uniform; rethrow anything that is not a singular-pivot exception.
    try
        _, _, _, du2, ipiv = gttrf!(dl, d, du)
        F.du2 = du2
        F.ipiv = ipiv
        F.info = 0
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        F.info = e.info
        F.du2 = fill!(_resize!(F.du2, max(n - 2, 0)), zero(T))
        F.ipiv = _identity_pivots!(_resize!(F.ipiv, n))
    end
    F.form = TRIDIAGONAL
    return F
end

# generic fallback: dense LU
function _factorize_tridiagonal!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T}
    _factorize_dense_lu!(F, A)
    F.form = TRIDIAGONAL
    return F
end

# --- banded ----------------------------------------------------------------

function _factorize_banded!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    n = F.n
    kl = F.kl
    ku = F.ku
    rows = 2kl + ku + 1
    AB = _ensure_band!(F, rows, n)
    fill!(AB, zero(T))
    @inbounds for j in 1:n
        for i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = A[i, j]
        end
    end
    try
        _, ipiv = gbtrf!(kl, ku, n, AB)
        F.ipiv = ipiv
        F.info = 0
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        F.info = e.info
        F.ipiv = _identity_pivots!(_resize!(F.ipiv, n))
    end
    F.form = BANDED
    return F
end

function _factorize_banded!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T}
    _factorize_dense_lu!(F, A)
    F.form = BANDED
    return F
end

# --- general / symmetric ---------------------------------------------------

# BlasFloat: try the symmetric specializations, otherwise plain LU.
function _factorize_general!(
        F::SpecializedLU{T}, A::AbstractMatrix{T},
        fallback_lu::Bool
    ) where {T <: BlasFloat}
    n = F.n
    if F.isherm
        # symmetric (real) or Hermitian (complex): attempt Cholesky.
        C = _ensure_fact!(F, n)
        copyto!(C, A)
        F.uplo = 'U'
        _, info = potrf!('U', C)
        if info == 0
            F.form = SYMMETRIC_POSITIVE_DEFINITE
            return F
        end
        # not positive definite: fall through to Bunch-Kaufman.
        copyto!(C, A)
        if T <: Real
            ipiv = _resize!(F.ipiv, n)
            _, _, info2 = sytrf!('U', C, ipiv)
            F.info = info2
            F.form = SYMMETRIC_INDEFINITE
        else
            ipiv = _resize!(F.ipiv, n)
            _, _, info2 = hetrf!('U', C, ipiv)
            F.info = info2
            F.form = HERMITIAN_INDEFINITE
        end
        return F
    elseif F.issym  # complex symmetric (Aᵀ = A) but not Hermitian
        C = _ensure_fact!(F, n)
        copyto!(C, A)
        F.uplo = 'U'
        ipiv = _resize!(F.ipiv, n)
        _, _, info2 = sytrf!('U', C, ipiv)
        F.info = info2
        F.form = SYMMETRIC_INDEFINITE
        return F
    else
        F.form = GENERAL
        if fallback_lu
            _factorize_dense_lu!(F, A)
        else
            F.factored = false   # leave the LU to the host
        end
        return F
    end
end

# generic element types: no symmetric specialization, plain LU.
function _factorize_general!(
        F::SpecializedLU{T}, A::AbstractMatrix{T},
        fallback_lu::Bool
    ) where {T}
    F.form = GENERAL
    if fallback_lu
        _factorize_dense_lu!(F, A)
    else
        F.factored = false
    end
    return F
end

# Shared dense-LU fallback. `lu!` dispatches to LAPACK getrf! for BlasFloat
# and to the generic factorization otherwise; we keep the raw factors + pivots
# so the workspace type stays concrete.
function _factorize_dense_lu!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T}
    n = F.n
    C = _ensure_fact!(F, n)
    copyto!(C, A)
    fact = lu!(C; check = false)
    F.ipiv = fact.ipiv
    F.info = fact.info
    return F
end

# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

"""
    ldiv!(x, F::SpecializedLU, b) -> x

Solve `A x = b` using the specialized factorization stored in `F`, writing
the result into `x`. Branches on `F.form` at runtime; every branch is
type-stable. `b` may be a vector or a matrix (multiple right-hand sides).
"""
function LinearAlgebra.ldiv!(x::AbstractVecOrMat, F::SpecializedLU, b::AbstractVecOrMat)
    F.factored || throw(
        ArgumentError(
            "SpecializedLU holds an unfactored GENERAL matrix (fallback_lu=false); " *
                "the host must supply the LU. Check `matrixform(F) == GENERAL` first."
        )
    )
    form = F.form
    if form == DIAGONAL
        _solve_diagonal!(x, F, b)
    elseif form == LOWER_TRIANGULAR
        _solve_triangular!(x, F, b, LowerTriangular)
    elseif form == UPPER_TRIANGULAR
        _solve_triangular!(x, F, b, UpperTriangular)
    elseif form == LOWER_BIDIAGONAL
        _solve_lower_bidiagonal!(x, F, b)
    elseif form == UPPER_BIDIAGONAL
        _solve_upper_bidiagonal!(x, F, b)
    elseif form == TRIDIAGONAL
        _solve_tridiagonal!(x, F, b)
    elseif form == BANDED
        _solve_banded!(x, F, b)
    elseif form == SYMMETRIC_POSITIVE_DEFINITE
        _solve_posdef!(x, F, b)
    elseif form == SYMMETRIC_INDEFINITE
        _solve_symmetric!(x, F, b)
    elseif form == HERMITIAN_INDEFINITE
        _solve_hermitian!(x, F, b)
    else
        _solve_dense_lu!(x, F, b)
    end
    return x
end

LinearAlgebra.ldiv!(F::SpecializedLU, b::AbstractVecOrMat) = ldiv!(b, F, b)

function Base.:\(F::SpecializedLU{T}, b::AbstractVecOrMat) where {T}
    x = similar(b, T, size(b))
    copyto!(x, b)
    return ldiv!(F, x)
end

# --- individual specialized solves ----------------------------------------

function _solve_diagonal!(x, F::SpecializedLU, b)
    d = F.dvec
    x .= b ./ d
    return x
end

function _solve_triangular!(x, F::SpecializedLU, b, ::Type{W}) where {W}
    x === b || copyto!(x, b)
    ldiv!(W(F.fact), x)
    return x
end

function _solve_lower_bidiagonal!(x, F::SpecializedLU{T}, b) where {T}
    n = F.n
    d = F.dvec
    dl = F.dl
    @inbounds for col in axes(b, 2)
        x[1, col] = b[1, col] / d[1]
        for i in 2:n
            x[i, col] = (b[i, col] - dl[i - 1] * x[i - 1, col]) / d[i]
        end
    end
    return x
end

function _solve_upper_bidiagonal!(x, F::SpecializedLU{T}, b) where {T}
    n = F.n
    d = F.dvec
    du = F.du
    @inbounds for col in axes(b, 2)
        x[n, col] = b[n, col] / d[n]
        for i in (n - 1):-1:1
            x[i, col] = (b[i, col] - du[i] * x[i + 1, col]) / d[i]
        end
    end
    return x
end

# tridiagonal: BlasFloat → gttrs!, generic → stored LU
function _solve_tridiagonal!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    gttrs!('N', F.dl, F.dvec, F.du, F.du2, F.ipiv, x)
    return x
end
_solve_tridiagonal!(x, F::SpecializedLU{T}, b) where {T} = _solve_dense_lu!(x, F, b)

# banded
function _solve_banded!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    gbtrs!('N', F.kl, F.ku, F.n, F.band, F.ipiv, x)
    return x
end
_solve_banded!(x, F::SpecializedLU{T}, b) where {T} = _solve_dense_lu!(x, F, b)

# symmetric positive definite (Cholesky)
function _solve_posdef!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    potrs!(F.uplo, F.fact, x)
    return x
end

# symmetric indefinite (Bunch-Kaufman, sytrf)
function _solve_symmetric!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    sytrs!(F.uplo, F.fact, F.ipiv, x)
    return x
end

# Hermitian indefinite (hetrf)
function _solve_hermitian!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    hetrs!(F.uplo, F.fact, F.ipiv, x)
    return x
end

# dense LU fallback (BlasFloat or generic): reconstruct the LU view and solve
function _solve_dense_lu!(x, F::SpecializedLU{T}, b) where {T}
    x === b || copyto!(x, b)
    lu = LU(F.fact, F.ipiv, F.info)
    ldiv!(lu, x)
    return x
end

# ---------------------------------------------------------------------------
# Determinant (cheap for structured forms)
# ---------------------------------------------------------------------------

@inline function _pivsign(ipiv::Vector{Int}, n::Int)
    s = 0
    @inbounds for k in 1:n
        ipiv[k] != k && (s += 1)
    end
    return isodd(s) ? -1 : 1
end

function LinearAlgebra.det(F::SpecializedLU{T}) where {T}
    F.factored || throw(
        ArgumentError(
            "SpecializedLU holds an unfactored GENERAL matrix (fallback_lu=false)"
        )
    )
    form = F.form
    n = F.n
    if form == DIAGONAL || form == LOWER_BIDIAGONAL || form == UPPER_BIDIAGONAL
        p = one(T)
        @inbounds for k in 1:n
            p *= F.dvec[k]
        end
        return p
    elseif form == LOWER_TRIANGULAR || form == UPPER_TRIANGULAR
        p = one(T)
        @inbounds for k in 1:n
            p *= F.fact[k, k]
        end
        return p
    elseif form == TRIDIAGONAL
        if T <: BlasFloat
            # gttrf overwrites the main diagonal with U's diagonal.
            p = one(T)
            @inbounds for k in 1:n
                p *= F.dvec[k]
            end
            return T(_pivsign(F.ipiv, n) * p)
        else
            return det(LU(F.fact, F.ipiv, F.info))
        end
    elseif form == BANDED
        if T <: BlasFloat
            # U's diagonal lives on row kl+ku+1 of the AB band storage.
            row = F.kl + F.ku + 1
            p = one(T)
            @inbounds for j in 1:n
                p *= F.band[row, j]
            end
            return T(_pivsign(F.ipiv, n) * p)
        else
            return det(LU(F.fact, F.ipiv, F.info))
        end
    elseif form == SYMMETRIC_POSITIVE_DEFINITE
        p = one(real(T))
        @inbounds for k in 1:n
            p *= real(F.fact[k, k])
        end
        return T(p * p)
    elseif form == SYMMETRIC_INDEFINITE
        # det(::BunchKaufman) infers as Union{real(T),T}; coerce to T so the
        # whole `det` stays type-stable for complex element types.
        return T(det(BunchKaufman(F.fact, F.ipiv, F.uplo, true, false, F.info)))
    elseif form == HERMITIAN_INDEFINITE
        return T(det(BunchKaufman(F.fact, F.ipiv, F.uplo, false, false, F.info)))
    else
        return det(LU(F.fact, F.ipiv, F.info))
    end
end

end # module

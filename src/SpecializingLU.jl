module SpecializingLU

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasInt
using LinearAlgebra.LAPACK: potrf!, potrs!, sytrs!, hetrs!, chkargsok
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using PrecompileTools: @setup_workload, @compile_workload

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

    # The two pieces of information have opposite access patterns, so they are
    # gathered in two passes, each with its own cheap early-exit:
    #
    #  * symmetry / Hermitian-ness compares `A[i, j]` with its transpose
    #    partner `A[j, i]` (a full row away — cache-hostile). It is computed
    #    first, in BLK×BLK tiles so both an `A[i, j]` strip and its transpose
    #    `A[j, i]` strip stay cache-resident; it bails the instant both flags
    #    die (which, for an unstructured matrix, happens in the very first
    #    tile). This is what makes the full symmetric scan cache-friendly.
    #
    #  * the lower/upper bandwidth is a pure column-major scan of `A[i, j]`
    #    (unit stride) with a per-column early-exit: once a nonzero band on
    #    both sides has ruled out triangular, the band is wider than `cutoff`
    #    (so not banded), and symmetry has already been ruled out, only GENERAL
    #    remains. The exact bandwidths reported on that GENERAL early-exit are
    #    order-dependent and not part of the contract.
    issym, isherm = _scan_symmetry(A, n)

    bandexit = !issym && !isherm
    @inbounds for j in 1:n
        for i in 1:n
            if !iszero(A[i, j])
                d = i - j
                if d > 0
                    d > kl && (kl = d)
                elseif d < 0
                    -d > ku && (ku = -d)
                end
            end
        end
        if bandexit && kl > 0 && ku > 0 && (kl + ku + 1) > cutoff
            return DetectionResult(GENERAL, kl, ku, false, false)
        end
    end

    form = _classify(kl, ku, n, cutoff)
    return DetectionResult(form, kl, ku, issym, isherm)
end

# Exact symmetry / Hermitian classification, tiled for cache locality. Compares
# `A[i, j]` with its transpose partner `A[j, i]` over the strict upper triangle
# in BLK×BLK tiles (both an `A[i, j]` block and its `A[j, i]` partner block are
# O(BLK²) elements and stay cache-resident), and checks the real-diagonal
# requirement for Hermitian-ness. Bails the moment both flags are dead.
function _scan_symmetry(A::AbstractMatrix, n::Int)
    issym = true
    isherm = true
    BLK = 512
    @inbounds for jb in 1:BLK:n
        jhi = min(jb + BLK - 1, n)
        for ib in 1:BLK:jb
            ihi = min(ib + BLK - 1, jhi)
            if ib == jb
                # diagonal tile: strictly-upper pairs within the block, plus
                # the real-diagonal check. Bail per column so an unstructured
                # matrix (flags die on the first off-diagonal pair) stops almost
                # immediately, while a symmetric matrix pays only one extra
                # branch per column.
                for j in jb:jhi
                    for i in ib:(j - 1)
                        aij = A[i, j]
                        aji = A[j, i]
                        issym &= (aij == aji)
                        isherm &= (aij == conj(aji))
                    end
                    isherm &= isreal(A[j, j])
                    (issym || isherm) || return (false, false)
                end
            else
                for j in jb:jhi
                    for i in ib:ihi
                        aij = A[i, j]
                        aji = A[j, i]
                        issym &= (aij == aji)
                        isherm &= (aij == conj(aji))
                    end
                end
                (issym || isherm) || return (false, false)
            end
        end
    end
    return (issym, isherm)
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
    # banded AB storage (2kl+ku+1) × n
    band::Matrix{T}
    # reusable LAPACK work buffer for the Bunch-Kaufman query+factor; sytrf!/
    # hetrf! otherwise allocate a fresh `work` (≈ n·nb) on every call.
    work::Vector{T}
end

function SpecializedLU{T}() where {T}
    R = real(T)
    return SpecializedLU{T, R}(
        GENERAL, 0, 0, 0, 'U', false, false, 0, false,
        Matrix{T}(undef, 0, 0), Int[],
        T[], T[], T[], T[],
        Matrix{T}(undef, 0, 0), T[]
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
        Matrix{T}(undef, 0, 0), T[]
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
    # Factorizing in place needs a field, so integer element types must be
    # promoted (an `Int` can't hold a fractional pivot). `LinearAlgebra.lutype`
    # is exactly the promotion `lu`/`\` apply: `Int`→Float64 but `Rational`,
    # `BigFloat`, `Dual`, etc. left untouched — so a rational solve stays exact
    # rather than being silently widened to Float64. It is a compile-time
    # constant, so this stays type-stable and copy-free for already-field inputs.
    S = LinearAlgebra.lutype(T)
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

# Tridiagonal LU with partial pivoting, hand-rolled to mirror LAPACK `gttrf`
# exactly (verified bit-for-bit against `gttrf!`/`gttrs!` and to machine
# precision against dense `lu`). This is used for *every* element type: it is
# allocation-free on the warm reuse path, matches LAPACK gttrf factorization
# speed for `BlasFloat` (the LAPACK wrapper allocates `du2`/`ipiv` each call),
# matches gttrs single-RHS solve speed, and beats gttrs several-fold on
# multi-RHS solves by vectorizing across right-hand-side columns. It also
# replaces the old O(n³) dense-LU fallback that generic (e.g. `BigFloat`)
# tridiagonals used to take.
#
# Storage convention (identical to `gttrf`): on output `dl` holds L's
# subdiagonal multipliers, `dvec`(=d) holds U's diagonal, `du` holds U's first
# superdiagonal, `du2` holds U's second superdiagonal (fill produced by a row
# interchange), and `ipiv[i] == i+1` iff rows i and i+1 were swapped at step i.
# Returns the LAPACK `info` (index of the first zero pivot, else 0).
function _gttrf!(dl::Vector{T}, d::Vector{T}, du::Vector{T}, du2::Vector{T}, ipiv::Vector{Int}, n::Int) where {T}
    @inbounds for i in 1:n
        ipiv[i] = i
    end
    @inbounds for i in 1:(n - 2)
        du2[i] = zero(T)
    end
    @inbounds for i in 1:(n - 1)
        # partial pivoting: interchange rows i and i+1 unless |d[i]| >= |dl[i]|.
        if abs(d[i]) >= abs(dl[i])
            if !iszero(d[i])
                fact = dl[i] / d[i]
                dl[i] = fact
                d[i + 1] -= fact * du[i]
            end
            # no interchange ⇒ no second-superdiagonal fill (du2[i] stays 0).
        else
            fact = d[i] / dl[i]
            d[i] = dl[i]
            dl[i] = fact
            tmp = du[i]
            du[i] = d[i + 1]
            d[i + 1] = tmp - fact * d[i + 1]
            if i < n - 1
                du2[i] = du[i + 1]
                du[i + 1] = -fact * du[i + 1]
            end
            ipiv[i] = i + 1
        end
    end
    info = 0
    @inbounds for i in 1:n
        if iszero(d[i])
            info = i
            break
        end
    end
    return info
end

function _factorize_tridiagonal!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T}
    n = F.n
    dl = _resize!(F.dl, max(n - 1, 0))
    d = _resize!(F.dvec, n)
    du = _resize!(F.du, max(n - 1, 0))
    du2 = _resize!(F.du2, max(n - 2, 0))
    ipiv = _resize!(F.ipiv, n)
    @inbounds for k in 1:n
        d[k] = A[k, k]
    end
    @inbounds for k in 1:(n - 1)
        dl[k] = A[k + 1, k]
        du[k] = A[k, k + 1]
    end
    F.info = _gttrf!(dl, d, du, du2, ipiv, n)
    F.form = TRIDIAGONAL
    return F
end

# --- banded ----------------------------------------------------------------

# Banded LU with partial pivoting straight into the AB band buffer (the
# 2kl+ku+1 LAPACK layout) plus F.ipiv, for EVERY element type. The unblocked
# `_banded_lu!` (LAPACK `gbtf2`) agrees with `gbtrf!`/`gbtrs!` bit-for-bit on
# BlasFloat (including the pivot vector) and — because the narrow bands that
# get classified `BANDED` don't benefit from LAPACK's blocking — is actually
# ~2× faster than `gbtrf!` while reusing F.ipiv (allocation-free warm path).
# It also gives generic (e.g. BigFloat) banded matrices O(n·(kl+ku)·kl)
# instead of the old O(n³) dense fallback, and drops the gbtrf!/gbtrs! dep.
function _factorize_banded!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T}
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
    ipiv = _resize!(F.ipiv, n)
    F.info = _banded_lu!(AB, kl, ku, n, ipiv)
    F.form = BANDED
    return F
end

# Unblocked banded LU with partial pivoting (LAPACK gbtf2). `AB` is the packed
# band, kv = kl+ku superdiagonals of U after fill. Row kv+1 holds the diagonal;
# rows kv+2..kv+1+kl hold the unit-lower multipliers; rows 1..kl hold the
# pivot-induced fill of U. `min(j+kv, n)` is the (always-safe) rightmost column
# the step touches. Returns the index of the first zero pivot (0 if nonsingular)
# to match the `info` contract of the BlasFloat path.
function _banded_lu!(AB::AbstractMatrix{T}, kl::Int, ku::Int, n::Int, ipiv::Vector{Int}) where {T}
    kv = kl + ku
    info = 0
    @inbounds for j in 1:n
        km = min(kl, n - j)
        juu = min(j + kv, n)
        jp = 0
        amax = abs(AB[kv + 1, j])
        for r in 1:km
            v = abs(AB[kv + 1 + r, j])
            if v > amax
                amax = v
                jp = r
            end
        end
        ipiv[j] = jp + j
        if jp != 0
            for jj in j:juu
                r1 = kv + 1 + j - jj
                r2 = r1 + jp
                AB[r1, jj], AB[r2, jj] = AB[r2, jj], AB[r1, jj]
            end
        end
        piv = AB[kv + 1, j]
        if iszero(piv)
            info == 0 && (info = j)
            continue
        end
        if km > 0
            inv_piv = inv(piv)
            for r in 1:km
                AB[kv + 1 + r, j] *= inv_piv
            end
            for jj in (j + 1):juu
                u = AB[kv + 1 + j - jj, jj]
                if !iszero(u)
                    for r in 1:km
                        AB[kv + 1 + (j + r) - jj, jj] -= AB[kv + 1 + r, j] * u
                    end
                end
            end
        end
    end
    return info
end

# --- general / symmetric ---------------------------------------------------

# Bunch-Kaufman factoring `C` in place, storing the pivots and info on `F`.
#
# The stdlib `sytrf!`/`hetrf!` wrappers run a workspace-size query then allocate
# a fresh `work` of length ≈ n·nb on *every* call (even the 3-arg preallocated-
# ipiv form). To keep the warm re-factor allocation-free we call the LAPACK
# routine directly through libblastrampoline, reusing both `F.ipiv` and a
# persistent `F.work`: the query writes the optimal lwork into `work[1]`, we
# grow `F.work` only if it is too small, then the real factor call reuses it.
# We do not throw on `info > 0` (a singular Bunch-Kaufman factor); the `info`
# stored on `F` is what `issuccess`/`det` read — matching the previous wrappers
# (which use `chkargsok`, i.e. only error on illegal arguments).
for (fname, kernel, elty) in (
        (:dsytrf_, :_sytrf_into!, :Float64),
        (:ssytrf_, :_sytrf_into!, :Float32),
        (:csytrf_, :_sytrf_into!, :ComplexF32),
        (:zsytrf_, :_sytrf_into!, :ComplexF64),
        (:chetrf_, :_hetrf_into!, :ComplexF32),
        (:zhetrf_, :_hetrf_into!, :ComplexF64),
    )
    @eval function $kernel(F::SpecializedLU{$elty}, C::AbstractMatrix{$elty}, uplo::AbstractChar)
        n = F.n
        ipiv = _resize!(F.ipiv, n)
        if n == 0
            F.info = 0
            return F
        end
        # ensure `F.work` has room for the lwork query slot, then query.
        work = length(F.work) >= 1 ? F.work : _resize!(F.work, 1)
        info = Ref{BlasInt}()
        lda = max(1, stride(C, 2))
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong,
            ),
            uplo, n, C, lda, ipiv, work, BlasInt(-1), info, 1
        )
        chkargsok(info[])
        lwork = BlasInt(real(work[1]))
        length(F.work) < lwork && _resize!(F.work, Int(lwork))
        work = F.work
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong,
            ),
            uplo, n, C, lda, ipiv, work, lwork, info, 1
        )
        chkargsok(info[])
        F.info = Int(info[])
        return F
    end
end

_bunchkaufman_into!(F::SpecializedLU{T}, C::AbstractMatrix{T}, ::Val{:sym}) where {T <: BlasFloat} =
    _sytrf_into!(F, C, F.uplo)
_bunchkaufman_into!(F::SpecializedLU{T}, C::AbstractMatrix{T}, ::Val{:herm}) where {T <: BlasFloat} =
    _hetrf_into!(F, C, F.uplo)

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
            _bunchkaufman_into!(F, C, Val(:sym))
            F.form = SYMMETRIC_INDEFINITE
        else
            _bunchkaufman_into!(F, C, Val(:herm))
            F.form = HERMITIAN_INDEFINITE
        end
        return F
    elseif F.issym  # complex symmetric (Aᵀ = A) but not Hermitian
        C = _ensure_fact!(F, n)
        copyto!(C, A)
        F.uplo = 'U'
        _bunchkaufman_into!(F, C, Val(:sym))
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

# LU factor of `C` (square) into the reused `F.ipiv`. The preallocated-ipiv
# `getrf!(C, ipiv; check)` wrapper only exists on Julia ≥ 1.11, and `lu!`/the
# 1-arg `getrf!` allocate a fresh `Vector{BlasInt}` every call — so we ccall
# `getrf` directly (it has no work buffer, just an output pivot vector) on all
# Julia versions to keep the warm re-factor allocation-free. `info > 0`
# (singular U) is recorded on `F`, not thrown (chkargsok only flags illegal args).
for (gname, elty) in (
        (:dgetrf_, :Float64), (:sgetrf_, :Float32),
        (:zgetrf_, :ComplexF64), (:cgetrf_, :ComplexF32),
    )
    @eval function _getrf_into!(F::SpecializedLU{$elty}, C::AbstractMatrix{$elty})
        n = F.n
        ipiv = _resize!(F.ipiv, n)
        n == 0 && (F.info = 0; return F)
        info = Ref{BlasInt}()
        lda = max(1, stride(C, 2))
        ccall(
            (@blasfunc($gname), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
            n, n, C, lda, ipiv, info
        )
        chkargsok(info[])
        F.info = Int(info[])
        return F
    end
end

# Shared dense-LU fallback (the GENERAL form). BlasFloat goes through the
# allocation-free `_getrf_into!` ccall; generic element types keep `lu!`. The
# raw factors + pivots are stored so the workspace type stays concrete.
function _factorize_dense_lu!(F::SpecializedLU{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    C = _ensure_fact!(F, F.n)
    copyto!(C, A)
    return _getrf_into!(F, C)
end

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
    cols = axes(b, 2)
    # The row recurrence (x[i] depends on x[i-1]) is serial. With a single
    # right-hand side that recurrence is the whole cost, so we keep the tight
    # column-major inner loop. With multiple right-hand sides the *columns*
    # are independent, so we hoist the column loop inside and mark it
    # @simd ivdep to vectorize across right-hand sides.
    if length(cols) == 1
        @inbounds begin
            col = first(cols)
            x[1, col] = b[1, col] / d[1]
            for i in 2:n
                x[i, col] = (b[i, col] - dl[i - 1] * x[i - 1, col]) / d[i]
            end
        end
    else
        @inbounds begin
            d1 = d[1]
            @simd ivdep for col in cols
                x[1, col] = b[1, col] / d1
            end
            for i in 2:n
                di = d[i]
                dli = dl[i - 1]
                @simd ivdep for col in cols
                    x[i, col] = (b[i, col] - dli * x[i - 1, col]) / di
                end
            end
        end
    end
    return x
end

function _solve_upper_bidiagonal!(x, F::SpecializedLU{T}, b) where {T}
    n = F.n
    d = F.dvec
    du = F.du
    cols = axes(b, 2)
    if length(cols) == 1
        @inbounds begin
            col = first(cols)
            x[n, col] = b[n, col] / d[n]
            for i in (n - 1):-1:1
                x[i, col] = (b[i, col] - du[i] * x[i + 1, col]) / d[i]
            end
        end
    else
        @inbounds begin
            dn = d[n]
            @simd ivdep for col in cols
                x[n, col] = b[n, col] / dn
            end
            for i in (n - 1):-1:1
                di = d[i]
                dui = du[i]
                @simd ivdep for col in cols
                    x[i, col] = (b[i, col] - dui * x[i + 1, col]) / di
                end
            end
        end
    end
    return x
end

# tridiagonal: hand-rolled `gttrs` (no-transpose) using the stored `gttrf`
# factors, for every element type. Mirrors LAPACK `dgttrs` bit-for-bit. The
# row recurrence (forward L+pivots, backward U) is inherently serial, so for a
# single right-hand side we keep the tight scalar loop (matching gttrs); for
# multiple right-hand sides the *columns* are independent, so the column loop
# is hoisted inside and marked `@simd ivdep` to vectorize across RHS.
function _solve_tridiagonal!(x, F::SpecializedLU{T}, b) where {T}
    x === b || copyto!(x, b)
    n = F.n
    dl = F.dl
    d = F.dvec
    du = F.du
    du2 = F.du2
    ipiv = F.ipiv
    cols = axes(x, 2)
    if length(cols) == 1
        col = first(cols)
        @inbounds begin
            for i in 1:(n - 1)
                dli = dl[i]
                if ipiv[i] == i
                    x[i + 1, col] -= dli * x[i, col]
                else
                    tmp = x[i, col]
                    x[i, col] = x[i + 1, col]
                    x[i + 1, col] = tmp - dli * x[i, col]
                end
            end
            x[n, col] /= d[n]
            if n > 1
                x[n - 1, col] = (x[n - 1, col] - du[n - 1] * x[n, col]) / d[n - 1]
            end
            for i in (n - 2):-1:1
                x[i, col] = (x[i, col] - du[i] * x[i + 1, col] - du2[i] * x[i + 2, col]) / d[i]
            end
        end
    else
        @inbounds begin
            for i in 1:(n - 1)
                dli = dl[i]
                if ipiv[i] == i
                    @simd ivdep for col in cols
                        x[i + 1, col] -= dli * x[i, col]
                    end
                else
                    @simd ivdep for col in cols
                        tmp = x[i, col]
                        x[i, col] = x[i + 1, col]
                        x[i + 1, col] = tmp - dli * x[i, col]
                    end
                end
            end
            dn = d[n]
            @simd ivdep for col in cols
                x[n, col] /= dn
            end
            if n > 1
                dnm1 = d[n - 1]
                dunm1 = du[n - 1]
                @simd ivdep for col in cols
                    x[n - 1, col] = (x[n - 1, col] - dunm1 * x[n, col]) / dnm1
                end
            end
            for i in (n - 2):-1:1
                di = d[i]
                dui = du[i]
                du2i = du2[i]
                @simd ivdep for col in cols
                    x[i, col] = (x[i, col] - dui * x[i + 1, col] - du2i * x[i + 2, col]) / di
                end
            end
        end
    end
    return x
end

# Banded solve against the factored AB band + pivots (every element type):
# pivot+forward L sweep then a banded back-substitution against U (bandwidth
# kv = kl+ku). Columns of a multi-RHS are independent and handled in the outer
# loop. Allocation-free.
function _solve_banded!(x, F::SpecializedLU{T}, b) where {T}
    x === b || copyto!(x, b)
    AB = F.band
    ipiv = F.ipiv
    kl = F.kl
    ku = F.ku
    n = F.n
    kv = kl + ku
    nrhs = size(x, 2)
    @inbounds for c in 1:nrhs
        for j in 1:n
            ip = ipiv[j]
            if ip != j
                x[j, c], x[ip, c] = x[ip, c], x[j, c]
            end
            km = min(kl, n - j)
            xj = x[j, c]
            for r in 1:km
                x[j + r, c] -= AB[kv + 1 + r, j] * xj
            end
        end
        for j in n:-1:1
            xj = x[j, c] / AB[kv + 1, j]
            x[j, c] = xj
            for i in max(1, j - kv):(j - 1)
                x[i, c] -= AB[kv + 1 + i - j, j] * xj
            end
        end
    end
    return x
end

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
        # `_gttrf!` (every element type) overwrites the main diagonal with U's
        # diagonal; det(A) = sign(P) * prod(U[k,k]).
        p = one(T)
        @inbounds for k in 1:n
            p *= F.dvec[k]
        end
        return T(_pivsign(F.ipiv, n) * p)
    elseif form == BANDED
        # Both the BlasFloat (gbtrf!) and generic (`_banded_lu!`) paths write the
        # factor into the AB band buffer with the same layout and absolute-index
        # pivots, so U's diagonal is row kl+ku+1 and `_pivsign` applies uniformly.
        row = F.kl + F.ku + 1
        p = one(T)
        @inbounds for j in 1:n
            p *= F.band[row, j]
        end
        return T(_pivsign(F.ipiv, n) * p)
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

# ---------------------------------------------------------------------------
# Precompilation — exercise every form and solve path for all four BlasFloat
# element types so the first real solve is fast (no first-call latency).
# ---------------------------------------------------------------------------

@setup_workload begin
    @compile_workload begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 6
            cv = (re, im) -> T <: Complex ? T(re, im) : T(re)
            zT = zero(T)

            Adiag = [i == j ? cv(i + 1, 0) : zT for i in 1:n, j in 1:n]
            Altri = [i < j ? zT : (i == j ? cv(n + i, 0) : cv(1 / (i + j), 0)) for i in 1:n, j in 1:n]
            Autri = [i > j ? zT : (i == j ? cv(n + i, 0) : cv(1 / (i + j), 0)) for i in 1:n, j in 1:n]
            Albid = [i == j ? cv(n + i, 0) : (i == j + 1 ? cv(0.5, 0) : zT) for i in 1:n, j in 1:n]
            Aubid = [i == j ? cv(n + i, 0) : (j == i + 1 ? cv(0.5, 0) : zT) for i in 1:n, j in 1:n]
            Atri = [i == j ? cv(n + i, 0) : (abs(i - j) == 1 ? cv(-0.5, 0) : zT) for i in 1:n, j in 1:n]
            Aband = [i == j ? cv(2n, 0) : (abs(i - j) <= 2 ? cv(0.25, 0) : zT) for i in 1:n, j in 1:n]

            # exactly Hermitian/symmetric matrices so detection routes to
            # potrf (SPD) / sytrf (symmetric or complex-symmetric indefinite) /
            # hetrf (Hermitian indefinite), not the general getrf path.
            Aspd = zeros(T, n, n)
            Asym = zeros(T, n, n)
            Aher = zeros(T, n, n)
            for j in 1:n, i in 1:j
                if i == j
                    Aspd[i, j] = cv(2n, 0)
                    Asym[i, j] = cv((-1)^i * n, 0)
                    Aher[i, j] = cv((-1)^i * n, 0)
                else
                    v = cv(1 / (i + j), 1 / (i + 2j + 1))
                    Aspd[i, j] = v;  Aspd[j, i] = conj(v)         # Hermitian PD
                    w = cv(1 / (i + j + 1), 1 / (i * j + 1))
                    Asym[i, j] = w;  Asym[j, i] = w               # (complex-)symmetric
                    Aher[i, j] = w;  Aher[j, i] = conj(w)         # Hermitian indefinite
                end
            end
            Agen = [i == j ? cv(2n, 0) : cv(1 / (i + 2j), 0) for i in 1:n, j in 1:n]

            b = ones(T, n)
            B = ones(T, n, 2)
            for A in (Adiag, Altri, Autri, Albid, Aubid, Atri, Aband, Aspd, Asym, Aher, Agen)
                F = specializinglu(A)
                F \ b
                F \ B
                ldiv!(similar(b), F, b)
                det(F)
            end
            detect_form(Agen)
            specializinglu(Agen; fallback_lu = false)
        end
    end
end

end # module

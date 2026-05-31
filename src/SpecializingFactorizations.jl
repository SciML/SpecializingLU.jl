module SpecializingFactorizations

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasInt
using LinearAlgebra.LAPACK: potrf!, potrs!, sytrs!, hetrs!, chkargsok
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using PrecompileTools: @setup_workload, @compile_workload

export MatrixForm,
    GENERAL, DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR,
    LOWER_BIDIAGONAL, UPPER_BIDIAGONAL, TRIDIAGONAL, BANDED,
    SYMMETRIC_POSITIVE_DEFINITE, SYMMETRIC_INDEFINITE, HERMITIAN_INDEFINITE
export SpecializedLU, specializinglu, specializinglu!, reserve!, detect_form, DetectionResult,
    matrixform, issuccess, isfactored
export QRStatus, QR_UNFACTORED, QR_FULLRANK, QR_DEFICIENT
export SpecializedQR, specializingqr, specializingqr!, structuralform

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
    SpecializedLU{T}(n::Integer; kl = 0, ku = 0, symmetric = false)

Construct a workspace with buffers pre-sized (via [`reserve!`](@ref)) for an
`n×n` matrix of element type `T`, so that subsequent factorizations and solves
at size ≤ `n` are allocation-free. The dense and `O(n)` buffers are always
reserved; pass `kl`/`ku` to also reserve the banded `AB` buffer, and
`symmetric = true` to reserve the Bunch–Kaufman LAPACK work buffer — giving a
fully upfront-allocated workspace for those forms too.
"""
function SpecializedLU{T}(n::Integer; kl::Integer = 0, ku::Integer = 0, symmetric::Bool = false) where {T}
    F = SpecializedLU{T}()
    return reserve!(F, Int(n); kl = Int(kl), ku = Int(ku), symmetric = symmetric)
end

"""
    reserve!(F::SpecializedLU, n; kl = 0, ku = 0, symmetric = false) -> F

Grow `F`'s buffers so a problem of size up to `n` (bandwidth `kl,ku`; and the
Bunch–Kaufman work buffer when `symmetric`) factors and solves with **zero**
allocations. Buffers only ever grow, so `reserve!`-ing the largest problem you
will solve makes every later `specializinglu!` + `ldiv!` at that size or
smaller allocation-free. Use this for hot loops / real-time use where all
allocation must happen upfront.
"""
function reserve!(F::SpecializedLU{T}, n::Integer; kl::Integer = 0, ku::Integer = 0, symmetric::Bool = false) where {T}
    nn = Int(n)
    _ensure_fact!(F, nn)                       # dense fact (LU / Cholesky / Bunch-Kaufman / triangular)
    _resize!(F.ipiv, nn)
    _resize!(F.dvec, nn)
    _resize!(F.dl, max(nn - 1, 0))
    _resize!(F.du, max(nn - 1, 0))
    _resize!(F.du2, max(nn - 2, 0))
    if kl > 0 || ku > 0
        _ensure_band!(F, 2 * Int(kl) + Int(ku) + 1, nn)
    end
    if symmetric
        _reserve_work!(F, nn)                  # Bunch-Kaufman work buffer (BlasFloat only)
    end
    return F
end

# Size F.work to the optimal Bunch-Kaufman workspace for an n×n problem via the
# LAPACK lwork query (no-op for non-BlasFloat element types, which use generic
# LU and need no work buffer).
_reserve_work!(F::SpecializedLU, ::Int) = F
for (fname, elty) in (
        (:dsytrf_, :Float64), (:ssytrf_, :Float32),
        (:csytrf_, :ComplexF32), (:zsytrf_, :ComplexF64),
    )
    @eval function _reserve_work!(F::SpecializedLU{$elty}, n::Int)
        n == 0 && return F
        C = _ensure_fact!(F, n)
        ipiv = _resize!(F.ipiv, n)
        wq = Ref{$elty}()
        info = Ref{BlasInt}()
        lda = max(1, stride(C, 2))
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong,
            ),
            'U', n, C, lda, ipiv, wq, BlasInt(-1), info, 1
        )
        lwork = max(1, Int(real(wq[])))
        length(F.work) < lwork && _resize!(F.work, lwork)
        return F
    end
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

# The dense `fact` buffer is kept EXACTLY n×n: the dense solves pass it straight
# to LAPACK (potrs!/sytrs!/hetrs!/getrs!/trsv), and on the Julia 1.10 LTS those
# wrappers allocate a small temporary when handed a strided `view` (the SubArray
# is not elided there), which would break the zero-allocation solve. A real
# n×n `Matrix` keeps the solve 0-alloc on every supported Julia version; the
# trade is that a change in the *dense* problem size reallocates this O(n²)
# buffer (negligible next to the factorization, and fixed-size reuse is 0-alloc).
function _ensure_fact!(F::SpecializedLU{T}, n::Int) where {T}
    if size(F.fact, 1) != n || size(F.fact, 2) != n
        F.fact = Matrix{T}(undef, n, n)
    end
    return F.fact
end

# The `band` buffer IS grow-only: its solve (`_solve_banded!`) is pure Julia, so
# a strided sub-view is allocation-free on every Julia version. It reallocates
# only when capacity is insufficient; shrinking the size/bandwidth reuses it.
function _ensure_band!(F::SpecializedLU{T}, rows::Int, n::Int) where {T}
    if size(F.band, 1) < rows || size(F.band, 2) < n
        F.band = Matrix{T}(undef, max(rows, size(F.band, 1)), max(n, size(F.band, 2)))
    end
    return @view F.band[1:rows, 1:n]
end

# Logical factor / band sub-matrix for the current problem. `fact` is exact-size
# so this is the buffer itself; `band` may be a larger capacity buffer.
@inline _factmat(F::SpecializedLU) = F.fact
@inline _bandmat(F::SpecializedLU) = @view F.band[1:(2 * F.kl + F.ku + 1), 1:F.n]

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
    ldiv!(W(_factmat(F)), x)
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
_solve_tridiagonal!(x, F::SpecializedLU{T}, b) where {T} =
    (x === b || copyto!(x, b); _tridiag_solve!(x, F.dl, F.dvec, F.du, F.du2, F.ipiv, F.n))

# Shared tridiagonal back-solve core (gttrs), operating in place on `x` (already
# loaded with the RHS) against the `_gttrf!` factor buffers. Reused by both
# SpecializedLU and SpecializedQR (the QR full-rank tridiagonal fast path).
function _tridiag_solve!(x, dl, d, du, du2, ipiv, n)
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
_solve_banded!(x, F::SpecializedLU{T}, b) where {T} =
    (x === b || copyto!(x, b); _banded_solve!(x, _bandmat(F), F.ipiv, F.kl, F.ku, F.n))

# Shared banded back-solve core (gbtrs), in place on `x` against the packed AB
# factor from `_banded_lu!`. Reused by SpecializedLU and the QR banded fast path.
function _banded_solve!(x, AB, ipiv, kl, ku, n)
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
    potrs!(F.uplo, _factmat(F), x)
    return x
end

# symmetric indefinite (Bunch-Kaufman, sytrf)
function _solve_symmetric!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    sytrs!(F.uplo, _factmat(F), F.ipiv, x)
    return x
end

# Hermitian indefinite (hetrf)
function _solve_hermitian!(x, F::SpecializedLU{T}, b) where {T <: BlasFloat}
    x === b || copyto!(x, b)
    hetrs!(F.uplo, _factmat(F), F.ipiv, x)
    return x
end

# dense LU fallback (BlasFloat or generic): reconstruct the LU view and solve
function _solve_dense_lu!(x, F::SpecializedLU{T}, b) where {T}
    x === b || copyto!(x, b)
    lu = LU(_factmat(F), F.ipiv, F.info)
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
        # `_banded_lu!` (all element types) writes the factor into the AB band
        # buffer with U's diagonal on row kl+ku+1 and absolute-index pivots, so
        # `_pivsign` applies uniformly.
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
        return T(det(BunchKaufman(_factmat(F), F.ipiv, F.uplo, true, false, F.info)))
    elseif form == HERMITIAN_INDEFINITE
        return T(det(BunchKaufman(_factmat(F), F.ipiv, F.uplo, false, false, F.info)))
    else
        return det(LU(_factmat(F), F.ipiv, F.info))
    end
end

# ===========================================================================
# Specialized QR — rank-revealing least-squares / minimum-norm solver
#
# A second specializing factorization living in the same module. Where
# `SpecializedLU` is a square solver that detects *structure*, `SpecializedQR`
# is a rectangular least-squares solver that detects *rank*: a column-pivoted,
# rank-revealing QR (LAPACK `geqp3`) reveals the numerical rank, and the solve
# returns the least-squares / minimum-norm solution for any shape — including
# singular, rank-deficient, and rectangular `A` — without ever throwing. Like
# `SpecializedLU` it is one concrete workspace type whose runtime branch is an
# enum (full-rank vs rank-deficient), so the pipeline is type-stable, and the
# warm solve is allocation-free (raw LAPACK ccalls reusing persistent buffers).
# ===========================================================================

"""
    QRStatus

Enum selecting, at runtime, which [`SpecializedQR`](@ref) solve path applies.
Like [`MatrixForm`](@ref) it is a value (an `Int8`) rather than a Julia type,
so [`specializingqr`](@ref) always returns the same concrete `SpecializedQR`.

`QR_FULLRANK` means full *column* rank (`rank == n`, so `n ≤ m`): the solve is
`Q'b` then a triangular solve with the `n×n` `R`. `QR_DEFICIENT` means
`rank < n` (rank-deficient *or* underdetermined full-row-rank): the solve uses
the complete-orthogonal (gelsy) path for the minimum-norm solution, or a
rank-truncated basic solution. `QR_UNFACTORED` marks a workspace deliberately
left unfactored (`fallback = false`) for the host to own.
"""
@enum QRStatus::Int8 begin
    QR_UNFACTORED = 0
    QR_FULLRANK = 1
    QR_DEFICIENT = 2
end

"""
    SpecializedQR{T,R}

A single, concrete, reusable workspace holding a rank-revealing column-pivoted
QR factorization (`A[:,p] = Q R`) of a possibly rectangular / rank-deficient
matrix. Its Julia type is fixed (it does not depend on the rank), so
constructing and solving with it is type-stable; the `status` field selects the
solve path at runtime.

`solve`/`ldiv!` returns the least-squares solution (minimizing `‖Ax-b‖`) for
full column rank, and the minimum-norm least-squares solution (matching
`qr(A, ColumnNorm()) \\ b` and `pinv(A)*b`) when rank-deficient — never
throwing on singular input. `T` is the element type; `R = real(T)`.
"""
mutable struct SpecializedQR{T, R}
    status::QRStatus
    form::MatrixForm     # structural form actually USED; GENERAL on the geqp3 path
    m::Int
    n::Int
    kl::Int              # band lower/upper bandwidth (TRIDIAGONAL / BANDED paths)
    ku::Int
    rank::Int
    info::Int            # LAPACK illegal-arg only; rank deficiency is NOT an error
    factored::Bool       # false ⇒ left unfactored for the host (fallback=false)
    minnorm::Bool        # solve policy on the BlasFloat deficient path
    rtol::R              # relative tolerance for rank revelation
    factors::Matrix{T}   # A overwritten by geqp3! (R + reflectors); exact m×n
    tau::Vector{T}       # geqp3 reflector scalars
    jpvt::Vector{BlasInt}  # geqp3 column permutation p
    tzfactors::Matrix{T} # tzrzf! of the leading rank×n trapezoid (min-norm path)
    tau2::Vector{T}      # tzrzf reflector scalars
    rhs::Matrix{T}       # padded RHS scratch: exact max(m,n) × nrhs
    work::Vector{T}      # persistent LAPACK work (geqp3/ormqr/tzrzf/ormrz)
    rwork::Vector{R}     # complex geqp3 column-norm work
    wmin::Vector{T}      # laic1 incremental-condition-estimator scratch
    wmax::Vector{T}
    dvec::Vector{T}      # DIAGONAL entries, or U's diagonal for TRIDIAGONAL
    dl::Vector{T}        # TRIDIAGONAL: L subdiagonal multipliers (gttrf)
    du::Vector{T}        # TRIDIAGONAL: U first superdiagonal
    du2::Vector{T}       # TRIDIAGONAL: U second superdiagonal (pivot fill)
    band::Matrix{T}      # BANDED: packed AB factor (2kl+ku+1 × n)
    ipiv::Vector{Int}    # TRIDIAGONAL / BANDED pivots (distinct from geqp3 jpvt)
    gbuf::Vector{T}      # contiguous scratch for the condition gate's laic1 sweep
end

function SpecializedQR{T}() where {T}
    R = real(T)
    return SpecializedQR{T, R}(
        QR_UNFACTORED, GENERAL, 0, 0, 0, 0, 0, 0, false, true, R(0),
        Matrix{T}(undef, 0, 0), T[], BlasInt[],
        Matrix{T}(undef, 0, 0), T[],
        Matrix{T}(undef, 0, 0), T[], R[], T[], T[], T[],
        T[], T[], T[], Matrix{T}(undef, 0, 0), Int[], T[]
    )
end

matrixform(F::SpecializedQR) = F.status

"""
    structuralform(F::SpecializedQR) -> MatrixForm

The structural [`MatrixForm`](@ref) that `F`'s solve actually uses: `DIAGONAL`,
`LOWER_TRIANGULAR`/`UPPER_TRIANGULAR` (also covering the bidiagonal forms, which
are triangular), or `GENERAL` for the dense rank-revealing `geqp3` path (used for
unstructured, symmetric, rectangular, near-singular/ill-conditioned-structured,
and `detect_structure = false` inputs). Introspection only — the solve,
`rank(F)`, and `issuccess(F)` are numerically identical regardless of path.
"""
structuralform(F::SpecializedQR) = F.form
LinearAlgebra.rank(F::SpecializedQR) = F.rank
# Rank deficiency is a valid minimum-norm solve, not a failure: only a LAPACK
# illegal-argument (info ≠ 0) or an unfactored workspace is unsuccessful.
LinearAlgebra.issuccess(F::SpecializedQR) = F.factored && F.info == 0
isfactored(F::SpecializedQR) = F.factored
Base.size(F::SpecializedQR) = (F.m, F.n)
Base.size(F::SpecializedQR, i::Integer) = i == 1 ? F.m : (i == 2 ? F.n : 1)
Base.eltype(::SpecializedQR{T}) where {T} = T

function Base.show(io::IO, F::SpecializedQR{T}) where {T}
    print(io, "SpecializedQR{$T} of size $(F.m)×$(F.n), rank $(F.rank), status = $(F.status)")
    return F.info == 0 || print(io, " (info=$(F.info))")
end

_default_rtol(::Type{T}, m::Int, n::Int) where {T} = min(m, n) * eps(real(float(T)))

# Buffer sizing. Like the LU `fact`, the dense buffers are kept EXACT-size: the
# raw LAPACK solve ccalls write into `rhs`, and a strided sub-view is not always
# elided on the 1.10 LTS (costing a small allocation), so exact-size real
# `Matrix`/`Vector` buffers keep the warm solve 0-allocation on every supported
# Julia version. A change in problem size reallocates them (negligible next to
# the factorization).
function _ensure_qr_factors!(F::SpecializedQR{T}, m::Int, n::Int) where {T}
    if size(F.factors, 1) != m || size(F.factors, 2) != n
        F.factors = Matrix{T}(undef, m, n)
    end
    return F.factors
end

function _ensure_qr_tzfactors!(F::SpecializedQR{T}, r::Int, n::Int) where {T}
    if size(F.tzfactors, 1) != r || size(F.tzfactors, 2) != n
        F.tzfactors = Matrix{T}(undef, r, n)
    end
    return F.tzfactors
end

function _ensure_qr_rhs!(F::SpecializedQR{T}, rows::Int, nrhs::Int) where {T}
    if size(F.rhs, 1) != rows || size(F.rhs, 2) != nrhs
        F.rhs = Matrix{T}(undef, rows, nrhs)
    end
    return F.rhs
end

# The BANDED `band` buffer is grow-only (its solve is pure Julia, so a strided
# sub-view is allocation-free on every Julia version), mirroring SpecializedLU.
function _ensure_qr_band!(F::SpecializedQR{T}, rows::Int, n::Int) where {T}
    if size(F.band, 1) < rows || size(F.band, 2) < n
        F.band = Matrix{T}(undef, max(rows, size(F.band, 1)), max(n, size(F.band, 2)))
    end
    return @view F.band[1:rows, 1:n]
end

# --- raw LAPACK ccalls (BlasFloat), reusing persistent buffers --------------
#
# The stdlib LAPACK wrappers (`geqp3!`, `ormqr!`, `tzrzf!`, `ormrz!`, `laic1!`)
# each allocate a fresh work buffer / return box on every call (measured: geqp3!
# ≈14 KB, ormqr! ≈33–67 KB, laic1! a few KB). To keep the warm solve and
# re-factor allocation-free we call the routines directly through
# libblastrampoline, reusing `F.work`/`F.tau`/`F.jpvt`/`F.wmin`/`F.wmax` with the
# same query-then-reuse pattern the LU kernels use for `sytrf`/`getrf`.

# geqp3: rank-revealing QR. `jpvt` MUST be zeroed before the call (a nonzero
# entry pins that column). Real and complex differ only in the rwork argument.
for (fname, elty, relty) in (
        (:dgeqp3_, :Float64, :Float64), (:sgeqp3_, :Float32, :Float32),
        (:zgeqp3_, :ComplexF64, :Float64), (:cgeqp3_, :ComplexF32, :Float32),
    )
    cmplx = !(eval(elty) <: Real)
    if cmplx
        @eval function _geqp3_into!(F::SpecializedQR{$elty})
            m, n = F.m, F.n
            A = F.factors
            (m == 0 || n == 0) && return F
            jpvt = F.jpvt
            tau = _resize!(F.tau, min(m, n))
            length(F.rwork) < 2n && _resize!(F.rwork, 2n)
            length(F.work) >= 1 || _resize!(F.work, 1)
            info = Ref{BlasInt}()
            lda = max(1, stride(A, 2))
            ccall(
                (@blasfunc($fname), libblastrampoline), Cvoid,
                (
                    Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                ),
                m, n, A, lda, jpvt, tau, F.work, BlasInt(-1), F.rwork, info
            )
            chkargsok(info[])
            lwork = max(1, Int(real(F.work[1])))
            length(F.work) < lwork && _resize!(F.work, lwork)
            ccall(
                (@blasfunc($fname), libblastrampoline), Cvoid,
                (
                    Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                ),
                m, n, A, lda, jpvt, tau, F.work, BlasInt(lwork), F.rwork, info
            )
            chkargsok(info[])
            return F
        end
    else
        @eval function _geqp3_into!(F::SpecializedQR{$elty})
            m, n = F.m, F.n
            A = F.factors
            (m == 0 || n == 0) && return F
            jpvt = F.jpvt
            tau = _resize!(F.tau, min(m, n))
            length(F.work) >= 1 || _resize!(F.work, 1)
            info = Ref{BlasInt}()
            lda = max(1, stride(A, 2))
            ccall(
                (@blasfunc($fname), libblastrampoline), Cvoid,
                (
                    Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                ),
                m, n, A, lda, jpvt, tau, F.work, BlasInt(-1), info
            )
            chkargsok(info[])
            lwork = max(1, Int(real(F.work[1])))
            length(F.work) < lwork && _resize!(F.work, lwork)
            ccall(
                (@blasfunc($fname), libblastrampoline), Cvoid,
                (
                    Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                ),
                m, n, A, lda, jpvt, tau, F.work, BlasInt(lwork), info
            )
            chkargsok(info[])
            return F
        end
    end
end

# ormqr/unmqr: apply Qᴴ (side='L'). `nrows` is the number of rows Q acts on (m),
# passed explicitly so the padded `C` buffer's leading dim is used as ldc — see
# the LU `fact` exact-size note: a slicing view would cost a small allocation.
for (fname, elty) in (
        (:dormqr_, :Float64), (:sormqr_, :Float32),
        (:zunmqr_, :ComplexF64), (:cunmqr_, :ComplexF32),
    )
    @eval function _ormqr_into!(
            F::SpecializedQR{$elty}, A::AbstractMatrix{$elty}, tau::Vector{$elty},
            k::Int, C::AbstractMatrix{$elty}, nrows::Int, trans::AbstractChar
        )
        m = nrows
        n = size(C, 2)
        (m == 0 || k == 0) && return C
        lda = max(1, stride(A, 2))
        ldc = max(1, stride(C, 2))
        length(F.work) >= 1 || _resize!(F.work, 1)
        info = Ref{BlasInt}()
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong,
            ),
            'L', trans, m, n, k, A, lda, tau, C, ldc, F.work, BlasInt(-1), info, 1, 1
        )
        chkargsok(info[])
        lwork = max(1, Int(real(F.work[1])))
        length(F.work) < lwork && _resize!(F.work, lwork)
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong,
            ),
            'L', trans, m, n, k, A, lda, tau, C, ldc, F.work, BlasInt(lwork), info, 1, 1
        )
        chkargsok(info[])
        return C
    end
end

# tzrzf/ztzrzf: complete the leading r×n trapezoid to r×r upper-triangular T11
# via an orthogonal Z (the gelsy minimum-norm step).
for (fname, elty) in (
        (:dtzrzf_, :Float64), (:stzrzf_, :Float32),
        (:ztzrzf_, :ComplexF64), (:ctzrzf_, :ComplexF32),
    )
    @eval function _tzrzf_into!(F::SpecializedQR{$elty}, A::AbstractMatrix{$elty}, tau::Vector{$elty})
        m, n = size(A)
        m == 0 && return A
        lda = max(1, stride(A, 2))
        length(F.work) >= 1 || _resize!(F.work, 1)
        info = Ref{BlasInt}()
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
            m, n, A, lda, tau, F.work, BlasInt(-1), info
        )
        chkargsok(info[])
        lwork = max(1, Int(real(F.work[1])))
        length(F.work) < lwork && _resize!(F.work, lwork)
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
            m, n, A, lda, tau, F.work, BlasInt(lwork), info
        )
        chkargsok(info[])
        return A
    end
end

# ormrz/unmrz: apply Zᴴ (side='L') to lift the r-vector to the min-norm n-vector.
for (fname, elty) in (
        (:dormrz_, :Float64), (:sormrz_, :Float32),
        (:zunmrz_, :ComplexF64), (:cunmrz_, :ComplexF32),
    )
    @eval function _ormrz_into!(
            F::SpecializedQR{$elty}, A::AbstractMatrix{$elty}, tau::Vector{$elty},
            k::Int, l::Int, C::AbstractMatrix{$elty}, nrows::Int, trans::AbstractChar
        )
        m = nrows
        n = size(C, 2)
        (m == 0 || k == 0) && return C
        lda = max(1, stride(A, 2))
        ldc = max(1, stride(C, 2))
        length(F.work) >= 1 || _resize!(F.work, 1)
        info = Ref{BlasInt}()
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong,
            ),
            'L', trans, m, n, k, l, A, lda, tau, C, ldc, F.work, BlasInt(-1), info, 1, 1
        )
        chkargsok(info[])
        lwork = max(1, Int(real(F.work[1])))
        length(F.work) < lwork && _resize!(F.work, lwork)
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong,
            ),
            'L', trans, m, n, k, l, A, lda, tau, C, ldc, F.work, BlasInt(lwork), info, 1, 1
        )
        chkargsok(info[])
        return C
    end
end

# trtrs: triangular solve (no work buffer). Solves the leading nA×nA block.
for (fname, elty) in (
        (:dtrtrs_, :Float64), (:strtrs_, :Float32),
        (:ztrtrs_, :ComplexF64), (:ctrtrs_, :ComplexF32),
    )
    @eval function _trtrs_into!(
            ::SpecializedQR{$elty}, uplo::AbstractChar, A::AbstractMatrix{$elty},
            B::AbstractMatrix{$elty}, nA::Int
        )
        nA == 0 && return B
        nrhs = size(B, 2)
        lda = max(1, stride(A, 2))
        ldb = max(1, stride(B, 2))
        info = Ref{BlasInt}()
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong, Clong,
            ),
            uplo, 'N', 'N', nA, nrhs, A, lda, B, ldb, info, 1, 1, 1
        )
        chkargsok(info[])
        return B
    end
end

# laic1/zlaic1: incremental condition estimator. `Ref`s do not escape, so the
# compiler elides them (0-alloc), exactly like the LU `sytrf` info/query Refs.
for (fname, elty, relty) in (
        (:dlaic1_, :Float64, :Float64), (:slaic1_, :Float32, :Float32),
        (:zlaic1_, :ComplexF64, :Float64), (:claic1_, :ComplexF32, :Float32),
    )
    @eval @inline function _laic1(
            ::Type{$elty}, job::Int, x::AbstractVector{$elty}, sest::$relty,
            w::AbstractVector{$elty}, gamma::$elty
        )
        j = length(x)
        sestpr = Ref{$relty}()
        s = Ref{$elty}()
        c = Ref{$elty}()
        ccall(
            (@blasfunc($fname), libblastrampoline), Cvoid,
            (
                Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{$relty}, Ptr{$elty},
                Ref{$elty}, Ref{$relty}, Ref{$elty}, Ref{$elty},
            ),
            job, j, x, sest, w, gamma, sestpr, s, c
        )
        return sestpr[], s[], c[]
    end
end

# --- work-buffer reservation ------------------------------------------------

_reserve_qr_work!(F::SpecializedQR, ::Int, ::Int; deficient::Bool = false) = F

# Size F.work to the max optimal lwork over geqp3/ormqr (and tzrzf/ormrz when
# `deficient`). No-op for non-BlasFloat element types (generic path, no work).
for (geqp3f, ormqrf, tzrzff, ormrzf, elty, relty) in (
        (:dgeqp3_, :dormqr_, :dtzrzf_, :dormrz_, :Float64, :Float64),
        (:sgeqp3_, :sormqr_, :stzrzf_, :sormrz_, :Float32, :Float32),
        (:zgeqp3_, :zunmqr_, :ztzrzf_, :zunmrz_, :ComplexF64, :Float64),
        (:cgeqp3_, :cunmqr_, :ctzrzf_, :cunmrz_, :ComplexF32, :Float32),
    )
    cmplx = !(eval(elty) <: Real)
    @eval function _reserve_qr_work!(F::SpecializedQR{$elty}, m::Int, n::Int; deficient::Bool = false)
        (m == 0 || n == 0) && return F
        mn = min(m, n)
        A = _ensure_qr_factors!(F, m, n)
        lda = max(1, stride(A, 2))
        length(F.work) >= 1 || _resize!(F.work, 1)
        $(cmplx ? :(length(F.rwork) < 2n && _resize!(F.rwork, 2n)) : :nothing)
        info = Ref{BlasInt}()
        tau = _resize!(F.tau, mn)
        jpvt = _resize!(F.jpvt, n)
        lwork = 1
        # geqp3 query
        $(
            if cmplx
                quote
                    ccall(
                        (@blasfunc($geqp3f), libblastrampoline), Cvoid,
                        (
                            Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                            Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                        ),
                        m, n, A, lda, jpvt, tau, F.work, BlasInt(-1), F.rwork, info
                    )
                end
            else
                quote
                    ccall(
                        (@blasfunc($geqp3f), libblastrampoline), Cvoid,
                        (
                            Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                            Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                        ),
                        m, n, A, lda, jpvt, tau, F.work, BlasInt(-1), info
                    )
                end
            end
        )
        lwork = max(lwork, Int(real(F.work[1])))
        # ormqr query (apply Qᴴ to a single column padded to max(m,n) rows)
        rhs = _ensure_qr_rhs!(F, max(m, n), 1)
        ldc = max(1, stride(rhs, 2))
        ccall(
            (@blasfunc($ormqrf), libblastrampoline), Cvoid,
            (
                Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong,
            ),
            'L', 'N', m, 1, mn, A, lda, tau, rhs, ldc, F.work, BlasInt(-1), info, 1, 1
        )
        lwork = max(lwork, Int(real(F.work[1])))
        if deficient
            tzf = _ensure_qr_tzfactors!(F, mn, n)
            ldt = max(1, stride(tzf, 2))
            tau2 = _resize!(F.tau2, mn)
            ccall(
                (@blasfunc($tzrzff), libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                mn, n, tzf, ldt, tau2, F.work, BlasInt(-1), info
            )
            lwork = max(lwork, Int(real(F.work[1])))
            ccall(
                (@blasfunc($ormrzf), libblastrampoline), Cvoid,
                (
                    Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong,
                ),
                'L', 'N', n, 1, mn, n - mn, tzf, ldt, tau2, rhs, ldc, F.work, BlasInt(-1), info, 1, 1
            )
            lwork = max(lwork, Int(real(F.work[1])))
        end
        lwork = max(1, lwork)
        length(F.work) < lwork && _resize!(F.work, lwork)
        return F
    end
end

"""
    reserve!(F::SpecializedQR, m, n; deficient = false, nrhs = 1) -> F

Grow `F`'s buffers so a problem of size up to `m×n` (with up to `nrhs`
right-hand sides) factors and solves with **zero** allocations. Pass
`deficient = true` to also reserve the complete-orthogonal (`tzrzf`) buffers and
the larger work needed by the rank-deficient minimum-norm solve. Buffers only
ever grow within the exact-size dense buffers' footprint, so reserving the
largest problem you will solve makes every later `specializingqr!` + `ldiv!` at
that size or smaller allocation-free.
"""
function reserve!(
        F::SpecializedQR{T}, m::Integer, n::Integer;
        deficient::Bool = false, nrhs::Integer = 1, kl::Integer = 0, ku::Integer = 0
    ) where {T}
    mm = Int(m)
    nn = Int(n)
    mn = min(mm, nn)
    _ensure_qr_factors!(F, mm, nn)
    _resize!(F.tau, mn)
    _resize!(F.jpvt, nn)
    _resize!(F.tau2, mn)
    _ensure_qr_rhs!(F, max(mm, nn), Int(nrhs))
    _resize!(F.wmin, mn)
    _resize!(F.wmax, mn)
    _resize!(F.dvec, mn)               # DIAGONAL storage / TRIDIAGONAL U diagonal
    _resize!(F.gbuf, mn)               # condition-gate laic1 scratch
    # structured tridiagonal/banded buffers (cheap O(n); always reserved)
    _resize!(F.dl, max(nn - 1, 0))
    _resize!(F.du, max(nn - 1, 0))
    _resize!(F.du2, max(nn - 2, 0))
    _resize!(F.ipiv, nn)
    if kl > 0 || ku > 0
        _ensure_qr_band!(F, 2 * Int(kl) + Int(ku) + 1, nn)
    end
    if T <: Complex
        _resize!(F.rwork, 2nn)
    end
    if deficient
        _ensure_qr_tzfactors!(F, mn, nn)
    end
    _reserve_qr_work!(F, mm, nn; deficient = deficient)
    return F
end

"""
    SpecializedQR{T}(m, n; deficient = false, nrhs = 1, kl = 0, ku = 0)

Construct a workspace with buffers pre-sized (via [`reserve!`](@ref)) for an
`m×n` matrix of element type `T`, so that subsequent factorizations and solves
at size ≤ `m×n` are allocation-free. Pass `kl`/`ku` to also reserve the banded
`AB` buffer for the `BANDED` structured fast path.
"""
SpecializedQR{T}(m::Integer, n::Integer; deficient::Bool = false, nrhs::Integer = 1, kl::Integer = 0, ku::Integer = 0) where {T} =
    reserve!(SpecializedQR{T}(), m, n; deficient = deficient, nrhs = nrhs, kl = kl, ku = ku)

# --- rank detection ---------------------------------------------------------

# BlasFloat: laic1 incremental condition estimator (the gelsy algorithm Julia's
# QRPivoted ldiv! uses), so the revealed rank — and hence the solution — matches
# `qr(A, ColumnNorm())` and `pinv`. Allocation-free (reuses F.wmin/F.wmax).
function _detect_rank!(F::SpecializedQR{T}) where {T <: BlasFloat}
    m, n = F.m, F.n
    mn = min(m, n)
    A = F.factors
    rcond = F.rtol
    (mn == 0 || length(A) == 0) && return 0
    R = real(T)
    smax = R(abs(A[1, 1]))
    smin = smax
    smax == 0 && return 0
    wmin = _resize!(F.wmin, mn)
    wmax = _resize!(F.wmax, mn)
    @inbounds for k in 1:mn
        wmin[k] = zero(T)
        wmax[k] = zero(T)
    end
    wmin[1] = one(T)
    wmax[1] = one(T)
    rnk = 1
    @inbounds while rnk < mn
        i = rnk + 1
        gamma = A[i, i]
        sminpr, s1, c1 = _laic1(T, 2, view(wmin, 1:rnk), smin, view(A, 1:rnk, i), gamma)
        smaxpr, s2, c2 = _laic1(T, 1, view(wmax, 1:rnk), smax, view(A, 1:rnk, i), gamma)
        smaxpr * rcond > sminpr && break
        for k in 1:rnk
            wmin[k] *= s1
            wmax[k] *= s2
        end
        wmin[i] = c1
        wmax[i] = c2
        smin = sminpr
        smax = smaxpr
        rnk += 1
    end
    return rnk
end

# Generic: diagonal threshold. Column pivoting makes |R[i,i]| non-increasing, so
# the rank is the count of diagonal entries above the tolerance.
function _detect_rank_generic(A::AbstractMatrix{T}, m::Int, n::Int, rtol) where {T}
    mn = min(m, n)
    mn == 0 && return 0
    rmax = abs(A[1, 1])
    rmax == 0 && return 0
    tol = rmax * rtol
    rnk = 0
    @inbounds for i in 1:mn
        abs(A[i, i]) > tol || break
        rnk += 1
    end
    return rnk
end

# --- factorization ----------------------------------------------------------

"""
    specializingqr(A; rtol = min(m,n)*eps, minnorm = true, fallback = true) -> SpecializedQR

Compute a rank-revealing column-pivoted QR factorization of the (possibly
rectangular / rank-deficient) matrix `A`, returning a [`SpecializedQR`](@ref)
workspace. Always returns the same concrete type regardless of `A`'s shape or
rank (type-stable).

`F \\ b` / `ldiv!(x, F, b)` then returns the least-squares solution
(minimizing `‖Ax-b‖`); when `A` is rank-deficient it returns the **minimum-norm**
least-squares solution (matching `qr(A, ColumnNorm()) \\ b` and `pinv(A)*b`) and
**never throws** on singular input. `rtol` sets the relative tolerance for rank
revelation; `minnorm = false` selects the cheaper rank-truncated *basic*
solution (free variables zeroed) on the `BlasFloat` path; `fallback = false`
leaves the workspace unfactored for a host that wants to own the QR.

Integer / `Rational` element types are promoted via `float` (QR requires square
roots, so an exact-rational QR is not possible); `BigFloat` and other generic
element types use a generic column-pivoted QR with a rank-truncated *basic*
solve (the generic path returns the basic, not the minimum-norm, solution).
"""
function specializingqr(A::AbstractMatrix{T}; kwargs...) where {T}
    S = float(T)
    F = SpecializedQR{S}()
    Af = S === T ? A : convert(AbstractMatrix{S}, A)
    return specializingqr!(F, Af; kwargs...)
end

"""
    specializingqr!(F::SpecializedQR, A; rtol = -1, minnorm = true, fallback = true,
                    detect_structure = true) -> F

Re-factor `A` into the existing workspace `F`, reusing (and growing only as
needed) `F`'s buffers. `rtol < 0` uses the default `min(m,n)*eps(real(T))`. See
[`specializingqr`](@ref) for the keyword semantics. `detect_structure = false`
disables the structured fast paths and forces the dense rank-revealing `geqp3`
path for every input (the pre-structure behavior).
"""
function specializingqr!(
        F::SpecializedQR{T}, A::AbstractMatrix{T};
        rtol::Real = -1, minnorm::Bool = true, fallback::Bool = true,
        detect_structure::Bool = true
    ) where {T}
    m, n = size(A)
    F.m = m
    F.n = n
    F.info = 0
    F.minnorm = minnorm
    F.form = GENERAL
    F.rtol = rtol < 0 ? _default_rtol(T, m, n) : real(T)(rtol)
    if !fallback
        F.factored = false
        F.status = QR_UNFACTORED
        F.rank = 0
        return F
    end
    F.factored = true
    _factorize_qr!(F, A, detect_structure)
    return F
end

# BlasFloat: structure-aware. Square inputs are classified by `detect_form`; a
# DIAGONAL matrix uses the exact O(n) rank-revealing diagonal solve, and a
# triangular/bidiagonal matrix uses a triangular solve *only* when a conservative
# condition gate certifies it is comfortably full rank (so the structured solve
# provably equals geqp3/pinv). Everything else — unstructured, symmetric,
# rectangular, near-singular structured, tridiagonal/banded — falls through to
# the dense rank-revealing geqp3 path, which guarantees the rank + min-norm +
# never-throw contract.
function _factorize_qr!(F::SpecializedQR{T}, A::AbstractMatrix{T}, detect::Bool) where {T <: BlasFloat}
    m, n = F.m, F.n
    if detect && m == n && m > 0
        d = detect_form(A)
        f = d.form
        if f == DIAGONAL
            return _factorize_qr_diagonal!(F, A)
        elseif f == LOWER_TRIANGULAR || f == UPPER_TRIANGULAR ||
                f == LOWER_BIDIAGONAL || f == UPPER_BIDIAGONAL
            C = _ensure_qr_factors!(F, n, n)
            copyto!(C, A)            # a triangular A IS its own R
            lower = f == LOWER_TRIANGULAR || f == LOWER_BIDIAGONAL
            if _gate_fullrank(F, C, lower)
                F.form = f
                F.info = 0
                F.rank = n
                F.status = QR_FULLRANK
                return F
            end
            # gate failed (near-singular / ill-conditioned): fall through to geqp3.
        elseif f == TRIDIAGONAL
            _factorize_qr_tridiagonal!(F, A)
            # Varah O(n) early-accept (diagonally dominant) skips the O(n²) gate.
            if F.info == 0 && (_varah_band_accept(A, n, 1, 1, F.rtol) || _gate_fullrank_band!(F, true))
                F.form = TRIDIAGONAL
                F.rank = n
                F.status = QR_FULLRANK
                return F
            end
        elseif f == BANDED
            _factorize_qr_banded!(F, A, d.kl, d.ku)
            if F.info == 0 && (_varah_band_accept(A, n, d.kl, d.ku, F.rtol) || _gate_fullrank_band!(F, false))
                F.form = BANDED
                F.rank = n
                F.status = QR_FULLRANK
                return F
            end
        end
    end
    # dense rank-revealing geqp3 path (F.form stays GENERAL)
    F.info = 0                     # clear any info a fallen-through structured factor set
    C = _ensure_qr_factors!(F, m, n)
    copyto!(C, A)
    _resize!(F.jpvt, n)
    fill!(F.jpvt, 0)            # 0 ⇒ free pivoting (true rank revelation)
    _geqp3_into!(F)
    r = _detect_rank!(F)
    F.rank = r
    F.status = r == n ? QR_FULLRANK : QR_DEFICIENT
    # Prepare the complete-orthogonal factor once per factorization (reused over
    # right-hand sides) for the minimum-norm deficient solve.
    if F.status == QR_DEFICIENT && F.minnorm && 0 < r < n
        tzf = _ensure_qr_tzfactors!(F, r, n)
        @inbounds for j in 1:n, i in 1:r
            tzf[i, j] = C[i, j]
        end
        _resize!(F.tau2, r)
        _tzrzf_into!(F, tzf, F.tau2)
    end
    return F
end

# Generic element types: generic column-pivoted QR stored in the same fields.
# Structure detection is a BlasFloat-only optimization; the generic path keeps
# its documented rank-truncated basic-solve contract (so `detect` is ignored).
function _factorize_qr!(F::SpecializedQR{T}, A::AbstractMatrix{T}, detect::Bool) where {T}
    m, n = F.m, F.n
    F.form = GENERAL
    fac = qr(A, ColumnNorm())
    C = _ensure_qr_factors!(F, m, n)
    copyto!(C, fac.factors)
    nt = length(fac.τ)
    _resize!(F.tau, nt)
    copyto!(F.tau, fac.τ)
    _resize!(F.jpvt, n)
    @inbounds for i in 1:n
        F.jpvt[i] = fac.p[i]
    end
    r = _detect_rank_generic(C, m, n, F.rtol)
    F.rank = r
    F.status = r == n ? QR_FULLRANK : QR_DEFICIENT
    return F
end

# --- structured fast paths (BlasFloat, square) ------------------------------

# DIAGONAL: the diagonal entries ARE the singular values, so the rank-revealing
# rank and the minimum-norm least-squares solution are both exact and O(n) — no
# QR, no fallback, singular-safe. rank = count(|dᵢ| ≥ max|d|·rtol) uses the
# INCLUSIVE ≥ to match the laic1 boundary the geqp3 path uses.
function _factorize_qr_diagonal!(F::SpecializedQR{T}, A::AbstractMatrix{T}) where {T}
    n = F.n
    d = _resize!(F.dvec, n)
    @inbounds for i in 1:n
        d[i] = A[i, i]
    end
    F.form = DIAGONAL
    F.info = 0
    dmax = zero(real(T))
    @inbounds for i in 1:n
        a = abs(d[i])
        a > dmax && (dmax = a)
    end
    tol = dmax * F.rtol
    r = 0
    if dmax != 0
        @inbounds for i in 1:n
            abs(d[i]) >= tol && (r += 1)
        end
    end
    F.rank = r
    F.status = r == n ? QR_FULLRANK : QR_DEFICIENT
    return F
end

# Conservative full-rank / well-conditioning gate for a SQUARE triangular (or
# bidiagonal, which is triangular) matrix `C`, which IS its own R. Returns true
# only when `C` is comfortably full rank: the laic1 incremental condition
# estimate must clear `rtol` with a safety `margin`, so the structured trtrs
# solve provably reproduces the geqp3/pinv rank and solution. A near-singular,
# ill-conditioned, or zero-pivot `C` returns false and routes to the dense
# rank-revealing geqp3 path. 0-alloc (reuses F.wmin/F.wmax; for a lower-
# triangular C, F.dvec doubles as a contiguous row-slice buffer — safe because
# the DIAGONAL and triangular paths are mutually exclusive). The margin is
# empirical (≥16 left zero unsafe cases over 20000 trials); geqp3 is the
# contract source of truth, so the gate is deliberately conservative.
const _QR_GATE_MARGIN = 32

function _gate_fullrank(F::SpecializedQR{T}, C::AbstractMatrix{T}, lower::Bool) where {T <: BlasFloat}
    n = F.n
    n == 0 && return false
    _first_zero_diag(C, n) == 0 || return false   # exact zero pivot ⇒ trtrs would throw
    Re = real(T)
    smax = Re(abs(C[1, 1]))
    smin = smax
    smax == 0 && return false
    thr = _QR_GATE_MARGIN * F.rtol
    wmin = _resize!(F.wmin, n)
    wmax = _resize!(F.wmax, n)
    g = _resize!(F.dvec, n)        # contiguous slice buffer for the raw laic1 ccall
    @inbounds for k in 1:n
        wmin[k] = zero(T)
        wmax[k] = zero(T)
    end
    wmin[1] = one(T)
    wmax[1] = one(T)
    @inbounds for rnk in 1:(n - 1)
        i = rnk + 1
        # The i-th column of the upper-triangular factor, copied into the
        # contiguous buffer `g` so the raw laic1 ccall sees unit stride (a
        # strided row/column view would either be a non-unit-stride ptr or a
        # type-unstable Union of two SubArray types). For a lower-triangular C we
        # estimate σ via Cᵀ (same singular values), whose i-th column is row i.
        for k in 1:rnk
            g[k] = lower ? C[i, k] : C[k, i]
        end
        w = view(g, 1:rnk)
        gamma = C[i, i]
        sminpr, s1, c1 = _laic1(T, 2, view(wmin, 1:rnk), smin, w, gamma)
        smaxpr, s2, c2 = _laic1(T, 1, view(wmax, 1:rnk), smax, w, gamma)
        smaxpr * thr > sminpr && return false
        for k in 1:rnk
            wmin[k] *= s1
            wmax[k] *= s2
        end
        wmin[i] = c1
        wmax[i] = c2
        smin = sminpr
        smax = smaxpr
    end
    return true
end

# TRIDIAGONAL / BANDED: reuse the LU solver's hand-rolled, non-throwing
# `_gttrf!` / `_banded_lu!` (which return `info` rather than throwing on a zero
# pivot) into the QR workspace's band buffers.
function _factorize_qr_tridiagonal!(F::SpecializedQR{T}, A::AbstractMatrix{T}) where {T}
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
    return F
end

function _factorize_qr_banded!(F::SpecializedQR{T}, A::AbstractMatrix{T}, kl::Int, ku::Int) where {T}
    n = F.n
    F.kl = kl
    F.ku = ku
    rows = 2kl + ku + 1
    AB = _ensure_qr_band!(F, rows, n)
    fill!(AB, zero(T))
    @inbounds for j in 1:n
        for i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = A[i, j]
        end
    end
    ipiv = _resize!(F.ipiv, n)
    F.info = _banded_lu!(AB, kl, ku, n, ipiv)
    return F
end

# Full-rank / well-conditioning gate for a TRIDIAGONAL / BANDED matrix, run on
# the *upper factor* U produced by `_gttrf!` / `_banded_lu!`: U is upper-
# triangular (banded), and cond(U) tracks cond(A) under partial pivoting, so the
# same laic1 incremental estimate + safety margin used for the triangular gate
# applies (verified: 0 unsafe over 5000 wildly-conditioned tridiagonals). The
# column of U above the diagonal is reconstructed from the compact storage into
# the contiguous `gbuf` (kept zero between steps via the reset trick), so the raw
# laic1 ccall sees unit stride. 0-alloc (reuses F.wmin/F.wmax/F.gbuf).
#
# Provable strict-diagonal-dominance early-accept, read straight from the
# ORIGINAL band of A (O(n·(kl+ku)), no buffers). By Varah's theorem a strictly
# diagonally dominant A has σ_min ≥ β = min_i(|aᵢᵢ|−Rᵢ) and σ_max ≤ α =
# max_i(|aᵢᵢ|+Rᵢ) (Rᵢ = the in-band off-diagonal row sum), so σ_min/σ_max ≥
# β/(√n·α). Accepting only when β/α ≥ n·rtol (n ≥ √n) guarantees the matrix
# clears geqp3's rtol threshold, so the structured solve provably equals geqp3 —
# the O(n²) laic1 sweep is skipped for the common (diagonally-dominant) case.
# It is a true LOWER bound, so it can never certify a rank-deficient matrix
# (verified: 0 unsafe over adversarial barely-dominant probes at n up to 800);
# a non-dominant A declines here and falls through to the laic1 gate, contract
# unchanged.
function _varah_band_accept(A::AbstractMatrix{T}, n::Int, kl::Int, ku::Int, rtol) where {T <: BlasFloat}
    n == 0 && return false
    Re = real(T)
    mind = Re(Inf)
    maxs = zero(Re)
    @inbounds for i in 1:n
        aii = abs(A[i, i])
        ri = zero(Re)
        for j in max(1, i - ku):min(n, i + kl)
            j == i && continue
            ri += abs(A[i, j])
        end
        aii > ri || return false           # not strictly diagonally dominant
        d = aii - ri
        d < mind && (mind = d)
        s = aii + ri
        s > maxs && (maxs = s)
    end
    return maxs > 0 && mind >= n * rtol * maxs
end

function _gate_fullrank_band!(F::SpecializedQR{T}, tridiag::Bool) where {T <: BlasFloat}
    n = F.n
    n == 0 && return false
    Re = real(T)
    kv = F.kl + F.ku
    u11 = tridiag ? F.dvec[1] : F.band[kv + 1, 1]
    smax = Re(abs(u11))
    smin = smax
    smax == 0 && return false
    thr = _QR_GATE_MARGIN * F.rtol
    wmin = _resize!(F.wmin, n)
    wmax = _resize!(F.wmax, n)
    g = _resize!(F.gbuf, n)
    @inbounds for k in 1:n
        wmin[k] = zero(T)
        wmax[k] = zero(T)
        g[k] = zero(T)
    end
    wmin[1] = one(T)
    wmax[1] = one(T)
    @inbounds for rnk in 1:(n - 1)
        i = rnk + 1
        # w = U[1:rnk, i] written into g[1:rnk] (g is otherwise all-zero)
        local lo, gamma
        if tridiag
            lo = max(1, i - 2)
            i - 1 >= 1 && (g[i - 1] = F.du[i - 1])
            i - 2 >= 1 && (g[i - 2] = F.du2[i - 2])
            gamma = F.dvec[i]
        else
            lo = max(1, i - kv)
            for r in lo:(i - 1)
                g[r] = F.band[kv + 1 + r - i, i]
            end
            gamma = F.band[kv + 1, i]
        end
        w = view(g, 1:rnk)
        sminpr, s1, c1 = _laic1(T, 2, view(wmin, 1:rnk), smin, w, gamma)
        smaxpr, s2, c2 = _laic1(T, 1, view(wmax, 1:rnk), smax, w, gamma)
        for r in lo:(i - 1)         # restore g to all-zero for the next step
            g[r] = zero(T)
        end
        smaxpr * thr > sminpr && return false
        for k in 1:rnk
            wmin[k] *= s1
            wmax[k] *= s2
        end
        wmin[i] = c1
        wmax[i] = c2
        smin = sminpr
        smax = smaxpr
    end
    return true
end

# --- solving ----------------------------------------------------------------

"""
    ldiv!(x, F::SpecializedQR, b) -> x

Solve the least-squares problem `min ‖A x - b‖` using the rank-revealing QR
stored in `F`, writing the solution into `x` (`length(x) == size(A, 2)`,
`length(b) == size(A, 1)`). For rank-deficient `A` this is the minimum-norm
least-squares solution (basic if `minnorm = false`, or on the generic path).
`b` may be a vector or a matrix (multiple right-hand sides). Never throws on
singular `A`.
"""
function LinearAlgebra.ldiv!(x::AbstractVecOrMat, F::SpecializedQR, b::AbstractVecOrMat)
    F.factored || throw(
        ArgumentError(
            "SpecializedQR holds an unfactored matrix (fallback=false); the host " *
                "must supply its own QR. Check `isfactored(F)` / `matrixform(F)` first."
        )
    )
    size(b, 1) == F.m || throw(DimensionMismatch("rhs has $(size(b, 1)) rows, expected $(F.m)"))
    size(x, 1) == F.n || throw(DimensionMismatch("solution has $(size(x, 1)) rows, expected $(F.n)"))
    size(x, 2) == size(b, 2) || throw(DimensionMismatch("x and b have different right-hand-side counts"))
    _qr_solve!(x, F, b)
    return x
end

# In-place: only valid for square systems (x and b share length). Rectangular
# shapes must use the 3-arg form with a properly-sized x.
function LinearAlgebra.ldiv!(F::SpecializedQR, b::AbstractVecOrMat)
    F.m == F.n || throw(
        DimensionMismatch("2-arg ldiv!(F, b) requires a square system; use ldiv!(x, F, b) with size(x,1)==$(F.n)")
    )
    return ldiv!(b, F, b)
end

function Base.:\(F::SpecializedQR{T}, b::AbstractVecOrMat) where {T}
    x = b isa AbstractVector ? Vector{T}(undef, F.n) : Matrix{T}(undef, F.n, size(b, 2))
    return ldiv!(x, F, b)
end

@inline function _scatter_perm!(x::AbstractVector, rhs::AbstractMatrix, jpvt, n::Int)
    @inbounds for i in 1:n
        x[jpvt[i]] = rhs[i, 1]
    end
    return x
end

@inline function _scatter_perm!(x::AbstractMatrix, rhs::AbstractMatrix, jpvt, n::Int)
    @inbounds for c in axes(x, 2), i in 1:n
        x[jpvt[i], c] = rhs[i, c]
    end
    return x
end

# Copy b into the padded rhs buffer (leading dim max(m,n)); rows m+1:end zeroed.
@inline function _load_rhs!(F::SpecializedQR{T}, b::AbstractVecOrMat) where {T}
    m = F.m
    nrhs = size(b, 2)
    rhs = _ensure_qr_rhs!(F, max(m, F.n), nrhs)
    fill!(rhs, zero(T))
    if b isa AbstractVector
        @inbounds for i in 1:m
            rhs[i, 1] = b[i]
        end
    else
        @inbounds for c in 1:nrhs, i in 1:m
            rhs[i, c] = b[i, c]
        end
    end
    return rhs
end

# DIAGONAL structured solve: the minimum-norm least-squares solution xᵢ = bᵢ/dᵢ
# for kept coordinates (|dᵢ| ≥ tol) and 0 for the dropped (free) ones. This is
# the QR min-norm contract (NOT the LU `b ./ d`, which gives Inf on a zero dᵢ):
# zeroing the free coordinates is the min-norm choice because a diagonal matrix
# has orthogonal columns. 0-alloc, Inf-safe (a sub-tolerance dᵢ is never divided).
function _qr_solve_diagonal!(x::AbstractVecOrMat, F::SpecializedQR{T}, b::AbstractVecOrMat) where {T <: BlasFloat}
    n = F.n
    d = F.dvec
    # Full-rank diagonal: the factorization already certified every |dᵢ| ≥ tol, so
    # the solve is the plain reciprocal xᵢ = bᵢ/dᵢ — no max-scan and no per-element
    # tolerance branch (both invariant in the diagonal data fixed at factor time).
    if F.status == QR_FULLRANK
        if b isa AbstractVector
            @inbounds for i in 1:n
                x[i] = b[i] / d[i]
            end
        else
            @inbounds for c in axes(x, 2), i in 1:n
                x[i, c] = b[i, c] / d[i]
            end
        end
        return x
    end
    dmax = zero(real(T))
    @inbounds for i in 1:n
        a = abs(d[i])
        a > dmax && (dmax = a)
    end
    # All-zero diagonal ⇒ rank 0 ⇒ zero solution (and tol would be 0, which would
    # otherwise make the `>=` keep — and divide by — the zero entries).
    if dmax == 0
        fill!(x, zero(T))
        return x
    end
    tol = dmax * F.rtol
    if b isa AbstractVector
        @inbounds for i in 1:n
            x[i] = abs(d[i]) >= tol ? b[i] / d[i] : zero(T)
        end
    else
        @inbounds for c in axes(x, 2), i in 1:n
            x[i, c] = abs(d[i]) >= tol ? b[i, c] / d[i] : zero(T)
        end
    end
    return x
end

# Triangular (or bidiagonal) structured solve. The factorization gate certified
# `A` is comfortably full rank and square, so `A` IS its own R and the unique
# solution from a single triangular solve equals the least-squares = minimum-norm
# solution. No column permutation (no pivoting), so no scatter. 0-alloc (trtrs
# has no work buffer; reuses the exact-size F.rhs buffer).
function _qr_solve_triangular!(x::AbstractVecOrMat, F::SpecializedQR{T}, b::AbstractVecOrMat, uplo::Char) where {T <: BlasFloat}
    n = F.n
    nrhs = size(b, 2)
    rhs = _ensure_qr_rhs!(F, max(F.m, n), nrhs)
    if b isa AbstractVector
        @inbounds for i in 1:n
            rhs[i, 1] = b[i]
        end
    else
        @inbounds for c in 1:nrhs, i in 1:n
            rhs[i, c] = b[i, c]
        end
    end
    _trtrs_into!(F, uplo, F.factors, rhs, n)
    if x isa AbstractVector
        @inbounds for i in 1:n
            x[i] = rhs[i, 1]
        end
    else
        @inbounds for c in 1:nrhs, i in 1:n
            x[i, c] = rhs[i, c]
        end
    end
    return x
end

# TRIDIAGONAL / BANDED full-rank solve (gated): copy the RHS into x and solve in
# place with the shared gttrs/gbtrs cores. Square (m == n), no permutation beyond
# the structured factor's own pivots, so x has the same shape as b.
function _qr_solve_tridiagonal!(x::AbstractVecOrMat, F::SpecializedQR{T}, b::AbstractVecOrMat) where {T <: BlasFloat}
    copyto!(x, b)
    _tridiag_solve!(x, F.dl, F.dvec, F.du, F.du2, F.ipiv, F.n)
    return x
end

function _qr_solve_banded!(x::AbstractVecOrMat, F::SpecializedQR{T}, b::AbstractVecOrMat) where {T <: BlasFloat}
    copyto!(x, b)
    AB = @view F.band[1:(2F.kl + F.ku + 1), 1:F.n]
    _banded_solve!(x, AB, F.ipiv, F.kl, F.ku, F.n)
    return x
end

function _qr_solve!(x::AbstractVecOrMat, F::SpecializedQR{T}, b::AbstractVecOrMat) where {T <: BlasFloat}
    if F.form == DIAGONAL
        return _qr_solve_diagonal!(x, F, b)
    elseif F.form == LOWER_TRIANGULAR || F.form == LOWER_BIDIAGONAL
        return _qr_solve_triangular!(x, F, b, 'L')
    elseif F.form == UPPER_TRIANGULAR || F.form == UPPER_BIDIAGONAL
        return _qr_solve_triangular!(x, F, b, 'U')
    elseif F.form == TRIDIAGONAL
        return _qr_solve_tridiagonal!(x, F, b)
    elseif F.form == BANDED
        return _qr_solve_banded!(x, F, b)
    end
    # dense rank-revealing geqp3 path (F.form == GENERAL)
    m, n, r = F.m, F.n, F.rank
    fill!(x, zero(T))
    r == 0 && return x
    rhs = _load_rhs!(F, b)
    trans = T <: Complex ? 'C' : 'T'
    _ormqr_into!(F, F.factors, F.tau, min(m, n), rhs, m, trans)
    if F.status == QR_FULLRANK
        _trtrs_into!(F, 'U', F.factors, rhs, n)
    elseif F.minnorm && r < n
        _trtrs_into!(F, 'U', F.tzfactors, rhs, r)
        nrhs = size(rhs, 2)
        @inbounds for c in 1:nrhs, i in (r + 1):n
            rhs[i, c] = zero(T)
        end
        _ormrz_into!(F, F.tzfactors, F.tau2, r, n - r, rhs, n, trans)
    else  # basic (rank-truncated) solution: free variables zeroed
        _trtrs_into!(F, 'U', F.factors, rhs, r)
        nrhs = size(rhs, 2)
        @inbounds for c in 1:nrhs, i in (r + 1):n
            rhs[i, c] = zero(T)
        end
    end
    _scatter_perm!(x, rhs, F.jpvt, n)
    return x
end

# Generic: rank-truncated basic solve (Julia's generic QRPivoted ldiv! is NOT
# rank-safe and blows up on singular input, so we do our own rank truncation).
function _qr_solve!(x::AbstractVecOrMat, F::SpecializedQR{T}, b::AbstractVecOrMat) where {T}
    m, n, r = F.m, F.n, F.rank
    fill!(x, zero(T))
    r == 0 && return x
    Q = LinearAlgebra.QRPackedQ(F.factors, F.tau)
    qtb = Q' * b
    Rtri = UpperTriangular(view(F.factors, 1:r, 1:r))
    if b isa AbstractVector
        y = Rtri \ view(qtb, 1:r)
        @inbounds for i in 1:r
            x[F.jpvt[i]] = y[i]
        end
    else
        y = Rtri \ qtb[1:r, :]
        @inbounds for c in axes(x, 2), i in 1:r
            x[F.jpvt[i], c] = y[i, c]
        end
    end
    return x
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

            # QR: full-rank square/over/under and a rank-deficient case, with
            # the minimum-norm and basic solve paths.
            Asq = [cv(i == j ? n + 1 : 1 / (i + 2j), i == j ? 0 : 1 / (i + j + 1)) for i in 1:n, j in 1:n]
            Aover = [cv(1 / (i + j), 1 / (i + 2j + 1)) for i in 1:(n + 2), j in 1:n]
            Aunder = [cv(1 / (i + j), 1 / (i + 2j + 1)) for i in 1:n, j in 1:(n + 2)]
            Adef = [cv(i, 0) * cv(j, 0) + cv((-1)^(i + j), 0) for i in 1:n, j in 1:n]
            for (A, mm, nn) in ((Asq, n, n), (Aover, n + 2, n), (Aunder, n, n + 2), (Adef, n, n))
                bq = ones(T, mm)
                xq = ones(T, nn)
                Q = specializingqr(A)
                Q \ bq
                ldiv!(xq, Q, bq)
                rank(Q)
                specializingqr(A; minnorm = false)
            end
            specializingqr(Asq; fallback = false)

            # QR structured fast paths: diagonal (incl. a zero ⇒ rank-deficient),
            # well-conditioned triangular (gate pass), and the geqp3 fallback.
            Aqdiag = [i == j ? cv(i + 2, 0) : zT for i in 1:n, j in 1:n]
            Aqdiag[2, 2] = zT
            Aqutri = [i <= j ? (i == j ? cv(n + i, 0) : cv(0.1, 0)) : zT for i in 1:n, j in 1:n]
            Aqltri = [i >= j ? (i == j ? cv(n + i, 0) : cv(0.1, 0)) : zT for i in 1:n, j in 1:n]
            Aqtri = [i == j ? cv(n + i, 0) : (abs(i - j) == 1 ? cv(0.1, 0) : zT) for i in 1:n, j in 1:n]
            Aqband = [i == j ? cv(2n, 0) : (abs(i - j) <= 2 ? cv(0.1, 0) : zT) for i in 1:n, j in 1:n]
            for A in (Aqdiag, Aqutri, Aqltri, Aqtri, Aqband)
                bq = ones(T, n)
                xq = ones(T, n)
                Q = specializingqr(A)
                Q \ bq
                ldiv!(xq, Q, bq)
                structuralform(Q)
            end
        end
    end
end

end # module

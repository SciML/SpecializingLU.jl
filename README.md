# SpecializingFactorizations.jl

A **type-stable**, single-workspace dense linear solver that cheaply detects
whether a dense matrix actually has special structure and, if so, solves with
the *specialized* factorization for that structure instead of a general
`O(n³)` LU.

It is the type-stable analogue of
[`SparseMatrixIdentification.jl`](https://github.com/SciML/SparseMatrixIdentification.jl):
that package detects structure by returning different *Julia types*
(`Diagonal`, `Tridiagonal`, …), which is inherently **type-unstable** — the
compiler cannot infer which factorization you will get. `SpecializingFactorizations`
instead tags the structure with a runtime **enum** and stores every
factorization in **one concrete workspace type**, so detection → factor →
solve is fully inferable and allocation-free on the hot path.

## Why

```julia
# SparseMatrixIdentification-style: return type depends on the data → unstable
f(A) = specialize(A)          # ::Diagonal OR ::Tridiagonal OR ::Matrix ... ?

# SpecializingFactorizations: always the same concrete type, branch chosen by an enum field
specializinglu(A)::SpecializedLU{T,R}     # always
```

### Detection cost — why `O(n²)`, and where `O(n)` actually applies

Certifying that a *dense* matrix has a structure is `Θ(n²)` in the worst case:
to prove a matrix is (say) diagonal you must read the entries that have to
vanish — a single hidden off-diagonal nonzero would change the answer, so any
correct algorithm must inspect `Ω(n²)` entries. There is **no** `O(n)`
certification for dense storage (that would require structural metadata, like a
`BandedMatrix` or a sparse format already carries).

So `detect_form` is `O(n²)` *worst case* — but:

- It **early-exits to ≈`O(n)`** for unstructured matrices: as soon as the
  nonzero pattern rules out triangular/banded *and* symmetry has already
  failed, it returns `GENERAL` without finishing the scan. Measured:
  `detect_form` on a dense general `n×n` takes ~2/4/8 µs at n = 500/1000/2000
  (linear), versus ~0.5/2.2/11 ms for a dense symmetric matrix that must be
  fully certified (quadratic).
- The actual `O(n)` wins are the specialized **solves** (diagonal, bidiagonal,
  tridiagonal) and the singular-pivot scan.
- Detection is always `≪` the `O(n³)` factorization it can replace.

## Detected forms and the specialized solve used

| `MatrixForm`                  | detection           | factorization / solve            | cost          |
|-------------------------------|---------------------|----------------------------------|---------------|
| `DIAGONAL`                    | `kl=ku=0`           | elementwise divide               | `O(n)`        |
| `LOWER_BIDIAGONAL`            | `kl=1, ku=0`        | forward substitution             | `O(n)`        |
| `UPPER_BIDIAGONAL`            | `kl=0, ku=1`        | back substitution                | `O(n)`        |
| `LOWER_TRIANGULAR`            | `ku=0`              | `trsv` (no factorization)        | `O(n²)`       |
| `UPPER_TRIANGULAR`            | `kl=0`              | `trsv` (no factorization)        | `O(n²)`       |
| `TRIDIAGONAL`                 | `kl=ku=1`           | LAPACK `gttrf`/`gttrs`           | `O(n)`        |
| `BANDED`                      | narrow band         | LAPACK `gbtrf`/`gbtrs`           | `O(n·b²)`     |
| `SYMMETRIC_POSITIVE_DEFINITE` | symmetric + Chol ok | LAPACK `potrf`/`potrs` (Cholesky)| `O(n³/3)`     |
| `SYMMETRIC_INDEFINITE`        | symmetric/complex-sym | LAPACK `sytrf`/`sytrs` (Bunch–Kaufman) | `O(n³/3)` |
| `HERMITIAN_INDEFINITE`        | Hermitian, not PD   | LAPACK `hetrf`/`hetrs`           | `O(n³/3)`     |
| `GENERAL`                     | otherwise           | LU fallback (`lu!`)              | `O(n³)`       |

The structural forms come from the cheap scan. The three symmetric forms are
*resolved during factorization*: a symmetric/Hermitian dense matrix attempts a
Cholesky (`potrf`); `info == 0` ⇒ positive definite, otherwise it demotes to
Bunch–Kaufman. This Cholesky attempt *is* the factorization, so no work is
wasted.

## Usage

```julia
using SpecializingFactorizations, LinearAlgebra

A = Tridiagonal(rand(99), rand(100) .+ 4, rand(99)) |> Matrix   # a dense Matrix
b = rand(100)

F = specializinglu(A)        # detect + specialized factorization
matrixform(F)                # TRIDIAGONAL
x = F \ b                    # specialized O(n) solve
ldiv!(x, F, b)               # in-place, zero allocation when warm
det(F)                       # O(n) for tridiagonal
issuccess(F)                 # false if singular (uniform across all forms)

# Reuse one workspace across many matrices of the same eltype (hot loops):
specializinglu!(F, A2)       # re-detect + re-factor into F, growing buffers as needed
```

### Allocation-free / real-time use

After the initial setup, the warm hot path is **zero-allocation** for every form
(both `specializinglu!` re-factor and `ldiv!` solve), on Julia 1.10 and 1.12. To
push *all* buffer allocation upfront — so nothing grows later — `reserve!` (or the
keyword constructor) pre-sizes every buffer, including the banded `AB` and the
Bunch–Kaufman work buffer:

```julia
F = SpecializedLU{Float64}(n; kl = 2, ku = 2, symmetric = true)  # presize everything
# ...or grow an existing workspace's buffers to a high-water mark:
reserve!(F, n; kl = 2, ku = 2, symmetric = true)

# now, in a hot loop, every re-factor + solve at size ≤ n allocates nothing:
for k in 1:nsteps
    specializinglu!(F, A_k)   # 0 allocations
    ldiv!(x, F, b_k)          # 0 allocations
end
```

Buffers are grow-only: the band and `O(n)` buffers never reallocate when the
problem shrinks or its bandwidth changes within capacity. (The dense `n×n`
factor buffer is kept exact-size — a real `Matrix` rather than a strided view —
so dense solves stay 0-allocation on the 1.10 LTS; it only reallocates if you
change the *dense* problem size, which is negligible next to the factorization.)

### Just the detector

```julia
d = detect_form(A)           # ::DetectionResult
d.form, d.kl, d.ku, d.issym, d.isherm
detect_form(A; bandwidth_cutoff = 8)   # tune when a band counts as "narrow"
```

## The single workspace

```julia
mutable struct SpecializedLU{T,R}
    form::MatrixForm   # which solve to dispatch (the only runtime branch)
    n, kl, ku::Int
    uplo::Char
    issym, isherm::Bool
    info::Int          # ≠ 0 ⇒ singular
    fact::Matrix{T}    # triangular / Cholesky / Bunch-Kaufman / LU storage
    ipiv::Vector{Int}
    dvec, dl, du, du2::Vector{T}   # diagonal / bidiagonal / tridiagonal storage
    band::Matrix{T}    # LAPACK AB band storage
end
```

All fields are concrete, so `SpecializedLU{Float64,Float64}` is a single
`isconcretetype`. Buffers are sized **lazily**: a diagonal or tridiagonal
matrix never allocates the `n×n` `fact`; the band buffer is only grown when a
narrow-banded matrix is seen. `ldiv!` is an `if/elseif` ladder on `form`; every
branch calls a concrete kernel and returns the same `x`, so the whole pipeline
infers (`@code_warntype` shows zero `::Union`/`::Any`).

## Element types

- **`BlasFloat`** (`Float32`/`Float64`/`ComplexF32`/`ComplexF64`): all forms use
  the dedicated LAPACK routines.
- **Other types** (e.g. `BigFloat`): the unconditionally-stable fast paths
  (diagonal, bidiagonal, triangular) run as pure Julia; the remaining forms use
  a generic dense LU fallback. (Cholesky/Bunch–Kaufman specialization is a
  `BlasFloat` optimization.)

## Avoiding the LU — handing `GENERAL` back to the host

The package is deliberately set up so it does **not** have to do the `GENERAL`
LU. A plain `lu!` does *not* reproduce `LinearSolve.jl`'s value-level choice
(`RFLUFactorization` for small `n`, AppleAccelerate / MKL kernels, …), so that
choice must stay on the host side. Pass `fallback_lu = false` and the package
detects + factors only the specialized forms, leaving `GENERAL` for you:

```julia
F = specializinglu(A; fallback_lu = false)
if matrixform(F) == GENERAL          # isfactored(F) == false here
    x = my_tuned_lu(A) \ b           # host's CPU-dependent LU
else
    x = F \ b                        # specialized solve
end
```

For a host that wants to inspect the form *before* allocating/factoring (and
avoid a second detection scan), use the pure detector and pass the result back
in:

```julia
d = detect_form(A)                   # cheap, no factorization
if d.form == GENERAL
    x = my_tuned_lu(A) \ b
else
    F = SpecializedLU{eltype(A)}(size(A, 1))
    specializinglu!(F, A, d; fallback_lu = false)   # reuse detection
    x = F \ b
end
```

This is exactly the shape a `LinearSolve.SpecializingDenseLUFactorization`
algorithm would take: cache a single `SpecializedLU{T}` workspace, run the
specialized factor/solve for matched forms, and route the `GENERAL` branch to
LinearSolve's own tuned dense LU. The mechanics of that wiring
(`init_cacheval` → `do_factorization` → the generic `solve!` calling
`ldiv!(u, F, b)`) are a thin LinearSolve-side adapter; the default
(`fallback_lu = true`) keeps `SpecializingFactorizations` a complete standalone solver.

## Limitations

- Square matrices only (it is an LU-style solver, not least-squares).
- Structure detection is **structural** (`!iszero`): a matrix that is
  *numerically* near-triangular but has tiny nonzeros off the band is treated
  as its true band. Pre-threshold the matrix if you want tolerant detection.
- A symmetric *banded* matrix uses the (general) banded solver, not a banded
  Cholesky.

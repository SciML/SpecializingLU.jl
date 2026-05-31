using SpecializingFactorizations
using LinearAlgebra
using Test

const SLU = SpecializingFactorizations

# Build representative matrices for each form (square, well-conditioned).
function make(form::MatrixForm, ::Type{T}, n::Int) where {T}
    R = real(T)
    rnd() = T <: Complex ? T(randn(R), randn(R)) : T(randn(R))
    rndv(k) = T[rnd() for _ in 1:k]
    if form == DIAGONAL
        return diagm(0 => rndv(n) .+ T(3))
    elseif form == LOWER_BIDIAGONAL
        return diagm(0 => rndv(n) .+ T(3), -1 => rndv(n - 1))
    elseif form == UPPER_BIDIAGONAL
        return diagm(0 => rndv(n) .+ T(3), 1 => rndv(n - 1))
    elseif form == LOWER_TRIANGULAR
        A = T.(tril(reshape([rnd() for _ in 1:(n * n)], n, n)))
        for k in 1:n
            A[k, k] += T(3)
        end
        return A
    elseif form == UPPER_TRIANGULAR
        A = T.(triu(reshape([rnd() for _ in 1:(n * n)], n, n)))
        for k in 1:n
            A[k, k] += T(3)
        end
        return A
    elseif form == TRIDIAGONAL
        return diagm(0 => rndv(n) .+ T(4), 1 => rndv(n - 1), -1 => rndv(n - 1))
    elseif form == BANDED
        return diagm(
            0 => rndv(n) .+ T(6), 1 => rndv(n - 1), -1 => rndv(n - 1),
            2 => rndv(n - 2), -2 => rndv(n - 2)
        )
    elseif form == SYMMETRIC_POSITIVE_DEFINITE
        M = reshape([rnd() for _ in 1:(n * n)], n, n)
        return M * M' + T(n) * I
    elseif form == SYMMETRIC_INDEFINITE
        M = reshape([rnd() for _ in 1:(n * n)], n, n)
        return T <: Complex ? (M + transpose(M)) : (M + M')  # complex-symmetric vs real-symmetric
    elseif form == HERMITIAN_INDEFINITE
        M = reshape([rnd() for _ in 1:(n * n)], n, n)
        return M + M'
    else # GENERAL
        return reshape([rnd() for _ in 1:(n * n)], n, n) + T(n) * I
    end
end

reltol(::Type{Float32}) = 1.0e-3
reltol(::Type{ComplexF32}) = 1.0e-3
reltol(::Type{T}) where {T} = 1.0e-8

@testset "SpecializingFactorizations.jl" begin

    @testset "detection: form + bandwidth" begin
        n = 10
        @test detect_form(diagm(0 => ones(n))).form == DIAGONAL
        let d = detect_form(diagm(0 => ones(n), -1 => ones(n - 1)))
            @test d.form == LOWER_BIDIAGONAL && d.kl == 1 && d.ku == 0
        end
        let d = detect_form(diagm(0 => ones(n), 1 => ones(n - 1)))
            @test d.form == UPPER_BIDIAGONAL && d.kl == 0 && d.ku == 1
        end
        @test detect_form(tril(ones(n, n))).form == LOWER_TRIANGULAR
        @test detect_form(triu(ones(n, n))).form == UPPER_TRIANGULAR
        let d = detect_form(diagm(0 => ones(n), 1 => ones(n - 1), -1 => ones(n - 1)))
            @test d.form == TRIDIAGONAL && d.kl == 1 && d.ku == 1
        end
        let d = detect_form(diagm(0 => ones(n), 2 => ones(n - 2), -2 => ones(n - 2)))
            @test d.form == BANDED && d.kl == 2 && d.ku == 2
        end
        @test detect_form(ones(n, n)).form == GENERAL
        # symmetry flags on a dense symmetric matrix
        S = randn(n, n); S = S + S'
        let d = detect_form(S)
            @test d.form == GENERAL && d.issym && d.isherm
        end
        # a wide band counts as GENERAL, not BANDED (cutoff)
        W = diagm(0 => ones(n)); for k in 1:(n - 1)
            W[k, k + 1] = 1; W[k + 1, k] = 1
        end
        for k in 1:(n - 3)
            W[k, k + 3] = 1; W[k + 3, k] = 1
        end
        @test detect_form(W; bandwidth_cutoff = 3).form == GENERAL
        @test detect_form(W; bandwidth_cutoff = 100).form == BANDED
        @test_throws DimensionMismatch detect_form(randn(3, 4))
    end

    @testset "correctness: $T, form=$form, n=$n" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            form in instances(MatrixForm),
            n in (1, 2, 7, 23)

        # Some forms are not meaningful / distinct at tiny n; skip degenerate combos.
        (n == 1 && form != DIAGONAL) && continue
        (n == 2 && form == BANDED) && continue
        (form == HERMITIAN_INDEFINITE && !(T <: Complex)) && continue

        A = make(form, T, n)
        b = T <: Complex ? rand(T, n) : randn(T, n)
        F = specializinglu(A)

        x = F \ b
        @test norm(A * x - b) / norm(b) < reltol(T)

        # matrix right-hand side
        B = T <: Complex ? rand(T, n, 3) : randn(T, n, 3)
        X = F \ B
        @test norm(A * X - B) / norm(B) < reltol(T)

        # in-place ldiv! into a separate output and onto the rhs
        y = similar(b); ldiv!(y, F, b)
        @test norm(A * y - b) / norm(b) < reltol(T)
        bc = copy(b); ldiv!(F, bc)
        @test norm(A * bc - b) / norm(b) < reltol(T)

        # determinant matches LinearAlgebra (skip tiny magnitudes)
        dref = det(A)
        if abs(dref) > 1.0e-6 && isfinite(dref)
            @test abs(det(F) - dref) / abs(dref) < (T <: Union{Float32, ComplexF32} ? 1.0e-2 : 1.0e-6)
        end
    end

    @testset "Hermitian detection requires a real diagonal (regression)" begin
        # complex matrix with conjugate-symmetric off-diagonals but a non-real
        # diagonal is NOT Hermitian; it must not be routed to Cholesky/hetrf.
        M = randn(ComplexF64, 6, 6)
        A = M + M'                 # genuinely Hermitian (real diagonal)
        @test detect_form(A).isherm == ishermitian(A) == true
        Abad = copy(A)
        for k in 1:6
            Abad[k, k] += 1.0im    # break Hermitian-ness via the diagonal
        end
        @test ishermitian(Abad) == false
        @test detect_form(Abad).isherm == false
        F = specializinglu(Abad)
        @test matrixform(F) ∉ (SYMMETRIC_POSITIVE_DEFINITE, HERMITIAN_INDEFINITE)
        b = rand(ComplexF64, 6)
        @test F \ b ≈ Abad \ b               # correct, not silently wrong
        @test det(F) ≈ det(Abad)
        # detect_form.isherm must agree with ishermitian on random complex inputs
        for _ in 1:5
            C = randn(ComplexF64, 7, 7)
            @test detect_form(C).isherm == ishermitian(C)
            Ch = C + C'
            @test detect_form(Ch).isherm == ishermitian(Ch) == true
        end
    end

    @testset "det type-stability for complex element types (regression)" begin
        n = 9
        M = rand(ComplexF64, n, n)
        Fh = specializinglu(Matrix(M + M' - 5I))        # Hermitian indefinite
        @test matrixform(Fh) == HERMITIAN_INDEFINITE
        @test @inferred(det(Fh)) isa ComplexF64
        Fs = specializinglu(Matrix(M + transpose(M)))   # complex symmetric
        @test matrixform(Fs) == SYMMETRIC_INDEFINITE
        @test @inferred(det(Fs)) isa ComplexF64
        @test Base.return_types(det, (typeof(Fh),)) == [ComplexF64]
    end

    @testset "resolved symmetric forms" begin
        n = 12
        spd = let M = randn(n, n)
            M * M' + n * I
        end
        @test matrixform(specializinglu(spd)) == SYMMETRIC_POSITIVE_DEFINITE
        sind = let M = randn(n, n)
            M + M'
        end
        @test matrixform(specializinglu(sind)) == SYMMETRIC_INDEFINITE
        herm = let M = randn(ComplexF64, n, n)
            M + M'
        end
        # could be SPD or indefinite depending on randomness; both are valid resolutions
        @test matrixform(specializinglu(herm)) in
            (SYMMETRIC_POSITIVE_DEFINITE, HERMITIAN_INDEFINITE)
        csym = let M = randn(ComplexF64, n, n)
            M + transpose(M)
        end
        @test matrixform(specializinglu(csym)) == SYMMETRIC_INDEFINITE
    end

    @testset "type stability — same concrete type for every form" begin
        n = 9
        types = MatrixForm[]
        for form in (
                DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR, LOWER_BIDIAGONAL,
                UPPER_BIDIAGONAL, TRIDIAGONAL, BANDED, SYMMETRIC_POSITIVE_DEFINITE,
                SYMMETRIC_INDEFINITE, GENERAL,
            )
            F = specializinglu(make(form, Float64, n))
            push!(types, F.form)
        end
        Fs = [
            specializinglu(make(f, Float64, n)) for f in
                (DIAGONAL, TRIDIAGONAL, GENERAL, SYMMETRIC_POSITIVE_DEFINITE)
        ]
        @test all(t -> t === typeof(Fs[1]), typeof.(Fs))
        @test typeof(Fs[1]) === SpecializedLU{Float64, Float64}
    end

    @testset "@inferred specializinglu / ldiv! / det / \\" begin
        n = 9
        for form in (
                DIAGONAL, TRIDIAGONAL, BANDED, SYMMETRIC_POSITIVE_DEFINITE,
                SYMMETRIC_INDEFINITE, GENERAL, LOWER_TRIANGULAR,
            )
            A = make(form, Float64, n)
            b = randn(n)
            F = @inferred specializinglu(A)
            @test F isa SpecializedLU{Float64, Float64}
            x = similar(b)
            @inferred ldiv!(x, F, b)
            @inferred det(F)
            @inferred F \ b
        end
        # complex too
        let A = make(GENERAL, ComplexF64, n), b = rand(ComplexF64, n)
            F = @inferred specializinglu(A)
            @inferred ldiv!(similar(b), F, b)
        end
    end

    @testset "zero allocations after setup (warm refactor + solve), all forms" begin
        # After the initial specializinglu setup, a reuse loop must allocate
        # NOTHING — neither the re-factorization nor the solve — for every form.
        n = 64
        @noinline refac_allocs(F, A) = (specializinglu!(F, A); @allocated specializinglu!(F, A))
        @noinline solve_allocs(F, x, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        @noinline solvem_allocs(F, X, B) = (ldiv!(X, F, B); @allocated ldiv!(X, F, B))
        for T in (Float64, ComplexF64)
            for form in (
                    DIAGONAL, LOWER_BIDIAGONAL, UPPER_BIDIAGONAL, LOWER_TRIANGULAR,
                    UPPER_TRIANGULAR, TRIDIAGONAL, BANDED, SYMMETRIC_POSITIVE_DEFINITE,
                    SYMMETRIC_INDEFINITE, GENERAL,
                )
                A = make(form, T, n)
                b = T <: Complex ? rand(T, n) : randn(T, n); x = similar(b)
                B = T <: Complex ? rand(T, n, 4) : randn(T, n, 4); X = similar(B)
                F = specializinglu(A)
                @test refac_allocs(F, A) == 0       # warm re-factorization: 0 bytes
                @test solve_allocs(F, x, b) == 0    # warm solve (vector): 0 bytes
                @test solvem_allocs(F, X, B) == 0   # warm solve (multi-RHS): 0 bytes
            end
        end
        # complex Hermitian-indefinite (hetrf) too
        let M = rand(ComplexF64, n, n), A = Matrix(M + M')
            b = rand(ComplexF64, n); x = similar(b)
            F = specializinglu(A)
            @test matrixform(F) in (HERMITIAN_INDEFINITE, SYMMETRIC_POSITIVE_DEFINITE)
            @test refac_allocs(F, A) == 0
            @test solve_allocs(F, x, b) == 0
        end
    end

    @testset "reserve! / grow-only buffers (upfront allocation)" begin
        # Warm-then-measure (steady state): the first call on any workspace pays
        # Julia's usual first-execution cost, which is not a buffer allocation —
        # what `reserve!` guarantees is that no buffer GROWS thereafter.
        @noinline refac(F, A) = (specializinglu!(F, A); @allocated specializinglu!(F, A))
        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        bandmat(n) = diagm(
            0 => fill(8.0, n), 1 => fill(0.5, n - 1), -1 => fill(0.5, n - 1),
            2 => fill(0.25, n - 2), -2 => fill(0.25, n - 2),
        )
        allforms(n) = (
            diagm(0 => randn(n) .+ 3.0),
            diagm(0 => randn(n) .+ 4.0, 1 => randn(n - 1), -1 => randn(n - 1)),  # tridiagonal
            bandmat(n),                                                          # banded
            (M = randn(n, n); Matrix(M * M' + n * I)),                           # SPD
            (M = randn(n, n); Matrix(M + M')),                                   # symmetric-indefinite
            randn(n, n) + n * I,                                                 # general
        )
        structured(n) = (
            diagm(0 => randn(n) .+ 3.0),
            diagm(0 => randn(n) .+ 4.0, 1 => randn(n - 1), -1 => randn(n - 1)),
            bandmat(n),
        )

        # reserve! everything; then every form re-factors + solves at 0 allocations.
        R = SpecializedLU{Float64}()
        reserve!(R, 256; kl = 2, ku = 2, symmetric = true)
        b = randn(256); x = similar(b)
        for A in allforms(256)
            @test refac(R, A) == 0
            @test solv(x, R, b) == 0
            @test norm(A * (R \ b) - b) / norm(b) < 1.0e-7
        end

        # The band + O(n) buffers are grow-only: SMALLER structured problems
        # through the same reserved workspace stay 0-alloc and correct (no realloc).
        bs = randn(64); xs = similar(bs)
        for A in structured(64)
            @test refac(R, A) == 0
            @test solv(xs, R, bs) == 0
            @test norm(A * (R \ bs) - bs) / norm(bs) < 1.0e-8
        end

        # The keyword constructor pre-sizes the band + Bunch-Kaufman work buffers
        # so banded / symmetric workspaces are upfront-allocated too.
        C = SpecializedLU{Float64}(128; kl = 2, ku = 2, symmetric = true)
        bc = randn(128); xc = similar(bc)
        for A in (bandmat(128), (M = randn(128, 128); Matrix(M + M')))
            @test refac(C, A) == 0
            @test solv(xc, C, bc) == 0
        end
    end

    @testset "workspace reuse (specializinglu!)" begin
        n = 16
        F = specializinglu(make(DIAGONAL, Float64, n))
        # reuse for a different form, same size
        for form in (TRIDIAGONAL, GENERAL, BANDED, SYMMETRIC_POSITIVE_DEFINITE, UPPER_TRIANGULAR)
            A = make(form, Float64, n)
            specializinglu!(F, A)
            b = randn(n)
            x = F \ b
            @test norm(A * x - b) / norm(b) < 1.0e-8
        end
        # reuse with a different size (buffers must grow)
        for m in (4, 40, 7)
            A = make(TRIDIAGONAL, Float64, m)
            specializinglu!(F, A)
            b = randn(m)
            @test norm(A * (F \ b) - b) / norm(b) < 1.0e-8
        end
        # reuse does not allocate a new workspace object identity wise
        F2 = specializinglu!(F, make(DIAGONAL, Float64, n))
        @test F2 === F
    end

    @testset "generic (non-BLAS) element type: BigFloat" begin
        n = 8
        setprecision(BigFloat, 128) do
            b = BigFloat.(randn(n))
            for form in (
                    DIAGONAL, LOWER_BIDIAGONAL, UPPER_BIDIAGONAL, LOWER_TRIANGULAR,
                    UPPER_TRIANGULAR, TRIDIAGONAL, BANDED, GENERAL,
                )
                A = make(form, BigFloat, n)
                F = specializinglu(A)
                @test F isa SpecializedLU{BigFloat, BigFloat}
                x = F \ b
                @test norm(A * x - b) / norm(b) < 1.0e-25
            end
            # symmetric dense falls back to GENERAL (LU) for generic eltype
            S = let M = BigFloat.(randn(n, n))
                M + M'
            end
            FS = specializinglu(S)
            @test matrixform(FS) == GENERAL
            @test norm(S * (FS \ b) - b) / norm(b) < 1.0e-25
        end
    end

    @testset "singular detection via issuccess (no throw)" begin
        # general singular (n>=3 so it is GENERAL, not banded/tridiagonal)
        G = [1.0 2.0 3.0; 2.0 4.0 6.0; 1.0 1.0 1.0]
        @test !issuccess(specializinglu(G))
        # singular tridiagonal (gttrf path) must record info, not throw
        Ts = [1.0 2.0; 2.0 4.0]
        @test matrixform(specializinglu(Ts)) == TRIDIAGONAL
        @test !issuccess(specializinglu(Ts))
        # singular diagonal / triangular (zero pivot scan)
        @test !issuccess(specializinglu(diagm(0 => [1.0, 0.0, 2.0])))
        Lz = [1.0 0 0; 2.0 0.0 0; 3.0 4.0 5.0]   # zero on diagonal ⇒ singular lower-tri
        @test matrixform(specializinglu(Lz)) == LOWER_TRIANGULAR
        @test !issuccess(specializinglu(Lz))
        # singular banded (gbtrf path): pentadiagonal on n=8 with a zero row
        Bz = diagm(
            0 => fill(2.0, 8), 1 => fill(0.5, 7), -1 => fill(0.5, 7),
            2 => fill(0.3, 6), -2 => fill(0.3, 6)
        )
        Bz[4, :] .= 0.0
        @test matrixform(specializinglu(Bz)) == BANDED
        @test !issuccess(specializinglu(Bz))
        # nonsingular sanity
        @test issuccess(specializinglu(diagm(0 => [1.0, 2.0])))
    end

    @testset "fallback_lu=false delegates GENERAL to the host" begin
        n = 12
        G = randn(n, n) + n * I        # unstructured
        F = specializinglu(G; fallback_lu = false)
        @test matrixform(F) == GENERAL
        @test !isfactored(F)
        @test !issuccess(F)
        b = randn(n)
        @test_throws ArgumentError ldiv!(similar(b), F, b)
        @test_throws ArgumentError F \ b
        @test_throws ArgumentError det(F)

        # specialized forms are STILL factored and solvable with fallback_lu=false
        for A in (
                diagm(0 => randn(n) .+ 3),
                diagm(0 => randn(n) .+ 4, 1 => randn(n - 1), -1 => randn(n - 1)),
                (M = randn(n, n); M * M' + n * I),
            )
            Fs = specializinglu(A; fallback_lu = false)
            @test isfactored(Fs)
            @test matrixform(Fs) != GENERAL
            @test norm(A * (Fs \ b) - b) / norm(b) < 1.0e-8
        end

        # type stability is unaffected by the keyword
        @test (@inferred specializinglu(G; fallback_lu = false)) isa SpecializedLU{Float64, Float64}
    end

    @testset "detection-first host pattern (no package LU)" begin
        # How a host (e.g. LinearSolve) would use the package: detect cheaply,
        # delegate GENERAL to its own LU, otherwise use the specialized solve.
        host_lu(A, b) = lu(A) \ b
        function host_solve(A, b)
            d = detect_form(A)
            if d.form == GENERAL
                return host_lu(A, b)            # host's CPU-dependent choice
            else
                F = SpecializedLU{eltype(A)}(size(A, 1))
                specializinglu!(F, A, d; fallback_lu = false)   # reuse detection, no re-scan
                return F \ b
            end
        end
        n = 15
        b = randn(n)
        for A in (
                diagm(0 => randn(n) .+ 4, 1 => randn(n - 1), -1 => randn(n - 1)),
                Matrix(UpperTriangular(randn(n, n) + 3I)),
                (M = randn(n, n); M * M' + n * I),
                randn(n, n) + n * I,
            )          # the GENERAL one → host LU
            @test norm(A * host_solve(A, b) - b) / norm(b) < 1.0e-8
        end
    end

    @testset "early-exit detection stays correct" begin
        n = 60
        # unstructured asymmetric → GENERAL (early-exits)
        @test detect_form(randn(n, n)).form == GENERAL
        # wide-band SYMMETRIC must NOT early-exit (stays symmetric ⇒ GENERAL+issym)
        S = randn(n, n); S = S + S'
        ds = detect_form(S)
        @test ds.form == GENERAL && ds.issym && ds.isherm
        @test matrixform(specializinglu(S)) == SYMMETRIC_POSITIVE_DEFINITE ||
            matrixform(specializinglu(S)) == SYMMETRIC_INDEFINITE
        # narrow banded asymmetric must NOT early-exit (band condition false) ⇒ BANDED
        B = diagm(
            0 => randn(n) .+ 6, 1 => randn(n - 1), -1 => randn(n - 1),
            2 => randn(n - 2), -2 => randn(n - 2)
        )
        B[1, 3] = 0.0   # make it slightly asymmetric without widening the band
        @test detect_form(B).form == BANDED
    end

    @testset "integer / rational eltypes promote like Base" begin
        A = [2 0 0; 0 3 0; 0 0 4]
        b = [3, 4, 5]
        F = @inferred specializinglu(A)
        @test F isa SpecializedLU{Float64, Float64}
        @test F \ b ≈ A \ b
        # a general integer matrix
        G = [4 1 0; 1 3 1; 0 1 2]
        @test specializinglu(G) \ b ≈ G \ b
        # n=0 and n=1 corner cases
        @test matrixform(specializinglu(zeros(0, 0))) == DIAGONAL
        @test (specializinglu(reshape([5.0], 1, 1)) \ [10.0]) ≈ [2.0]
    end

    @testset "tridiagonal kernel matches lu(Tridiagonal) incl. pivoting" begin
        for T in (Float64, Float32, ComplexF64, ComplexF32)
            for trial in 1:3
                n = 120
                rv(k) = T <: Complex ? rand(T, k) : randn(T, k)
                dl, d, du = rv(n - 1), rv(n), rv(n - 1)
                trial == 2 && (d .*= T(1.0e-6); dl .*= T(1.0e3); du .*= T(1.0e3))  # pivot stress
                trial == 3 && (d[60] = zero(T))                                    # forces interchange
                A = Matrix(Tridiagonal(dl, d, du))
                b = T <: Complex ? rand(T, n) : randn(T, n)
                # bit-for-bit agreement with LinearAlgebra's tridiagonal LU
                @test specializinglu(A) \ b == Tridiagonal(dl, d, du) \ b
                B = T <: Complex ? rand(T, n, 5) : randn(T, n, 5)
                @test specializinglu(A) \ B == Tridiagonal(dl, d, du) \ B    # multi-RHS too
            end
        end
    end

    @testset "generic tridiagonal/banded are O(n) (not dense LU), exact for Rational" begin
        # BigFloat tridiagonal / banded now use the hand-rolled kernels, not a
        # dense O(n^3) fallback; verify form + accuracy.
        setprecision(BigFloat, 128) do
            n = 60
            T = Matrix(Tridiagonal(BigFloat.(randn(n - 1)), BigFloat.(randn(n) .+ 4), BigFloat.(randn(n - 1))))
            Bnd = Matrix(Tridiagonal(BigFloat.(fill(0.5, n - 1)), BigFloat.(fill(8.0, n)), BigFloat.(fill(0.5, n - 1)))) +
                diagm(2 => BigFloat.(fill(0.25, n - 2)), -2 => BigFloat.(fill(0.25, n - 2)))
            b = BigFloat.(randn(n))
            @test matrixform(specializinglu(T)) == TRIDIAGONAL
            @test matrixform(specializinglu(Bnd)) == BANDED
            @test norm(T * (specializinglu(T) \ b) - b) / norm(b) < 1.0e-30
            @test norm(Bnd * (specializinglu(Bnd) \ b) - b) / norm(b) < 1.0e-30
        end
        # Rational stays exact (lutype promotion, not float widening) across forms
        Atri = Matrix(Tridiagonal([1 // 2, 1 // 3], [2 // 1, 3 // 1, 2 // 1], [1 // 5, 1 // 7]))
        br = [1 // 1, 2 // 1, 3 // 1]
        Fr = specializinglu(Atri)
        @test Fr isa SpecializedLU{Rational{Int}, Rational{Int}}
        @test matrixform(Fr) == TRIDIAGONAL
        @test Fr \ br == Atri \ br                # exact rational solve
        @test Atri * (Fr \ br) == br
    end

    @testset "agreement with LinearAlgebra structured solves" begin
        n = 20
        dl = randn(n - 1); d = randn(n) .+ 4; du = randn(n - 1)
        A = Matrix(Tridiagonal(dl, d, du))
        b = randn(n)
        @test specializinglu(A) \ b ≈ Tridiagonal(dl, d, du) \ b
        L = Matrix(LowerTriangular(randn(n, n) + 3I))
        @test specializinglu(L) \ b ≈ LowerTriangular(L) \ b
    end
end

# ===========================================================================
# SpecializedQR — rank-revealing least-squares / minimum-norm solver
# ===========================================================================

# correct-typed random matrix/vector, and an exact-rank-r factor product.
_qrand(::Type{T}, dims...) where {T <: Complex} = T.(complex.(randn(dims...), randn(dims...)))
_qrand(::Type{T}, dims...) where {T <: Real} = T.(randn(dims...))
_lowrank(::Type{T}, m, n, r) where {T} = _qrand(T, m, r) * _qrand(T, r, n)
_reltol(::Type{T}) where {T} = (real(T) === Float32 ? 1.0f-3 : 1.0e-8)

@testset "SpecializedQR (rank-revealing least-squares)" begin

    @testset "full-rank LS matches qr(ColumnNorm) and pinv: $T, $shape" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            shape in (:square, :over, :under)

        m, n = shape === :square ? (5, 5) : (shape === :over ? (7, 4) : (4, 7))
        A = _qrand(T, m, n)
        b = _qrand(T, m)
        F = specializingqr(A)
        x = F \ b
        @test length(x) == n
        # full COLUMN rank (rank == n) is QR_FULLRANK; an underdetermined system
        # at full ROW rank (rank == m < n) takes the min-norm path → QR_DEFICIENT.
        @test matrixform(F) == (n <= m ? QR_FULLRANK : QR_DEFICIENT)
        @test rank(F) == min(m, n)
        @test issuccess(F)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        # the normal equations hold: Aᴴ(Ax-b) ≈ 0
        @test norm(A' * (A * x - b)) < _reltol(T) * max(1, norm(A)^2)
    end

    @testset "rank-deficient min-norm matches pinv: $T, $shape" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            shape in (:square, :over, :under)

        m, n = shape === :square ? (6, 6) : (shape === :over ? (7, 5) : (5, 8))
        r = 3
        A = _lowrank(T, m, n, r)
        b = _qrand(T, m)
        F = specializingqr(A)              # minnorm = true (default)
        x = F \ b
        @test length(x) == n
        @test matrixform(F) == QR_DEFICIENT
        @test rank(F) == r
        @test issuccess(F)                 # rank deficiency is NOT a failure
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        # min-norm: no other LS solution has smaller norm (compare to basic)
        xbasic = specializingqr(A; minnorm = false) \ b
        @test norm(A' * (A * xbasic - b)) < _reltol(T) * max(1, norm(A)^2)  # valid LS
        @test norm(x) <= norm(xbasic) + _reltol(T)                          # min-norm ≤ basic
    end

    @testset "singular / zero / rank-0 / empty never throw: $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        for (m, n) in ((3, 3), (2, 4), (4, 2))
            A = zeros(T, m, n)
            b = _qrand(T, m)
            F = specializingqr(A)
            x = F \ b
            @test iszero(x)
            @test length(x) == n
            @test rank(F) == 0
            @test issuccess(F)
            @test x ≈ pinv(A) * b
        end
        # empty inputs
        let F = specializingqr(zeros(T, 0, 0))
            @test length(F \ zeros(T, 0)) == 0
            @test rank(F) == 0
        end
        let F = specializingqr(zeros(T, 0, 3))     # 0×3: x has length 3, all zero
            x = F \ zeros(T, 0)
            @test length(x) == 3 && iszero(x)
        end
    end

    @testset "multiple right-hand sides: $T" for T in (Float64, ComplexF64)
        A = _lowrank(T, 7, 5, 3)
        B = _qrand(T, 7, 4)
        F = specializingqr(A)
        X = F \ B
        @test size(X) == (5, 4)
        @test X ≈ pinv(A) * B rtol = _reltol(T)
        # column-wise consistency with single-RHS solves
        for c in 1:4
            @test X[:, c] ≈ F \ B[:, c] rtol = _reltol(T)
        end
    end

    @testset "type stability — one concrete type for every shape/rank" begin
        for T in (Float64, ComplexF64)
            for A in (_qrand(T, 5, 5), _qrand(T, 7, 4), _qrand(T, 4, 7), _lowrank(T, 6, 6, 2))
                @test (@inferred specializingqr(A)) isa SpecializedQR{T, real(T)}
            end
            A = _qrand(T, 7, 4); b = _qrand(T, 7); F = specializingqr(A)
            x = Vector{T}(undef, 4)
            @test (@inferred ldiv!(x, F, b)) === x
            @test (@inferred F \ b) isa Vector{T}
        end
    end

    @testset "zero allocations after setup (warm solve + refactor): $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        @noinline refac(F, A) = (specializingqr!(F, A); @allocated specializingqr!(F, A))
        m, n = 8, 5
        # full column rank
        let A = _qrand(T, m, n), b = _qrand(T, m)
            F = SpecializedQR{T}(m, n)
            specializingqr!(F, A)
            x = Vector{T}(undef, n)
            @test matrixform(F) == QR_FULLRANK
            @test solv(x, F, b) == 0
            @test refac(F, A) == 0
        end
        # rank-deficient (min-norm path: tzrzf/ormrz), buffers reserved upfront
        let A = _lowrank(T, m, n, 2), b = _qrand(T, m)
            F = SpecializedQR{T}(m, n; deficient = true)
            specializingqr!(F, A)
            x = Vector{T}(undef, n)
            @test matrixform(F) == QR_DEFICIENT
            @test solv(x, F, b) == 0
            @test refac(F, A) == 0
        end
    end

    @testset "reserve! makes smaller subsequent solves 0-alloc" begin
        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        F = SpecializedQR{Float64}()
        reserve!(F, 64, 32; deficient = true, nrhs = 1)
        for (m, n, r) in ((64, 32, 32), (40, 20, 8), (50, 25, 25))
            A = r == n ? _qrand(Float64, m, n) : _lowrank(Float64, m, n, r)
            b = randn(m); x = Vector{Float64}(undef, n)
            specializingqr!(F, A)
            @test solv(x, F, b) == 0
            @test norm(A' * (A * (F \ b) - b)) < 1.0e-7 * max(1, norm(A)^2)
        end
    end

    @testset "rtol keyword controls revealed rank" begin
        # A matrix with one deliberately tiny (but nonzero) singular value.
        U, _ = qr(randn(6, 6)); V, _ = qr(randn(6, 6))
        s = [1.0, 0.5, 0.25, 0.1, 1.0e-9, 1.0e-12]
        A = Matrix(U) * Diagonal(s) * Matrix(V)'
        @test rank(specializingqr(A; rtol = 1.0e-6)) == 4   # drops the 1e-9 and 1e-12
        @test rank(specializingqr(A; rtol = 1.0e-10)) == 5  # keeps 1e-9, drops 1e-12
        @test rank(specializingqr(A; rtol = 1.0e-14)) == 6  # keeps all
    end

    @testset "agreement with Base \\ for tall full-rank (LS)" begin
        A = randn(20, 8); b = randn(20)
        @test specializingqr(A) \ b ≈ A \ b rtol = 1.0e-8
        Ac = randn(ComplexF64, 15, 6); bc = randn(ComplexF64, 15)
        @test specializingqr(Ac) \ bc ≈ Ac \ bc rtol = 1.0e-8
    end

    @testset "generic (non-BLAS) element type: BigFloat" begin
        # rank-deficient: Julia's generic QRPivoted \\ blows up here (≈4e76);
        # our rank-truncated fallback returns a valid LS solution and never throws.
        let A = BigFloat[1 1; 1 1], b = BigFloat[2, 3]
            F = specializingqr(A)
            x = F \ b
            @test rank(F) == 1
            @test norm(A' * (A * x - b)) < 1.0e-60
            @test all(isfinite, x)
        end
        # full-rank overdetermined: matches the exact reference
        let A = BigFloat.(randn(6, 3)), b = BigFloat.(randn(6))
            @test norm(specializingqr(A) \ b - (A \ b)) < 1.0e-60
        end
        # zero matrix: zeros, no throw
        let F = specializingqr(zeros(BigFloat, 3, 3))
            @test iszero(F \ BigFloat.(randn(3)))
            @test rank(F) == 0
        end
    end

    @testset "integer / rational eltypes promote (QR needs sqrt)" begin
        Ai = [2 1 0; 1 3 1; 0 1 2]; bi = [1, 2, 3]
        F = specializingqr(Ai)
        @test eltype(F) === Float64
        @test F \ bi ≈ Float64.(Ai) \ Float64.(bi)
        Ar = Rational{Int}[1 2; 2 4]; br = Rational{Int}[1, 2]
        Fr = specializingqr(Ar)        # promotes to Float64 (no exact rational QR)
        @test eltype(Fr) === Float64
        @test rank(Fr) == 1
        @test norm(Float64.(Ar)' * (Float64.(Ar) * (Fr \ br) - Float64.(br))) < 1.0e-10
    end

    @testset "fallback = false leaves the QR to the host" begin
        A = randn(6, 4)
        F = specializingqr(A; fallback = false)
        @test !isfactored(F)
        @test matrixform(F) == QR_UNFACTORED
        @test_throws ArgumentError ldiv!(zeros(4), F, ones(6))
    end

    @testset "dimension mismatches throw" begin
        A = randn(6, 4); F = specializingqr(A)
        @test_throws DimensionMismatch ldiv!(zeros(4), F, ones(5))   # wrong rhs rows
        @test_throws DimensionMismatch ldiv!(zeros(3), F, ones(6))   # wrong x rows
        @test_throws DimensionMismatch ldiv!(F, ones(6))             # 2-arg needs square
    end

    # ----- structure specialization (detect_form reused by the QR solver) -----

    # A well-conditioned, diagonally-dominant instance of each structured form.
    function _struct_mat(form::MatrixForm, ::Type{T}, n::Int) where {T}
        dom() = T <: Complex ? T(n + 3, 0) : T(n + 3)
        off() = T <: Complex ? T(0.1, 0.1) : T(0.1)
        if form == DIAGONAL
            return Matrix(Diagonal(T[dom() for _ in 1:n]))
        elseif form == UPPER_TRIANGULAR
            return T[i < j ? off() : (i == j ? dom() : zero(T)) for i in 1:n, j in 1:n]
        elseif form == LOWER_TRIANGULAR
            return T[i > j ? off() : (i == j ? dom() : zero(T)) for i in 1:n, j in 1:n]
        elseif form == UPPER_BIDIAGONAL
            return Matrix(Bidiagonal(T[dom() for _ in 1:n], T[off() for _ in 1:(n - 1)], :U))
        elseif form == LOWER_BIDIAGONAL
            return Matrix(Bidiagonal(T[dom() for _ in 1:n], T[off() for _ in 1:(n - 1)], :L))
        elseif form == TRIDIAGONAL
            return Matrix(Tridiagonal(T[off() for _ in 1:(n - 1)], T[dom() for _ in 1:n], T[off() for _ in 1:(n - 1)]))
        else # BANDED (pentadiagonal)
            return diagm(
                0 => T[dom() for _ in 1:n],
                1 => T[off() for _ in 1:(n - 1)], -1 => T[off() for _ in 1:(n - 1)],
                2 => T[off() for _ in 1:(n - 2)], -2 => T[off() for _ in 1:(n - 2)],
            )
        end
    end

    @testset "structured forms match pinv & geqp3: $T, $form" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            form in (
                DIAGONAL, UPPER_TRIANGULAR, LOWER_TRIANGULAR,
                UPPER_BIDIAGONAL, LOWER_BIDIAGONAL, TRIDIAGONAL, BANDED,
            )

        n = 9
        A = _struct_mat(form, T, n)
        b = _qrand(T, n)
        F = specializingqr(A)
        x = F \ b
        @test structuralform(F) == form          # took the structured fast path
        @test rank(F) == n
        @test matrixform(F) == QR_FULLRANK
        @test issuccess(F)
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        # indistinguishable from the dense rank-revealing path:
        Fg = specializingqr(A; detect_structure = false)
        @test structuralform(Fg) == GENERAL
        @test rank(F) == rank(Fg)
        @test x ≈ Fg \ b rtol = _reltol(T)
        # multi-RHS
        B = _qrand(T, n, 3)
        @test F \ B ≈ pinv(A) * B rtol = _reltol(T)
    end

    @testset "DIAGONAL rank-revealing + singular (pure structured): $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        # exact zeros and a sub-tolerance entry ⇒ rank deficiency, no fallback
        d = T[2, 0, 3, 0, 4]
        d[5] = T(maximum(abs, d)) * (5 * eps(real(T)) / 4)  # sub-tolerance ⇒ dropped
        A = Matrix(Diagonal(d))
        b = _qrand(T, 5)
        F = specializingqr(A)
        x = F \ b
        @test structuralform(F) == DIAGONAL          # still structured (no fallback)
        @test rank(F) == 2
        @test issuccess(F)                           # deficiency is success
        @test all(isfinite, x)
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        # all-zero diagonal ⇒ rank 0, zero solution, no throw
        let Z = zeros(T, 4, 4), F0 = specializingqr(Z)
            @test rank(F0) == 0
            @test structuralform(F0) == DIAGONAL
            @test iszero(F0 \ _qrand(T, 4))
        end
        # the rank matches geqp3 across a randomized stress of zeros/tiny entries
        for _ in 1:50
            dd = _qrand(T, 8)
            for k in 1:8
                rand() < 0.3 && (dd[k] = zero(T))
            end
            AA = Matrix(Diagonal(dd))
            @test rank(specializingqr(AA)) == rank(specializingqr(AA; detect_structure = false))
        end
    end

    @testset "ill-conditioned / singular structured fall back to geqp3 (contract)" begin
        # graded ill-conditioned upper-triangular: gate fails ⇒ geqp3, and the
        # rank + solution stay identical to the dense rank-revealing reference.
        n = 12
        A = triu(ones(n, n))
        for i in 1:n
            A[i, i] = 10.0^(-1.4 * (i - 1))            # cond ≈ 1e15, past the rtol boundary
        end
        b = randn(n)
        F = specializingqr(A)
        Fg = specializingqr(A; detect_structure = false)
        @test structuralform(F) == GENERAL            # gate failed ⇒ fell back
        @test rank(F) == rank(Fg)                     # same revealed rank as geqp3
        @test F \ b ≈ Fg \ b rtol = 1.0e-8            # same solution as geqp3
        @test F \ b ≈ pinv(A) * b rtol = 1.0e-7
        @test all(isfinite, F \ b)

        # exact zero on a triangular diagonal: a plain trtrs would THROW; we must
        # fall back to geqp3 and return the finite min-norm solution.
        let U = [2.0 1.0 3.0; 0.0 0.0 4.0; 0.0 0.0 5.0], bz = [1.0, 2.0, 3.0]
            Fz = specializingqr(U)
            @test structuralform(Fz) == GENERAL
            @test all(isfinite, Fz \ bz)
            @test Fz \ bz ≈ pinv(U) * bz rtol = 1.0e-9
            @test rank(Fz) == 2
        end

        # rank-deficient tridiagonal (two identical rows ⇒ rank 2 of 3): the
        # hand-rolled gttrf would otherwise divide by a zero pivot; must fall back.
        let A = Matrix(Tridiagonal([1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0])), bt = [1.0, 2.0, 3.0]
            Ft = specializingqr(A)
            @test structuralform(Ft) == GENERAL
            @test all(isfinite, Ft \ bt)
            @test Ft \ bt ≈ pinv(A) * bt rtol = 1.0e-9
            @test rank(Ft) == rank(specializingqr(A; detect_structure = false))
        end
        # ill-conditioned banded full-rank-but-past-rtol: gate must route to geqp3.
        let nb = 30
            A = diagm(
                0 => 10.0 .^ range(0, -16; length = nb),
                1 => fill(1.0, nb - 1), -1 => fill(1.0, nb - 1),
                2 => fill(1.0, nb - 2), -2 => fill(1.0, nb - 2),
            )
            bb = randn(nb)
            Fb = specializingqr(A)
            Fbg = specializingqr(A; detect_structure = false)
            @test rank(Fb) == rank(Fbg)
            @test Fb \ bb ≈ Fbg \ bb rtol = 1.0e-7
            @test all(isfinite, Fb \ bb)
        end
    end

    @testset "structured paths: 0 allocations (warm solve + refactor): $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        @noinline refac(F, A) = (specializingqr!(F, A); @allocated specializingqr!(F, A))
        n = 10
        for form in (
                DIAGONAL, UPPER_TRIANGULAR, LOWER_TRIANGULAR,
                UPPER_BIDIAGONAL, LOWER_BIDIAGONAL, TRIDIAGONAL, BANDED,
            )
            A = _struct_mat(form, T, n)
            b = _qrand(T, n)
            # reserve the band buffer for the BANDED path (kl=ku=2 pentadiagonal)
            F = form == BANDED ? SpecializedQR{T}(n, n; kl = 2, ku = 2) : SpecializedQR{T}(n, n)
            specializingqr!(F, A)
            x = Vector{T}(undef, n)
            @test structuralform(F) == form
            @test solv(x, F, b) == 0
            @test refac(F, A) == 0
        end
    end

    @testset "structured paths: type stability and escape hatches" begin
        for T in (Float64, ComplexF64)
            for form in (DIAGONAL, UPPER_TRIANGULAR, LOWER_TRIANGULAR)
                A = _struct_mat(form, T, 6)
                @test (@inferred specializingqr(A)) isa SpecializedQR{T, real(T)}
                F = specializingqr(A)
                x = Vector{T}(undef, 6); b = _qrand(T, 6)
                @test (@inferred ldiv!(x, F, b)) === x
            end
        end
        # detect_structure = false forces the dense path even for a diagonal A,
        # and returns the identical solution.
        let A = Matrix(Diagonal(randn(6) .+ 3.0)), b = randn(6)
            @test structuralform(specializingqr(A; detect_structure = false)) == GENERAL
            @test specializingqr(A) \ b ≈ specializingqr(A; detect_structure = false) \ b
        end
        # rectangular input must NOT call detect_form (it is square-only / throws)
        @test structuralform(specializingqr(randn(7, 4))) == GENERAL
        @test structuralform(specializingqr(randn(4, 7))) == GENERAL
    end

    @testset "band gate: Varah early-accept and laic1 fallback agree with geqp3: $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        rt = _reltol(T)
        n = 16
        # strongly diagonally dominant ⇒ the O(n)/O(n·b) Varah early-accept fires
        # (no laic1 sweep); NOT diagonally dominant but still full rank ⇒ Varah
        # declines and the O(n²) laic1 gate accepts. Both must equal geqp3/pinv.
        for (kl, ku) in ((1, 1), (2, 2))
            dl = T[_qrand(T, 1)[1] for _ in 1:(n - 1)]
            dom = Matrix(Tridiagonal(dl .* T(0.2), T[_qrand(T, 1)[1] + T(8) for _ in 1:n], dl .* T(0.2)))
            if kl == 2
                dom = dom + diagm(2 => fill(T(0.1), n - 2), -2 => fill(T(0.1), n - 2))
            end
            # off-diagonals larger than the diagonal ⇒ not diagonally dominant
            nondom = Matrix(
                Tridiagonal(
                    T[_qrand(T, 1)[1] + T(3) for _ in 1:(n - 1)],
                    T[_qrand(T, 1)[1] * T(0.5) for _ in 1:n],
                    T[_qrand(T, 1)[1] + T(3) for _ in 1:(n - 1)],
                )
            )
            for A in (dom, nondom)
                b = _qrand(T, n)
                F = specializingqr(A)
                Fg = specializingqr(A; detect_structure = false)
                @test structuralform(F) in (TRIDIAGONAL, BANDED)
                @test rank(F) == rank(Fg)                 # Varah/laic1 path == geqp3 rank
                @test F \ b ≈ pinv(A) * b rtol = rt
                @test F \ b ≈ Fg \ b rtol = rt
            end
        end
        # a barely-non-dominant-but-rank-deficient tridiagonal must still fall to
        # geqp3 (Varah declines; laic1/gttrf detect the deficiency): never a
        # spurious full-rank Varah accept.
        let A = Matrix(Tridiagonal([1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0])), bz = [1.0, 2.0, 3.0]
            F = specializingqr(A)
            @test structuralform(F) == GENERAL
            @test rank(F) == 2
            @test F \ bz ≈ pinv(A) * bz rtol = 1.0e-9
        end
    end
end

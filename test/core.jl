using SpecializingLU
using LinearAlgebra
using Test

const SLU = SpecializingLU

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

@testset "SpecializingLU.jl" begin

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

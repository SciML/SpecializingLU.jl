using SpecializingFactorizations, Aqua, Test

@testset "Aqua quality assurance" begin
    Aqua.test_all(SpecializingFactorizations)
end

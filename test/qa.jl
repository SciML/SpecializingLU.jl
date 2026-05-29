using SpecializingLU, Aqua, Test

@testset "Aqua quality assurance" begin
    Aqua.test_all(SpecializingLU)
end

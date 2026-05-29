using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @safetestset "Core" begin
            include("core.jl")
        end
    end

    if GROUP == "All" || GROUP == "QA"
        @safetestset "Quality Assurance (Aqua)" begin
            include("qa.jl")
        end
    end
end

using Test
using ForceCouplingMethod

# The public surface is the two-phase API: the cold-path
# constructor `FFCMConfig`, the hot-path operator `mobility!`, and the
# matrix-free `FFCMMobility`. The seven internal step functions stay reachable as
# `ForceCouplingMethod.spread_forces!` for tests and advanced use, but exporting them
# would leak the internal decomposition and contradict the deep-module intent.
@testset "Only the two-phase public API is exported" begin
    # names() lists everything public, exported or not, and the module itself.
    public_names = setdiff(names(ForceCouplingMethod), [:ForceCouplingMethod])
    exported = filter(name -> Base.isexported(ForceCouplingMethod, name), public_names)
    @test sort(exported) == sort([:FFCMConfig, :mobility!, :FFCMMobility])
end

# Calibration is a cold-path tool people call once, by its full name, so it is part of the
# public interface without adding to what `using ForceCouplingMethod` brings into scope.
@testset "Calibration is public without being exported" begin
    @test Base.ispublic(ForceCouplingMethod, :calibrate)
    @test !Base.isexported(ForceCouplingMethod, :calibrate)
end

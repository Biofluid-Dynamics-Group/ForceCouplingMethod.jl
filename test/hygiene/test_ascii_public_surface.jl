using Test
using ForceCouplingMethod
using LinearAlgebra: mul!

# A terminal user must be able to type every name the public API requires, so
# the exported names and the keyword arguments of the public methods are
# ASCII. Unicode stays in internal identifiers and rendered documentation,
# where the paper's math symbols (σ, Σ, μ) are preferred for readability —
# this audit deliberately sweeps neither internal names nor docstring prose.

# The parametric keyword constructor `FFCMConfig{T}(; …)` is registered on the
# parameterised type, so `methods(FFCMConfig)` alone would miss it; a concrete
# instantiation is queried as well.
function _public_keyword_names()
    method_lists = (
        methods(FFCMConfig),
        methods(FFCMConfig{Float64}),
        methods(mobility!),
        methods(FFCMMobility),
        methods(ForceCouplingMethod.calibrate),
        filter(m -> m.module === ForceCouplingMethod, collect(methods(mul!))),
        filter(m -> m.module === ForceCouplingMethod, collect(methods(*))),
    )
    keyword_names = Symbol[]
    for method_list in method_lists, method in method_list
        append!(keyword_names, Base.kwarg_decl(method))
    end
    return keyword_names
end

@testset "Exported names are typeable in any terminal (ASCII)" begin
    @test all(isascii ∘ String, names(ForceCouplingMethod))
end

@testset "Public keyword arguments are typeable in any terminal (ASCII)" begin
    keyword_names = _public_keyword_names()
    # If method introspection ever stopped seeing the constructors, the ASCII
    # sweep would pass vacuously; pin known keywords to catch that.
    @test :kernel_widths_ratio in keyword_names
    @test :viscosity in keyword_names
    @test :fft_planning in keyword_names
    @test :relative_tolerance in keyword_names
    @test all(isascii ∘ String, keyword_names)
end

# Reference barycenter_socp / analyze_socp results from GraphTransportation.jl.
using GraphTransportation, LinearAlgebra, Printf
fmt(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"
Q, π = triangle_markov_chain()
G = MarkovGraph(Q, π)
refs = [[2.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 2.0]]
λ = [0.5, 0.3, 0.2]
for (name, θ) in (("GEOMETRIC", GeometricMean()), ("ARITHMETIC", ArithmeticMean()),
                  ("HARMONIC", HarmonicMean()), ("QUADLOG8", QuadLogMean(8)))
    Gθ = MarkovGraph(G; mean=θ)
    ν, J, geos = GraphTransportation.barycenter_socp(Gθ, refs, λ; N=6)
    println("BSOCP_", name, "_NU = ", fmt(ν))
    println("BSOCP_", name, "_J = ", @sprintf("%.17g", J))
    println("BSOCP_", name, "_W2S = ", fmt([g.W2 for g in geos]))
    λ̂ = vec(GraphTransportation.analyze_socp(Gθ, ν, refs; N=6))
    println("BSOCP_", name, "_LAMBDA_HAT = ", fmt(λ̂))
    λ̂m = vec(GraphTransportation.analyze_socp(Gθ, ν, refs; N=6, convention=:momentum))
    println("BSOCP_", name, "_LAMBDA_HAT_MOMENTUM = ", fmt(λ̂m))
end

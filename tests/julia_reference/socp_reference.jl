# Reference geodesic_socp solutions from GraphTransportation.jl, per admissible mean.
using GraphTransportation, LinearAlgebra, Printf
Q, π = grid_markov_chain(3)
hops = GraphTransportation._bfs_hops(MarkovGraph(Q, π))
mu = reduce(hcat, (begin v = exp.(-2 .* hops[k, :]); v ./ sum(v) end for k in (1, 3, 8)))
refs = [mu[:, s] ./ π for s in 1:3]
fmt(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"
for (name, θ) in (("GEOMETRIC", GeometricMean()), ("ARITHMETIC", ArithmeticMean()),
                  ("HARMONIC", HarmonicMean()), ("QUADLOG6", QuadLogMean(6)))
    G = MarkovGraph(Q, π; mean=θ)
    sol = GraphTransportation.geodesic_socp(G, refs[1], refs[2]; N=4)
    println("SOCP_", name, "_W2 = ", @sprintf("%.17g", sol.W2))
    println("SOCP_", name, "_RHO_MID = ", fmt(sol.ρ[:, 3]))
    println("SOCP_", name, "_M0 = ", fmt(sol.m0))
    println("SOCP_", name, "_PHI0 = ", fmt(sol.φ0))
    println("SOCP_", name, "_PHI1 = ", fmt(sol.φ1))
    println("SOCP_", name, "_STATUS = ", sol.status)
end

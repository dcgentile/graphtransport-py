# Reference values from GraphTransportation.jl's unified API (method=:sinkhorn).
using GraphTransportation, LinearAlgebra, Printf
Q, π = grid_markov_chain(3)
G = MarkovGraph(Q, π)
cost = ground_cost(G, :shortest_path)
hops = GraphTransportation._bfs_hops(G)
mu = reduce(hcat, (begin v = exp.(-2 .* hops[k, :]); v ./ sum(v) end for k in (1, 3, 8)))
refs = [mu[:, s] ./ π for s in 1:3]          # densities w.r.t. π
fmt(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"

ν, J, info = barycenter(G, refs, [0.5, 0.3, 0.2]; method=:sinkhorn, cost=cost, epsilon=0.1, iters=256)
println("BARY_NU = ", fmt(ν))
println("BARY_J = ", @sprintf("%.17g", J))
println("BARY_MARGINAL_ERRORS = ", fmt(info.marginal_errors))

sol = geodesic(G, refs[1], refs[2]; method=:sinkhorn, cost=cost, epsilon=0.1, N=4, iters=256)
println("GEO_W2 = ", @sprintf("%.17g", sol.W2))
println("GEO_RHO_COL2 = ", fmt(sol.ρ[:, 3]))   # t = 0.5
println("GEO_RHO_SHAPE = ", size(sol.ρ))
println("TRANSPORT_COST = ", @sprintf("%.17g", transport_cost(G, refs[1], refs[2]; method=:sinkhorn, cost=cost, epsilon=0.1, N=4, iters=256)))

λ̂ = analysis(G, ν, refs; method=:sinkhorn, cost=cost, epsilon=0.1, iters=256)
println("ANALYSIS_LAMBDA = ", fmt(λ̂))

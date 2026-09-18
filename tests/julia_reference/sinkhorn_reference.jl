# Reference values from GraphTransportation.jl for graphtransport-py's tests.
using GraphTransportation, LinearAlgebra, Printf

Q, π = grid_markov_chain(3)
G = MarkovGraph(Q, π)
cost = ground_cost(G, :shortest_path)
dcost = ground_cost(G, :diffusion)

# three probability vectors peaked at nodes 1, 3, 8 (0-indexed: 0, 2, 7)
hops = GraphTransportation._bfs_hops(G)
mu = reduce(hcat, (begin v = exp.(-2 .* hops[k, :]); v ./ sum(v) end for k in (1, 3, 8)))

fmt(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"

println("DIFFUSION_COST_ROW0 = ", fmt(dcost[1, :]))
println("MU = [", join((fmt(mu[:, s]) for s in 1:3), ",\n      "), "]")

lam = [0.5, 0.3, 0.2]
p = sinkhorn_barycenter(lam, mu, nothing, cost, 0.1; iters=256)
println("BARYCENTER_EPS01_ITERS256 = ", fmt(p))

p2 = sinkhorn_barycenter([0.25, 0.75], mu[:, 1:2], nothing, cost, 0.05; iters=64)
println("BARYCENTER2_EPS005_ITERS64 = ", fmt(p2))

# Step 5 references: gradient of the barycentric loss and simplex regression
alpha = [0.2, -0.1, 0.3]
q = p  # target = the eps=0.1 barycenter above
g = GraphTransportation.loss_gradient(alpha, mu, cost, q, 0.1; iters=40)
println("LOSS_GRADIENT_ALPHA_ITERS40 = ", fmt(g))
E = GraphTransportation.barycentric_loss(alpha, mu, q, cost, 0.1; iters=40)
println("BARYCENTRIC_LOSS_ALPHA_ITERS40 = ", @sprintf("%.17g", E))
_, w = GraphTransportation.sinkhorn_differentiate(lam, mu, q, cost, 0.1, 40)
println("W_LAMBDA_ITERS40 = ", fmt(w))
lamhat = simplex_regression(mu, q, cost, 0.1; iters=256)
println("SIMPLEX_REGRESSION_LAMBDA = ", fmt(lamhat))

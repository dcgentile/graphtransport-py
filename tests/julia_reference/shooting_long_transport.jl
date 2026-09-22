# Reference log_map (single shooting) values from GraphTransportation.jl on long transports:
# Gaussian bumps exp(-|x-c|^2/8) + 0.05 on an n x n grid, from corner (0, 0) to (shift, shift).
# Julia's Jacobian is ForwardDiff's; the Python port matches these values only with an exact
# Jacobian (its forward-difference Jacobian stalled on every one of them).
using GraphTransportation, LinearAlgebra, Printf
for (k, shift) in ((10, 9), (12, 8), (12, 11), (16, 15))
    G = MarkovGraph(grid_markov_chain(k)...)
    xy = [((i - 1) % k, (i - 1) ÷ k) for i in 1:k^2]
    bump(c) = (r = [exp(-((x - c[1])^2 + (y - c[2])^2) / 8) + 0.05 for (x, y) in xy]; r ./ dot(r, G.π))
    r = log_map(G, bump((0, 0)), bump((shift, shift)))
    @printf("(%d, %d): (%.17g, %d),\n", k, shift, r.W2, r.iters)
end

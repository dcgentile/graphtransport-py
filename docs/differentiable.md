# Differentiable geodesics

`geodesic` and `transport_cost` take torch tensors and return tensors. With
shooting, gradients flow back to both endpoints through `W2`, the density path,
the momenta and the potentials.

```python
import torch
import graphtransport as gt

G = gt.MarkovGraph(*gt.grid_markov_chain(5))
pi = torch.tensor(G.pi)
xy = torch.tensor([(i % 5, i // 5) for i in range(G.n)], dtype=torch.float64)
target = torch.exp(-((xy - 4) ** 2).sum(1) / 2) + 0.05  # a bump in the corner
target = target / (target @ pi)

logits = torch.zeros(G.n, dtype=torch.float64, requires_grad=True)
optimiser = torch.optim.Adam([logits], lr=0.3)
for step in range(30):
    optimiser.zero_grad()
    W = gt.transport_cost(G, torch.softmax(logits, 0) / pi, target)
    W.backward()
    optimiser.step()
```

<figure markdown>
![Gradient descent moves a uniform density onto a bump](assets/gradient-light.png#only-light)
![Gradient descent moves a uniform density onto a bump](assets/gradient-dark.png#only-dark)
![The transport distance during descent](assets/gradient-curve-light.png#only-light){ width="420" }
![The transport distance during descent](assets/gradient-curve-dark.png#only-dark){ width="420" }
<figcaption>Adam on softmax logits, minimising the transport distance itself to a
bump in the corner of a 5×5 grid.</figcaption>
</figure>

The gradients are exact for the discrete problem the solver solves. They come
from the implicit function theorem at the solution, not from differentiating
through Newton's iterations, so they agree with finite differences of the
returned values.

- Gradients are **first order**: differentiating a gradient again
  (`create_graph=True`) raises.
- Where `rhoA == rhoB`, `transport_cost`'s gradient is 0, the subgradient at its
  minimum, as for `torch.linalg.norm`.
- The SOCP and Sinkhorn methods are not differentiable here. With inputs that
  require grad they raise, and so does shooting where it would otherwise fall
  back to the SOCP.
- Tensors are computed in float64 on the CPU. `barycenter` and `analysis` do not
  take tensors yet.

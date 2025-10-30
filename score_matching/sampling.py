import torch

"""
Langevin dynamics
After training, our model is able to produce an approximation of
the gradient of the probabiliity.
Therefore, we could use this to generate data by relying on a simple gradient ascent
from a given point by using an initial sample,
and then using the gradient information to find a local maximum with a step size we 
take in the direction of the gradient (akin to the learning rate).
"""
def sample_simple(model, x, n_steps=20, eps=1e-3):
    x_sequence = [x[None].detach().cpu()]
    for s in range(n_steps):
        print(f"step {s}")
        x = x + eps * model(x)
        x_sequence.append(x[None].detach().cpu())
    return torch.cat(x_sequence)

"""
However, the previous procedure does not produce a true sample from p(x).
In order to obtain such a sample, we can rely on a special case of Langevin dynamics.
In this case, Langevin dynamics can produce true samples from the density p(x), 
by relying only on Gradient_x log p(x).
The sampling is defined in a way very similar to MCMC approaches, by applying 
recursively:
x_{t+1} = x_t + eps/2. * Gradient_x log p(x_t) + sqrt(eps) * z_t
where z_t is standard Gaussian noise.

It has been shown in Welling et al. (2011) that under eps -> 0, t -> infinity that 
x_t converges to an exact sample from p(x). 
This is a key idea behind the score-based generative modeling approach.
"""

"""
In order to implement this sampling procedure, we can once again start from 
x_0 ~ N(0,1) and progressively anneal eps -> 0 at each step to obtain true samples from p(x).
"""
def sample_langevin(model, x, n_steps=100, eps=1e-2):
    x_sequence = [x[None].detach().cpu()]
    for s in range(n_steps):
        print(f"step {s}")
        z = torch.randn_like(x)
        x = x + eps / 2. * model(x) + torch.sqrt(torch.tensor(eps)) * z
        x_sequence.append(x[None].detach().cpu())
    return torch.cat(x_sequence)

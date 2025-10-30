import torch
import torch.autograd as autograd
from utils import jacobian

def score_matching(model, samples, train=False):
    samples.requires_grad_(True)
    # Compute the model output
    logp = model(samples)
    # Compute the norm loss
    norm_loss = torch.norm(logp, dim=-1) ** 2 / 2.
    # Compute the Jacobian loss
    jacob_mat = jacobian(model, samples)
    tr_jacobian_loss = torch.diagonal(jacob_mat, dim1=-2, dim2=-1).sum(-1)
    return (tr_jacobian_loss + norm_loss).mean(-1)

"""
The previously defined score matching with this loss is not scalable
to high-dimensional data, nor deep networks, because of the computation of the trace 
of the jacobian tr(gradient_x F(x)).
Indeed, the computation of the Jacobian is a O(N^2 + N) operation, 
thus not being suitable for high-dimensional problems, even with the optimized solution
proposed in the previous code.
More recently, Song et al. [ 7 ] (sliced score matching) proposed to use random 
projections to approximate the computation of tr(gradient_x F(x)). 
 """
def sliced_score_matching(model, samples):
    samples.requires_grad_(True)
    # Construct random vectors
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
    # Compute the optimized vector-product jacobian
    logp, jvp = autograd.functional.jvp(model, samples, vectors, create_graph=True)
    # Compute the norm loss
    norm_loss = (logp * vectors) ** 2 / 2.
    # Compute the Jacobian loss
    v_jvp = jvp * vectors
    jacob_loss = v_jvp
    loss = jacob_loss + norm_loss
    return loss.mean(-1).mean(-1)

# Denoising score matching
"""
Originally, the notion of denoising score matching was discussed by Vincent
 in the context of denoising auto-encoders.
 In that case, this allows to completely remove the use of the gradient_x F_theta(x)
 in the  computation of score matching.
 To do so, we can first corrupt the input point x with a given noise vector, leading 
 to a distribution q_sigma (x_tilde | x).
 Then, score matching can be used to estimate the score of this perturbed data distribution.
 It has been shown that the optimal network that approximates 
 can be found by minimizing the denoising score matching objective (see doc).
"""
def denoising_score_matching(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss

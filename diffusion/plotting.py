import torch
from matplotlib import pyplot as plt


def plot_data_samples(data, fig_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.4)
    plt.title("Swiss roll points")
    plt.savefig(fig_dir / "swiss_roll.png")
    plt.close()


@torch.no_grad()
def plot_forward_diffusion(
    q_sample,
    x0,
    steps,
    fig_dir,
):
    # total steps to be visualized
    pics = 10
    assert steps % pics == 0, f"steps {steps} must be divisible by pics {pics}"
    step = steps // pics
    assert isinstance(step, int), (
        f"steps {steps} must be an integer divisible by {pics}, got {step}"
    )
    idxs = torch.arange(0, steps, step).long()
    fig, axs = plt.subplots(1, pics, figsize=(3*pics, 3))
    # x0 is the initial state of the data
    for i, t_i in enumerate(idxs):
        print(f'Plotting forward diffusion step {i+1}/{pics}, t={t_i}')
        tt = torch.full((x0.shape[0],), t_i)
        xt = q_sample(x0, t=tt)
        axs[i].scatter(xt[:, 0], xt[:, 1], s=3)
        axs[i].set_title(f"t={int(t_i)}")
        axs[i].set_axis_off()
    plt.suptitle("Forward diffusion process q")
    plt.savefig(fig_dir / f"forward_diffusion_{steps}.png")
    plt.close()


@torch.no_grad()
def plot_sampling_trajectory(
    sampling_trajectory,
    n_steps,
    epoch,
    fig_dir,
    num_snapshots=10,
):
    """
    Plot the reverse sampling trajectory showing denoising steps.

    Parameters
    ----------
    sampling_trajectory : list of torch.Tensor
        List of tensors containing samples at each timestep, from T to 0.
        Length should be n_steps + 1.
    n_steps : int
        Total number of diffusion steps.
    epoch : int
        Current training iteration/epoch for labeling.
    fig_dir : Path
        Directory to save the plot.
    num_snapshots : int, optional
        Number of snapshots to visualize (default: 10).
    """
    step = n_steps // num_snapshots
    idxs = torch.arange(0, n_steps + 1, step).long().tolist()

    fig, axs = plt.subplots(1, num_snapshots + 1, figsize=(3 * (num_snapshots + 1), 3))

    for i, t_idx in enumerate(idxs):
        xt = sampling_trajectory[t_idx]
        axs[i].scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=3)
        axs[i].set_title(f"t={t_idx}")
        axs[i].set_axis_off()

    plt.suptitle(f"Denoising process samples at iter {epoch} (EMA)")
    plt.savefig(fig_dir / f"epoch_{epoch}_ddpm_mean_sampling_trajectory.png")
    plt.close()


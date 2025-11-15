import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from helper_functions import hdr_plot_style

hdr_plot_style()

# Sample a batch from the swiss roll
def sample_batch(size, noise=1.0):
    x, _ = make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

def plot_swiss_roll(n=10**4, fpath="./figures/swiss_roll.png"):
    """Plot it"""
    data = sample_batch(n).T
    plt.figure(figsize=(16, 12))
    plt.scatter(*data, alpha=0.5, color='red', edgecolor='white', s=40)
    plt.savefig(fpath)
    plt.close()

if __name__ == "__main__":
    plot_swiss_roll()

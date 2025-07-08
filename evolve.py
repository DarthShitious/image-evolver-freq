import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import data
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from synthesis import *
from utils import *
from analysis import *


class Evolver:

    def __init__(self, config_path:str, results_dir:str):
        self.config = load_config(config_path=config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.results_dir = results_dir
        self.reconds_dir = os.path.join(results_dir, "recons")

        # Target image
        self.target = load_image(
            path=self.config.target_image,
            reduce_factor=self.config.reduce_factor
        ).to(self.device)
        C, H, W = self.target.shape

        # Instantiate Model (Genotype -> Phenotype)
        self.model = WaveletImageSynthesis(
            image_shape=self.target.shape,
            num_tiles_per_dim=self.config.spatial_depth,
            num_scales=self.config.scale_depth
        ).to(self.device)

        # Load from checkpoint if it exists
        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            self.ckpt = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.pop = self.ckpt["pop"]
            self.history = self.ckpt["history"]
            self.start_gen = self.ckpt["gen"] + 1
            print(f"Loaded checkpoint from '{self.config.checkpoint_path}'. Resuming at {self.start_gen}.")
        else:
            self.pop = torch.randn(self.config.pop_size, C, self.model.num_wavelet_kernels, device=self.device) #* 1e-6
            self.history = []
            self.start_gen = 1

        # Number of learnable parameters
        self.num_params = self.model.num_params()

        # Number of pixels in target image
        self.num_pixels = C*H*W

        # Compression Ratio
        self.comp_ratio = self.num_params / self.num_pixels

        # Print compression ratio
        print(f"Learnable parameters: {self.num_params:d}")
        print(f"Image Pixels:         {self.num_pixels:d}")
        print(f"Compression Ratio:    {self.comp_ratio:0.4f}")

    def fitness(self, coeffs):
        recon = self.model(coeffs)
        return torch.pow(recon - self.target, 2).mean(-1).mean(-1).mean(-1).detach().cpu().detach()

    def evaluate(self):
        # Evaluate Fitnesses
        fitnesses = self.fitness(self.pop)

        # Rank
        topk_vals, topk_idx = torch.topk(fitnesses, self.config.parent_k, largest=False)

        # Select Breeding pool
        self.parents = self.pop[topk_idx]

        # Record Best
        self.best_mse = topk_vals.max().item()
        self.history.append(self.best_mse)

    def breed(self):
        # Create the next generation
        next_pop = []

        # Elitism
        for i in range(self.config.num_elites):
            next_pop.append(self.parents[i].clone())

        # Crossover + Mutation
        while len(next_pop) < self.config.pop_size:
            i, j = np.random.choice(self.config.parent_k, size=2, replace=False)
            p1, p2 = self.parents[i], self.parents[j]
            point = np.random.randint(1, self.model.num_wavelet_kernels)
            child = torch.cat([p1[:, :point], p2[:, point:]], dim=1)
            mask  = (torch.rand_like(child) < self.config.mut_rate).float()
            noise = torch.randn_like(child) * self.config.mut_scale
            next_pop.append(child + mask * noise)

        self.pop = torch.stack(next_pop, dim=0)

    def analysis(self, gen:int):

        print(f"Gen {gen}/{self.config.generations} | Best MSE: {self.best_mse:.6f}")

        # Synthesize image with best chromosome
        best_coeffs = self.parents[0].unsqueeze(0)
        with torch.no_grad():
            recon = self.model(best_coeffs).cpu()[0].permute(1, 2, 0).numpy()
            orig  = self.target.cpu().permute(1, 2, 0).numpy()

        self.plot_recon(orig, recon, f"{recons_dir}/evolved_{gen:06d}.png")

    def plot_recon(self, orig, recon, path):

        fig, ax = plt.subplots(1,2,figsize=(10, 5))
        ax[0].imshow(orig);  ax[0].set_title("Orig");  ax[0].axis('off')
        ax[1].imshow(recon); ax[1].set_title("Best"); ax[1].axis('off')
        plt.suptitle(
            f"Wavelets: {self.config.spatial_depth**2:d} | Freqs/Wavelet: {self.config.scale_depth:d} | Total Params: {self.num_params:d} | Image Dims ({self.target.shape[-3]}, {self.target.shape[-2]}, {self.target.shape[-1]}) | Num Pixels: {self.num_pixels:d} | Ratio: {self.comp_ratio}", fontsize=8)
        plt.savefig(path)
        plt.close(fig)

    #     plt.figure()
    #     plt.plot(history)
    #     plt.xlabel("Generation")
    #     plt.ylabel("Best MSE")
    #     plt.title("GA Training Progress")
    #     plt.savefig(f"{results_dir}/evolution_progress.png")
    #     plt.close()

    #     # Plot histograms
    #     compare_rgb_hist(
    #         img1=recon,
    #         img2=target[0].cpu().numpy(),
    #         labels=["Recon", "Target"],
    #         percentile=99.9,
    #         path=os.path.join(hists_dir, f"evolved_hists_{gen:05d}.png")
    #     )

    #     # Plot Slice
    #     plot_slice(
    #         img1=recon.transpose(2, 0, 1),
    #         img2=target[0].cpu().numpy(),
    #         labels=["Recon", "Target"],
    #         path=os.path.join(slices_dir, f"evolved_slice_{gen:05d}.png")
    #     )

    #     # Save Checkpoint
    #     checkpoint_savepath = os.path.join(results_dir, "checkpoint.pth")
    #     torch.save({
    #         "pop": pop,
    #         "history": history,
    #         "gen": gen
    #     }, checkpoint_savepath)
    #     print(f"Saved checkpoint to '{checkpoint_savepath}' at gen {gen}")

    # if gen % (10 * analyze_every) == 0 or gen == generations:
    #     # Make videos
    #     pngs_to_mp4(recons_dir, f"{recons_dir}/evolved_recons.mp4", fps=15)
    #     pngs_to_mp4(hists_dir, f"{hists_dir}/evolved_hists.mp4", fps=4)
    #     pngs_to_mp4(slices_dir, f"{slices_dir}/trained_slices.mp4", fps=4)

if __name__ == "__main__":

    # Time stamp for results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), "evolved", timestamp)
    recons_dir = os.path.join(results_dir, "recons")
    hists_dir = os.path.join(results_dir, "hists")
    slices_dir = os.path.join(results_dir, "slices")
    for p in [results_dir, recons_dir, hists_dir, slices_dir]:
        os.makedirs(p, exist_ok=True)

    # Instantiate evolver
    evolver = Evolver(
        config_path="ga_config.yaml",
        results_dir=results_dir
    )

    # Evolution Loop
    for gen in trange(evolver.start_gen, evolver.config.generations+1):

        # Test fitnesses
        evolver.evaluate()

        # Breed
        evolver.breed()

        # Analysis
        if gen == 1 or gen % evolver.config.analyze_every == 0:
            evolver.analysis(gen=gen)










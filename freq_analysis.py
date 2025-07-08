import os
# import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
from scipy.signal import welch


def plot_rgb_power_spectra(image:np.ndarray, title:str="", path:str=""):

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Input must be an RGB image of shape (3, H, W).")

    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(8, 6))

    for i, color in enumerate(colors):
        channel = image[i, ...].astype(float)

        # Compute 2D FFT and shift zero frequency to center
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        # Compute the 2D power spectrum
        psd2D = np.abs(fshift) ** 2

        # Radially average the 2D power spectrum to get a 1D profile
        H, W = psd2D.shape
        center = [H // 2, W // 2]
        Y, X = np.indices((H, W))
        r = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        r = r.astype(int)

        # Bin and average the PSD values at each radius
        tbin = np.bincount(r.ravel(), psd2D.ravel())
        nr = np.bincount(r.ravel())
        radial_psd = tbin / np.maximum(nr, 1)

        # Create a radius vector
        radial_frequencies = np.arange(len(radial_psd))

        # Plot on log-log
        plt.loglog(radial_frequencies[1:], radial_psd[1:], color=color, label=f'{color} channel')

    # Plot fit guess
    fit = lambda f: radial_psd[1:].max()/((f - 1)**2.75)
    plt.plot(
        radial_frequencies,
        fit(radial_frequencies),
        color="black",
        linewidth=2
    )

    plt.xlabel('Spatial frequency (pixels⁻¹)')
    plt.ylabel('Power density')
    plt.title(f'Power Spectral Density: {title}')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.ylim(1e-3, 2*radial_psd[1:].max())
    plt.tight_layout()
    if path:
        plt.savefig(path)



if __name__ == "__main__":

    # Make save directory
    save_dir = "image_freq_analysis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load image
    image_path = "images/Checkerboard_patten.png"
    img = read_image("./images/Checkerboard_pattern.png") / 255.0
    C, img_H, img_W = img.shape

    # If single-channel, replicate it across RGB
    if C == 1:
        img = img.repeat(3, 1, 1)

    # Convert to numpy array
    img = img.detach().cpu().numpy()

    # Frequency Analysis
    plot_rgb_power_spectra(image=img, title="Checkerboard_pattern.png", path=f"{save_dir}/Checkerboard_pattern_psd.png")
import torch
import torch.nn as nn
import math

class HardLeakySigmoid(torch.nn.Module):
    """
        HardLeakySigmoid activation function.

        This activation behaves like a linear function in the central region
        [-1, 1], and applies a small “leak” outside that range to avoid zero
        gradients. Finally, it rescales the output to the [0, 1] interval
        with a half-wave rectification.

        Args:
            slope (float): Slope of the linear leak for inputs outside [-1, 1].
                        Defaults to 0.01.

        Shape:
            - Input: (…, *) where * means any number of additional dimensions.
            - Output: Same shape as input.

        Forward computation:
            1. For x ≤ -1: f = slope * (x + 1) - 1
            2. For -1 < x ≤ 1: f = x
            3. For x > 1: f = slope * (x - 1) + 1
            4. Output = max(0.5 * (f + 1), slope * x)

        Returns:
            torch.Tensor: Activated output, same shape as input.
        """
    def __init__(self, slope=0.01):
        super().__init__()
        self.m = slope

    def forward(self, x):
        f = torch.where(
            x <= -1,
            self.m * (x + 1) - 1,
            torch.where(
                x > 1,
                self.m * (x - 1) + 1,
                x
            )
        )
        return torch.maximum(0.5 * (f + 1), self.m * x)


class WaveletImageSynthesis(nn.Module):
    def __init__(
        self,
        image_shape,
        num_tiles_per_dim,
        num_scales,
        x_extent=1.0,
        y_extent=1.0,
        device=None
    ):
        super().__init__()
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.num_tiles_per_dim = num_tiles_per_dim
        self.num_scales = num_scales
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # B is batch size, C/H/W are kept as single letters per your preference
        C, H, W = image_shape
        B = 1

        # Build coordinate grids
        x_linspace = torch.linspace(-x_extent, x_extent, W, device=self.device)
        y_linspace = torch.linspace(-y_extent, y_extent, H, device=self.device)
        coord_grid_y, coord_grid_x = torch.meshgrid(y_linspace, x_linspace, indexing='ij')
        coord_grid_x = coord_grid_x.view(1, 1, 1, H, W)
        coord_grid_y = coord_grid_y.view(1, 1, 1, H, W)

        # Compute tile centers
        tile_size_x = 2 * x_extent / num_tiles_per_dim
        tile_size_y = 2 * y_extent / num_tiles_per_dim
        tile_centers_x = -x_extent + tile_size_x * (0.5 + torch.arange(num_tiles_per_dim, device=self.device))
        tile_centers_y = -y_extent + tile_size_y * (0.5 + torch.arange(num_tiles_per_dim, device=self.device))
        self.register_buffer('tile_centers_x', tile_centers_x)
        self.register_buffer('tile_centers_y', tile_centers_y)

        # Frequencies & wavelengths
        self.frequencies = torch.tensor([2 ** (n / 2) for n in range(num_scales)], device=self.device)
        wavelengths = 1.0 / self.frequencies
        safe_wavelengths = wavelengths.abs().clamp(min=1e-4)
        self.register_buffer('cycles_per_pixel', 1.0 / safe_wavelengths)
        self.register_buffer('wavelengths', wavelengths)

        # Build wavelet basis
        S = num_tiles_per_dim
        Sd = num_scales
        tcx = self.tile_centers_x.view(S, 1, 1, 1, 1)
        tcy = self.tile_centers_y.view(1, S, 1, 1, 1)
        cpp = self.cycles_per_pixel.view(1, 1, Sd, 1, 1)

        freq_grid_x = (coord_grid_x - tcx) * cpp
        freq_grid_y = (coord_grid_y - tcy) * cpp
        sine_basis   = self.sawtooth_wavepacket_sine_sep(freq_grid_x, freq_grid_y)
        cosine_basis = self.sawtooth_wavepacket_cosine_sep(freq_grid_x, freq_grid_y)
        wavelet_basis = torch.cat([sine_basis, cosine_basis], dim=2)  # (S, S, 2*Sd, H, W)

        # Flatten and broadcast
        self.num_wavelet_kernels = num_tiles_per_dim * num_tiles_per_dim * num_scales * 2
        self.wavelet_basis = (
            wavelet_basis
            .reshape(self.num_wavelet_kernels, H, W)
            .unsqueeze(0)
            .expand(B, -1, -1, -1)
            .to(self.device)
        )

        # Output activation
        self.hls = HardLeakySigmoid(slope=0.01)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------------ #
    def sawtooth_wavepacket_sine_radial(self, u, v):
        N = 12; S = 1
        saw = 0
        r = torch.sqrt(u**2 + v**2)
        for n in range(1, N+1):
            saw += (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*r)
        env = torch.exp(-0.5 * (r/S)**2)
        return saw * env

    def sawtooth_wavepacket_cosine_radial(self, u, v):
        N = 12; S = 1
        saw = 0
        r = torch.sqrt(u**2 + v**2)
        for n in range(1, N+1):
            saw += (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*(r + 0.25))
        env = torch.exp(-0.5 * (r/S)**2)
        return saw * env

    def sawtooth_wavepacket_sine_sep(self, u, v):
        N = 8; S = 1
        r = torch.sqrt(u**2 + v**2)
        saw_u = saw_v = 0
        for n in range(1, N+1):
            coef = (2/torch.pi) * (((-1)**(n+1))/n)
            saw_u += coef * torch.sin(2*torch.pi*n*u)
            saw_v += coef * torch.sin(2*torch.pi*n*v)
        env = torch.exp(-0.5 * (r/S)**2)
        return saw_u * saw_v * env

    def sawtooth_wavepacket_cosine_sep(self, u, v):
        N = 8; S = 1
        r = torch.sqrt(u**2 + v**2)
        saw_u = saw_v = 0
        for n in range(1, N+1):
            coef = (2/torch.pi) * (((-1)**(n+1))/n)
            saw_u += coef * torch.sin(2*torch.pi*n*(u + 0.25))
            saw_v += coef * torch.sin(2*torch.pi*n*(v + 0.25))
        env = torch.exp(-0.5 * (r/S)**2)
        return saw_u * saw_v * env

    # ------------------------------------------------------------------------ #
    def forward(self, wavelet_coeffs) -> torch.Tensor:
        B, C, _ = wavelet_coeffs.shape
        with torch.no_grad():
            synthesized_image = torch.einsum('bcn, bnhw -> bchw', wavelet_coeffs, self.wavelet_basis)
            coeff_norm = torch.abs(wavelet_coeffs).sum(dim=2).view(B, C, 1, 1)
            synthesized_image = self.hls(synthesized_image)
        return synthesized_image













# class WaveletImageSynthesis(nn.Module):
#     def __init__(
#         self,
#         target_shape,
#         spatial_depth,
#         scale_depth,
#         boundx=1.0,
#         boundy=1.0,
#         device=None
#         ):

#         super().__init__()
#         self.boundx = boundx
#         self.boundy = boundy
#         self.spatial_depth = spatial_depth
#         self.scale_depth = scale_depth
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

#         # Target shape
#         C, H, W = target_shape
#         B = 1

#         # Build mesh grid
#         xs = torch.linspace(-self.boundx, self.boundx, W, device=device)
#         ys = torch.linspace(-self.boundy, self.boundy, H, device=device)
#         grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

#         # Reshape for broadcasting
#         grid_x = grid_x.view(1, 1, 1, H, W)
#         grid_y = grid_y.view(1, 1, 1, H, W)

#         # Compute tile widths
#         tile_width_x = 2 * boundx / spatial_depth
#         tile_width_y = 2 * boundy / spatial_depth

#         # Compute centers of tiles
#         shifts_x = -boundx + tile_width_x * (0.5 + torch.arange(spatial_depth))
#         shifts_y = -boundy + tile_width_y * (0.5 + torch.arange(spatial_depth))

#         # Register as buffers
#         self.register_buffer('shifts_x', shifts_x)
#         self.register_buffer('shifts_y', shifts_y)

#         # Scale factors along x and y
#         self.freqs = torch.Tensor([1.0 * 2**(n/2) for n in range(scale_depth)])
#         scales = 1 / self.freqs
#         scales_safe = scales.abs().clamp(min=1e-4)
#         self.register_buffer('inv_scales', (1.0 / scales_safe))
#         self.register_buffer('scales', scales)

#         S, Sd = self.spatial_depth, self.scale_depth
#         sx = self.shifts_x.view(S, 1, 1, 1, 1)
#         sy = self.shifts_y.view(1, S, 1, 1, 1)
#         isc = self.inv_scales.view(1, 1, Sd, 1, 1)

#         # Vectorized wavelet bank: sine & cosine phases
#         u = (grid_x - sx) * isc
#         v = (grid_y - sy) * isc
#         basis_sin = self.sawtooth_wavepacket_sine_sep(u, v)
#         basis_cos = self.sawtooth_wavepacket_cosine_sep(u, v)
#         basis = torch.cat([basis_sin, basis_cos], dim=2)  # (S, S, 2*Sd, H, W)

#         # Learnable parameters: double for sine and cosine phases
#         self.num_bases = spatial_depth * spatial_depth * scale_depth * 2  # factor 2 for sine & cosine
#         # self.wavelet_coefficients = nn.Parameter(torch.randn(1, 3, self.num_bases))

#         # Flatten and broadcast
#         self.basis = basis.reshape(self.num_bases, H, W).unsqueeze(0).expand(B, -1, -1, -1).to(self.device)

#         # Output Activation
#         self.hls = HardLeakySigmoid(slope=0.01)

#     def num_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

#     def sawtooth_wavepacket_sine_radial(self, u, v):
#         N = 12
#         S = 1
#         sawtooth = 0
#         r = torch.sqrt(u**2 + v**2)
#         for n in range(1, N + 1):
#             sawtooth = sawtooth + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*r)
#         env = torch.exp(-0.5 * (r/S)**2)
#         return sawtooth * env

#     def sawtooth_wavepacket_cosine_radial(self, u, v):
#         N = 12
#         S = 1
#         sawtooth = 0
#         r = torch.sqrt(u**2 + v**2)
#         for n in range(1, N + 1):
#             sawtooth = sawtooth + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*(r + 0.25))
#         env = torch.exp(-0.5 * (r/S)**2)
#         return sawtooth * env

#     def sawtooth_wavepacket_sine_sep(self, u, v):
#         N = 8
#         S = 1
#         r = torch.sqrt(u**2 + v**2)
#         sawtooth_u = 0
#         sawtooth_v = 0
#         for n in range(1, N + 1):
#             sawtooth_u = sawtooth_u + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*u)
#             sawtooth_v = sawtooth_v + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*v)
#         env = torch.exp(-0.5 * (r/S)**2)
#         return sawtooth_u * sawtooth_v * env

#     def sawtooth_wavepacket_cosine_sep(self, u, v):
#         N = 8
#         S = 1
#         r = torch.sqrt(u**2 + v**2)
#         sawtooth_u = 0
#         sawtooth_v = 0
#         for n in range(1, N + 1):
#             sawtooth_u = sawtooth_u + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*(u + 0.25))
#             sawtooth_v = sawtooth_v + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin(2*torch.pi*n*(v + 0.25))
#         env = torch.exp(-0.5 * (r/S)**2)
#         return sawtooth_u * sawtooth_v * env

#     # ------------------------------------------------------------------------ #
#     def forward(self, coeffs) -> torch.Tensor:
#         B, C, _ = coeffs.shape
#         with torch.no_grad():
#             # Synthesize image
#             out = torch.einsum('bcn, bnhw -> bchw', coeffs, self.basis)
#             norm = torch.abs(coeffs).sum(dim=2).view(B, C, 1, 1)
#             # out = out / (norm + 1e-9)
#             out = self.hls(out)
#         return out



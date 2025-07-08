


def ricker(self, u, v):
    return (1 - 2 * (u**2 + v**2)) * torch.exp(-(u**2 + v**2))

def sinegaussian(self, u, v):
    return torch.sin(u) * torch.sin(v) * torch.exp(-(u**2 + v**2))

def sinc(self, u, v):
    return torch.sinc(u) * torch.sinc(v)

def sinc_radial(self, u, v):
    r = (u**2 + v**2)**0.5
    return torch.sinc(r)


def cosinc_radial(self, u, v):
    # Cosine-phase counterpart of the radial sinc wavelet
    r = (u**2 + v**2)**0.5
    pi_r = math.pi * r
    # Avoid division by zero
    cosr = torch.where(r == 0, torch.ones_like(r), torch.cos(pi_r) / pi_r)
    return cosr

def sinc_taxi(self, u, v):
    r = torch.abs(u) + torch.abs(v)
    return torch.sinc(r)

def sine_radial(self, u, v):
    r = torch.sqrt(u**2 + v**2)
    return torch.sin(2 * torch.pi * r)

def grid(self, u, v):
    return torch.sin(2 * torch.pi * u) + torch.sin(2 * torch.pi * v)

def morlet_1d(self, x):
    s = torch.Tensor([5.0]).to(self.device)
    c = 1 / torch.sqrt(1 + torch.exp(-s**2) - 2 * torch.exp(-(3/4) * s**2))
    K = torch.exp(-(s**2 / 2))
    return c * torch.pi**(-1/4) * torch.exp(-(x**2)/2) * (torch.exp(1j * s * x) - K)

def morlet_2d(self, u, v):
    return torch.real(self.morlet_1d((u**2 + v**2)**0.5))

def wavepacket_sine(self, u, v):
    s = 1
    r = torch.sqrt(u**2 + v**2)
    g = torch.exp(-0.5 * (r/s)**2)
    f = torch.sin(2*torch.pi*r)
    return g * f

def wavepacket_cosine(self, u, v):
    s = 1
    r = torch.sqrt(u**2 + v**2)
    g = torch.exp(-0.5 * (r/s)**2)
    f = torch.cos(2*torch.pi*r)
    return g * f

def sawtooth_wavepacket_sine(self, u, v, phase=0):
    N = 4
    S = 1
    sawtooth_u = 0
    sawtooth_v = 0
    for n in range(1, N + 1):
        sawtooth_u = sawtooth_u + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin((2*torch.pi*n*(u + phase/(2*torch.pi))))
        sawtooth_v = sawtooth_v + (2/torch.pi) * (((-1)**(n+1))/n) * torch.sin((2*torch.pi*n*(v + phase/(2*torch.pi))))
    env_u = torch.exp(-0.5 * (u/S)**2)
    env_v = torch.exp(-0.5 * (v/S)**2)
    return sawtooth_u * env_u * sawtooth_v * env_v

def sawtooth_wavepacket_cosine(self, u, v):
    return self.sawtooth_wavepacket_sine(u, v, phase=torch.pi/2)
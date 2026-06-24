import torch

_RGB2XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ2RGB = torch.inverse(_RGB2XYZ)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])  # d65 reference white

_EPS = 0.008856  # (6/29)**3
_KAPPA = 903.3  # (29/3)**3


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(
        c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp(min=0) ** 2.4
    )


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(min=0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    # (B,3,H,W) sRGB image in [0,1] to CIELab.
    m = _RGB2XYZ.to(rgb.device, rgb.dtype)
    lin = _srgb_to_linear(rgb)
    xyz = torch.einsum("ij,bjhw->bihw", m, lin)
    xyz = xyz / _WHITE.to(rgb.device, rgb.dtype).view(1, 3, 1, 1)

    f = torch.where(
        xyz > _EPS, xyz.clamp(min=1e-8) ** (1 / 3), (_KAPPA * xyz + 16) / 116
    )
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    # (B,3,H,W) CIELab image back to sRGB in [0,1].
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def f_inv(t: torch.Tensor) -> torch.Tensor:
        t3 = t**3
        return torch.where(t3 > _EPS, t3, (116 * t - 16) / _KAPPA)

    xyz = torch.cat([f_inv(fx), f_inv(fy), f_inv(fz)], dim=1)
    xyz = xyz * _WHITE.to(lab.device, lab.dtype).view(1, 3, 1, 1)
    m = _XYZ2RGB.to(lab.device, lab.dtype)
    lin = torch.einsum("ij,bjhw->bihw", m, xyz)
    return _linear_to_srgb(lin).clamp(0, 1)


class ABQuantizer:
    def __init__(
        self,
        grid_size: int = 16,
        ab_range: float = 110.0,
        temperature: float = 0.38,
        device=None,
    ):
        self.grid_size = grid_size
        self.ab_range = ab_range
        self.step = 2 * ab_range / grid_size
        self.num_classes = grid_size * grid_size
        self.temperature = temperature

        edges = torch.linspace(-ab_range, ab_range, grid_size + 1)
        centers_1d = (edges[:-1] + edges[1:]) / 2  # (grid_size,)
        ga, gb = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        # row-major flatten: class index = ai * grid_size + bi
        self.centers = torch.stack([ga.reshape(-1), gb.reshape(-1)], dim=1)  # (C, 2)
        if device is not None:
            self.centers = self.centers.to(device)

    def ab_to_index(self, ab: torch.Tensor) -> torch.Tensor:
        # (B,2,H,W) continuous ab to (B,H,W) long class indices.
        a, b = ab[:, 0], ab[:, 1]
        ai = ((a + self.ab_range) / self.step).floor().clamp(0, self.grid_size - 1)
        bi = ((b + self.ab_range) / self.step).floor().clamp(0, self.grid_size - 1)
        return (ai * self.grid_size + bi).long()

    def soft_decode(
        self, logits: torch.Tensor, temperature: float | None = None
    ) -> torch.Tensor:
        # (B,C,H,W) logits -> (B,2,H,W) continuous ab (annealed-mean decode).
        if temperature is None:
            temperature = self.temperature
        prob = torch.softmax(logits / temperature, dim=1)
        centers = self.centers.to(logits.device, logits.dtype)
        a = torch.einsum("bchw,c->bhw", prob, centers[:, 0])
        b = torch.einsum("bchw,c->bhw", prob, centers[:, 1])
        return torch.stack([a, b], dim=1)

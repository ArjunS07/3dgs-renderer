import torch
from plyfile import PlyData
from threed_util import q_to_3d_rotation_matrix, scaling_vector_to_mats

# Config
LAMBDA = 0.2
EPSILON_ALPHA = 1e-4
DENSIFICATION_ITER = 100
ALPHA_TO_ZERO_N = 3000
TAU_POS = 2e-4
PHI = 1.6


class GaussianScene:
    def __init__(
        self,
        max_sh_degree: int = 4,
        lambda_weight: float = LAMBDA,
        epsilon_alpha: float = EPSILON_ALPHA,
        densification_iter: int = DENSIFICATION_ITER,
        tau_pos: float = TAU_POS,
        phi_factor: float = PHI,
        alpha_to_zero_n: int = ALPHA_TO_ZERO_N,
        device="cpu",
    ):
        self.mu_x = torch.tensor([])
        self.mu_y = torch.tensor([])
        self.mu_z = torch.tensor([])
        self.scaling_s = torch.tensor([])
        self.rotation_q = torch.tensor([])
        self.alpha = torch.tensor([])

        self.sh_coefficients = torch.tensor([])

        self.max_sh_degree = max_sh_degree

        # Optimization
        self.lambda_weight = lambda_weight
        self.epsilon_alpha = epsilon_alpha
        self.densification_iterations = densification_iter
        self.tau_pos = tau_pos
        self.phi = PHI
        self.alpha_to_zero_n = alpha_to_zero_n

        self.device = device

    def calc_n_gaussians(self):
        return self.mu_x.shape[0]

    def calc_covariance_3d(self):
        # Express RS S^T R^T as RS (RS)^T to save a matmul
        R = q_to_3d_rotation_matrix(self.rotation_q)
        S = scaling_vector_to_mats(self.scaling_s)
        RS = R @ S
        covariances = RS @ RS.transpose(-1, -2)
        return covariances

    def save_to_ply(self, filename: str):
        """ """
        ...

    def init_full_scene_from_ply(self, filename: str):
        ply = PlyData.read(filename)
        v = ply["vertex"]
        self.mu_x = torch.tensor(v["x"], device=self.device)
        self.mu_y = torch.tensor(v["y"], device=self.device)
        self.mu_z = torch.tensor(v["z"], device=self.device)

        scales = torch.stack(
            [
                torch.tensor(v["scale_0"]),
                torch.tensor(v["scale_1"]),
                torch.tensor(v["scale_2"]),
            ],
            dim=-1,
        ).to(self.device)
        self.scaling_s = torch.exp(scales)

        rots = torch.stack(
            [
                torch.tensor(v["rot_0"]),
                torch.tensor(v["rot_1"]),
                torch.tensor(v["rot_2"]),
                torch.tensor(v["rot_3"]),
            ],
            dim=-1,
        ).to(self.device)

        self.rotation_q = rots / rots.norm(dim=-1, keepdim=True)

        self.alpha = torch.sigmoid(torch.tensor(v["opacity"], device=self.device))

        dc = torch.stack(
            [
                torch.tensor(v["f_dc_0"]),
                torch.tensor(v["f_dc_1"]),
                torch.tensor(v["f_dc_2"]),
            ],
            dim=-1,
        )

        rest_names = sorted(
            [p.name for p in v.properties if p.name.startswith("f_rest_")],
            key=lambda n: int(n.split("_")[-1]),
        )
        rest = torch.stack([torch.tensor(v[name]) for name in rest_names], dim=-1)
        n_rest_per_channel = len(rest_names) // 3
        rest_reshaped = rest.reshape(-1, 3, n_rest_per_channel).permute(
            0, 2, 1
        )  # (N, K, 3)
        dc_expanded = dc.unsqueeze(1)  # (N, 1, 3)
        self.sh_coefficients = torch.cat([dc_expanded, rest_reshaped], dim=1).to(
            self.device
        )

    def init_from_colmap(self): ...


class MaskedSceneView:
    MASKED_ATTRS = [
        "covariance_3d",
        "cov_camera_3d",
        "cov_2d",
        "mu_3d",
        "mu_camera_3d",
        "mu_2d",
        "pi",
        "pj",
        "viewdep_colors",
        "sh_coefficients",
        "alpha",
        "stddev_2d",
        "radius_px_2d",
    ]

    def __init__(self, scene: GaussianScene, device):
        """
        The mask is 1 where we want to retain the value and 0 when we want to drop
        """
        self.scene = scene
        self.mask = torch.ones(
            scene.calc_n_gaussians(), dtype=torch.bool, device=device
        )
        self.tile_masks = []

    def __setattr__(self, name: str, value):
        if name in MaskedSceneView.MASKED_ATTRS:
            try:
                mask = super().__getattribute__("mask")
                # enforce full n tensor
                if (
                    isinstance(value, torch.Tensor)
                    and value.shape[0] == int(mask.sum())
                    and int(mask.sum()) < mask.shape[0]
                ):
                    full_shape = (mask.shape[0],) + tuple(value.shape[1:])
                    full = torch.zeros(
                        full_shape, dtype=value.dtype, device=value.device
                    )
                    full[mask] = value
                    value = full
            except AttributeError:
                pass
            super().__setattr__(f"_{name}", value)
        else:
            super().__setattr__(name, value)

    def __getattribute__(self, name: str):
        if name in MaskedSceneView.MASKED_ATTRS:
            value = super().__getattribute__(f"_{name}")
            mask = super().__getattribute__("mask")
            return value[mask]
        else:
            return super().__getattribute__(name)

    @property
    def N(self) -> int:
        return int(self.mask.sum())

    def update_mask(self, new_mask):
        """
        New mask length is going to be less than the old mask shape
        Over the indices where the old mask was true
        Set the value to old & new
        """
        old_mask = self.mask
        assert new_mask.shape[0] == old_mask.sum()
        updated_mask = torch.zeros_like(old_mask)
        updated_mask[old_mask] = new_mask
        self.mask = updated_mask

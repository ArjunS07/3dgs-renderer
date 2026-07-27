import torch


def q_to_3d_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternion vectors [s, x, y, z] into 4x4 rotation matrices.
    Input: q of shape (B, 4)
    Output: R of shape (B, 4, 4)
    """
    # convert to unit quarternion
    q = q / q.norm(dim=-1, keepdim=True)
    qs, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Don't need the homogeneous 4th row here
    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
    R[:, 0, 1] = 2 * (qx * qy - qz * qs)
    R[:, 0, 2] = 2 * (qx * qz + qy * qs)

    R[:, 1, 0] = 2 * (qx * qy + qz * qs)
    R[:, 1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
    R[:, 1, 2] = 2 * (qy * qz - qx * qs)

    R[:, 2, 0] = 2 * (qx * qz - qy * qs)
    R[:, 2, 1] = 2 * (qy * qz + qx * qs)
    R[:, 2, 2] = 1 - 2 * qx * qx - 2 * qy * qy

    return R


def scaling_vector_to_mats(scaling_s: torch.Tensor):
    return torch.diag_embed(scaling_s)

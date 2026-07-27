import torch
import triton
import triton.language as tl
from gaussian_scene import GaussianScene, MaskedSceneView
from spherical_harmonics import evaluate_sh
from torchvision.utils import save_image

# Numerical stability constants
NEAR_PLANE_Z_THRESHOLD = 0.01
TILE_SIZE = 16
DEVICE = triton.runtime.driver.active.get_active_torch_device()
EPSILON = 1.0 / 255
ALPHA_CLAMP_MAX = 0.99


"""
We add 0.3 to the diagonal of the projected 2-D covariance matrix.
This is following the original 3DGS implementation.
While not explicitly discussed in the Kerbl et al. paper, Mip-Splatting and `gsplat` both detail this

Refer to A.4 of `gsplat`: https://arxiv.org/pdf/2409.06765
"""
S_ADD_TO_COV2D_DIAG = 0.3


def camera_coords_to_screen(mu_2d, fx, fy, cx, cy):
    p_i = fx * mu_2d[..., 0] + cx - 0.5
    p_j = fy * mu_2d[..., 1] + cy - 0.5
    return torch.stack([p_i, p_j], dim=-1)


def camera_intrinsics_to_info(camera_intrinscs, w, h):
    fx, fy = camera_intrinscs[0, 0], camera_intrinscs[1, 1]
    cx, cy = camera_intrinscs[0, 2], camera_intrinscs[1, 2]
    xmin, xmax = -cx / fx, (w - cx) / fx
    ymin, ymax = -cy / fy, (h - cy) / fy
    return fx, fy, cx, cy, xmin, xmax, ymin, ymax


@triton.jit
def rasterizer_kernel(
    mu_screen_ptr,
    inv_cov_ptr,
    alpha_ptr,
    colors_ptr,
    tile_gaussian_start_ptr,  # (n_tiles,)
    n_tiles_x,
    image_w,
    image_h,
    color_output_ptr,
    TILE_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    ALPHA_CLAMP_MAX: tl.constexpr,
):
    """
    Input:
        `tile_gaussian_start_ptr`: a 1D array containing the offset into the packed array where that tile's gaussians begin
    """
    tile_id = tl.program_id(0)
    pixel_id = tl.program_id(1)

    tile_origin_x = (tile_id % n_tiles_x) * TILE_SIZE
    tile_origin_y = (tile_id // n_tiles_x) * TILE_SIZE

    rel_x = pixel_id % TILE_SIZE
    rel_y = pixel_id // TILE_SIZE
    abs_x = tile_origin_x + rel_x
    abs_y = tile_origin_y + rel_y

    start = tl.load(tile_gaussian_start_ptr + tile_id)
    # Get the offset upto the next tile, so the difference is the count of this tile's gaussians
    count = tl.load(tile_gaussian_start_ptr + tile_id + 1) - start

    T = 1.0
    C_r, C_g, C_b = 0.0, 0.0, 0.0

    for n in range(count):
        idx = start + n
        mu_x = tl.load(mu_screen_ptr + 2 * idx)
        mu_y = tl.load(mu_screen_ptr + 2 * idx + 1)
        ic00 = tl.load(inv_cov_ptr + 3 * idx)
        ic01 = tl.load(inv_cov_ptr + 3 * idx + 1)
        ic11 = tl.load(inv_cov_ptr + 3 * idx + 2)
        c_r = tl.load(colors_ptr + 3 * idx)
        c_g = tl.load(colors_ptr + 3 * idx + 1)
        c_b = tl.load(colors_ptr + 3 * idx + 2)
        alpha = tl.load(alpha_ptr + idx)

        """
        We weight alpha by the Gaussian PDF value at the pixel center, which is given by:
            G(x) = exp(-0.5 * (x - mu)^T @ inv_cov @ (x - mu)))

        x^T Sigma x = (x, y) (a, b \\ b, d)(x \\ y)
                    = (x, y) (ax + by \\ bx + dy)
                    = (ax^2 + 2byx + dy^2)
        """
        diff_x = abs_x - mu_x
        diff_y = abs_y - mu_y
        exponent = -0.5 * (
            ic00 * diff_x * diff_x + 2 * ic01 * diff_x * diff_y + ic11 * diff_y * diff_y
        )

        a = alpha * tl.exp(exponent)
        valid = (a > EPSILON) & (T >= 1e-4)
        a = tl.minimum(a, ALPHA_CLAMP_MAX)

        C_r += tl.where(valid, T * a * c_r, 0.0)
        C_g += tl.where(valid, T * a * c_g, 0.0)
        C_b += tl.where(valid, T * a * c_b, 0.0)
        T *= tl.where(valid, 1 - a, 1.0)

    # if / return early doesn't work in triton, so define a mask for whether the tile is in the bounds of the image
    out_offset = (abs_y * image_w + abs_x) * 3
    in_bounds = (abs_x < image_w) & (abs_y < image_h)
    tl.store(color_output_ptr + out_offset, C_r, mask=in_bounds)
    tl.store(color_output_ptr + out_offset + 1, C_g, mask=in_bounds)
    tl.store(color_output_ptr + out_offset + 2, C_b, mask=in_bounds)


def rasterize_from_camera_view(
    scene: GaussianScene,
    P,
    fx,
    fy,
    cx,
    cy,
    xmin,
    xmax,
    ymin,
    ymax,
    w,
    h,
    cov_3d,
    save_to_file_path: str,
    device=torch.device("cpu"),
):
    assert P.shape == (4, 4)
    R, t = P[0:3, 0:3], P[0:3, 3].flatten()

    view = MaskedSceneView(scene, device=device)
    view.cov_3d = cov_3d
    view.sh_coefficients = scene.sh_coefficients
    view.alpha = scene.alpha

    view.mu_3d = torch.stack([scene.mu_x, scene.mu_y, scene.mu_z], dim=-1)  # (N, 3)
    view.mu_camera_3d = (view.mu_3d - t) @ R  # (N, 3), C2W: p_cam = R^T (p_world - t)

    # CULL 1: Cull Gaussians with means close to the viewing plane
    near_plane_mask = view.mu_camera_3d[:, 2] > NEAR_PLANE_Z_THRESHOLD
    view.update_mask(near_plane_mask)

    # Calculate 2-D means in camera space and pixel space
    view.mu_2d = (
        torch.stack([view.mu_camera_3d[:, 0], view.mu_camera_3d[:, 1]], dim=-1)
        / view.mu_camera_3d[:, 2][:, None]
    )
    pi_pj = camera_coords_to_screen(view.mu_2d, fx, fy, cx, cy)
    view.pi, view.pj = pi_pj[:, 0], pi_pj[:, 1]

    # CULL 2: Cull Gaussians with projected means far outside the 2-D view frustum
    view_frustum_mask = (
        (view.mu_2d[:, 0] > xmin)
        & (view.mu_2d[:, 0] < xmax)
        & (view.mu_2d[:, 1] > ymin)
        & (view.mu_2d[:, 1] < ymax)
    )
    view.update_mask(view_frustum_mask)

    # Calculate 2-D covariance
    R_batch = R.unsqueeze(0)  # (1, 3, 3)
    view.cov_camera_3d = (
        R_batch.transpose(-2, -1)  # 1, 3, 3
        @ view.cov_3d  # N_filtered, 3, 3
        @ R_batch  # 1, 3, 3
    )  # N, 3, 3
    J = torch.zeros(view.N, 2, 3, device=device)
    z = view.mu_camera_3d[:, 2]
    x = view.mu_camera_3d[:, 0]
    y = view.mu_camera_3d[:, 1]
    J[..., 0, 0] = fx / z
    J[..., 1, 1] = fy / z
    J[..., 0, 2] = -fx * x / (z**2)
    J[..., 1, 2] = -fy * y / (z**2)

    view.cov_2d = J @ view.cov_camera_3d @ J.transpose(-2, -1)

    # CULL 3: Cull Gaussians with less than estimated 99% confidence of intersecting the view frustum. Approximate this as 3 stddev from the mean in each direction
    var_max = (
        torch.maximum(view.cov_2d[:, 0, 0], view.cov_2d[:, 1, 1]) + S_ADD_TO_COV2D_DIAG
    )
    radius_px = (3 * torch.sqrt(var_max)).ceil().int()  # (N,)
    frustum_intersection_mask = (
        (view.pi + radius_px >= 0)
        & (view.pi - radius_px < w)
        & (view.pj + radius_px >= 0)
        & (view.pj - radius_px < h)
    )
    view.update_mask(frustum_intersection_mask)

    # Calculate view-dependent colors
    camera_pos_world = t  # (3,), translation is the camera position in world space
    unit_view_dirs = view.mu_3d - camera_pos_world
    unit_view_dirs = unit_view_dirs / unit_view_dirs.norm(dim=-1, keepdim=True)

    view.viewdep_colors = evaluate_sh(
        sh_coefficients=view.sh_coefficients,
        view_dirs=unit_view_dirs,
        max_degree=scene.max_sh_degree,
    )

    # Calculate the inverse 2-D covariance analytically
    a = view.cov_2d[:, 0, 0] + S_ADD_TO_COV2D_DIAG
    b = view.cov_2d[:, 0, 1]
    # c = view.cov_2d[:, 1, 0] # not needed - symmetric off-diagonal
    d = view.cov_2d[:, 1, 1] + S_ADD_TO_COV2D_DIAG
    det = ((a * d) - (b**2)).clamp(min=1e-12)
    inv_cov_selected_3 = torch.stack(
        [
            d / det,
            -b / det,
            a / det,
        ],
        dim=-1,
    )

    """
    Now, we prepare the tile rasterizer.
    1. Expand the list of Gaussians so that it is sum^{n_gaussians} t_i,
    where t is the number of tiles whose extent intersects the confidence interval of the i-th Gaussian
    2. Prepare 64-bit sort keys. Keys are tile_idx << 32 + projected depth
    3. argsort list of gaussians by keys
    4. find tile boundaries in list using high key bits
    5. gather and pack buffers using keys
    then we can launch kernels
    """

    """
    Steps 1-2. Goal:
        - Let M be the number of Gaussian-tile overlap pairs
        - Then produce two parallel arrays of length M:
            1. `pairs_gaussian_idx`
            2. `pairs_tile_id`
            where entry `k` means the gaussian pairs_gaussian_idx[k]
            overlaps with tile pairs_tile_id[k]
    """

    n_tiles_x = (w + TILE_SIZE - 1) // TILE_SIZE
    n_tiles_y = (h + TILE_SIZE - 1) // TILE_SIZE
    n_tiles = n_tiles_x * n_tiles_y

    # (N,) arrays giving the indices of the overlapping tiles. Make sure to clamp to prevent assignment to nonexistent titles
    tx_min = ((view.pi - radius_px) / TILE_SIZE).floor().long().clamp(0, n_tiles_x - 1)
    ty_min = ((view.pj - radius_px) / TILE_SIZE).floor().long().clamp(0, n_tiles_y - 1)
    tx_max = ((view.pi + radius_px) / TILE_SIZE).floor().long().clamp(0, n_tiles_x - 1)
    ty_max = ((view.pj + radius_px) / TILE_SIZE).floor().long().clamp(0, n_tiles_y - 1)

    n_tiles_per_gaussian = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)  # N,
    # Repeat the idx of each gaussian by the number of tiles it covers
    pairs_gaussian_idx = torch.repeat_interleave(
        torch.arange(view.N, device=device), n_tiles_per_gaussian
    )  # M,

    """
    Now, for every Gaussian, we know:
        1. The boundaries of the tiles it overlaps with
        2. The number of tiles it overlaps with
    Goal: enumerate all (tx, ty) pairs within that box

    Avoid a naive implementation that would loop over the 2-D range and append to pairs_tile_id
    """
    tx_min_repeated = tx_min.repeat_interleave(n_tiles_per_gaussian)
    ty_min_repeated = ty_min.repeat_interleave(n_tiles_per_gaussian)

    # gaussians are contiguous, so we can calculate the number of tiles they intersect with by taking the min and max delta in ecah direction
    tx_counts = tx_max - tx_min + 1  # N,

    tx_counts_repeated = tx_counts.repeat_interleave(n_tiles_per_gaussian)  # M,

    # Local idx is a running index of length M which resets to 0 at the beginning of each Gaussian's block
    base_idx = torch.repeat_interleave(
        # Stores the index of where each Gaussian starts
        torch.cat(
            [torch.tensor([0], device=device), n_tiles_per_gaussian[:-1].cumsum(0)]
        ),
        # And repeat it for every Gaussian
        n_tiles_per_gaussian,
    )
    local_idx = torch.arange(pairs_gaussian_idx.shape[0], device=device) - base_idx

    # Get 2-D index of tiles
    local_tx, local_ty = (
        local_idx % tx_counts_repeated,
        local_idx // tx_counts_repeated,
    )  # M, and M,

    # Convert to global space, since tx_min_repeated and ty_min_repeated are in global coords
    pairs_tx = tx_min_repeated + local_tx  # M,
    pairs_ty = ty_min_repeated + local_ty  # M,
    # Flatten into row-major format
    pairs_tile_id = pairs_ty * n_tiles_x + pairs_tx  # (M,)

    """
    Step 2: Generate sort keys
    """
    depth = view.mu_camera_3d[:, 2]  # (N,)
    depth_pairs = depth[pairs_gaussian_idx]  # M,
    depth_bits = depth_pairs.view(torch.int32).to(torch.int64)
    # Bit-shift tile_id into high bits of a 64-bit integer
    sort_keys = (pairs_tile_id.long() << 32) | depth_bits  # (M,)

    """
    3. Sort Gaussians by keys.

    Torch should automatically use a Radix sort here
    """
    # reorder pairs by sort keys
    order = torch.argsort(sort_keys)
    sorted_tile_ids = pairs_tile_id[order]
    sorted_gaussian_idx = pairs_gaussian_idx[order]

    """
    At this point, the entries in (sorted_tile_ids, sorted_gaussian_idx)
    are sorted by tile and depth

    Our goal is to produce a 1D array of length n_tiles containing, for each tile, the offset into the M-length arrays where that tile's gaussians begin
    """
    tile_gaussian_start = torch.full(
        (n_tiles + 1,), len(order), device=device, dtype=torch.long
    )  # fill with arbitrary value that results in error when read

    # find boundaries in sorted_tile_ids
    changes = torch.cat(
        [
            torch.tensor([True], device=device),
            sorted_tile_ids[1:] != sorted_tile_ids[:-1],
        ]
    )  # (M, ) bool
    change_positions = torch.where(changes)[
        0
    ]  # indices in changes where the value is true
    change_tile_ids = sorted_tile_ids[
        change_positions
    ]  # get the tile ID that starts at each of these positions
    tile_gaussian_start[change_tile_ids] = (
        change_positions  # store the change positions for all tiles with gaussians associated with them. leave a placeholder value everywhere else.
    )
    # an empty tile should have the same start as the next non-empty tile (so count = 0)
    for i in range(n_tiles - 1, -1, -1):
        if tile_gaussian_start[i] == len(order):
            tile_gaussian_start[i] = tile_gaussian_start[i + 1]

    """
    5. Launch kernel
    """
    color_output = torch.zeros(h, w, 3, device=device)

    mu_screen = torch.stack([view.pi, view.pj], dim=-1)  # (N, 2) pixel-space
    rasterizer_kernel[(n_tiles, TILE_SIZE * TILE_SIZE)](
        mu_screen[sorted_gaussian_idx].contiguous(),  # M, 2
        inv_cov_selected_3[sorted_gaussian_idx].contiguous(),  # M, 3
        view.alpha[sorted_gaussian_idx].contiguous(),  # M,
        view.viewdep_colors[sorted_gaussian_idx].contiguous(),  # M, 3
        tile_gaussian_start,
        n_tiles_x,
        w,
        h,
        color_output,
        TILE_SIZE=TILE_SIZE,
        EPSILON=EPSILON,
        ALPHA_CLAMP_MAX=ALPHA_CLAMP_MAX,
    )
    save_image(color_output.permute(2, 0, 1), save_to_file_path)


def render_all_camera_views(
    scene: GaussianScene,
    all_camera_extrinsics: torch.Tensor,  # one for each view
    camera_intrinsics_K: torch.Tensor,  # same for all views
    img_dims: tuple[int, int],
    save_to_files,
    device=torch.device("cpu"),
):
    assert len(all_camera_extrinsics) == len(save_to_files), (
        "Number of poses must patch number of file names"
    )

    assert camera_intrinsics_K.shape == (3, 3)

    h, w = img_dims
    fx, fy, cx, cy, xmin, xmax, ymin, ymax = camera_intrinsics_to_info(
        camera_intrinsics_K, w, h
    )

    # Cache the current 3D-covariance, which requires matmuls
    cov_3d = scene.calc_covariance_3d()
    N = scene.calc_n_gaussians()

    for i, P in enumerate(all_camera_extrinsics):
        rasterize_from_camera_view(
            scene=scene,
            P=P,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            w=w,
            h=h,
            cov_3d=cov_3d,
            device=device,
            save_to_file_path=save_to_files[i],
        )

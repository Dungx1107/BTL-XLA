import cv2
import numpy as np
import color_merge_cpp
import radix_sort_custom
import os


# =========================
# VARIANCE
# =========================
def _variance(points):
    p = np.asarray(points, dtype=np.float32)
    mu = np.mean(p, axis=0)
    diff = p - mu
    return np.mean(np.sum(diff * diff, axis=1))


# =========================
# OCTREE SPLIT WITH VAR STOP
# =========================
def _split(points, depth=0, var_threshold=200.0, max_depth=6, min_size=8):
    n = len(points)

    if n <= min_size or depth >= max_depth:
        return [np.mean(points, axis=0)]

    if _variance(points) < var_threshold:
        return [np.mean(points, axis=0)]

    bit = 7 - depth

    octants = (
        (((points[:, 0] >> bit) & 1) << 2) |
        (((points[:, 1] >> bit) & 1) << 1) |
        ((points[:, 2] >> bit) & 1)
    )

    result = []
    for oid in range(8):
        mask = octants == oid
        if np.any(mask):
            result.extend(
                _split(points[mask], depth + 1, var_threshold, max_depth, min_size)
            )

    return result


# =========================
# PALETTE EXTRACT
# =========================
def get_palette(patch):
    pixels = patch.reshape(-1, 3).astype(np.uint8)

    palette = _split(
        pixels,
        depth=0,
        var_threshold=200,
        max_depth=6,
        min_size=8
    )

    palette = np.asarray(palette, dtype=np.uint8)

    palette = color_merge_cpp.merge(palette, 50.0)

    return palette


# =========================
# VISUALIZATION PIPELINE
# =========================
def build_debug_image(img_path, tile_size=128, gap=10, out_path="debug_output.png"):
    import numpy as np
    import cv2
    import math

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    cells = []

    # =========================
    # CREATE 4 CELLS
    # =========================
    for _ in range(4):

        i = np.random.randint(0, max(1, h - tile_size))
        j = np.random.randint(0, max(1, w - tile_size))

        patch = img[i:i+tile_size, j:j+tile_size]
        palette = get_palette(patch)

        ph, pw_img = patch.shape[:2]

        # =========================
        # palette -> square blocks
        # =========================
        box = 12

        if len(palette) == 0:
            palette_vis = np.ones((ph, pw_img, 3), dtype=np.uint8) * 255
        else:
            k = len(palette)

            grid_w = int(math.ceil(math.sqrt(k)))
            grid_h = int(math.ceil(k / grid_w))

            grid = np.ones((grid_h * box, grid_w * box, 3), dtype=np.uint8) * 255

            for idx, color in enumerate(palette):
                y = idx // grid_w
                x = idx % grid_w

                grid[y*box:(y+1)*box, x*box:(x+1)*box] = color

            palette_vis = cv2.resize(
                grid,
                (pw_img, ph),
                interpolation=cv2.INTER_NEAREST
            )

        # =========================
        # FORCE SEPARATION (PATCH | GAP | PALETTE)
        # =========================
        white_gap = np.ones((ph, gap, 3), dtype=np.uint8) * 255

        cell = np.hstack([patch, white_gap, palette_vis])

        cells.append(cell)

    # =========================
    # 2x2 GRID
    # =========================
    row1 = np.hstack([cells[0], cells[1]])
    row2 = np.hstack([cells[2], cells[3]])

    # =========================
    # FIX WIDTH (NO VSTACK ERROR EVER)
    # =========================
    max_w = max(row1.shape[1], row2.shape[1])

    def pad_row(r):
        h, w = r.shape[:2]
        if w < max_w:
            pad = np.ones((h, max_w - w, 3), dtype=np.uint8) * 255
            r = np.hstack([r, pad])
        return r

    row1 = pad_row(row1)
    row2 = pad_row(row2)

    out = np.vstack([row1, row2])

    # =========================
    # 🔥 VERY THICK BLACK GRID LINES (x5 strength)
    # =========================
    def draw_grid_lines(img):
        img = img.copy()
        h, w = img.shape[:2]

        color = (0, 0, 0)

        # x5 thickness (very bold)
        thickness = 20

        x_mid = w // 2
        y_mid = h // 2

        img[:, x_mid - thickness//2 : x_mid + thickness//2] = color
        img[y_mid - thickness//2 : y_mid + thickness//2, :] = color

        return img

    out = draw_grid_lines(out)

    cv2.imwrite(out_path, out)
    print(f"[OK] saved -> {out_path}")


# =========================
# RUN
# =========================
build_debug_image("images/03.png")
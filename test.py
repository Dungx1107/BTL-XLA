import numpy as np
import radix_sort_custom
import color_merge_cpp


import numpy as np

def origin_color_sampling(
    image,
    n_samples=20,
    sample_stride=10,
    rs_init=0.5,
    rs_decay=0.98,
    rs_min=0.001
):
    """
    Origin Poisson-Disk style color sampling (baseline)

    Args:
        image: uint8 HxWx3 or float [0,1]
        n_samples: số màu cần lấy
    Returns:
        (K,3) float32 in [0,1]
    """

    if image.dtype != np.uint8:
        img = (image * 255).astype(np.uint8)
    else:
        img = image

    pixels = img.reshape(-1, 3)

    # giảm noise + tăng tốc giống origin code
    unique_colors = np.unique(pixels[::sample_stride], axis=0)

    if len(unique_colors) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    samples = []
    rs = rs_init

    rng = np.random.default_rng()

    while len(samples) < n_samples:

        color = unique_colors[
            rng.integers(len(unique_colors))
        ]

        if (
            not samples
            or np.all(
                np.linalg.norm(
                    np.asarray(samples) - color,
                    axis=1
                ) > 2 * rs
            )
        ):
            samples.append(color)

        rs *= rs_decay

        if rs < rs_min:
            break

    # nếu thiếu thì fill random
    while len(samples) < n_samples:
        samples.append(
            unique_colors[
                rng.integers(len(unique_colors))
            ]
        )

    return np.asarray(samples, dtype=np.float32) / 255.0


def _split(points, dim=0):
    points = radix_sort_custom.sort_by_dim(
        np.asarray(points, dtype=np.uint8),
        dim
    )

    n = len(points)
    q1 = n // 3
    q2 = 2 * n // 3

    if dim == 2:
        return [
            np.mean(points[:q1], axis=0),
            np.mean(points[q1:q2], axis=0),
            np.mean(points[q2:], axis=0),
        ]

    return (
        _split(points[:q1], dim + 1)
        + _split(points[q1:q2], dim + 1)
        + _split(points[q2:], dim + 1)
    )


def select_colors(
    image,
    color_dis_threshold=10.0
):
    """
    Args:
        image: uint8 HxWx3

    Returns:
        palette: uint8 (K,3)
    """

    pixels = image.reshape(-1, 3)

    palette = _split(pixels)

    palette = np.asarray(
        np.round(palette),
        dtype=np.uint8
    )

    palette = color_merge_cpp.merge(
        palette,
        color_dis_threshold
    )

    return np.asarray(
        palette,
        dtype=np.uint8
    )



import numpy as np
import cv2


# -----------------------------
# YOUR FUNCTIONS (keep as-is)
# -----------------------------
# origin_color_sampling(...)
# _split(...)
# select_colors(...)


# -----------------------------
# DRAW PALETTE GRID
# -----------------------------
def draw_palette(colors, patch=25, cols=8):
    if len(colors) == 0:
        return np.zeros((patch, patch, 3), dtype=np.uint8)

    rows = (len(colors) + cols - 1) // cols

    canvas = np.ones((rows * patch, cols * patch, 3), dtype=np.uint8) * 255

    for i, c in enumerate(colors):
        r = i // cols
        col = i % cols

        y0, y1 = r * patch, (r + 1) * patch
        x0, x1 = col * patch, (col + 1) * patch

        canvas[y0:y1, x0:x1] = c

    return canvas


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run(image_path, out_path="compare.png"):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # 1. palettes
    # -------------------------
    origin = origin_color_sampling(img, n_samples=20)
    newver = select_colors(img, color_dis_threshold=30.0)

    origin = (origin * 255).astype(np.uint8)
    newver = newver.astype(np.uint8)

    # -------------------------
    # 2. draw palettes
    # -------------------------
    origin_grid = draw_palette(origin)
    new_grid = draw_palette(newver)

    # -------------------------
    # 3. layout right panel
    # -------------------------
    text_h = 30

    def add_title(img, text):
        h, w = img.shape[:2]
        canvas = np.ones((h + text_h, w, 3), dtype=np.uint8) * 255
        canvas[text_h:, :, :] = img

        cv2.putText(
            canvas,
            text,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        return canvas

    origin_block = add_title(origin_grid, "origin")
    new_block = add_title(new_grid, "new version")

    right = np.vstack([origin_block, new_block])

    # -------------------------
    # 4. resize left image
    # -------------------------
    h_right = right.shape[0]
    h_img, w_img = img.shape[:2]

    scale = h_right / h_img
    new_w = int(w_img * scale)

    left = cv2.resize(img, (new_w, h_right))

    # -------------------------
    # 5. combine
    # -------------------------
    out = np.hstack([left, right])

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out)

    print("Saved:", out_path)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run("images/03.png", "compare.png")
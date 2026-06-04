import cv2
import numpy as np

from filters.constant_time_bilateral_2 import ConstantTimeBilateral
from filters.constant_time_bilateral_4 import BoxSplitVer


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def put_label(img, text):
    """Add label on top of image"""
    out = img.copy()

    # add white padding on top
    h, w = out.shape[:2]
    canvas = np.ones((h + 40, w, 3), dtype=np.uint8) * 255

    canvas[40:, :] = out

    cv2.putText(
        canvas,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return canvas


def make_3col_grid(img1, img2, img3):
    """
    Stack 3 images horizontally with same height
    """

    h = min(img1.shape[0], img2.shape[0], img3.shape[0])

    def resize(img):
        new_w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (new_w, h))

    img1 = resize(img1)
    img2 = resize(img2)
    img3 = resize(img3)

    # add labels
    img1 = put_label(img1, "Original")
    img2 = put_label(img2, "Origin Filter")
    img3 = put_label(img3, "New Filter")

    # recompute height after label
    h = img1.shape[0]

    w = img1.shape[1] + img2.shape[1] + img3.shape[1]

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    x = 0
    for img in [img1, img2, img3]:
        canvas[:, x:x + img.shape[1]] = img
        x += img.shape[1]

    return canvas


def run(image_path, output_path="compare.png"):

    img = load_image(image_path)

    # Original
    original = img

    # Origin filter
    origin_filter = ConstantTimeBilateral(
        n_samples=20,
        sigma_s=35,
        sigma_r=0.08
    )
    origin_result = origin_filter.apply(img)

    # New filter
    new_filter = BoxSplitVer(
        color_dis_threshold=30.0,
        sigma_s=35,
        sigma_r=0.08
    )
    new_result = new_filter.apply(img)

    # Build output
    out = make_3col_grid(original, origin_result, new_result)

    cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run("images/03.png")
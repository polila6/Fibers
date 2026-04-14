import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, box
from shapely.strtree import STRtree
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = Path("images")


def get_possible_overlaps(tree, ellipse_shapes, ellipse_shape):
    """Return candidate overlapping geometries across Shapely versions."""

    query_result = tree.query(ellipse_shape)
    if len(query_result) == 0:
        return []

    first_result = query_result[0]
    if isinstance(first_result, (int, np.integer)):
        return [ellipse_shapes[index] for index in query_result]

    return query_result


def generate_fiber_image(
    b_ini=20.5,
    b_end=200,
    width=510,
    height=310,
    a=20,
    phi_range=(0, 90),
    p=0.30,
    max_attempts=20000,
    output_dir=DEFAULT_OUTPUT_DIR,
    show_image=False,
):
    """Generate one synthetic fiber image and return its orientation tensor.

    Parameters
    ----------
    b_ini : float
        Minimum major-axis length sampled for each ellipse.
    b_end : float
        Maximum major-axis length sampled for each ellipse.
    width : int
        Image width in arbitrary units.
    height : int
        Image height in arbitrary units.
    a : float
        Nominal minor-axis length. The actual value is sampled within a
        tolerance of +/- 0.5.
    phi_range : tuple[float, float]
        Minimum and maximum ellipse orientation angles in degrees.
    p : float
        Target fraction of the image area to be filled by ellipses.
    max_attempts : int
        Maximum number of ellipse placement attempts before stopping.
    output_dir : str or Path
        Directory where the generated image is saved.
    show_image : bool
        If True, show the generated image interactively before closing it.
    """

    fig, ax = plt.subplots()

    # Use a black background so the white fibers are clearly visible.
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Match the axes to the requested image dimensions and hide decorations.
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0)

    image_bounds = box(0, 0, width, height)
    target_area = width * height * p
    current_area = 0
    ellipses = []
    ellipse_shapes = []

    attempts = 0
    ellipse_count = 0
    sum_sin2_beta = 0
    sum_cos2_beta_sin2_phi = 0
    sum_cos2_beta_cos2_phi = 0

    max_b = b_end

    while current_area < target_area and attempts < max_attempts:
        # Sample ellipse dimensions within the user-defined ranges.
        a_random = random.uniform(a - 0.5, a + 0.5)
        b_random = random.uniform(b_ini, b_end)

        # Keep the sampled values physically meaningful.
        a_random = max(0.1, min(a_random, width * 2))
        b_random = max(a_random, min(b_random, height * 2))

        # Allow ellipses to extend beyond the image so cropped fibers can appear.
        x = random.uniform(-max_b, width + max_b)
        y = random.uniform(-max_b, height + max_b)
        phi = random.uniform(*phi_range)

        ellipse_shape = Point(0, 0).buffer(1, resolution=128)
        ellipse_shape = scale(ellipse_shape, a_random / 2, b_random / 2)
        ellipse_shape = rotate(ellipse_shape, phi, use_radians=False)
        ellipse_shape = translate(ellipse_shape, x, y)

        overlap = False
        if ellipse_shapes:
            tree = STRtree(ellipse_shapes)
            possible_overlaps = get_possible_overlaps(
                tree, ellipse_shapes, ellipse_shape
            )

            for other in possible_overlaps:
                if ellipse_shape.intersects(other):
                    overlap = True
                    break

        if not overlap:
            intersection_area = ellipse_shape.intersection(image_bounds).area
            if intersection_area > 0:
                current_area += intersection_area
                ellipse_count += 1

                sin_beta_i = min(1.0, a_random / b_random)
                beta_i = np.arcsin(sin_beta_i)

                sin2_beta_i = np.sin(beta_i) ** 2
                cos2_beta_i = np.cos(beta_i) ** 2

                phi_i_rad = np.deg2rad(phi)
                sin2_phi_i = np.sin(phi_i_rad) ** 2
                cos2_phi_i = np.cos(phi_i_rad) ** 2

                # Accumulate the orientation tensor terms.
                sum_sin2_beta += sin2_beta_i
                sum_cos2_beta_sin2_phi += cos2_beta_i * sin2_phi_i
                sum_cos2_beta_cos2_phi += cos2_beta_i * cos2_phi_i

            ellipses.append({"x": x, "y": y, "a": a_random, "b": b_random, "phi": phi})
            ellipse_shapes.append(ellipse_shape)

        attempts += 1

    if ellipse_count > 0:
        a_xx = sum_sin2_beta / ellipse_count
        a_yy = sum_cos2_beta_sin2_phi / ellipse_count
        a_zz = sum_cos2_beta_cos2_phi / ellipse_count
    else:
        a_xx = a_yy = a_zz = 0

    for ellipse_data in ellipses:
        ellipse = Ellipse(
            (ellipse_data["x"], ellipse_data["y"]),
            width=ellipse_data["a"],
            height=ellipse_data["b"],
            angle=ellipse_data["phi"],
            edgecolor="white",
            facecolor="white",
            linewidth=0,
        )
        ax.add_patch(ellipse)

    plt.gca().invert_yaxis()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{a_xx:.4f},{a_yy:.4f},{a_zz:.4f}.jpg"

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black",
    )

    if show_image:
        plt.show()

    plt.close(fig)
    return a_xx, a_yy, a_zz


def parse_args():
    """Parse command-line arguments for batch image generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic fiber images and save the orientation tensor "
            "components in the output filename."
        )
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--b-ini",
        type=float,
        default=20.5,
        help="Minimum major-axis length sampled for each ellipse.",
    )
    parser.add_argument(
        "--b-end",
        type=float,
        default=200,
        help="Maximum major-axis length sampled for each ellipse.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=510,
        help="Image width in arbitrary units.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=310,
        help="Image height in arbitrary units.",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=20,
        help="Nominal minor-axis length sampled within a tolerance of +/- 0.5.",
    )
    parser.add_argument(
        "--phi-min",
        type=float,
        default=0,
        help="Minimum ellipse orientation angle in degrees.",
    )
    parser.add_argument(
        "--phi-max",
        type=float,
        default=90,
        help="Maximum ellipse orientation angle in degrees.",
    )
    parser.add_argument(
        "--target-area-fraction",
        type=float,
        default=0.30,
        help="Fraction of the image area that should be covered by ellipses.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20000,
        help="Maximum number of ellipse placement attempts per image.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated images are saved.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Display each generated image before closing it.",
    )
    return parser.parse_args()


def validate_args(args):
    """Validate the user-provided command-line arguments."""

    if args.num_images < 1:
        raise ValueError("--num-images must be at least 1.")
    if args.b_ini <= 0 or args.b_end <= 0:
        raise ValueError("--b-ini and --b-end must be positive.")
    if args.b_ini > args.b_end:
        raise ValueError("--b-ini cannot be greater than --b-end.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.a <= 0:
        raise ValueError("--a must be positive.")
    if args.phi_min > args.phi_max:
        raise ValueError("--phi-min cannot be greater than --phi-max.")
    if not 0 < args.target_area_fraction <= 1:
        raise ValueError("--target-area-fraction must be in the range (0, 1].")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")


def main():
    """Run batch image generation using the command-line configuration."""

    args = parse_args()
    validate_args(args)

    start_time = time.time()

    for _ in tqdm(range(args.num_images), desc="Generating images"):
        generate_fiber_image(
            b_ini=args.b_ini,
            b_end=args.b_end,
            width=args.width,
            height=args.height,
            a=args.a,
            phi_range=(args.phi_min, args.phi_max),
            p=args.target_area_fraction,
            max_attempts=args.max_attempts,
            output_dir=args.output_dir,
            show_image=args.show_image,
        )

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"{args.num_images} image(s) generated in {elapsed_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()

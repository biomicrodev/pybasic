import argparse
from pathlib import Path

from bmd_perf.profiling import timed_ctx
from tifffile import tifffile

from src.pybasic.compute import compute
from src.pybasic.utils import _query_yes_no, _flatten_paths


def parse_args() -> argparse.Namespace:
    filename = Path(__file__).name

    parser = argparse.ArgumentParser(
        description=f"{filename} - Generate background and shading correction images for microscopy images"
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help="paths to images and/or folders containing images",
    )
    parser.add_argument(
        "--and-darkfield",
        dest="compute_darkfield",
        action="store_true",
        help="compute darkfield (in addition to flatfield)",
    )
    parser.add_argument(
        "--iter-axes",
        metavar="N",
        type=int,
        nargs="+",
        help="if multichannel images, must specify axes to iterate over (axes other than YX)",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="shorthand for --iter-axes but in the special case of RGB images",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=str,
        default=".",
        help="output folder; current folder if not specified",
    )
    # parser.add_argument(
    #     "--log",
    #     dest="log",
    #     metavar="PATH",
    #     type=str,
    #     help="log file for debugging; saved to output location",
    # )
    parser.add_argument(
        "--flatfield-reg",
        metavar="VALUE",
        dest="flatfield_reg",
        type=float,
        help="flatfield regularization parameter",
    )
    parser.add_argument(
        "--darkfield-reg",
        metavar="VALUE",
        dest="darkfield_reg",
        type=float,
        help="darkfield regularization parameter",
    )
    parser.add_argument(
        "--working-size",
        metavar="VALUE",
        type=int,
        default=128,
        help="resize both image axes to this",
    )
    parser.add_argument(
        "--max-images",
        metavar="VALUE",
        type=int,
        default=500,
        help="maximum number of images to load",
    )
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="log messages",
    )
    return parser.parse_args()


def _check_output_files(out: Path, compute_darkfield: bool):
    flat_exists = (out / "flatfield.tiff").exists()
    dark_exists = compute_darkfield and (out / "darkfield.tiff").exists()

    msg = None
    if flat_exists and dark_exists:
        msg = "Flatfield and darkfield images already exist. Overwrite?"
    elif flat_exists and not dark_exists:
        msg = "Flatfield image already exists. Overwrite?"

    if msg is None:
        return
    overwrite = _query_yes_no(msg, default="no")
    if overwrite:
        return
    else:
        raise RuntimeError("File already exists; quitting")


def run():
    args = parse_args()

    # check if destination files already exist before doing anything
    dst = Path(args.out)
    try:
        _check_output_files(dst, args.compute_darkfield)
    except RuntimeError:
        return

    # normalize paths; sort, then take first max_images images, for reproducibility
    # purposes
    paths = [Path(p) for p in args.images]
    paths = list(set(paths))  # make unique
    paths = sorted(list(_flatten_paths(paths, depth=1)))
    paths = paths[: args.max_images]

    flatfield, darkfield = compute(
        paths,
        iter_axes=args.iter_axes,
        rgb=args.rgb,
        working_size=args.working_size,
        flatfield_reg=args.flatfield_reg,
        darkfield_reg=args.darkfield_reg,
        compute_darkfield=args.compute_darkfield,
        verbose=args.verbose,
    )

    dst.mkdir(exist_ok=True, parents=True)

    with timed_ctx("Written", verbose=args.verbose):
        tifffile.imwrite(dst / "flatfield.tiff", flatfield, compression="zlib")
        tifffile.imwrite(dst / "darkfield.tiff", darkfield, compression="zlib")


if __name__ == "__main__":
    run()
"""Input validation and reproducible random state helpers."""

import hashlib
import math
import random
from contextlib import contextmanager
from pathlib import Path


def get_img_list(root):
    root = Path(root).expanduser().resolve()
    extensions = {".png", ".jpg", ".jpeg"}
    if root.is_file() and root.suffix.lower() in extensions:
        return [root]
    if root.is_dir():
        # Preserve the source's PNG, JPG, JPEG grouping and ordering.
        return [
            p
            for ext in (".png", ".jpg", ".jpeg")
            for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() == ext
        ]
    return []


def process_text(prompt=None, prompt_file=None):
    if prompt is not None:
        return [prompt]
    prompts = Path(prompt_file).read_text(encoding="utf-8").splitlines()
    # Preserve the source DIV2K/DAPE parser, including its prefix on plain lines.
    return [
        "a high quality photo of " + text if text else "a high quality photo"
        for line in prompts
        for text in [line.split(": ")[-1].strip()]
    ]


def validate_inputs(args):
    """Validate before loading Torch, model weights or creating output directories."""
    for key in ("img_path", "db_img_dir", "db_dir", "workdir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.prompt_file is not None:
        args.prompt_file = args.prompt_file.expanduser().resolve()
    for key in ("deg_scale", "measurement_noise", "cfg_scale", "step_size", "step_size_inv"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{key} must be finite and nonnegative.")
    if args.img_size <= 0 or args.img_size % 16:
        raise ValueError("--img_size must be a positive multiple of 16 for SD3.")
    if args.NFE <= 0 or args.NFE_inv <= 0 or args.db_size <= 0:
        raise ValueError("--NFE, --NFE_inv and --db_size must be positive.")
    if args.num_samples == 0 or args.num_samples < -1:
        raise ValueError("--num_samples must be -1 (all) or a positive integer.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32).")
    if args.task == "sr_avgpool":
        if (
            args.deg_scale < 2
            or not args.deg_scale.is_integer()
            or args.img_size % int(args.deg_scale)
        ):
            raise ValueError(
                "For sr_avgpool, --deg_scale must be an integer >=2 dividing --img_size."
            )
    elif args.task == "deblur_motion":
        if not args.deg_scale.is_integer() or args.deg_scale < 3 or int(args.deg_scale) % 2 != 1:
            raise ValueError("For deblur_motion, --deg_scale must be an odd integer >=3.")
        if args.deg_scale // 2 >= args.img_size:
            raise ValueError("The motion kernel radius must be smaller than --img_size.")
    elif args.task == "cs_blockbased":
        if args.img_size % 32 or not (1 / 1024 <= args.deg_scale <= 1):
            raise ValueError(
                "For cs_blockbased, size must be divisible by 32 and ratio in [1/1024, 1]."
            )
    if args.measurement_noise is None:
        args.measurement_noise = args.deg_scale if args.task == "denoising" else 0.05
    images = get_img_list(args.img_path)
    if not images:
        raise ValueError(
            "--img_path must be an image or a directory containing PNG/JPG/JPEG files."
        )
    images = images[: args.num_samples] if args.num_samples > 0 else images
    db_images = get_img_list(args.db_img_dir)
    if not Path(args.db_img_dir).is_dir() or not db_images:
        raise ValueError("--db_img_dir must be a nonempty image directory.")
    prompts = process_text(args.prompt, args.prompt_file)
    if not prompts or (args.prompt_file is not None and len(prompts) < len(images)):
        raise ValueError("The prompt file must have at least one line per selected input image.")
    if len(prompts) > len(images) and args.prompt_file is not None:
        prompts = prompts[: len(images)]
    if args.prompt is not None:
        prompts = prompts * len(images)
    output = args.workdir / f"{args.task}_{args.deg_scale}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose another --workdir.")
    return images, prompts, db_images, output


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def isolated_seed(seed):
    """Cache misses and cache hits must leave inference random state identical."""
    import numpy as np
    import torch

    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            set_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

"""Run measurement-DINO retrieval, RA-DGI inversion and FlowDPS reconstruction."""

import argparse
import json
from pathlib import Path

from util import validate_inputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--img_path", type=Path, required=True, help="Image file or input image directory"
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="Literal prompt shared by all inputs")
    prompts.add_argument(
        "--prompt_file", type=Path, help="One source-format prompt per input image"
    )
    parser.add_argument("--db_img_dir", type=Path, required=True, help="Retrieval image directory")
    parser.add_argument("--db_dir", type=Path, default=Path("db"), help="Generated retrieval cache")
    parser.add_argument("--workdir", type=Path, default=Path("results"))
    parser.add_argument(
        "--task",
        choices=["sr_avgpool", "deblur_motion", "denoising", "cs_blockbased"],
        default="sr_avgpool",
    )
    parser.add_argument(
        "--deg_scale",
        type=float,
        default=None,
        help="SR factor, blur kernel size, denoising std or CS measurement ratio",
    )
    parser.add_argument(
        "--measurement_noise",
        type=float,
        default=None,
        help="Defaults to deg_scale for denoising, otherwise 0.05",
    )
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--db_size", type=int, default=1000)
    parser.add_argument("--NFE", type=int, default=10, help="Reconstruction steps")
    parser.add_argument("--NFE_inv", type=int, default=10, help="RA-DGI inversion steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--step_size", type=float, default=15.0)
    parser.add_argument("--step_size_inv", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--efficient_memory",
        action="store_true",
        help="Precompute text embeddings and release text encoders",
    )
    return parser


def run(args, images, prompts, db_images, output):
    # Keep --help and argument validation usable without the model dependencies.
    import torch
    from functions.degradation import get_degradation
    from retrieval import DINO_MODEL, load_image, make_db, search_db
    from sd3_sampler import RADGISolver
    from torchvision import transforms
    from torchvision.utils import save_image
    from tqdm import tqdm
    from transformers import pipeline
    from util import isolated_seed, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RA-DGI inference requires a CUDA GPU. Install CUDA-enabled PyTorch and check torch.cuda.is_available()."
        )
    device = "cuda"
    set_seed(args.seed)
    solver = RADGISolver(device=device)
    for encoder in (solver.text_enc_1, solver.text_enc_2, solver.text_enc_3):
        encoder.to(device)
    with torch.no_grad():
        null_emb = solver.encode_prompt([""])
        embeddings = None
        if args.efficient_memory:
            embeddings = {}
            for prompt in tqdm(dict.fromkeys(prompts), desc="Encoding prompts"):
                embeddings[prompt] = [x.cpu() for x in solver.encode_prompt([prompt])]
    if args.efficient_memory:
        # Delete the loop variable as well so the last encoder is actually freed.
        del encoder, solver.text_enc_1, solver.text_enc_2, solver.text_enc_3
        torch.cuda.empty_cache()
    solver.vae.to(device)
    solver.transformer.to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
        ]
    )
    # Dedicated seed makes the sampled operator independent of model loading and prompts.
    with isolated_seed(args.seed):
        operator = get_degradation(args.task, args.img_size, args.deg_scale, device)
    dino = pipeline(
        task="image-feature-extraction", model=DINO_MODEL, dtype=torch.bfloat16, device=-1
    )
    index, paths = make_db(args, db_images, transform, operator, device, dino)
    for folder in ("input", "recon", "label", "retrieval"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config["images"] = [
        {"input": str(path), "prompt": prompt} for path, prompt in zip(images, prompts)
    ]
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    retrieved = []
    for i, (path, prompt) in enumerate(tqdm(list(zip(images, prompts)), desc="Solving")):
        # Cache hits and earlier images cannot change this input index's random stream.
        with isolated_seed((args.seed + i) % 2**32):
            image = load_image(path, transform, device)
            measurement = operator.A(image)
            measurement = measurement + args.measurement_noise * torch.randn_like(measurement)
            reference, reference_path = search_db(
                index,
                paths,
                measurement,
                operator,
                args.task,
                args.img_size,
                dino,
                transform,
                device,
            )
            with torch.no_grad():
                prompt_emb = (
                    embeddings[prompt] if embeddings is not None else solver.encode_prompt([prompt])
                )
            result = solver.sample(
                measurement,
                operator,
                args.task,
                reference,
                prompt_emb,
                null_emb,
                NFE=args.NFE,
                NFE_inv=args.NFE_inv,
                cfg_scale=args.cfg_scale,
                step_size=args.step_size,
                step_size_inv=args.step_size_inv,
            )
        # Preserve the original save_image(normalize=True) output convention.
        outputs = {
            "input": operator.At(measurement).reshape(image.shape),
            "recon": result,
            "label": image,
            "retrieval": reference,
        }
        for folder, tensor in outputs.items():
            save_image(tensor, output / folder / f"{i:04d}.png", normalize=True)
        retrieved.append(
            {"output": f"{i:04d}.png", "input": str(path), "retrieval": str(reference_path)}
        )
        (output / "retrieval.json").write_text(json.dumps(retrieved, indent=2), encoding="utf-8")
    print(f"Results saved to {output.resolve()}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.deg_scale is None:
        args.deg_scale = {
            "sr_avgpool": 16.0,
            "deblur_motion": 81.0,
            "denoising": 0.7,
            "cs_blockbased": 0.3,
        }[args.task]
    try:
        validated = validate_inputs(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    run(args, *validated)


if __name__ == "__main__":
    main()

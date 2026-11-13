"""Measurement-space DINO features and exact FAISS L2 retrieval."""

import json
import random
from importlib.metadata import version

import faiss
import numpy as np
import torch
from functions.degradation import operator_fingerprint
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from util import file_hash, isolated_seed

DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_image(path, transform, device):
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0).to(device) * 2 - 1


def measurement_image(measurement, operator, task, img_size):
    lift = operator.A_pinv if task == "sr_avgpool" else operator.At
    return lift(measurement).reshape(1, 3, img_size, img_size)


@torch.no_grad()
def extract_feature(measurement, operator, task, img_size, dino):
    image = measurement_image(measurement, operator, task, img_size)
    # Same mapping and uint8 conversion as the original; deliberately no clamping.
    image = ToPILImage()(((image.squeeze(0) + 1) / 2).cpu())
    feature = np.asarray(dino(image), dtype=np.float32).reshape(1, -1)
    if not np.isfinite(feature).all():
        raise ValueError("DINO produced a non-finite feature.")
    return np.ascontiguousarray(feature)


def make_db(args, db_images, transform, operator, device, dino):
    root = args.db_img_dir.resolve()
    # Record content hashes for all candidates: changes also affect subset selection.
    candidates = [
        {"path": p.relative_to(root).as_posix(), "sha256": file_hash(p)} for p in db_images
    ]
    selected = random.Random(args.seed).sample(candidates, min(args.db_size, len(candidates)))
    manifest = {
        "format": 2,
        "feature": "measurement_dino_flatten_l2",
        "model": DINO_MODEL,
        "feature_device": str(dino.device),
        "feature_dtype": str(dino.model.dtype),
        "transformers": version("transformers"),
        "torchvision": version("torchvision"),
        "img_size": args.img_size,
        "task": args.task,
        "deg_scale": args.deg_scale,
        "measurement_noise": args.measurement_noise,
        "seed": args.seed,
        "db_size": args.db_size,
        "operator_sha256": operator_fingerprint(operator),
        "candidates": candidates,
        "selected": selected,
    }
    # Features depend on the degradation applied to the reference images (paper Sec. 3.2).
    # Other feature-generation settings are checked in metadata, not directory names.
    cache = args.db_dir / f"{args.task}_{args.deg_scale:g}_noise_{args.measurement_noise:g}"
    index_path, meta_path = cache / "index.faiss", cache / "meta.json"
    if index_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["manifest"] == manifest and file_hash(index_path) == meta["index_sha256"]:
                index = faiss.read_index(str(index_path))
                if index.ntotal == len(selected):
                    print(f"Loaded retrieval cache: {cache}")
                    return index, [root / item["path"] for item in selected]
        except (ValueError, KeyError, RuntimeError, OSError):
            pass
    print(f"Building retrieval cache: {cache}")
    features = []
    with isolated_seed(args.seed):
        for item in tqdm(selected, desc="Building retrieval DB"):
            image = load_image(root / item["path"], transform, device)
            y = operator.A(image)
            y = y + args.measurement_noise * torch.randn_like(y)
            features.append(extract_feature(y, operator, args.task, args.img_size, dino))
    features = np.concatenate(features, axis=0)
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    cache.mkdir(parents=True, exist_ok=True)
    # Metadata is the commit marker. An interrupted write will rebuild next time.
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps({"manifest": manifest, "index_sha256": file_hash(index_path)}, indent=2),
        encoding="utf-8",
    )
    return index, [root / item["path"] for item in selected]


def search_db(index, paths, measurement, operator, task, img_size, dino, transform, device):
    feature = extract_feature(measurement, operator, task, img_size, dino)
    if feature.shape[1] != index.d:
        raise ValueError("DINO feature dimension differs from the cached database.")
    _, neighbors = index.search(feature, k=1)
    nearest = int(neighbors[0, 0])
    if not 0 <= nearest < len(paths):
        raise ValueError("No valid retrieval result; rebuild the DB cache.")
    return load_image(paths[nearest], transform, device), paths[nearest]

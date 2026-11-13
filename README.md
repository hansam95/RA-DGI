# [WACV 2027] Retrieval-Augmented Initialization with Dual-Guided Inversion for Flow-Based Inverse Problem Solvers

[![arXiv](https://img.shields.io/badge/arXiv-2610.41695-red)](https://arxiv.org/abs/2610.41695)

> **Retrieval-Augmented Initialization with Dual-Guided Inversion for Flow-Based Inverse Problem Solvers**<br>
> Hansam Cho, Seoung Bum Kim<br>
>
> **Abstract**: <br>
Flow models have recently emerged as powerful generative priors for solving inverse problems. Despite their success, most existing flow-based inverse problem solvers initialize the reconstruction process with random Gaussian noise. However, recent studies in image generation have shown that the choice of initial noise can significantly influence generation outcomes. The role of initialization as a distinct design choice in flow-based inverse problem solving remains comparatively underexplored. In this work, we systematically evaluate noise initialization strategies while holding the downstream flow-based inverse solver fixed. We propose Retrieval-Augmented Initialization with Dual-Guided Inversion (RA-DGI), a method that constructs improved initial noise by leveraging retrieved proxy clean images and dual-guided inversion. Extensive experiments across multiple inverse problems demonstrate that the quality of the initial noise substantially affects reconstruction fidelity and perceptual performance. RA-DGI outperforms standard random initialization and achieves superior results compared to inference-time scaling strategies, while requiring less computational overhead. These findings highlight initialization as a critical yet previously overlooked factor in flow-based inverse solvers.

## Description

Official implementation of **Retrieval-Augmented Initialization with Dual-Guided Inversion for Flow-Based Inverse Problem Solvers (WACV 2027)**.

This repository is based on the official [FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS) code. RA-DGI retrieves a proxy clean image using measurement-space DINOv3 features, constructs initial noise through dual-guided inversion, and reconstructs the image with FlowDPS and Stable Diffusion 3 Medium.

![RA-DGI reconstructions and inference-time scaling comparison](images/teaser.jpg)

## Setup

Use Python 3.10–3.12 and a CUDA GPU.

```bash
git clone https://github.com/hansam95/RA-DGI.git
cd RA-DGI
conda create -n ra-dgi python=3.10 -y
conda activate ra-dgi
# CUDA 12.8 example; choose the PyTorch index appropriate for your system.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Request access to [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) and [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), accept their terms, and authenticate with your Hugging Face account:

```bash
huggingface-cli login
```

You may also set `HF_TOKEN`. Weights download on first use. DINOv3 runs on CUDA in bfloat16, and SD3 runs on CUDA in float16. `--efficient_memory` precomputes prompt embeddings and releases the text encoders to reduce GPU memory usage.

## RA-DGI + Inverse Problems

The following examples use the included cat image and three retrieval images:

```text
samples/
  input/flickr_cat_000008.jpg
  retrieval/
    0.jpg
    1.jpg
    10.jpg
```

These small samples are for checking execution. For experiments, replace the image paths with your own dataset and retrieval pool. The default DB size is 1,000, capped at the number of available retrieval images; these examples therefore use three images.

### Super-resolution

16x average-pool downsampling with measurement noise of 0.05.

```bash
python solve.py --img_path samples/input --prompt "a photo of a close-up face of a cat" \
  --db_img_dir samples/retrieval --task sr_avgpool --deg_scale 16 \
  --measurement_noise 0.05 --efficient_memory --workdir results/sr
```

### Motion deblurring

Motion blur with an 81x81 kernel and measurement noise of 0.05.

```bash
python solve.py --img_path samples/input --prompt "a photo of a close-up face of a cat" \
  --db_img_dir samples/retrieval --task deblur_motion --deg_scale 81 \
  --measurement_noise 0.05 --efficient_memory --workdir results/deblur
```

### Denoising

Additive Gaussian noise with a standard deviation of 0.7.

```bash
python solve.py --img_path samples/input --prompt "a photo of a close-up face of a cat" \
  --db_img_dir samples/retrieval --task denoising --deg_scale 0.7 \
  --efficient_memory --workdir results/denoise
```

### Block compressive sensing

32x32 blocks with a measurement ratio of 0.3 and measurement noise of 0.05. Set `--deg_scale 0.1` for the measurement ratio reported in Section 4.1 of the paper.

```bash
python solve.py --img_path samples/input --prompt "a photo of a close-up face of a cat" \
  --db_img_dir samples/retrieval --task cs_blockbased --deg_scale 0.3 \
  --measurement_noise 0.05 --efficient_memory --workdir results/cs
```

All examples use 10 reconstruction steps, 10 inversion steps, image size 768, and CFG 2. They use Bash line continuations; in PowerShell, put each command on one line or replace `\` with the PowerShell backtick. Choose a new `--workdir` when repeating a run: nonempty output directories are rejected.

<details>
<summary>Custom images and prompts</summary>

`--img_path` accepts an image or a directory, and `--db_img_dir` accepts a directory of clean retrieval images. Paths may be absolute. Both directories support PNG, JPG, and JPEG, including uppercase extensions, without recursion. Images are read as sorted PNG files, then sorted JPG files, then sorted JPEG files. They are resized on the shorter edge and center-cropped to `--img_size`.

Use `--prompt` for a literal prompt shared by all inputs, or `--prompt_file` for one line per input. These arguments are mutually exclusive. The prompt-file parser follows the original DAPE/DIV2K convention: take the text after the last `: `, strip whitespace, and prepend `a high quality photo of `. For example, `0000: a cat` becomes `a high quality photo of a cat`. An empty line becomes `a high quality photo`. An already complete sentence also receives the prefix, so use `--prompt` to preserve literal text.

There must be at least one prompt-file line per selected input. Trailing lines are ignored when processing a subset with `--num_samples`.

</details>

<details>
<summary>Arguments</summary>

| Argument | Default | Meaning |
| --- | --- | --- |
| `--img_size` | 768 | Square image size; multiple of 16, or 32 for block CS |
| `--NFE` | 10 | Reconstruction steps |
| `--NFE_inv` | 10 | Inversion steps |
| `--cfg_scale` | 2.0 | Reconstruction CFG; inversion CFG is 1 |
| `--step_size` / `--step_size_inv` | 15.0 / 15.0 | Measurement guidance step sizes; three inner updates |
| `--db_size` | 1000 | Seeded random subset, capped at available DB images |
| `--db_dir` | `db` | Retrieval cache directory |
| `--num_samples` | -1 | All inputs; use a positive value for a subset |
| `--seed` | 0 | Random seed |
| `--deg_scale` | Per task | 16, 81, 0.7, or 0.3 for the tasks above |
| `--measurement_noise` | Per task | `deg_scale` for denoising, otherwise 0.05; measured in the [-1, 1] image domain |

SR factors must be integers >=2 that divide the image size. Motion kernel sizes must be odd integers >=3 with radius smaller than the image size. Block CS uses 32x32 blocks and a measurement ratio in [1/1024, 1].

</details>

<details>
<summary>Outputs and retrieval DB</summary>

```text
results/sr/sr_avgpool_16.0/
  input/0000.png
  recon/0000.png
  label/0000.png
  retrieval/0000.png
  config.json
  retrieval.json
```

`input` is the backprojection of the noisy measurement, `label` is the preprocessed clean image, and `retrieval` is the selected proxy image. PNGs use the original `save_image(normalize=True)` convention. `config.json` records inputs and prompts; `retrieval.json` records the selected reference.

Each degradation has its own DB, using flattened DINOv3 features and exact FAISS L2 search:

```text
db/
  sr_avgpool_16_noise_0.05/
  deblur_motion_81_noise_0.05/
  denoising_0.7_noise_0.7/
  cs_blockbased_0.3_noise_0.05/
```

Each folder contains `index.faiss` and `meta.json`. Matching feature-generation settings reuse the existing DB without extracting DB features again or rewriting the files. Input images, prompts, and reconstruction steps do not invalidate the DB.

Image size, DB image hashes, seed, model and library settings, and the actual sampled blur kernel or CS basis are checked in metadata. Mismatched, incomplete, or corrupt caches are rebuilt in the same directory. Use another `--db_dir` to keep multiple DB variants for the same degradation. Caches from earlier nested layouts must be rebuilt in this layout.

DB construction preserves inference random state, each input has its own random stream, and the inversion scheduler is reset per image. These changes remove cache and previous-image dependencies; historical experiment outputs may differ even though the original update equations are retained.

</details>

The four tasks and DB reuse were checked on an NVIDIA RTX 5090 with PyTorch 2.7.0+cu128, one input image, and three retrieval images. This execution check does not reproduce full-dataset paper metrics.

## Acknowledgements

This repository is built upon the official [FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS) implementation. We also thank [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [DINOv3](https://github.com/facebookresearch/dinov3), [FAISS](https://github.com/facebookresearch/faiss), and [motionblur](https://github.com/LeviBorodenko/motionblur) for their open-source contributions. See [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution and licensing notes.

## Citation

```bibtex
@inproceedings{cho2027radgi,
  title={Retrieval-Augmented Initialization with Dual-Guided Inversion for Flow-Based Inverse Problem Solvers},
  author={Cho, Hansam and Kim, Seoung Bum},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```

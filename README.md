# RA-DGI

Implementation of measurement-DINO retrieval, RA-DGI inversion, and FlowDPS reconstruction with Stable Diffusion 3 Medium.

## Setup

Use Python 3.10–3.12 and a CUDA GPU. The original environment used PyTorch 2.7.0, torchvision 0.22.0, Diffusers 0.30.1, and Transformers 4.57.6. This release keeps those model-library versions.

```bash
conda create -n ra-dgi python=3.10 -y
conda activate ra-dgi
# Example for CUDA 12.8; use the PyTorch index appropriate for your system.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Request access to both model repositories and accept their terms using your Hugging Face account:

- [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)

Then authenticate with `huggingface-cli login` (or set `HF_TOKEN`). Weights download on first use. DINO features are extracted on CUDA in bfloat16, matching the original code's GPU selection when CUDA is available. SD3 runs on CUDA in float16. `--efficient_memory` releases the text encoders after precomputing prompt embeddings; no fixed VRAM capacity is claimed.

## Prepare images

Provide your own input and retrieval images. Data, weights and cached features are not included.

```text
data/
  inputs/
    0000.png
    0001.png
  prompts.txt
  retrieval/
    image_0000.jpg
    image_0001.jpg
    ...
```

Both directories support PNG, JPG and JPEG, including uppercase extensions. Files are read directly from the directory, without recursion. The input order follows the original code: sorted PNG files, then sorted JPG files, then sorted JPEG files. Images are resized on the shorter edge and center-cropped to `--img_size`.

Use `--prompt "a photo of a cat"` to share a literal prompt across all inputs, or `--prompt_file data/prompts.txt` for one line per input. These arguments are mutually exclusive.

The prompt-file parser preserves the original DAPE/DIV2K behavior: take the text after the last `: `, strip whitespace, and prepend `a high quality photo of `. An empty line becomes `a high quality photo`. Thus `0000: a cat` becomes `a high quality photo of a cat`; an already complete sentence also receives that prefix. Use `--prompt` when you want the exact text without this conversion. There must be at least one line per selected input; trailing lines are ignored when processing a subset with `--num_samples`.

## Run RA-DGI

Run commands from this repository directory. Paths may also be absolute, so existing datasets can stay in their current location. All examples include retrieval, RA-DGI inversion with DPS measurement guidance and prior guidance, and FlowDPS reconstruction.

### Super-resolution (16x average pooling)

```bash
python solve.py --img_path data/inputs --prompt_file data/prompts.txt \
  --db_img_dir data/retrieval --task sr_avgpool --deg_scale 16 \
  --measurement_noise 0.05 --efficient_memory --workdir results/sr
```

### Motion deblurring (81x81 kernel)

```bash
python solve.py --img_path data/inputs --prompt_file data/prompts.txt \
  --db_img_dir data/retrieval --task deblur_motion --deg_scale 81 \
  --measurement_noise 0.05 --efficient_memory --workdir results/deblur
```

### Denoising (Gaussian noise standard deviation 0.7)

```bash
python solve.py --img_path data/inputs --prompt_file data/prompts.txt \
  --db_img_dir data/retrieval --task denoising --deg_scale 0.7 \
  --efficient_memory --workdir results/denoise
```

### Block compressive sensing (measurement ratio 0.3)

```bash
python solve.py --img_path data/inputs --prompt_file data/prompts.txt \
  --db_img_dir data/retrieval --task cs_blockbased --deg_scale 0.3 \
  --measurement_noise 0.05 --efficient_memory --workdir results/cs
```

The examples use Bash line continuations. In PowerShell, put the command on a single line or replace `\` with the PowerShell backtick.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--img_size` | 768 | Square image size; multiple of 16, or 32 for block CS |
| `--NFE` | 10 | Reconstruction steps |
| `--NFE_inv` | 10 | Inversion steps |
| `--cfg_scale` | 2.0 | Reconstruction CFG; inversion CFG remains 1 |
| `--step_size` / `--step_size_inv` | 15.0 / 15.0 | Measurement guidance step sizes; three inner updates |
| `--db_size` | 1000 | Seeded random subset, capped at the available DB images |
| `--db_dir` | `db` | Retrieval cache directory |
| `--num_samples` | -1 | All inputs; use a positive number to process a subset |
| `--seed` | 0 | Random seed |
| `--deg_scale` | Per task | 16, 81, 0.7, or 0.3 for the four tasks above |
| `--measurement_noise` | Per task | `deg_scale` for denoising, otherwise 0.05; measured in the [-1, 1] image domain |

SR factors must be integers >=2 that divide the image size. Motion kernel sizes must be odd integers >=3 with radius smaller than the image size. Block CS uses 32x32 blocks and a measurement ratio in [1/1024, 1].

## Outputs and retrieval cache

```text
results/sr/sr_avgpool_16.0/
  input/0000.png
  recon/0000.png
  label/0000.png
  retrieval/0000.png
  config.json
  retrieval.json
```

`input` is the original operator's backprojection of the noisy measurement; `label` is the preprocessed clean image. PNGs use the original `save_image(normalize=True)` convention. `config.json` maps inputs to prompts; `retrieval.json` records the selected reference for each result. Choose a new `--workdir` for each run: existing nonempty task directories are rejected to avoid mixing results.

The retrieval database uses the original flattened DINO pipeline output, without feature normalization, and exact FAISS L2 nearest-neighbor search. Features are computed from degraded DB images using the same operator as the query. Cache metadata records image content hashes, selected relative paths, task, image size, noise, seed, feature-library versions, DINO device and dtype, and the actual sampled blur kernel or CS basis. Mismatched or incomplete caches are rebuilt under a new settings key. Metadata uses relative paths so the image directory can be relocated.

DB construction preserves the inference random state. Measurements and sampling use a separate seed per input index, and each image resets the inversion scheduler shift before reconstruction sets it to 4. These fixes remove cache-hit and previous-image state dependencies; full-run PNGs are not promised to be bit-identical to historical experiments. The original RA-DGI/FlowDPS update equations, original motion backprojection (identity), DINO conversion and text-tokenization logic are retained.

## Code and acknowledgements

- `solve.py`: input validation, model setup and inference.
- `sd3_sampler.py`: SD3 encoding, RA-DGI inversion and FlowDPS sampling.
- `retrieval.py`: measurement-DINO features, cache and L2 retrieval.
- `functions/`: the four measurement operators and motion kernel.
- `util.py`: input helpers and random-state handling.

Baseline methods, ablations, evaluation scripts, notebooks and training code are excluded. See [THIRD_PARTY.md](THIRD_PARTY.md) for source attribution and licensing notes. Paper metadata and citation details are intentionally left for the authors to supply.

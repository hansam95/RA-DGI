# Source attribution

This release was extracted from the author's local RA-DGI research code. SD3 prompt encoding, latent scaling, dual-guidance inversion, FlowDPS sampling and the SVD measurement operators retain that implementation's calculations.

## Base implementation

RA-DGI is based on the official [FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS) code for *FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems* (ICCV 2025), by Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye. RA-DGI adds measurement-space DINOv3 retrieval and dual-guided inversion to initialize the FlowDPS reconstruction process. This attribution does not assign a new license to retained upstream code.

## Paper figure and sample images

`images/teaser.jpg` is rendered from the author's supplied RA-DGI teaser PDF. `samples/input/flickr_cat_000008.jpg` is copied from the AFHQ images in the author's research data; `samples/retrieval/0.jpg`, `1.jpg`, and `10.jpg` are copied from the author's retrieval image pool. These four images are included for execution examples. Image rights remain with their respective owners; no additional image license is assigned here.

## Motion kernel

`functions/motionblur.py` retains the kernel-generation portion of the local `motionblur/motionblur.py`; display helpers, image application and the standalone demo are removed. The duplicate local `utils/motionblur.py` identifies the source as [LeviBorodenko/motionblur](https://github.com/LeviBorodenko/motionblur). That attribution is preserved in the released module.

The inspected upstream repository root and the local RA-DGI tree do not contain a license file for this code. No license grant is inferred or added for it. The author should resolve the redistribution license with its owner before public distribution.

## Measurement operators

`functions/svd_operators.py` retains the base measurement interface, block CS, denoising and average-pool super-resolution implementations from the local source, excluding methods for other solvers. The source contains no copyright header or license notice identifying their upstream origin. No new license is assigned to these retained sections.

The motion convolution and identity backprojection are extracted from the local `functions/measurements.py` and `utils/blur_util.py`. Their random kernel initialization sequence is preserved.

## Libraries and model weights

This implementation uses [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch) and [FAISS](https://github.com/facebookresearch/faiss). They are installed as dependencies, rather than vendored here. Stable Diffusion 3 Medium and DINOv3 weights remain subject to their respective model licenses and access terms.

No repository-wide license is added because the provided RA-DGI source has no such file. The MIT licenses in the author's separate OSASIS and NMG repositories are not applied to RA-DGI by assumption.

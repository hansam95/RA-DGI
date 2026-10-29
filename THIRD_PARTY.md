# Source attribution

This release was extracted from the author's local DGI research code. SD3 prompt encoding, latent scaling, dual-guidance inversion, FlowDPS sampling and the SVD measurement operators retain that implementation's calculations.

## Motion kernel

`functions/motionblur.py` retains the kernel-generation portion of the local `motionblur/motionblur.py`; display helpers, image application and the standalone demo are removed. The duplicate local `utils/motionblur.py` identifies the source as [LeviBorodenko/motionblur](https://github.com/LeviBorodenko/motionblur). That attribution is preserved in the released module.

The inspected upstream repository root and the local DGI tree do not contain a license file for this code. No license grant is inferred or added for it. The author should resolve the redistribution license with its owner before public distribution.

## Measurement operators

`functions/svd_operators.py` retains the base measurement interface, block CS, denoising and average-pool super-resolution implementations from the local source, excluding methods for other solvers. The source contains no copyright header or license notice identifying their upstream origin. No new license is assigned to these retained sections.

The motion convolution and identity backprojection are extracted from the local `functions/measurements.py` and `utils/blur_util.py`. Their random kernel initialization sequence is preserved.

## Libraries and model weights

This implementation uses [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch) and [FAISS](https://github.com/facebookresearch/faiss). They are installed as dependencies, rather than vendored here. Stable Diffusion 3 Medium and DINOv3 weights remain subject to their respective model licenses and access terms.

No repository-wide license is added because the provided DGI source has no such file. The MIT licenses in the author's separate OSASIS and NMG repositories are not applied to DGI by assumption.

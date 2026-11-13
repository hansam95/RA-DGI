# Validation

Release preparation: 2026-10-27, Windows, Python 3.12, CPU only.

## Completed checks

- Syntax compilation of every released Python file and `solve.py --help`.
- Imports of the released solver, retrieval and degradation modules with PyTorch 2.7.0+cpu, torchvision 0.22.0+cpu, NumPy 2.1.2, Diffusers 0.30.1, Transformers 4.57.6 and FAISS 1.13.2.
- Nine offline CPU regression tests: prompt parsing and input ordering; invalid arguments and prompt counts; task constraints and output collision handling; validation before model loading; shapes and input gradients for all four operators; isolated random state; FAISS cache creation/reuse/invalidation/corruption recovery; a real randomly initialized small DINOv3 pipeline in CPU bfloat16; and retrieval → inversion → reconstruction with a differentiable toy solver for all four tasks.
- Direct comparison against the original source loaded into memory: the four operators' forward/backprojection results and SR pseudoinverse were exactly equal. DPS updates, two-step RA-DGI inversion and two-step FlowDPS reconstruction were also exactly equal for each task with fixed seeds, identical scheduler state, and a toy VAE/vector field (`rtol=0`, `atol=0`). These comparisons validate the retained equations, not pretrained-model quality.
- SHA-256 comparison of the original Python, shell and notebook files and `.gitignore` before and after preparation. No source-code changes were made.

To rerun the portable offline tests after installing the dependencies and PyTorch:

```bash
python tests/test_ra_dgi.py -v
```

The direct original-source comparison was a release-preparation check; it is not part of the portable test suite, because the public directory has no dependency on the author's original research directory.

## Remaining validation

This machine has no available CUDA device. A full inference run using pretrained SD3 Medium and pretrained DINOv3, GPU memory consumption, reconstruction quality and paper metrics have not been tested. Run one input with `--num_samples 1 --efficient_memory` on the target CUDA machine before running the full dataset.

Per-input random streams, isolated cache generation and resetting inversion scheduler state intentionally remove order/cache dependencies in the original runner. Historical full experiment images may differ even though the core updates agree under matching inputs and random state.

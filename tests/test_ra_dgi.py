"""CPU smoke/regression tests; no network access or model weights required."""

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import solve
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from functions.degradation import get_degradation, operator_fingerprint
from PIL import Image
from retrieval import extract_feature, load_image, make_db, search_db
from sd3_sampler import RADGISolver
from torchvision import transforms
from util import isolated_seed, process_text, set_seed, validate_inputs


class TinySolver(RADGISolver):
    """Differentiable identity VAE and a deterministic toy vector field."""

    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float16
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        self.inversion_shift = 1.0

    def encode(self, image):
        return image.to(self.dtype).clone()

    def decode(self, latent):
        return latent

    def predict_vector(self, z, t, prompt, pooled):
        return 0.1 * z + 0.01 * prompt.mean() + t.reshape(-1, 1, 1, 1).to(z.dtype) / 10000


class TinyDino:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        # Retain a token-like output shape; retrieval must flatten all dimensions.
        array = np.asarray(image, dtype=np.float32) / 255
        return [[array.mean(axis=(0, 1)).tolist(), array.std(axis=(0, 1)).tolist()]]


class RADGIRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for name, color in [("a.png", (20, 40, 60)), ("b.JPG", (180, 160, 140))]:
            Image.new("RGB", (40, 36), color).save(self.images / name)
        self.prompts = self.root / "prompts.txt"
        self.prompts.write_text("a: cat\nb: dog\n", encoding="utf-8")
        self.args = solve.build_parser().parse_args(
            [
                "--img_path",
                str(self.images),
                "--db_img_dir",
                str(self.images),
                "--prompt_file",
                str(self.prompts),
                "--img_size",
                "32",
                "--deg_scale",
                "2",
                "--db_dir",
                str(self.root / "db"),
                "--workdir",
                str(self.root / "results"),
            ]
        )
        self.transform = transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
        )

    def test_input_order_prompts_and_defaults(self):
        images, prompts, _, _ = validate_inputs(self.args)
        self.assertEqual([p.name for p in images], ["a.png", "b.JPG"])
        self.assertEqual(prompts, ["a high quality photo of cat", "a high quality photo of dog"])
        self.assertEqual((self.args.NFE, self.args.NFE_inv, self.args.step_size), (10, 10, 15))
        self.prompts.write_text("\nplain sentence\n", encoding="utf-8")
        self.assertEqual(
            process_text(prompt_file=self.prompts),
            ["a high quality photo", "a high quality photo of plain sentence"],
        )
        self.assertEqual(process_text(prompt="literal"), ["literal"])

    def test_invalid_inputs(self):
        for key, value in [
            ("NFE", 0),
            ("NFE_inv", -1),
            ("db_size", 0),
            ("num_samples", 0),
            ("img_size", 33),
            ("deg_scale", 3.0),
            ("measurement_noise", -1),
            ("cfg_scale", float("nan")),
            ("seed", -1),
        ]:
            args = SimpleNamespace(**vars(self.args))
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_inputs(args)
        self.prompts.write_text("only one\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_inputs(self.args)
        self.args.num_samples = 1
        self.assertEqual(len(validate_inputs(self.args)[0]), 1)
        empty = self.root / "empty"
        empty.mkdir()
        self.args.db_img_dir = empty
        with self.assertRaises(ValueError):
            validate_inputs(self.args)

    def test_task_constraints_and_output_collision(self):
        for task, scale in [("deblur_motion", 4.0), ("cs_blockbased", 0.0), ("cs_blockbased", 1.1)]:
            args = SimpleNamespace(**vars(self.args))
            args.task, args.deg_scale = task, scale
            with self.subTest(task=task, scale=scale), self.assertRaises(ValueError):
                validate_inputs(args)
        self.args.task, self.args.deg_scale = "denoising", 0.7
        _, _, _, output = validate_inputs(self.args)
        self.assertEqual(self.args.measurement_noise, 0.7)
        output.mkdir(parents=True)
        (output / "existing").touch()
        with self.assertRaises(ValueError):
            validate_inputs(self.args)

    def test_cli_validation_precedes_loading_models(self):
        with patch("solve.run") as run, self.assertRaises(SystemExit):
            solve.main(
                [
                    "--img_path",
                    str(self.root / "missing"),
                    "--prompt",
                    "cat",
                    "--db_img_dir",
                    str(self.images),
                ]
            )
        run.assert_not_called()

    def test_degradation_shapes_gradients_and_fingerprints(self):
        for task, scale in [
            ("sr_avgpool", 2.0),
            ("deblur_motion", 9.0),
            ("denoising", 0.7),
            ("cs_blockbased", 0.3),
        ]:
            with self.subTest(task=task), isolated_seed(9):
                operator = get_degradation(task, 32, scale, "cpu")
                image = torch.randn(1, 3, 32, 32, requires_grad=True)
                y = operator.A(image)
                self.assertEqual(operator.At(y).numel(), image.numel())
                y.square().sum().backward()
                self.assertTrue(torch.isfinite(image.grad).all())
                self.assertGreater(image.grad.abs().sum().item(), 0)
                self.assertEqual(operator_fingerprint(operator), operator_fingerprint(operator))
        with isolated_seed(0):
            first = get_degradation("cs_blockbased", 32, 0.3, "cpu")
        with isolated_seed(1):
            second = get_degradation("cs_blockbased", 32, 0.3, "cpu")
        self.assertNotEqual(operator_fingerprint(first), operator_fingerprint(second))

    def test_isolated_rng(self):
        set_seed(99)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        set_seed(99)
        with isolated_seed(1):
            random.random(), np.random.rand(), torch.rand(100)
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_real_dinov3_pipeline_cpu_bfloat16(self):
        # Exercise the installed model/pipeline APIs without downloading weights.
        from transformers import (
            DINOv3ViTConfig,
            DINOv3ViTImageProcessorFast,
            DINOv3ViTModel,
            pipeline,
        )

        config = DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        model = DINOv3ViTModel(config).to(torch.bfloat16).eval()
        processor = DINOv3ViTImageProcessorFast(size={"height": 32, "width": 32})
        dino = pipeline(
            "image-feature-extraction",
            model=model,
            image_processor=processor,
            dtype=torch.bfloat16,
            device=-1,
        )
        operator = get_degradation("denoising", 32, 0.7, "cpu")
        feature = extract_feature(torch.zeros(1, 3, 32, 32), operator, "denoising", 32, dino)
        self.assertEqual(feature.shape, (1, 5 * 32))
        self.assertEqual(feature.dtype, np.float32)
        self.assertTrue(np.isfinite(feature).all())

    def test_retrieval_build_load_invalidation_and_corruption(self):
        _, _, db_images, _ = validate_inputs(self.args)
        operator = get_degradation("sr_avgpool", 32, 2, "cpu")
        dino = TinyDino()
        set_seed(123)
        before = torch.get_rng_state().clone()
        index, paths = make_db(self.args, db_images, self.transform, operator, "cpu", dino)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(dino.calls, 2)
        loaded, loaded_paths = make_db(self.args, db_images, self.transform, operator, "cpu", dino)
        self.assertEqual(dino.calls, 2)
        self.assertEqual(paths, loaded_paths)
        np.testing.assert_array_equal(index.reconstruct_n(), loaded.reconstruct_n())
        image = load_image(paths[0], self.transform, "cpu")
        reference, path = search_db(
            loaded,
            paths,
            operator.A(image),
            operator,
            "sr_avgpool",
            32,
            dino,
            self.transform,
            "cpu",
        )
        self.assertEqual(path, paths[0])
        torch.testing.assert_close(image, reference)
        self.assertEqual(
            extract_feature(operator.A(image), operator, "sr_avgpool", 32, dino).shape, (1, 6)
        )
        self.args.measurement_noise = 0.2
        make_db(self.args, db_images, self.transform, operator, "cpu", dino)
        self.assertEqual(len(list(self.args.db_dir.iterdir())), 2)
        # Incomplete/corrupt caches rebuild without changing the inference RNG.
        for index_path in self.args.db_dir.glob("*/index.faiss"):
            index_path.write_bytes(b"broken")
        rebuilt, _ = make_db(self.args, db_images, self.transform, operator, "cpu", dino)
        self.assertEqual(rebuilt.ntotal, 2)
        Image.new("RGB", (40, 36), (5, 6, 7)).save(self.images / "a.png")
        make_db(self.args, db_images, self.transform, operator, "cpu", dino)
        self.assertEqual(len(list(self.args.db_dir.iterdir())), 3)


    def test_toy_end_to_end_retrieval_inversion_reconstruction(self):
        _, _, db_images, _ = validate_inputs(self.args)
        for task, scale in [
            ("sr_avgpool", 2),
            ("deblur_motion", 9),
            ("denoising", 0.7),
            ("cs_blockbased", 0.3),
        ]:
            with self.subTest(task=task), isolated_seed(12):
                self.args.task, self.args.deg_scale = task, scale
                operator = get_degradation(task, 32, scale, "cpu")
                dino = TinyDino()
                index, paths = make_db(self.args, db_images, self.transform, operator, "cpu", dino)
                image = load_image(paths[0], self.transform, "cpu")
                measurement = operator.A(image) + 0.01 * torch.randn_like(operator.A(image))
                reference, _ = search_db(
                    index, paths, measurement, operator, task, 32, dino, self.transform, "cpu"
                )
                solver = TinySolver()
                embeddings = [
                    torch.ones(1, 2, dtype=torch.float16),
                    torch.ones(1, 2, dtype=torch.float16),
                ]
                null = [torch.zeros_like(x) for x in embeddings]
                with isolated_seed(42):
                    result = solver.sample(
                        measurement,
                        operator,
                        task,
                        reference,
                        embeddings,
                        null,
                        NFE=2,
                        NFE_inv=2,
                        step_size=0.1,
                        step_size_inv=0.1,
                    )
                with isolated_seed(42):
                    repeated = solver.sample(
                        measurement,
                        operator,
                        task,
                        reference,
                        embeddings,
                        null,
                        NFE=2,
                        NFE_inv=2,
                        step_size=0.1,
                        step_size_inv=0.1,
                    )
                self.assertEqual(tuple(result.shape), (1, 3, 32, 32))
                self.assertTrue(torch.isfinite(result).all())
                torch.testing.assert_close(result, repeated, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

"""SD3, dual-guidance inversion and FlowDPS reconstruction."""

import math

import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm


class StableDiffusion3Base:
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        device="cuda",
        dtype=torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        self.scheduler = pipe.scheduler
        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3
        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        del pipe
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.inversion_shift = self.scheduler.config.shift

    def encode_prompt(self, prompt: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        """
        text_clip1_ids = self.tokenizer_1(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        text_clip1_emb = self.text_enc_1(
            text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True
        )
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=self.text_enc_1.device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(
            dtype=self.dtype, device=self.text_enc_1.device
        )
        text_clip2_ids = self.tokenizer_2(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        text_clip2_emb = self.text_enc_2(
            text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True
        )
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=self.text_enc_2.device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(
            dtype=self.dtype, device=self.text_enc_2.device
        )
        text_t5_ids = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=self.text_enc_3.device)
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)
        return (prompt_emb, pooled_prompt_emb)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(
            hidden_states=z,
            timestep=t,
            pooled_projections=pooled_emb,
            encoder_hidden_states=prompt_emb,
            return_dict=False,
        )[0]
        return v


class RADGISolver(StableDiffusion3Base):
    """The proposed method only; inference uses one image at a time."""

    def data_consistency(self, z0t, operator, measurement, task, stepsize=15.0):
        z0t = z0t.detach().requires_grad_(True)
        for _ in range(3):
            x0t = self.decode(z0t).float()
            lift = operator.A_pinv if task == "sr_avgpool" else operator.At
            loss = torch.linalg.norm((lift(measurement) - lift(operator.A(x0t))).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0].to(z0t.dtype)
            z0t = z0t - stepsize * grad
        return z0t.detach()

    def inversion(
        self,
        retrieved_img,
        operator,
        measurement,
        task,
        prompt_emb,
        null_emb,
        NFE_inv=10,
        step_size_inv=15.0,
    ):
        # The original RA-DGI path uses inversion CFG=1, independently of sampling CFG.
        prompt, pooled = prompt_emb
        with torch.no_grad():
            z = self.encode(retrieved_img.to(device=self.device, dtype=self.dtype))

        # Reset per image: reconstruction's shift=4 must not leak into inversion.
        self.scheduler.config.shift = self.inversion_shift
        self.scheduler.set_timesteps(NFE_inv, device=self.device)
        timesteps = torch.cat([self.scheduler.timesteps, torch.zeros(1, device=self.device)]).flip(
            0
        )
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        for i, t in enumerate(tqdm(timesteps[:-1], desc="RA-DGI inversion")):
            with torch.no_grad():
                pred_v = self.predict_vector(z, t.expand(z.shape[0]), prompt, pooled)
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            z0t = z - sigma * pred_v
            z1t = z + (1 - sigma) * pred_v
            z0y = self.data_consistency(z0t, operator, measurement, task, step_size_inv)
            z0t = (1 - sigma) * z0t + sigma * z0y
            z1t = math.sqrt(sigma) * z1t + math.sqrt(1 - sigma) * torch.randn_like(z1t)
            z = z0t + sigma_next * (z1t - z0t)
        return z

    def sample(
        self,
        measurement,
        operator,
        task,
        retrieved_img,
        prompt_emb,
        null_emb,
        NFE=10,
        NFE_inv=10,
        cfg_scale=2.0,
        step_size=15.0,
        step_size_inv=15.0,
    ):
        prompt, pooled = [x.to(self.device) for x in prompt_emb]
        null_prompt, null_pooled = [x.to(self.device) for x in null_emb]
        z = self.inversion(
            retrieved_img,
            operator,
            measurement,
            task,
            [prompt, pooled],
            [null_prompt, null_pooled],
            NFE_inv,
            step_size_inv,
        )
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        for i, t in enumerate(tqdm(timesteps, desc="FlowDPS reconstruction")):
            timestep = t.expand(z.shape[0])
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt, pooled)
                pred_null = (
                    self.predict_vector(z, timestep, null_prompt, null_pooled)
                    if cfg_scale != 1
                    else 0.0
                )
            vector = pred_null + cfg_scale * (pred_v - pred_null)
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else 0.0
            z0t = z - sigma * vector
            z1t = z + (1 - sigma) * vector
            delta = sigma - sigma_next
            z0y = self.data_consistency(z0t, operator, measurement, task, step_size)
            z0y = (1 - sigma) * z0t + sigma * z0y
            noise = math.sqrt(sigma_next) * z1t + math.sqrt(1 - sigma_next) * torch.randn_like(z1t)
            z = z0y + (sigma - delta) * (noise - z0y)
        with torch.no_grad():
            return self.decode(z)

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import io
from typing import List
import base64
from PIL import Image
import torch
from cog import BasePredictor, Input, Path

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

WEIGHTS_DIR = "model_weights"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.device = "cuda"
        self.xm = load_model("transmitter", cache_dir=WEIGHTS_DIR, device=self.device)
        self.text_model = load_model(
            "text300M", cache_dir=WEIGHTS_DIR, device=self.device
        )
        self.image_model = load_model(
            "image300M", cache_dir=WEIGHTS_DIR, device=self.device
        )
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for generating the 3D model, ignored if an image is provide below",
            default=None,
        ),
        image: Path = Input(
            description="An synthetic view image for generating the 3D modeld",
            default=None,
        ),
        guidance_scale: float = Input(
            description="Set the scale for guidanece", default=15.0
        ),
        batch_size: int = Input(description="Number of output", default=1),
        render_mode: str = Input(
            description="Choose a render mode", choices=["nerf", "stf"], default="nerf"
        ),
        render_size: int = Input(
            description="Set the size of the a renderer, higher values take longer to render",
            default=128,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        assert prompt or image, "Please provide prompt of image"
        model = self.image_model if image else self.text_model

        if image:
            model_kwargs = dict(images=[load_image(str(image))] * batch_size)
        else:
            model_kwargs = dict(texts=[prompt] * batch_size)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        cameras = create_pan_cameras(render_size, self.device)
        output = []
        for i, latent in enumerate(latents):
            images = decode_latent_images(
                self.xm, latent, cameras, rendering_mode=render_mode
            )
            writer = io.BytesIO()
            images[0].save(
                writer,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
            writer.seek(0)
            data = base64.b64encode(writer.read()).decode("ascii")

            filename = "/tmp/out_{i}.gif"
            with open(filename, "wb") as f:
                f.write(writer.getbuffer())
            output.append(Path(filename))
        return output

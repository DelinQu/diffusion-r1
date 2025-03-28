# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available
from grpo_config import GRPOConfig
from grpo_sd_trainer import GRPOTrainer
from modeling_sd_base import DefaultDDPOStableDiffusionPipeline
from datetime import datetime
import logging
from pathlib import Path
import torchvision.transforms.functional as TF

@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion"`):
            HuggingFace repo to save model weights to.
        hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
            Hugging Face model ID for aesthetic scorer model weights.
        hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
            Hugging Face model filename for aesthetic scorer model weights.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether to use LoRA.
    """
    pretrained_model: str = field(
        default="../pretrained/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="../pretrained/ddpo-aesthetic-predictor",
        metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("../pretrained/clip-vit-large-patch14", local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained("../pretrained/clip-vit-large-patch14", local_files_only=True)
        self.mlp = MLP()
        cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if is_torch_npu_available():
        scorer = scorer.npu()
    elif is_torch_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra",
    "spider",
    "bird",
    "sheep",
    "deer",
    "cow",
    "goat",
    "lion",
    "frog",
    "chicken",
    "duck",
    "goose",
    "bee",
    "pig",
    "turkey",
    "fly",
    "llama",
    "camel",
    "bat",
    "gorilla",
    "hedgehog",
    "kangaroo",
]


def prompt_fn():
    return np.random.choice(animals), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    if global_step % 25 != 0: return
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        # result[f"{prompt:.25} | {reward:.2f}"] = F.interpolate(image, size=256).squeeze(0).float()
        result[f"{prompt:.25} | {reward:.2f}"] = TF.resize(image.unsqueeze(0).float(), (256, 256))

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging_dir = f"{training_args.logdir}/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}_{training_args.run_name}"
    Path(logging_dir).mkdir(parents=True, exist_ok=True)

    training_args.project_kwargs = {
        "logging_dir": logging_dir,
        "automatic_checkpoint_naming": True,
        "total_limit": training_args.num_checkpoint_limit,
        "project_dir": logging_dir,
    }

    if training_args.log_with is not None:
        training_args.tracker_kwargs = {training_args.log_with: {"name": training_args.run_name, "dir": logging_dir}}

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = GRPOTrainer(
        training_args,
        aesthetic_scorer(script_args.hf_hub_aesthetic_model_id, script_args.hf_hub_aesthetic_model_filename),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

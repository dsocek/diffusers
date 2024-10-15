#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState

from diffusers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler
)


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="CompVis/stable-diffusion-v1-4",
        type=str,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        default="lllyasviel/sd-controlnet-canny",
        type=str,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--scheduler",
        default="ddim",
        choices=["default", "euler_discrete", "euler_ancestral_discrete", "ddim", "flow_match_euler_discrete"],
        type=str,
        help="Name of scheduler",
    )
    parser.add_argument(
        "--timestep_spacing",
        default="linspace",
        choices=["linspace", "leading", "trailing"],
        type=str,
        help="The way the timesteps should be scaled.",
    )
    # Pipeline arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="An image of a squirrel in Picasso style",
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--prompts_2",
        type=str,
        nargs="*",
        default=None,
        help="The second prompt or prompts to guide the image generation (applicable to SDXL and SD3).",
    )
    parser.add_argument(
        "--prompts_3",
        type=str,
        nargs="*",
        default=None,
        help="The third prompt or prompts to guide the image generation (applicable to SD3).",
    )
    parser.add_argument(
        "--base_image",
        type=str,
        default=None,
        help=("Path to inpaint base image"),
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        default=None,
        help=("Path to inpaint mask image"),
    )
    parser.add_argument(
        "--control_image",
        type=str,
        default=None,
        help=("Path to the controlnet conditioning image"),
    )
    parser.add_argument(
        "--control_preprocessing_type",
        type=str,
        default="canny",
        help=(
            "The type of preprocessing to apply on contol image. Only `canny` is supported."
            " Defaults to `canny`. Set to unsupported value to disable preprocessing."
        ),
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The number of images in a batch.")
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="The height in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="The width in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=(
            "Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
            " Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,"
            " usually at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        nargs="*",
        default=None,
        help="The prompt or prompts not to guide the image generation.",
    )
    parser.add_argument(
        "--negative_prompts_2",
        type=str,
        nargs="*",
        default=None,
        help="The second prompt or prompts not to guide the image generation (applicable to SDXL and SD3).",
    )
    parser.add_argument(
        "--negative_prompts_3",
        type=str,
        nargs="*",
        default=None,
        help="The third prompt or prompts not to guide the image generation (applicable to SD3).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Corresponds to parameter eta (  ) in the DDIM paper: https://arxiv.org/abs/2010.02502.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )

    parser.add_argument(
        "--pipeline_save_dir",
        type=str,
        default=None,
        help="The directory where the generation pipeline will be saved.",
    )
    parser.add_argument(
        "--image_save_dir",
        type=str,
        default="./stable-diffusion-generated-images",
        help="The directory where images will be saved.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    parser.add_argument("--bf16", action="store_true", help="Whether to perform generation in bf16 precision.")
    parser.add_argument(
        "--ldm3d", action="store_true", help="Use LDM3D to generate an image and a depth map from a given text prompt."
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=None,
        help="Number of steps to ignore for throughput calculation.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        type=int,
        default=0,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        type=int,
        default=0,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument("--distributed", action="store_true", help="Use distributed inference on multi-cards")
    parser.add_argument(
        "--unet_adapter_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--text_encoder_adapter_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--lora_id",
        default=None,
        type=str,
        help="Path to lora id",
    )
    parser.add_argument(
        "--use_cpu_rng",
        action="store_true",
        help="Enable deterministic generation using CPU Generator",
    )
    parser.add_argument(
        "--quant_mode",
        default="disable",
        type=str,
        help="Quantization mode 'measure', 'quantize', 'quantize-mixed' or 'disable'",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="The file with prompts (for large number of images generation).",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use eager torch.compile() model",
    )
    args = parser.parse_args()

    # Select stable diffuson pipeline based on input
    sdxl_models = ["stable-diffusion-xl", "sdxl"]
    sd3_models = ["stable-diffusion-3"]
    flux_models = ["FLUX.1-dev", "FLUX.1-schnell"]
    sdxl = True if any(model in args.model_name_or_path for model in sdxl_models) else False
    sd3 = True if any(model in args.model_name_or_path for model in sd3_models) else False
    flux = True if any(model in args.model_name_or_path for model in flux_models) else False
    controlnet = True if args.control_image is not None else False
    inpainting = True if (args.base_image is not None) and (args.mask_image is not None) else False

    # Set the scheduler
    kwargs = {"timestep_spacing": args.timestep_spacing}

    if args.scheduler == "euler_discrete":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "euler_ancestral_discrete":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler", **kwargs)
    elif args.scheduler == "flow_match_euler_discrete":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    else:
        scheduler = None

    # Set pipeline class instantiation options
    kwargs = {}

    if scheduler is not None:
        kwargs["scheduler"] = scheduler

    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    # Set pipeline call options
    kwargs_call = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "output_type": args.output_type,
        "profiling_warmup_steps": args.profiling_warmup_steps,
        "profiling_steps": args.profiling_steps,
    }

    if args.width > 0 and args.height > 0:
        kwargs_call["width"] = args.width
        kwargs_call["height"] = args.height

    if args.use_cpu_rng:
        kwargs_call["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)
    else:
        kwargs_call["generator"] = None

    if args.throughput_warmup_steps is not None:
        kwargs_call["throughput_warmup_steps"] = args.throughput_warmup_steps

    negative_prompts = args.negative_prompts
    if args.distributed:
        distributed_state = PartialState()
        if args.negative_prompts is not None:
            with distributed_state.split_between_processes(args.negative_prompts) as negative_prompt:
                negative_prompts = negative_prompt
    kwargs_call["negative_prompt"] = negative_prompts

    if sdxl or sd3 or flux:
        prompts_2 = args.prompts_2
        if args.distributed and args.prompts_2 is not None:
            with distributed_state.split_between_processes(args.prompts_2) as prompt_2:
                prompts_2 = prompt_2
        kwargs_call["prompt_2"] = prompts_2

    if sdxl or sd3:
        negative_prompts_2 = args.negative_prompts_2
        if args.distributed and args.negative_prompts_2 is not None:
            with distributed_state.split_between_processes(args.negative_prompts_2) as negative_prompt_2:
                negative_prompts_2 = negative_prompt_2
        kwargs_call["negative_prompt_2"] = negative_prompts_2

    if sd3:
        prompts_3 = args.prompts_3
        negative_prompts_3 = args.negative_prompts_3
        if args.distributed and args.prompts_3 is not None:
            with distributed_state.split_between_processes(args.prompts_3) as prompt_3:
                prompts_3 = prompt_3
        if args.distributed and args.negative_prompts_3 is not None:
            with distributed_state.split_between_processes(args.negative_prompts_3) as negative_prompt_3:
                negative_prompts_3 = negative_prompt_3
        kwargs_call["prompt_3"] = prompts_3
        kwargs_call["negative_prompt_3"] = negative_prompts_3

    if inpainting:
        from diffusers.utils import load_image

        init_image = load_image(args.base_image)
        mask_image = load_image(args.mask_image)
        kwargs_call["image"] = init_image
        kwargs_call["mask_image"] = mask_image

    if controlnet:
        from diffusers.utils import load_image
        from PIL import Image

        control_image = load_image(args.control_image)
        if args.control_preprocessing_type == "canny":
            # Generate Canny image for ControlNet
            import cv2

            image = np.array(control_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            control_image = Image.fromarray(image)
        kwargs_call["image"] = control_image

    kwargs_call["quant_mode"] = args.quant_mode

    # Instantiate a Stable Diffusion pipeline class
    if sdxl:
        # SDXL pipelines
        if controlnet:
            # Import SDXL+ControlNet pipeline
            raise ValueError("SDXL+ControlNet pipeline is not currenly supported")

        elif inpainting:
            # Import SDXL Inpainting pipeline
            from diffusers import AutoPipelineForInpainting

            pipeline = AutoPipelineForInpainting.from_pretrained(args.model_name_or_path, **kwargs)

        else:
            # Import SDXL pipeline
            from diffusers import StableDiffusionXLPipeline

            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )
            if args.lora_id:
                pipeline.load_lora_weights(args.lora_id)

    elif sd3:
        # SD3 pipelines
        if controlnet:
            # Import SD3+ControlNet pipeline
            raise ValueError("SD3+ControlNet pipeline is not currenly supported")
        elif inpainting:
            # Import SD3 Inpainting pipeline
            raise ValueError("SD3 Inpainting pipeline is not currenly supported")
        else:
            # Import SD3 pipeline
            from diffusers import StableDiffusion3Pipeline

            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )
    elif flux:
        # Flux pipelines
        if controlnet:
            # Import Flux+ControlNet pipeline
            raise ValueError("Flux+ControlNet pipeline is not currenly supported")
        elif inpainting:
            # Import FLux Inpainting pipeline
            raise ValueError("Flux Inpainting pipeline is not currenly supported")
        else:
            # Import Flux pipeline
            #from diffusers import FluxPipeline
            from diffusers import FluxPipeline

            pipeline = FluxPipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )

    else:
        # SD pipelines (SD1.x, SD2.x)
        if controlnet:
            # SD+ControlNet pipeline
            from diffusers import ControlNetModel

            from diffusers import StableDiffusionControlNetPipeline

            model_dtype = torch.bfloat16 if args.bf16 else None
            controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=model_dtype)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                args.model_name_or_path,
                controlnet=controlnet,
                **kwargs,
            )
            if args.lora_id:
                pipeline.load_lora_weights(args.lora_id)

        elif inpainting:
            # SD Inpainting pipeline
            from diffusers import AutoPipelineForInpainting

            pipeline = AutoPipelineForInpainting.from_pretrained(args.model_name_or_path, **kwargs)

        else:
            # SD pipeline
            if not args.ldm3d:
                from diffusers import StableDiffusionPipeline

                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model_name_or_path,
                    **kwargs,
                )

                if args.unet_adapter_name_or_path is not None:
                    from peft import PeftModel

                    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.unet_adapter_name_or_path)
                    pipeline.unet = pipeline.unet.merge_and_unload()

                if args.text_encoder_adapter_name_or_path is not None:
                    from peft import PeftModel

                    pipeline.text_encoder = PeftModel.from_pretrained(
                        pipeline.text_encoder, args.text_encoder_adapter_name_or_path
                    )
                    pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()

            else:
                # SD LDM3D use-case
                from diffusers import StableDiffusionLDM3DPipeline as StableDiffusionPipeline

                if args.model_name_or_path == "CompVis/stable-diffusion-v1-4":
                    args.model_name_or_path = "Intel/ldm3d-4c"
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model_name_or_path,
                    **kwargs,
                )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # If prompts file is specified override prompts from the file
    if args.prompts_file is not None:
        lines = []
        with open(args.prompts_file, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        args.prompts = lines

    # Generate Images using a Stable Diffusion pipeline
    #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    pipeline = pipeline.to("cuda")
    #pipeline.enable_vae_slicing()
    if args.use_torch_compile:
        pipeline = torch.compile(pipeline)
    if args.distributed:
        with distributed_state.split_between_processes(args.prompts) as prompt:
            outputs = pipeline(prompt=prompt, **kwargs_call)
    else:
        outputs = pipeline(prompt=args.prompts, **kwargs_call)

    # Save the pipeline in the specified directory if not None
    if args.pipeline_save_dir is not None:
        save_dir = args.pipeline_save_dir
        if args.distributed:
            save_dir = f"{args.pipeline_save_dir}_{distributed_state.process_index}"
        pipeline.save_pretrained(save_dir)

    # Save images in the specified directory if not None and if they are in PIL format
    if args.image_save_dir is not None:
        if args.output_type == "pil":
            image_save_dir = Path(args.image_save_dir)
            if args.distributed:
                image_save_dir = Path(f"{image_save_dir}_{distributed_state.process_index}")

            image_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving images in {image_save_dir.resolve()}...")
            if args.ldm3d:
                for i, rgb in enumerate(outputs.rgb):
                    rgb.save(image_save_dir / f"rgb_{i+1}.png")
                for i, depth in enumerate(outputs.depth):
                    depth.save(image_save_dir / f"depth_{i+1}.png")
            else:
                for i, image in enumerate(outputs.images):
                    image.save(image_save_dir / f"image_{i+1}.png")
        else:
            logger.warning("--output_type should be equal to 'pil' to save images in --image_save_dir.")


if __name__ == "__main__":
    main()

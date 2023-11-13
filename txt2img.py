import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

def sdxl_txt2img(config):

    if config.seed is not None:
        torch.manual_seed(config.seed)

    pipe = DiffusionPipeline.from_pretrained(
        config.stable_diffusion_checkpoint,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        scheduler=EulerDiscreteScheduler(
            **config.scheduler_kwargs
        ),
    )


    if config.offload_to_cpu:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if config.compile_model:
        pipe.unet = torch.compile(
            pipe.unet, mode="reduce-overhead", fullgraph=True
        )

    refiner = DiffusionPipeline.from_pretrained(
        config.refiner_checkpoint,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        scheduler=EulerDiscreteScheduler(
            **config.scheduler_kwargs
        ),
    )

    # Offload to CPU in case of OOM
    if config.offload_to_cpu:
        refiner.enable_model_cpu_offload()
    else:
        refiner.to("cuda")

    if config.compile_model:
        refiner.unet = torch.compile(
            refiner.unet, mode="reduce-overhead", fullgraph=True
        )

    if not config.use_ensemble_of_experts:
        latent = pipe(
            prompt=config.prompt_1,
            prompt_2=config.prompt_2,
            negative_prompt=config.negative_prompt_1,
            negative_prompt_2=config.negative_prompt_2,
            output_type="latent",
            width = config.width, 
            height = config.height,
            seed = config.seed,
            guidance_scale = config.guidance_scale,
            num_inference_steps = config.num_inference_steps,
            num_images_per_prompt = config.num_images_per_prompt
        )

        latent_images = latent.images

        # unrefined_image = postprocess_latent(latent)
        refined_images = []
        for latent_image in latent_images:
            refined_image = refiner(
                prompt=config.prompt_1,
                prompt_2=config.prompt_2,
                negative_prompt=config.negative_prompt_1,
                negative_prompt_2=config.negative_prompt_2,
                image=latent_image[None, :],
            ).images[0]

            refined_images.append(refined_image)

    return refined_images
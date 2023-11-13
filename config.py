import wandb

def configs():
    wandb.init(project="stable-diffusion-xl", job_type="text-to-image")
    config = wandb.config

    config.stable_diffusion_checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
    config.refiner_checkpoint = "stabilityai/stable-diffusion-xl-refiner-1.0"
    config.offload_to_cpu = False
    config.compile_model = False
    config.prompt_1 = "Astronaut in a jungle"
    config.prompt_2 = "cold color palette, muted colors, detailed, 8k"
    config.negative_prompt_1 = "oversaturated, ugly, render, cartoon, grain, low-res, kitsch, bad, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, "
    config.negative_prompt_2 = "loversaturated, ugly, render, cartoon, grain, low-res, kitsch, bad, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, "
    config.seed = -1
    config.width = 1024
    config.height = 1024
    config.use_ensemble_of_experts = True
    config.num_inference_steps = 100
    config.num_refinement_steps = 150
    config.high_noise_fraction = 0.8
    config.guidance_scale = 7
    config.num_images_per_prompt = 1
    config.scheduler_kwargs = {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear", # one of ["linear", "scaled_linear"]
        "beta_start": 0.00085,
        "interpolation_type": "linear", # one of ["linear", "log_linear"]
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon", # one of ["epsilon", "sample", "v_prediction"]
        "steps_offset": 1,
        "timestep_spacing": "leading", # one of ["linspace", "leading"]
        "trained_betas": None,
        "use_karras_sigmas": False,
    }

    return config

def set_configs(config, prompt_1, prompt_2, negative_prompt_1, negative_prompt_2, width, height, seed, guidance_scale, num_inference_steps, num_images_per_prompt):

    if prompt_1 != "": config.prompt_1 = prompt_1
    if prompt_2 != "": 
        config.prompt_2 = prompt_2
    else:
        config.prompt_2 = config.prompt_1 

    if negative_prompt_1 != "": config.negative_prompt_1 += negative_prompt_1
    if negative_prompt_2 != "": 
        config.negative_prompt_2 += negative_prompt_2
    else:
        config.negative_prompt_2 = config.negative_prompt_1 
        
    if width != 0: config.width = width
    if height != 0: config.height = height

    if seed != -1: config.seed = seed 
    if guidance_scale != -1: config.guidance_scale = guidance_scale 

    if num_inference_steps != 0: 
        config.num_refinement_steps = num_inference_steps
        config.num_inference_steps = config.num_refinement_steps // 3 * 2

    if num_images_per_prompt != 0:
        config.num_images_per_prompt = num_images_per_prompt

    return config
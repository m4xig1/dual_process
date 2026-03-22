import diffusers
import numpy as np
from omegaconf import OmegaConf
import torch
import transformers

from diffusers.training_utils import compute_density_for_timestep_sampling
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoProcessor

# ===========================
#        Load Models
# ===========================
def check_vram(device, min_vram_gb=None):
    if min_vram_gb is not None:
        free, total = torch.cuda.mem_get_info(device)
        if free / 1024**3 < min_vram_gb:
            raise ValueError(f"Not enough VRAM on device {device}. Please set pipe_kwargs.pipe_device and vlm_kwargs.vlm_device to different GPUs. You need at least {min_vram_gb}GB of VRAM.")

def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def load_vlm(vlm_device, vlm_id, vlm_cls, **vlm_kwargs):
    check_vram(vlm_device, vlm_kwargs.pop("min_vram_gb", None))
    vlm_kwargs = {k: eval(v) if ("dtype" in k and v != "auto") else v for k, v in vlm_kwargs.items()}
    vlm_cls = getattr(transformers, vlm_cls)
    vlm = vlm_cls.from_pretrained(vlm_id, device_map=vlm_device, **vlm_kwargs)
    vlm_processor = AutoProcessor.from_pretrained(vlm_id)
    # Disable image splitting
    if hasattr(vlm_processor.image_processor, "do_image_splitting"):
        vlm_processor.image_processor.do_image_splitting = False
    # Turn off requires grad
    set_requires_grad(vlm, False)
    return vlm, vlm_processor

def load_pipe(pipe_device, pipe_id, pipe_cls, scheduler_config={}, **pipe_kwargs):
    check_vram(pipe_device, pipe_kwargs.pop("min_vram_gb", None))
    pipe_kwargs = {k: eval(v) if ("dtype" in k and v != "auto") else v for k, v in pipe_kwargs.items()}
    pipe_cls = getattr(diffusers, pipe_cls)
    pipe = pipe_cls.from_pretrained(pipe_id, **pipe_kwargs)
    # Disable progress bar
    pipe.set_progress_bar_config(disable=True)
    # Disable dynamic shifting
    pipe.scheduler.config.use_dynamic_shifting = False
    # Disable checker
    pipe.safety_checker = None
    # Turn off requires grad
    for module in pipe.config.keys():
        module = getattr(pipe, module)
        if isinstance(module, torch.nn.Module):
          set_requires_grad(module, False)
    pipe.to(pipe_device)
    scheduler_cls = scheduler_config.get("eval_scheduler_cls") or scheduler_config.get("train_scheduler_cls")
    if scheduler_cls is not None:
        pipe.scheduler = getattr(diffusers, scheduler_cls).from_config(pipe.scheduler.config)
    pipe.scheduler_config = scheduler_config
    return pipe

def load_config(config_names):
    if type(config_names) is not list:
        config_names = [config_names]
    config = OmegaConf.create({})
    for config_name in config_names:
        config = OmegaConf.merge(config, OmegaConf.load(config_name))
    config = OmegaConf.to_container(config, resolve=True)
    return config

# ===========================
#        LoRA Helpers
# ===========================
def get_pipe_cls(pipe):
    class_name = str(type(pipe))
    if "Flux" in class_name:
        return "flux"
    elif "StableDiffusion" in class_name:
        return "sd"
    elif "Sana" in class_name:
        return "sana"
    else:
        return None
    
def get_backbone(pipe):
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    elif hasattr(pipe, "unet"):
        return pipe.unet
    else:
        return pipe
    
def create_lora(pipe, lora_lr, lora_name="default", **lora_kwargs):
    backbone = get_backbone(pipe)
    backbone.delete_adapters(lora_name)
    backbone.add_adapter(
        LoraConfig(**lora_kwargs), 
        adapter_name=lora_name
    )
    params = {"params": [p for name, p in backbone.named_parameters() if "lora" in name], "lr": lora_lr}
    return params

def toggle_lora(pipe, weights):
    backbone = get_backbone(pipe)
    if hasattr(backbone, "peft_config"):
        adapter_names = backbone.active_adapters()
        backbone.set_adapters(adapter_names=adapter_names, weights=weights)

class LoraManager:
    def __init__(self, pipe, enter_weights=1, exit_weights=1):
        self.pipe = pipe
        self.enter_weights = enter_weights
        self.exit_weights = exit_weights

    def __enter__(self):
        toggle_lora(self.pipe, self.enter_weights)

    def __exit__(self, exc_type, exc_value, traceback):
        toggle_lora(self.pipe, self.exit_weights)

def save_weights(pipe, output_file):
    backbone = get_backbone(pipe)
    if hasattr(backbone, "peft_config"):
        torch.save(get_peft_model_state_dict(backbone), f"{output_file}.pt")

def load_weights(pipe, output_file):
    backbone = get_backbone(pipe)
    state_dict = torch.load(f"{output_file}.pt")
    state_dict = {n.replace(".weight", ".default.weight"): p for n, p in state_dict.items()}
    backbone.load_state_dict(state_dict, strict=False)

# ===========================
#  General Pipeline Helpers
# ===========================
def renormalize(x, range_a, range_b):
    min_a, max_a = range_a
    min_b, max_b = range_b
    return ((x - min_a) / (max_a - min_a)) * (max_b - min_b) + min_b

def create_callback_interrupt(stop_i):
    # Runs stop_i steps then early stops
    def callback_on_step_end(self, i, t, callback_kwargs):
        assert stop_i > 0
        if i >= (stop_i - 1):
            self._interrupt = True
        else:
            self._interrupt = False
        return {}
    return callback_on_step_end

@torch.no_grad()
def run_pipe(pipe, generator_kwargs, prompt_kwargs={}, stop_i=None, **kwargs):
    pipe_kwargs = {**generator_kwargs, **kwargs}
    prompt_remapping = {
        "pooled_projections": "pooled_prompt_embeds", 
        "encoder_hidden_states": "prompt_embeds", 
        "encoder_attention_mask": "prompt_attention_mask"
    }
    for k, v in prompt_remapping.items():
        if k in prompt_kwargs:
            pipe_kwargs[v] = prompt_kwargs[k]
    if "prompt_embeds" in pipe_kwargs:
        prompt_embeds = pipe_kwargs["prompt_embeds"]
        # Handle classifier-free guidance
        assert prompt_embeds.shape[0] <= 2, "Only one prompt supported in this mode due to cfg handling"
        if prompt_embeds.shape[0] == 2:
            pipe_kwargs["prompt_embeds"] = prompt_embeds[1][None, ...]
            pipe_kwargs["negative_prompt_embeds"] = prompt_embeds[0][None, ...]
            if "prompt_attention_mask" in pipe_kwargs:
                prompt_attention_mask = pipe_kwargs["prompt_attention_mask"]
                pipe_kwargs["prompt_attention_mask"] = prompt_attention_mask[1][None, ...]
                pipe_kwargs["negative_prompt_attention_mask"] = prompt_attention_mask[0][None, ...]
            pipe_kwargs["prompt"] = None
            pipe_kwargs["negative_prompt"] = None
    if stop_i is not None:
        callback_interrupt = create_callback_interrupt(stop_i)
        pipe_kwargs["callback_on_step_end"] = callback_interrupt
    if stop_i is None or stop_i > 0:
        return pipe(**pipe_kwargs).images
    else:
        return pipe_kwargs.get("latents")

def get_train_scheduler(pipe):
    scheduler_cls = pipe.scheduler_config["train_scheduler_cls"]
    scheduler_obj = getattr(diffusers, scheduler_cls)

    # Create a dummy scheduler to get valid config keys from pipe.scheduler.config
    scheduler_config = pipe.scheduler.config
    dummy_scheduler = scheduler_obj()
    valid_keys = set(dummy_scheduler.config.keys())
    valid_config = {k: v for k, v in scheduler_config.items() if k in valid_keys}
    
    # Create scheduler
    scheduler = scheduler_obj(**valid_config)
    scheduler._class_name = scheduler_cls

    # Set timesteps
    num_timesteps = pipe.scheduler_config["num_train_timesteps"]
    if scheduler_cls == "FlowMatchEulerDiscreteScheduler":
        sigmas = np.linspace(1.0, 1 / num_timesteps, num_timesteps)
        scheduler.set_timesteps(sigmas=sigmas)
    else:
        scheduler.set_timesteps(num_timesteps)
    return scheduler

def get_timestep(pipe):
    train_scheduler = get_train_scheduler(pipe)
    timesteps = train_scheduler.timesteps.to(pipe.device)
    u = compute_density_for_timestep_sampling(
        weighting_scheme="none",
        batch_size=1,
    )
    indices = (u * len(timesteps)).long()
    t = timesteps[indices]
    i = indices
    i = i.long().item()
    return i, t

# ===========================
#    Flux Pipeline Helpers
# ===========================
def get_flux_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    backbone = get_backbone(pipe)
    num_channels_latents = backbone.config.in_channels // 4
    height, width = generator_kwargs["height"], generator_kwargs["width"]
    height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    if not pack:
        shape = (batch_size, num_channels_latents, height, width)
    else:
        shape = (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return shape

def encode_flux_text(pipe, prompts, keys, device):
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device
        )
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
        }
    return prompt_kwargs

def get_flux_latent_image_ids(pipe, generator_kwargs, batch_size=1):
    device, dtype = pipe.device, pipe.dtype
    b, c, h, w = get_flux_latent_shape(pipe, generator_kwargs, batch_size, pack=False)
    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size,
        h // 2, 
        w // 2,
        device,
        dtype,
    )
    return latent_image_ids

def get_flux_guidance(pipe, generator_kwargs):
    if pipe.transformer.config.guidance_embeds:
        device = pipe.device
        guidance = torch.tensor([generator_kwargs["guidance_scale"]]).to(device)
        return guidance
    else:
        return None

def run_flux_forward(pipe, generator_kwargs, latents, t, forward_kwargs):
    latent_image_ids = get_flux_latent_image_ids(pipe, generator_kwargs)
    guidance = get_flux_guidance(pipe, generator_kwargs)
    forward_kwargs = {
        "hidden_states": latents,
        "timestep": (t / 1000),
        "guidance": guidance,
        "img_ids": latent_image_ids,
        "return_dict": False,
        **forward_kwargs
    }
    model_pred = pipe.transformer(**forward_kwargs)[0]
    return model_pred, forward_kwargs

# ===========================
#     SD Pipeline Helpers
# ===========================
def get_sd_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    backbone = get_backbone(pipe)
    num_channels_latents = backbone.config.in_channels
    height, width = generator_kwargs["height"], generator_kwargs["width"]
    height = int(height) // pipe.vae_scale_factor
    width = int(width) // pipe.vae_scale_factor
    shape = (batch_size, num_channels_latents, height, width)
    return shape

def encode_sd_text(pipe, prompts, keys, device):
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "encoder_hidden_states": prompt_embeds,
        }
    return prompt_kwargs

def run_sd_forward(pipe, generator_kwargs, latents, t, forward_kwargs, sample_key="sample"):
    backbone = get_backbone(pipe)
    b = forward_kwargs["encoder_hidden_states"].shape[0] // 2
    latents = latents.expand(b * 2, *latents.shape[1:])
    t = t.expand(latents.shape[0])
    forward_kwargs = {
        sample_key: latents,
        "timestep": t,
        "return_dict": False,
        **forward_kwargs
    }
    noise_pred = backbone(**forward_kwargs)[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    model_pred = noise_pred_uncond + generator_kwargs["guidance_scale"] * (noise_pred_text - noise_pred_uncond)
    forward_kwargs["hidden_states"] = forward_kwargs.pop(sample_key)[:b]
    return model_pred, forward_kwargs

# ===========================
#    Sana Pipeline Helpers
# ===========================
def get_sana_latent_shape(pipe, generator_kwargs, batch_size=1, pack=False):
    return get_sd_latent_shape(pipe, generator_kwargs, batch_size, pack)

def encode_sana_text(pipe, prompts, keys, device):
    prompt_kwargs = {}
    for key, prompt in zip(keys, prompts):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        prompt_kwargs[f"{key}_prompt_kwargs"] = {
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_attention_mask
        }
    return prompt_kwargs

def run_sana_forward(pipe, generator_kwargs, latents, t, forward_kwargs):
    return run_sd_forward(pipe, generator_kwargs, latents, t, forward_kwargs, sample_key="hidden_states")
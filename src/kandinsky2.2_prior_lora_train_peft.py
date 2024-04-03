import argparse
import math
import wandb
import numpy as np

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoPipelineForText2Image, DDPMScheduler, PriorTransformer
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from piq import FID
from PIL import Image
from peft import LoraConfig
from peft import get_peft_model
from peft.utils import get_peft_model_state_dict
from transformers import (CLIPImageProcessor, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)
from requests.exceptions import ConnectionError

from config.config import get_cfg_defaults
from dataset import ClefKandinskyPriorDataset, MyFIDDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        default="Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis",
        required=False,
        help="Wandb project name",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        required=False,
        help="Wandb run name",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=False,
        help=('Path to config file')
    )

    args = parser.parse_args()

    return args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main():
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config_path:
        cfg.merge_from_file(args.config_path)
    cfg.freeze()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
        mixed_precision=cfg['ACCELERATOR']['MIXED_PRECISION'],
        log_with=cfg['ACCELERATOR']['LOG_WITH'],
    )
    set_seed(cfg['SYSTEM']['RANDOM_SEED'])

    noise_scheduler = DDPMScheduler.from_pretrained(cfg['EXPERIMENT']['DECODER_PATH'], subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'], subfolder="tokenizer")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'],  subfolder="image_encoder")
    image_processor = CLIPImageProcessor.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'],  subfolder="image_processor")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'], subfolder="text_encoder")
    prior = PriorTransformer.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'], subfolder="prior")

    image_encoder.requires_grad_(False)
    prior.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    prior.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    prior_lora_config = LoraConfig(
        r=cfg['LORA']['RANK'],
        lora_alpha=cfg['LORA']['ALPHA'],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    prior.add_adapter(prior_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, prior.parameters())

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_dataset = ClefKandinskyPriorDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], tokenizer, image_processor, cfg['EXPERIMENT']['PROMPTS_FILE_NAME'])

    def collate_fn(examples):
        clip_pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.contiguous_format).float()
        text_input_ids = torch.stack([example["text_input_ids"] for example in examples])
        text_mask = torch.stack([example["text_mask"] for example in examples])
        return {"clip_pixel_values": clip_pixel_values, "text_input_ids": text_input_ids, "text_mask": text_mask}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS']
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg['ACCELERATOR']['ACCUMULATION_STEPS'])
    max_train_steps = cfg['TRAIN']['EPOCHS'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
            name=cfg['SCHEDULER']['NAME'],
            optimizer=optimizer,
            num_warmup_steps=cfg['SCHEDULER']['WARMUP_STEPS'] * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
            num_training_steps=max_train_steps * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg['EXPERIMENT']['PROJECT_NAME'], config=cfg, init_kwargs={"wandb": {"name": cfg['EXPERIMENT']['NAME']}})

    clip_mean = prior.clip_mean.clone()
    clip_std = prior.clip_std.clone()
    prior, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        prior, optimizer, train_dataloader, lr_scheduler
    )

    clip_mean = clip_mean.to(weight_dtype).to(accelerator.device)
    clip_std = clip_std.to(weight_dtype).to(accelerator.device)

    total_batch_size = cfg['TRAIN']['BATCH_SIZE'] * accelerator.num_processes * cfg['ACCELERATOR']['ACCUMULATION_STEPS']

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(first_epoch, cfg['TRAIN']['EPOCHS']):
        prior.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(prior):
                text_input_ids, text_mask, clip_images = (
                    batch["text_input_ids"],
                    batch["text_mask"],
                    batch["clip_pixel_values"].to(weight_dtype),
                )
                with torch.no_grad():
                    text_encoder_output = text_encoder(text_input_ids)
                    prompt_embeds = text_encoder_output.text_embeds
                    text_encoder_hidden_states = text_encoder_output.last_hidden_state
                    image_embeds = image_encoder(clip_images).image_embeds
                    # Sample noise that we'll add to the image_embeds
                    noise = torch.randn_like(image_embeds)
                    bsz = image_embeds.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=image_embeds.device
                    )
                    timesteps = timesteps.long()
                    image_embeds = (image_embeds - clip_mean) / clip_std
                    noisy_latents = noise_scheduler.add_noise(image_embeds, noise, timesteps)

                    target = image_embeds

                # Predict the noise residual and compute loss
                model_pred = prior(
                    noisy_latents,
                    timestep=timesteps,
                    proj_embedding=prompt_embeds,
                    encoder_hidden_states=text_encoder_hidden_states,
                    attention_mask=text_mask,
                ).predicted_image_embedding
                # model_pred.requires_grad = True
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['BATCH_SIZE'])).mean()
                train_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers, cfg['LORA']['MAX_GRADIENT_NORM'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"prior_train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                    break

        if accelerator.is_main_process:
            if (epoch + 1) % cfg['TRAIN']['VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['EPOCHS'] - 1:
                try:
                    pipeline = AutoPipelineForText2Image.from_pretrained(
                        cfg['EXPERIMENT']['DECODER_PATH'],
                        prior_prior=accelerator.unwrap_model(prior),
                        torch_dtype=weight_dtype,
                        local_files_only=True
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    generator = torch.Generator(device=accelerator.device)
                    generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])

                    image = pipeline(cfg['EXPERIMENT']['VALIDATION_PROMPT'], num_inference_steps=30, generator=generator).images[0]
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            tracker.log({"Prior validation": wandb.Image(image, caption=cfg['EXPERIMENT']['VALIDATION_PROMPT'])})

                    del pipeline
                except ConnectionError as error:
                    print('Unable to Validate: Connection Error')
                torch.cuda.empty_cache()


    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     prior = prior.to(torch.float32)
    #     prior.save_attn_procs(cfg['EXPERIMENT']['LORA_PRIOR_WEIGHTS_OUTPUT_DIR'])
    accelerator.end_training()


if __name__ == "__main__":
    main()

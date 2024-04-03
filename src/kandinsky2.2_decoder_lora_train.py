import argparse
import math
import wandb

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoPipelineForText2Image, DDPMScheduler, UNet2DConditionModel, VQModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnAddedKVProcessor
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from piq import FID
from PIL import Image
from transformers import (CLIPImageProcessor, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)
from requests.exceptions import ConnectionError

from config.config import get_cfg_defaults
from dataset import ClefKandinskyDecoderDataset


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
    image_processor = CLIPImageProcessor.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'], subfolder="image_processor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg['EXPERIMENT']['PRIOR_PATH'], subfolder="image_encoder")
    vae = VQModel.from_pretrained(cfg['EXPERIMENT']['DECODER_PATH'], subfolder="movq")
    unet = UNet2DConditionModel.from_pretrained(cfg['EXPERIMENT']['DECODER_PATH'], subfolder="unet")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnAddedKVProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=cfg['LORA']['RANK'],
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_dataset = ClefKandinskyDecoderDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], image_processor, resolution=cfg['TRAIN']['IMAGE_RESOLUTION'])
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        clip_pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "clip_pixel_values": clip_pixel_values}

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

    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    total_batch_size = cfg['TRAIN']['BATCH_SIZE'] * accelerator.num_processes * cfg['ACCELERATOR']['ACCUMULATION_STEPS']

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(first_epoch, cfg['TRAIN']['EPOCHS']):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                images = batch["pixel_values"].to(weight_dtype)
                clip_images = batch["clip_pixel_values"].to(weight_dtype)
                latents = vae.encode(images).latents
                image_embeds = image_encoder(clip_images).image_embeds
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                target = noise

                added_cond_kwargs = {"image_embeds": image_embeds}

                model_pred = unet(noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs).sample[:, :4]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['BATCH_SIZE'])).mean()
                train_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers.parameters(), cfg['LORA']['MAX_GRADIENT_NORM'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"decoder_train_loss": train_loss}, step=global_step)
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
                        unet=accelerator.unwrap_model(unet),
                        torch_dtype=weight_dtype,
                        local_files_only=True
                    )
                    pipeline.prior_prior.load_attn_procs('/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/lora_weights')
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    generator = torch.Generator(device=accelerator.device)
                    generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])

                    image = pipeline(cfg['EXPERIMENT']['VALIDATION_PROMPT'], num_inference_steps=30, generator=generator).images[0]
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            tracker.log({"Decoder validation": wandb.Image(image, caption=cfg['EXPERIMENT']['VALIDATION_PROMPT'])})

                    del pipeline
                except ConnectionError as error:
                    print('Unable to Validate: Connection Error')
                torch.cuda.empty_cache()


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(cfg['EXPERIMENT']['LORA_DECODER_WEIGHTS_OUTPUT_DIR'])
    accelerator.end_training()


if __name__ == "__main__":
    main()

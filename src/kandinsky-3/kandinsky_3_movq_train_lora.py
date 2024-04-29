import argparse
import math
import wandb
import numpy as np
import os
from lpips import LPIPS
import torchvision

import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoPipelineForText2Image, DDPMScheduler, UNet2DConditionModel, VQModel, AutoencoderKL, ConfigMixin
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnAddedKVProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import compute_snr
from tqdm import tqdm
from piq import FID, ssim
from PIL import Image
from peft import LoraConfig, PeftModel
from peft import get_peft_model
from peft.utils import get_peft_model_state_dict
from transformers import (CLIPImageProcessor, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)
from requests.exceptions import ConnectionError

from config.medfusion_config import get_cfg_defaults
from medfusion_dataset import ClefMedfusionVAEDataset

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
        cfg.merge_from_file(cfg['SYSTEM']['CONFIG_PATH'] + args.config_path)
    cfg.freeze()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
        mixed_precision=cfg['ACCELERATOR']['MIXED_PRECISION'],
        log_with=cfg['ACCELERATOR']['LOG_WITH'],
    )

    set_seed(cfg['SYSTEM']['RANDOM_SEED'])

    vae = AutoencoderKL.from_config(cfg['EXPERIMENT']['MEDFUSION_VAE_CONFIG_PATH'])
    vae.requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_dataset = ClefMedfusionVAEDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], resolution=cfg['TRAIN']['IMAGE_RESOLUTION'])
    val_dataset = ClefMedfusionVAEDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], train=False, load_to_ram=True)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS'],
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg['TRAIN']['VAL_BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS'],
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

    vae, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    lpips_loss_fn = LPIPS(net="alex").to(accelerator.device)

    for epoch in range(first_epoch, cfg['TRAIN']['EPOCHS']):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                target = batch["pixel_values"]

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                posterior = vae.encode(target).latent_dist
                z = posterior.mode()
                pred = vae.decode(z).sample

                kl_loss = posterior.kl().mean()
                mse_loss = F.mse_loss(pred, target, reduction="mean")
                lpips_loss = lpips_loss_fn(pred, target).mean()

                loss = (
                    mse_loss + cfg['TRAIN']['LPIPS_SCALE'] * lpips_loss + cfg['TRAIN']['KL_SCALE'] * kl_loss
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['BATCH_SIZE'])).mean()
                train_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"vae_train_loss": train_loss}, step=global_step)

                if accelerator.is_main_process: 
                    if (global_step) % cfg['TRAIN']['IMAGE_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps:
                        
                        vae_model = accelerator.unwrap_model(vae)
                        real_images = []
                        reconstructed_images = []
                        for _, sample in enumerate(val_dataloader):
                            x = sample["pixel_values"]
                            reconstructions = vae_model(x).sample
                            real_images.append(sample["pixel_values"].cpu())
                            reconstructed_images.append(reconstructions.cpu().clamp(-1, 1))
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                tracker.log(
                                    {
                                        "Original": wandb.Image(torchvision.utils.make_grid(real_images[0], nrow=3)),
                                        "Reconstruction": wandb.Image(torchvision.utils.make_grid(reconstructed_images[0], nrow=3))
                                    }
                                )

                        del vae_model
                        torch.cuda.empty_cache()

                global_step += 1
                train_loss = 0.0
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(cfg['EXPERIMENT']['MEDFUSION_VAE_OUTPUT_DIR'], subfolder=cfg['EXPERIMENT']['MEDFUSION_VAE_SUBFOLDER'])

    accelerator.end_training()


if __name__ == "__main__":
    main()

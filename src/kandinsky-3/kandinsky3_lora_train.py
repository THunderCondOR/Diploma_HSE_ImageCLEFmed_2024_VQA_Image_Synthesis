import argparse
import math
import wandb
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, gather, gather_object
from diffusers import DiffusionPipeline, Kandinsky3Pipeline, DDPMScheduler, VQModel, Kandinsky3UNet, VQModel, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnAddedKVProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft import get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm
from piq import FID, ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from requests.exceptions import ConnectionError

from config.config import get_cfg_defaults
from dataset import ClefKandinsky3Dataset, MyFIDDataset


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

    noise_scheduler = DDPMScheduler.from_pretrained(cfg['EXPERIMENT']['KANDINSKY3_PATH'], subfolder="scheduler", local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(cfg['EXPERIMENT']['KANDINSKY3_PATH'], subfolder="text_encoder", use_safetensors=True, variant="fp16", local_files_only=True)
    movq = VQModel.from_pretrained(cfg['EXPERIMENT']['KANDINSKY3_PATH'], subfolder="movq", use_safetensors=True, variant="fp16", local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(cfg['EXPERIMENT']['KANDINSKY3_PATH'], subfolder="tokenizer", local_files_only=True)
    unet = Kandinsky3UNet.from_pretrained(cfg['EXPERIMENT']['KANDINSKY3_PATH'], subfolder="unet", use_safetensors=True, variant="fp16", local_files_only=True)
    unet.requires_grad_(False)
    movq.requires_grad_(False)
    text_encoder.requires_grad_(False)

    for param in unet.parameters():
        param.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=cfg['LORA']['RANK'],
        lora_alpha=cfg['LORA']['ALPHA'],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    movq.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    #unet.add_adapter(unet_lora_config)
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.to(torch.float32)
    
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_dataset = ClefKandinsky3Dataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], tokenizer, resolution=cfg['TRAIN']['IMAGE_RESOLUTION'], load_to_ram=False)
    val_dataset = ClefKandinsky3Dataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], tokenizer, resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], train=False, load_to_ram=True)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

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

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
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
                with torch.no_grad():
                    latents = movq.encode(batch["pixel_values"].to(dtype=torch.float32)).latents # * movq.config.scaling_factor

                    noise = torch.randn_like(latents)

                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    print(timesteps.shape)

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                    if cfg['SCHEDULER']['PREDICTION_TYPE'] is not None:
                        noise_scheduler.register_to_config(prediction_type=cfg['SCHEDULER']['PREDICTION_TYPE'])

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, batch["attention_mask"], return_dict=False)[0] #[:, :4]

                if cfg['SCHEDULER']['SNR_GAMMA'] == -1:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, cfg['SCHEDULER']['SNR_GAMMA'] * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['BATCH_SIZE'])).mean()
                train_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers, cfg['LORA']['MAX_GRADIENT_NORM'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"kandinsky3_train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    if (global_step) % cfg['TRAIN']['IMAGE_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps:
                        try:
                            pipeline = Kandinsky3Pipeline.from_pretrained(
                                cfg['EXPERIMENT']['KANDINSKY3_PATH'],
                                unet=accelerator.unwrap_model(unet),
                                scheduler=accelerator.unwrap_model(noise_scheduler),
                                tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                movq=movq,
                                torch_dtype=weight_dtype,
                                variant="fp16",
                                use_safetensors=True,
                                local_files_only=True
                            )
                            #pipeline = pipeline.to(accelerator.device)
                            pipeline.set_progress_bar_config(disable=True)
                            
                            #generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])

                            pipeline.to(accelerator.device)
                            generator = torch.Generator(device=accelerator.device)

                            val_images = pipeline(cfg['EXPERIMENT']['VALIDATION_PROMPT'], num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'], height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], num_images_per_prompt=4, generator=generator).images
                            log_dict = {"Kandinsky-3 validation": wandb.Image(image_grid(val_images, 2, 2), caption=cfg['EXPERIMENT']['VALIDATION_PROMPT'])}

                            if cfg['EXPERIMENT']['FID_VALIDATION'] and ((global_step) % cfg['TRAIN']['FID_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps):
                                fake_images = []
                                for val_batch in tqdm(val_dataloader):
                                    fake_images += pipeline(val_batch['prompts'], num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'], height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], generator=generator).images

                                del pipeline 

                                real_images = torch.stack(val_dataset.images)

                                real_dataset = MyFIDDataset(real_images)
                                fake_dataset = MyFIDDataset(fake_images, fake=True)

                                ssim_value = ssim(fake_dataset.data, real_images, data_range=1.)
                                fid_metric = FID()

                                real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=100)
                                fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=100)
                                
                                real_feats = fid_metric.compute_feats(real_loader, device=accelerator.device)
                                fake_feats = fid_metric.compute_feats(fake_loader, device=accelerator.device)
                                
                                fid_value = fid_metric(real_feats, fake_feats).item()

                                #-----------------------------------------------------------
                                fake_images = fake_dataset.data.to(accelerator.device)
                                real_images = real_images.to(accelerator.device)
                                torchmetrics_fid = FrechetInceptionDistance(normalize=True)
                                torchmetrics_fid.to(accelerator.device)
                                torchmetrics_fid.update(real_images, real=True)
                                torchmetrics_fid.update(fake_images, real=False)

                                log_dict['Torchmetrics_FID'] = float(torchmetrics_fid.compute())
                                del torchmetrics_fid
                                #-----------------------------------------------------------

                                log_dict['FID'] = fid_value
                                log_dict['SSIM'] = ssim_value
                            else:
                                del pipeline
                            
                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(log_dict)

                        except ConnectionError as error:
                            print('Unable to Validate: Connection Error')
                        torch.cuda.empty_cache()

                global_step += 1
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(
                os.path.join(cfg['EXPERIMENT']['KANDINSKY3_LORA_WEIGHTS_OUTPUT_DIR'], "unet"),
                safe_serialization=False
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()

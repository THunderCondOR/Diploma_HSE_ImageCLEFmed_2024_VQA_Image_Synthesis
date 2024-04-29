import argparse
import math
import wandb
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, gather, gather_object
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from tqdm import tqdm
from piq import FID, multi_scale_ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from requests.exceptions import ConnectionError
from torchvision.utils import make_grid
from config.medfusion_config import get_cfg_defaults
from dataset import ClefMedfusionUnetDataset, MyFIDDataset

from medical_diffusion.models.pipelines.diffusion_pipeline import DiffusionPipeline
from medical_diffusion.models.estimators.unet import UNet
from medical_diffusion.models.noise_schedulers.gaussian_scheduler import GaussianNoiseScheduler
from medical_diffusion.models.embedders.time_embedder import TimeEmbbeding
from medical_diffusion.models.embedders.cond_embedders import TextEmbedder, LabelEmbedder
from medical_diffusion.models.embedders.latent_embedders import VAE


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

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


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

    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True, model_max_length=64)

    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    #cond_embedder = TextEmbedder(text_encoder, emb_dim=768)
    # cond_embedder = LabelEmbedder
    # cond_embedder_kwargs = {
    #     'text_encoder': text_encoder,
    #     'emb_dim': 1024,
    #     'num_classes': tokenizer.vocab_size,
    #     'input_dim': tokenizer.model_max_length,
    #     'hidden_dim': 768
    # }

    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }

    unet = UNet(in_ch=8,
                out_ch=8,
                spatial_dims=2,
                hid_chs=[256, 256, 512, 1024],
                kernel_sizes=[3, 3, 3, 3],
                strides=[1, 2, 2, 2],
                time_embedder=time_embedder,
                time_embedder_kwargs = time_embedder_kwargs,
                deep_supervision=False,
                use_res_block=True,
                use_attention='diffusers',
                cond_emb_hidden_dim=768)
    
    unet.to(accelerator.device, dtype=torch.float32)

    noise_scheduler = GaussianNoiseScheduler(timesteps=1000,
                                             beta_start=0.002,
                                             beta_end=0.02,
                                             schedule_strategy='scaled_linear')

    vae = VAE(
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs =[64, 128, 256, 512], 
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention='none',
    )

    vae_checkpoint = torch.load('/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/src/original_medfusion/trained_models/vae/vae.pt')
    vae.load_state_dict(vae_checkpoint['state_dict'])
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_dataset = ClefMedfusionUnetDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], tokenizer, resolution=cfg['TRAIN']['IMAGE_RESOLUTION'], load_to_ram=False)
    val_dataset = ClefMedfusionUnetDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], tokenizer, resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'], train=False, load_to_ram=True)
    
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

    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    pipeline = DiffusionPipeline(noise_scheduler=noise_scheduler,
                                 noise_estimator=unet,
                                 tokenizer=tokenizer,
                                 latent_embedder=vae,
                                 text_encoder=text_encoder,
                                 estimator_objective='x_T',
                                 estimate_variance=False, 
                                 use_self_conditioning=False, 
                                 use_ema=False,
                                 classifier_free_guidance_dropout=0, # Disable during training by setting to 0
                                 do_input_centering=False,
                                 clip_x0=False,
                                 sample_every_n_steps=1000,
                                 device=accelerator.device
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

    val_texts = ['Generate an image with 1 finding.',
                                         'Generate an image containing text.',
                                         'Generate an image with an abnormality with the colors pink, red and white.',
                                         'Generate an image not containing a green/black box artefact.',
                                         'Generate an image containing a polyp.',
                                         'Generate an image containing a green/black box artefact.',
                                         'Generate an image with no polyps.',
                                         'Generate an image with an an instrument located in the lower-right and lower-center.',
                                         'Generate an image containing a green/black box artefact.']
    #val_condition = tokenizer(val_texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    polyp_texts = ['generate an image containing a polyp'] * 4
    #polyp_condition = tokenizer(['generate an image containing a polyp'] * 4, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    val_epoch_loss = 0.0
    val_epoch_batch_sum = 0.0
    best_fid = 1e10
    for epoch in range(first_epoch, cfg['TRAIN']['EPOCHS']):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                loss = pipeline.step(batch['pixel_values'], batch['input_ids'], batch['attention_mask'])

                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['BATCH_SIZE'])).mean()
                train_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']
                val_epoch_loss += train_loss * batch['pixel_values'].shape[0]
                val_epoch_batch_sum += batch['pixel_values'].shape[0]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg['LORA']['MAX_GRADIENT_NORM'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"medfusion_unet_step_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    if (global_step) % cfg['TRAIN']['IMAGE_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps:
                        try:
                            unwrapped_unet = accelerator.unwrap_model(pipeline.noise_estimator)
                            unwrapped_unet.eval()
                            val_pipeline = DiffusionPipeline(
                                 noise_scheduler=noise_scheduler,
                                 noise_estimator=unwrapped_unet,
                                 tokenizer=tokenizer,
                                 latent_embedder=vae,
                                 text_encoder=text_encoder,
                                 estimator_objective='x_T',
                                 estimate_variance=False, 
                                 use_self_conditioning=False, 
                                 use_ema=False,
                                 classifier_free_guidance_dropout=0, # Disable during training by setting to 0
                                 do_input_centering=False,
                                 clip_x0=False,
                                 sample_every_n_steps=1000,
                                 device=accelerator.device
                                 )
                            val_pipeline = val_pipeline.to(accelerator.device)
                            
                            val_images = normalize_tensor(val_pipeline.sample(num_samples=9,
                                                                              img_size=[64, 64],
                                                                              prompts=val_texts,
                                                                              #   condition=val_condition['input_ids'].to(accelerator.device),
                                                                              #   condition_mask=val_condition['attention_mask'].to(accelerator.device),
                                                                              steps=150,
                                                                              guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach())
                            
                            polyp_images = normalize_tensor(val_pipeline.sample(num_samples=4,
                                                                                img_size=[64, 64],
                                                                                prompts=polyp_texts,
                                                                                # condition=polyp_condition['input_ids'].to(accelerator.device),
                                                                                # condition_mask=polyp_condition['attention_mask'].to(accelerator.device),
                                                                                steps=150,
                                                                                guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach())
                            
                            log_dict = {"Medfusion Unet validation": wandb.Image(make_grid(val_images, nrow=3), caption=';'.join(val_texts)),
                                        "Polyp Validation": wandb.Image(make_grid(polyp_images, nrow=2), caption='generate an image containing a polyp'),
                                        "Val Epoch Loss:": val_epoch_loss / val_epoch_batch_sum}
                            
                            val_epoch_loss = 0.0
                            val_epoch_batch_sum = 0.0

                            if cfg['EXPERIMENT']['FID_VALIDATION'] and ((global_step) % cfg['TRAIN']['FID_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps):
                                fake_images = []
                                for val_batch in tqdm(val_dataloader):
                                    fake_images += [normalize_tensor(val_pipeline.sample(num_samples=val_batch['images'].shape[0],
                                                                                         img_size=[64, 64],
                                                                                         prompts=val_batch['texts'],
                                                                                         #  condition=val_batch['input_ids'],
                                                                                         #  condition_mask=val_batch['attention_mask'],
                                                                                         steps=150,
                                                                                         guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']))]

                                del val_pipeline 

                                real_images = torch.stack(val_dataset.images).to(accelerator.device)
                                fake_images = torch.cat(fake_images)

                                real_dataset = MyFIDDataset(real_images)
                                fake_dataset = MyFIDDataset(fake_images)

                                ssim_value = multi_scale_ssim(fake_dataset.data, real_images, data_range=1.)
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
                                del real_images
                                del fake_images
                                #-----------------------------------------------------------

                                log_dict['FID'] = fid_value
                                log_dict['MS_SSIM'] = ssim_value

                                if fid_value < best_fid:
                                    torch.save({'model_state_dict': unwrapped_unet.state_dict()},
                                                os.path.join(cfg['EXPERIMENT']['MEDFUSION_UNET_OUTPUT_DIR'], cfg['EXPERIMENT']['MEDFUSION_UNET_SUBFOLDER'], cfg['EXPERIMENT']['MEDFUSION_UNET_CKPT_NAME']))
                                    best_fid = fid_value
                            else:
                                del val_pipeline
                            
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
    # if accelerator.is_main_process:
    #     unwrapped_unet = accelerator.unwrap_model(unet)
    #     torch.save({'state_dict': unwrapped_unet.state_dict()},
    #                 os.path.join(cfg['EXPERIMENT']['MEDFUSION_VAE_OUTPUT_DIR'], cfg['EXPERIMENT']['MEDFUSION_VAE_SUBFOLDER'], 'vae.pt'))
    accelerator.end_training()


if __name__ == "__main__":
    main()
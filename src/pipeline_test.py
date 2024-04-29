from diffusers import DiffusionPipeline, AutoPipelineForText2Image, PriorTransformer, Kandinsky3Pipeline
from diffusers.loaders import LoraLoaderMixin
from peft import PeftModel
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from piq import FID
from dataset import ClefKandinskyDecoderDataset, MyFIDDataset
from tqdm import tqdm
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# prior_ = None

print('fp_16 + 256 + inf 200')

pipeline = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", local_files_only=True, torch_dtype=torch.float16)
pipeline.prior_prior = PeftModel.from_pretrained(pipeline.prior_prior, '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/trained_lora_weights/lora_weights_prior', subfolder='prior_rank16_2')
pipeline.unet = PeftModel.from_pretrained(pipeline.unet, '/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/trained_lora_weights/lora_weights_decoder', subfolder='unet')
pipeline = pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=True)
generator = torch.Generator(device='cuda')
generator = generator.manual_seed(49275923)


val_dataset = ClefKandinskyDecoderDataset('/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/data/train', image_processor=None, resolution=256, train=False, load_to_ram=True)

val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=100,
        num_workers=2,
)

fake_images = []
for val_batch in tqdm(val_dataloader):
    fake_images += pipeline(val_batch['prompts'], num_inference_steps=100, height=256, width=256, generator=generator).images

grid = image_grid(fake_images[:9], 3, 3)
grid.save('pipeline_test_image_grid_256.jpg')

del pipeline               

real_images = torch.stack(val_dataset.images)

real_dataset = MyFIDDataset(real_images)
fake_dataset = MyFIDDataset(fake_images, fake=True)

fid_metric = FID()

real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=100)
fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=100)

real_feats = fid_metric.compute_feats(real_loader, device='cuda')
fake_feats = fid_metric.compute_feats(fake_loader, device='cuda')

fid_value = fid_metric(real_feats, fake_feats).item()

print(fid_value)
print('OK')
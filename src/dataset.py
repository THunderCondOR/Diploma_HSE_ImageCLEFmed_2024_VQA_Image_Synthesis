import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torch

from torch.utils.data import Dataset


class ClefKandinsky3Dataset(Dataset):
    def __init__(self,
                 path,
                 tokenizer,
                 resolution=64,
                 csv_filename='prompt-gt.csv',
                 train=True,
                 load_to_ram=False,
                 val_size = 0.025,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.load_to_ram = load_to_ram
        self.tokenizer = tokenizer
        self.train = train
        self.resolution = resolution
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')
        
        self.train_size = int(self.prompts_info.shape[0] * (1 - val_size))
        train_indexes = self.random_state.choice(self.prompts_info.shape[0], size=self.train_size, replace=False)

        if train:
            self.texts = self.prompts_info.iloc[train_indexes, 0].tolist()
            self.prompts = tokenizer(
                self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.image_files = self.prompts_info.iloc[train_indexes, 1].tolist()
        else:
            self.val_size = self.prompts_info.shape[0] - self.train_size
            val_indexes = np.delete(np.arange(self.prompts_info.shape[0]), train_indexes)
            self.texts = self.prompts_info.iloc[val_indexes, 0].tolist()
            self.prompts = tokenizer(
                self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.image_files = self.prompts_info.iloc[val_indexes, 1].tolist()

        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files):
        images = []
        for filename in tqdm(image_files):
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            images += [image]

        return images

    def __len__(self):
        return len(self.prompts['input_ids'])

    def __getitem__(self, item):
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.image_files[item]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
        transform = T.Compose([T.CenterCrop(min(image.size)),
                               T.Resize(size=(self.resolution, self.resolution), interpolation=InterpolationMode.BICUBIC),
                               T.ToTensor()])
        if self.train:
            return {"pixel_values": transform(image) / 127.5 - 1,
                    "input_ids": self.prompts['input_ids'][item],
                    "attention_mask": self.prompts['attention_mask'][item].bool()}
        else:
            return {"image": self.transform(image), "prompt": self.texts[item]}


class ClefKandinskyPriorDataset(Dataset):
    def __init__(self,
                 path,
                 tokenizer,
                 image_processor,
                 resolution=64,
                 csv_filename='prompt-gt.csv',
                 train=True,
                 load_to_ram=False,
                 val_size = 0.025,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.load_to_ram = load_to_ram
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.train = train
        self.transform = T.Compose([T.Resize(size=(resolution, resolution)), T.ToTensor()])
        
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')
        
        self.train_size = int(self.prompts_info.shape[0] * (1 - val_size))
        train_indexes = self.random_state.choice(self.prompts_info.shape[0], size=self.train_size, replace=False)

        if train:
            self.texts = self.prompts_info.iloc[train_indexes, 0].tolist()
            self.prompts = tokenizer(
                self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.image_files = self.prompts_info.iloc[train_indexes, 1].tolist()
        else:
            self.val_size = self.prompts_info.shape[0] - self.train_size
            val_indexes = np.delete(np.arange(self.prompts_info.shape[0]), train_indexes)
            self.texts = self.prompts_info.iloc[val_indexes, 0].tolist()
            self.prompts = tokenizer(
                self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.image_files = self.prompts_info.iloc[val_indexes, 1].tolist()

        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files):
        images = []
        for filename in tqdm(image_files):
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            images += [image]

        return images

    def __len__(self):
        return len(self.prompts['input_ids'])

    def __getitem__(self, item):
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.image_files[item]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
        if self.train:
            return {"clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0),
                    "text_input_ids": self.prompts['input_ids'][item],
                    "text_mask": self.prompts['attention_mask'][item].bool()}
        else:
            return {"image": self.transform(image), "prompt": self.texts[item]}


class ClefKandinskyDecoderDataset(Dataset):
    def __init__(self,
                 path,
                 image_processor,
                 csv_filename='prompt-gt.csv',
                 resolution=256,
                 train=True,
                 load_to_ram=False,
                 val_size = 0.025,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.resolution = resolution
        self.load_to_ram = load_to_ram
        self.image_processor = image_processor
        self.train = train
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')
        all_files_names = self.prompts_info['Filename'].unique()
        
        self.train_size = int(len(all_files_names) * (1 - val_size))
        train_indexes = self.random_state.choice(len(all_files_names), size=self.train_size, replace=False)

        if train:
            self.image_files = all_files_names[train_indexes].tolist()
        else:
            self.val_size = len(all_files_names) - self.train_size
            val_indexes = np.delete(np.arange(len(all_files_names)), train_indexes)
            self.image_files = all_files_names[val_indexes].tolist()

        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files):
        images = []
        for filename in tqdm(image_files):
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            images += [image]

        return images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.image_files[item]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
        transform = T.Compose([T.CenterCrop(min(image.size)),
                               T.Resize(size=(self.resolution, self.resolution), interpolation=InterpolationMode.BICUBIC),
                               T.ToTensor()])
        
        if self.train:
            return {"pixel_values": transform(image).to(torch.float32) / 127.5 - 1,
                    "clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)}
        else:
            return {"image": self.transform(image), "prompt": self.texts[item]}


class MyFIDDataset(Dataset):
    def __init__(self, data, tensors=True):
        super().__init__()
        
        self.tensors = tensors
        self.transform = T.Compose([T.Resize(size=(64, 64)), T.ToTensor()])
        if tensors:
            self.data = torch.cat(data, dim=0)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.tensors:
            return {'images': self.data[item]}
        else:
            return {'images': self.transform(self.data[item])}
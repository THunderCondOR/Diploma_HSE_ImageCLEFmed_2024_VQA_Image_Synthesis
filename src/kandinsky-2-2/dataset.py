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
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.load_to_ram = load_to_ram
        self.tokenizer = tokenizer
        self.train = train
        self.resolution = resolution

        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')
        
        self.prompts_info['original_index'] = self.prompts_info.index
        indexes = self.prompts_info.groupby('Filename').apply(lambda x: x.sample(1, random_state=self.random_state)).sample(2000)['original_index'].to_numpy()
        
        if train:
            indexes = np.delete(np.arange(self.prompts_info.shape[0]), indexes)
        
        self.texts = self.prompts_info.iloc[indexes, 0].tolist()
        self.prompts = tokenizer(
            self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.image_ids, self.image_files = pd.factorize(self.prompts_info.iloc[indexes, 1])

        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files, desc='Loading images'):
        images = []
        for filename in image_files:
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            transform = T.Compose([T.CenterCrop(min(image.size)),
                                   T.Resize(self.resolution),
                                   T.ToTensor()])
            images += [transform(image)]

        return images

    def __len__(self):
        return len(self.prompts['input_ids'])

    def __getitem__(self, item):
        if self.load_to_ram:
            image = self.images[self.image_ids[item]]
        else:
            filename = self.image_files[self.image_ids[item]]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            transform = T.Compose([T.RandomCrop(min(image.size)),
                                   T.Resize(self.resolution),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            image = transform(image)
        if self.train:
            return {"pixel_values": image,
                    "input_ids": self.prompts['input_ids'][item],
                    "attention_mask": self.prompts['attention_mask'][item].bool()}
        else:
            return {"images": image, "prompts": self.texts[item]}


class ClefKandinskyPriorDataset(Dataset):
    def __init__(self,
                 path,
                 tokenizer,
                 image_processor,
                 csv_filename='prompt-gt.csv',
                 resolution=64,
                 train=True,
                 load_to_ram=True,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.load_to_ram = load_to_ram
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.train = train
        self.resolution = resolution
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')
        
        self.prompts_info['original_index'] = self.prompts_info.index
        indexes = self.prompts_info.groupby('Filename').apply(lambda x: x.sample(1, random_state=self.random_state)).sample(2000)['original_index'].to_numpy()
        
        if train:
            indexes = np.delete(np.arange(self.prompts_info.shape[0]), indexes)
        
        self.texts = self.prompts_info.iloc[indexes, 0].tolist()
        self.prompts = tokenizer(
            self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.image_ids, self.image_files = pd.factorize(self.prompts_info.iloc[indexes, 1])

        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files):
        images = []
        for filename in image_files:
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            if self.train:
                images += [self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)]
            else:
                transform = T.Compose([T.CenterCrop(min(image.size)),
                                       T.Resize(self.resolution),
                                       T.ToTensor()])
                images += [transform(image)]

        return images

    def __len__(self):
        return len(self.prompts['input_ids'])

    def __getitem__(self, item):
        if self.load_to_ram:
            image = self.images[self.image_ids[item]]
        else:
            filename = self.image_files[self.image_ids[item]]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            if self.train:
                image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
            else:
                transform = T.Compose([T.RandomCrop(min(image.size)),
                                       T.Resize(self.resolution),
                                       T.ToTensor(),
                                       T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                image = transform(image)
        if self.train:
            return {"clip_pixel_values": image,
                    "text_input_ids": self.prompts['input_ids'][item],
                    "text_mask": self.prompts['attention_mask'][item].bool()}
        else:
            return {"images": image, "prompts": self.texts[item]}


class ClefKandinskyDecoderDataset(Dataset):
    def __init__(self,
                 path,
                 image_processor,
                 csv_filename='prompt-gt.csv',
                 resolution=256,
                 train=True,
                 load_to_ram=False,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.resolution = resolution
        self.load_to_ram = load_to_ram
        self.image_processor = image_processor
        self.train = train
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')

        self.image_files = self.prompts_info['Filename'].unique()
        
        if not train:
            self.prompts_info['original_index'] = self.prompts_info.index
            val_indexes = self.prompts_info.groupby('Filename').apply(lambda x: x.sample(1, random_state=self.random_state)).sample(2000)['original_index'].to_numpy()
            self.texts = self.prompts_info.iloc[val_indexes, 0].tolist()
            self.image_ids, self.image_files = pd.factorize(self.prompts_info.iloc[val_indexes, 1])
        
        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files):
        images = []
        for filename in image_files:
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            transform = T.Compose([T.CenterCrop(min(image.size)),
                                   T.Resize(self.resolution),
                                   T.ToTensor(),
                                   ])
            images += [transform(image)]

        return images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        if self.train:
            filename = self.image_files[item]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
            transform = T.Compose([T.RandomCrop(min(image.size)),
                                    T.Resize(self.resolution),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
            return {"pixel_values": transform(image).to(torch.float32),
                    "clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)}
        else:
            return {"prompts": self.texts[item]}


class AIROGSKandinskyDecoderDataset(Dataset):
    def __init__(self,
                 path,
                 image_processor,
                 csv_filename='train_labels.csv',
                 resolution=256,
                 train=True,
                 load_to_ram=False,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.resolution = resolution
        self.load_to_ram = load_to_ram
        self.image_processor = image_processor
        self.train = train
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0)

        self.image_files = self.prompts_info['challenge_id'].unique()
        
        if not train:
            clef_info = pd.read_csv('/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/data/image_clef/train/prompt-gt.csv', header=0, delimiter=';')
            clef_info['original_index'] = clef_info.index
            val_indexes = clef_info.groupby('Filename').apply(lambda x: x.sample(1, random_state=self.random_state)).sample(2000)['original_index'].to_numpy()
            self.texts = clef_info.iloc[val_indexes, 0].tolist()
            self.image_ids, self.image_files = pd.factorize(clef_info.iloc[val_indexes, 1])
        
        if load_to_ram:
            self.images = self._load_images(self.image_files)
        

    def _load_images(self, image_files):
        images = []
        for filename in image_files:
            image = Image.open(f'/home/mvchaychuk/Diploma_HSE_ImageCLEFmed_2024_VQA_Image_Synthesis/data/image_clef/train/images/{filename}').convert('RGB')
            transform = T.Compose([T.CenterCrop(min(image.size)),
                                   T.Resize(self.resolution),
                                   T.ToTensor(),
                                   ])
            images += [transform(image)]

        return images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        if self.train:
            filename = self.image_files[item]
            image = Image.open(f'{self.path}/images/{filename}.jpg').convert('RGB')
            transform = T.Compose([T.RandomCrop(min(image.size)),
                                    T.Resize(self.resolution),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
            return {"pixel_values": transform(image).to(torch.float32),
                    "clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)}
        else:
            return {"prompts": self.texts[item]}

class MyFIDDataset(Dataset):
    def __init__(self, data, fake=False):
        super().__init__()
        self.fake = fake
        self.to_tensor = T.ToTensor()
        self.data = data
        if fake:
            tensor_data = []
            for img in data:
                tensor_data += [self.to_tensor(img)]
            self.data = torch.stack(tensor_data)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {'images': self.data[item]}
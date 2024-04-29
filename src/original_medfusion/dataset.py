import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torch

from torch.utils.data import Dataset

class ClefMedfusionVAEDataset(Dataset):
    def __init__(self,
                 path,
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
        self.train = train
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')

        self.image_files = self.prompts_info['Filename'].unique()
        
        if not train:
            self.prompts_info['original_index'] = self.prompts_info.index
            val_indexes = self.prompts_info.groupby('Filename').apply(lambda x: x.sample(1, random_state=self.random_state)).sample(9)['original_index'].to_numpy()
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
        filename = self.image_files[item]
        image = Image.open(f'{self.path}/images/{filename}').convert('RGB')
        transform = T.Compose([T.CenterCrop(min(image.size)),
                               T.Resize(self.resolution),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        return {"pixel_values": transform(image).to(torch.float32)}
    

class ClefMedfusionUnetDataset(Dataset):
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
            return {"images": image, "input_ids": self.prompts['input_ids'][item], "texts": self.texts[item]}
        

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
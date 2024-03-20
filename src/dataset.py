import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset


class ClefPriorDataset(Dataset):
    def __init__(self,
                 path,
                 tokenizer,
                 image_processor,
                 csv_filename='prompt-gt.csv',
                 train=True,
                 load_to_ram=False,
                 val_size = 0.15,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.path = path
        self.load_to_ram = load_to_ram
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        self.to_tensor = T.ToTensor()
        
        self.prompts_info = pd.read_csv(self.path + f'/{csv_filename}', header=0, delimiter=';')
        
        self.train_size = int(self.prompts_info.shape[0] * (1 - val_size))
        train_indexes = self.random_state.choice(self.prompts_info.shape[0], size=self.train_size, replace=False)

        if train:
            self.prompts = tokenizer(
                self.prompts_info.iloc[train_indexes, 0].tolist(), max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            self.image_files = self.prompts_info.iloc[train_indexes, 1].tolist()
        else:
            self.val_size = self.prompts_info.shape[0] - self.train_size
            val_indexes = np.delete(np.arange(self.prompts_info.shape[0]), train_indexes)
            self.prompts = tokenizer(
                self.prompts_info.iloc[val_indexes, 0].tolist(), max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
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
        return len(self.prompts)

    def __getitem__(self, item):
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.image_files[item]
            image = Image.open(f'{self.path}/images/{filename}').convert('RGB')

        return {"clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values,
                "text_input_ids": self.prompts['input_ids'][item],
                "text_mask": self.prompts['attention_mask'][item].bool()}
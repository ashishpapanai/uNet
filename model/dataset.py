import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        #rint(self.images)
        #print(img_path)
        mask_path = os.path.join(
            self.mask_dir, self.images[idx].replace(".jpg", "_mask.gif"))
        #print(mask_path)
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert(
            'L'), dtype=np.float32)  # convert to grayscale
        mask[mask == 255.0] = 1.0

        if self.transform != None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
    


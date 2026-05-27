import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

class ImageMeasurementDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Numerical feature columns
        self.feature_cols = [
            'gender', 'side', 'len_foot', 'len_arch', 'len_mm_med', 'len_mm_lat',
            'len_latdors', 'len_arch_lat', 'len_arch_med', 'wid_fore', 'wid_heel',
            'wid_instep', 'wid_meta', 'size_eu', 'arch_med', 'arch_reg', 'arch_lat',
            'arch_index', 'pron_angle'
        ]
        self.label_col = 'GT_label_arch'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract image basename
        basename = row['pdf'].replace('_Report.pdf', '')
        image_path = os.path.join(self.image_dir, f"{basename}_Orig.png")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Extract numerical features and label
        features = torch.tensor(row[self.feature_cols].astype(float).values, dtype=torch.float32)
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        
        return image, features, label




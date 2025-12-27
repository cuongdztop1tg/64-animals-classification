from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class ImageDataLoader:
    def __init__(self, train_dir: str, test_dir: str, batch_size: int=32, img_size=(128, 128), num_workers: int=2):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        
        self.train_transform = transforms.Compose([
            transforms.Resize(self.img_size),

            # Data Augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),

            # Tensor and Normalize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_size),

            # Tensor and Normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_loader(self):
        train_dataset = ImageFolder(
            root=self.train_dir,
            transform=self.train_transform
        )

        test_dataset = ImageFolder(
            root=self.test_dir,
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        return train_loader, test_loader
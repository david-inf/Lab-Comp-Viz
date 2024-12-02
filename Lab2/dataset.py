import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as T
# TODO: import torchvision.transforms.v2 as v2


class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        # full dataset (X, y): X=[N, 32, 32, 3], y=[N]
        self.dataset = datasets.CIFAR10(
            root="../../data", train=True, download=True  # (X, y)
        )
        self.quick_preprocess = T.Compose([
            T.ToTensor()
        ])
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # number of samples

    def __getitem__(self, idx):
        """
        Returns two views of the original image at `idx`.
        Augmentation will be performed if provided.
        """
        image, label = self.dataset[idx]

        # if isinstance(image, str):
        #     image = read_image(image)
        # else:
        #     # da un numpy.ndarray trasforma in una PIL Image
        #     # per le trasformazioni voglio PIL Image come input
        #     from PIL import Image
        #     image = Image.fromarray(image.astype('uint8'), 'RGB')

        if self.transform:
            # in input vuole una PIL Image
            image1 = self.transform(image)  # view1
            image2 = self.transform(image)  # view2
        else:
            image1 = self.quick_preprocess(image)
            image2 = self.quick_preprocess(image)

        return image1, image2, label


class AugmentedImageDataset(CustomImageDataset):
    """ Augmentation pipeline over CIFAR10 dataset """
    def __init__(self):
        super().__init__()

        s, size = 1, 32
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.pipeline = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image1 = self.pipeline(image)
        image2 = self.pipeline(image)

        return image1, image2, label


class MakeDataLoaders():
    def __init__(self, data, batch_size, num_workers=2):
        # data: Dataset object
        self.loader = DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

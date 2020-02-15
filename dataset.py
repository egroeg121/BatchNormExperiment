import torchvision
import torch
import torchvision.transforms as transforms
import logging

class loaders():
    def __init__(self,data_dir = './data',num_workers = 0,batch_size=1):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    def train_loader(self):
        transforms = self.get_transforms(mode='train')

        logging.debug(f"Loading train loader")

        train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms,)

        return torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,)


    def test_loader(self):
        transforms = self.get_transforms(mode='test')

        test_set = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms)

        logging.debug(f"Loading train loader")

        return torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,)



    def get_transforms(self,mode):
        transform_list = []

        # Transforms for special modes
        if mode=='train':
            pass
        elif mode=='test':
            pass
        elif mode=='val':
            pass

        # Default Transforms
        transform_list.extend([
            transforms.transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

        return transforms.Compose(transform_list)

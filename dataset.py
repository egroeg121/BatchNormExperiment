import torchvision
import torch
import logging

class loaders():
    def __init__(self,transforms,data_dir = './data',num_workers = 0,batch_size=1):

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainset  = torchvision.datasets.CIFAR10(root=data_dir,
                                                      train=True,
                                                      download=True,
                                                      transform=transforms
                                                      )
        self.testset = torchvision.datasets.CIFAR10(root=data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform=transforms
                                                    )


    def get_loader(self,mode):
        if mode=='train':
            return self.train_loader(self.batch_size,self.num_workers)
        elif mode=='test':
            return self.test_loader()
        else:
            print("Incorrect Mode")
            exit()

    def train_loader(self,batch_size,num_workers):
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True,
                                           )
    def test(self,batch_size,num_workers):
        return torch.utils.data.DataLoader(self.testset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False,
                                           )
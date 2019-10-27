import torch
from torch import nn
import numpy as np
import torchvision
import argparse

from models import BasicConvNet
import dataset

def main():
    loaders = dataset.loaders(num_workers=2,batch_size=2)

    model = BasicConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epoch = 10

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loaders.train_loader(), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



    print('Finished Training')


def train_loop():
    pass


if __name__ == "__main__":
    main()
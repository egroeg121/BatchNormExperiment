import logging
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models import BasicConvNet
import dataset

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Starting")
    loaders = dataset.loaders(num_workers=2,batch_size=2)
    train_loader = loaders.train_loader()

    tag = datetime.now().strftime("%Y_%m_%d__%H_%M")
    tb = SummaryWriter(log_dir="./tb/"+tag)

    model = BasicConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epoch = 10
    logging.info(f"Training for {num_epoch} epochs.")
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        logging.debug(f"Start epoch {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss
            logging.debug(f"Loss {i}/{len(train_loader)} : {loss} ")
            tb.add_scalar(f"Loss/Epoch_{epoch}",loss,i)

            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch} loss: {running_loss/len(train_loader)}")
        tb.add_scalar(f"Loss/Cross_Entropy",running_loss/len(train_loader),epoch)



    print('Finished Training')


def train_loop():
    pass


if __name__ == "__main__":
    main()
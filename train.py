import logging
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models import BasicConvNet
import dataset

logging.basicConfig(level=logging.INFO)

num_epochs = 50
batch_size = 256
cuda = True

tag = datetime.now().strftime("%Y_%m_%d__%H_%M")
tb = SummaryWriter(log_dir="./tb/"+tag)

def train_epoch(model,train_loader,loss_func, optimizer, epoch: int):
    running_loss = 0.0
    model = model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        if cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        running_loss += loss
        logging.debug(f"Train Loss {i}/{len(train_loader)} : {loss} ")

        loss.backward()
        optimizer.step()

    logging.info(f"Train epoch {epoch} loss: {running_loss / len(train_loader)}")
    tb.add_scalar(f"Train/Cross_Entropy", running_loss / len(train_loader), epoch)

def val_epoch(model,test_loader,loss_func, epoch: int):
    model.eval()
    val_loss = 0.0
    running_acc = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_acc += (predicted == labels).sum().item()
            val_loss += loss
            logging.debug(f"Val Loss {i}/{len(test_loader)} : {loss} ")

        logging.info(f"Val epoch {epoch} loss: {val_loss / len(test_loader)}")
        tb.add_scalar(f"Val/Cross_Entropy", val_loss / len(test_loader), epoch)

        logging.info(f"Val accuracy Epoch {epoch}: {running_acc / (batch_size*len(test_loader))}")
        tb.add_scalar(f"Val/Accuracy", running_acc / (batch_size*len(test_loader)), epoch)

def train(model: nn.Module, train_dataloader,test_dataloader, loss_func, optimizer, max_epoch: int  = 100):
    logging.info(f"Training for {max_epoch} epochs.")
    for epoch in range(max_epoch):
        logging.debug(f"Start epoch {epoch}")
        train_epoch(model,train_dataloader,loss_func,optimizer,epoch)
        val_epoch(model,test_dataloader,loss_func,epoch)

def main(cuda=True):
    logging.info("Starting")
    loaders = dataset.loaders(num_workers=4, batch_size=batch_size)
    train_loader = loaders.train_loader()
    test_loader = loaders.test_loader()



    model = BasicConvNet()
    loss_func = nn.CrossEntropyLoss()
    if cuda and torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

    train(model, train_loader,test_loader, loss_func, optimizer, max_epoch = num_epochs)





    print('Finished Training')




if __name__ == "__main__":
    main()
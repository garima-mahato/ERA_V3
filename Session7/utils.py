import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

def set_device():
    """set device as with/without cuda based on availability

    Returns:
        string: returns "cuda" f cuda is available else "cpu"
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

##################### Visualtion Utilities #########################

def view_data(data_loader):
    """Gives a visualization of sample data taken from the pytorch data loader

    Args:
        data_loader (torch.utils.data.DataLoader): Wraps an iterable around the Dataset to enable easy access to the samples.
    """
    batch_data, batch_label = next(iter(data_loader)) 
    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def vis_train_test_comp_graphs(train_losses, train_acc, test_losses, test_acc):
    """Creates graphs for Training Loss and Accuracy, Test Loss and Accuracy.

    Args:
        train_losses (list): training loss
        train_acc (list): training accuracy
        test_losses (list): test loss
        test_acc (list): test accuracy
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

###################### Train and Test Functionalities #####################

from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
    """Counts the number of correct predictions made, i.e. prediction=ground truth

    Args:
        pPrediction (tensor): prediction made by the model
        pLabels (tensor): ground truth

    Returns:
        int: count of correct predictions
    """
    return pPrediction.argmax(dim=1, keepdim=True).eq(pLabels.view_as(pPrediction.argmax(dim=1, keepdim=True))).sum().item()

def train(model, device, train_loader, optimizer, train_acc, train_losses):
    """Trains the model

    Args:
        model (torch.nn.Module): pytorch model
        device (_type_): cuda or cpu
        train_loader (torch.utils.data.DataLoader): data iterator on training data
        optimizer (torch.optim): optimizer function
        train_acc (list): stores accuracy of each batch
        train_losses (list): stores loss of each batch
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, test_acc, test_losses):
    """_summary_

    Args:
        model (torch.nn.Module): pytorch model
        device (_type_): cuda or cpu
        test_loader (torch.utils.data.DataLoader): data iterator on test data
        test_acc (list): stores accuracy of each batch
        test_losses (list): stores loss of each batch
    """
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import math
import numpy
import glob
import torchvision
import matplotlib.pyplot as plt
import io
import glob
import os
from shutil import move, copy
from os.path import join
from os import listdir, rmdir
import time
import numpy as np
import random

###############################################################

def save_network(path, model_file_name, model, train_losses, test_losses, train_acc, test_acc):
    print("\n Saving trained model and parameters...")
    torch.save(model.state_dict(), path+"/"+model_file_name+".pth")
    # save train and test losses and accuracies
    train_test_data = {"Training Loss": train_losses, "Test Loss": test_losses, "Training Accuracy": train_acc, "Test Accuracy": test_acc}
    torch.save(train_test_data, path+"/"+model_file_name+"_train_test_params.pt")

def load_network(path, model_file_name, model, device):
    print("\n Loading trained model...")
    model = model.to(device)
    model.load_state_dict(torch.load(path+"/"+model_file_name+".pth"))


###############################################################

def find_custom_dataset_mean_std(DATA_PATH, cuda):
  num_of_inp_channels = 3
  simple_transforms = transforms.Compose([
                                          transforms.ToTensor()
                                        ])
  exp = datasets.ImageFolder(DATA_PATH+"/train_set", transform=simple_transforms)
  dataloader_args = dict(shuffle=True, batch_size=256, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
  loader = torch.utils.data.DataLoader(exp, **dataloader_args)

  mean = 0.0
  for images, _ in loader:
      batch_samples = images.size(0) 
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
  mean = mean / len(loader.dataset)

  var = 0.0
  for images, _ in loader:
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      var += ((images - mean.unsqueeze(1))**2).sum([0,2])
  std = torch.sqrt(var / (len(loader.dataset)*224*224))

  # print("means: {}".format(mean))
  # print("stdevs: {}".format(std))
  # print('transforms.Normalize(mean = {}, std = {})'.format(mean, std))

  return tuple(mean.numpy().astype(numpy.float32)), tuple(std.numpy().astype(numpy.float32))

def find_cifar10_normalization_values(data_path='./data'):
  num_of_inp_channels = 3
  simple_transforms = transforms.Compose([
                                        transforms.ToTensor()
                                       ])
  exp = datasets.CIFAR10(data_path, train=True, download=True, transform=simple_transforms)
  data = exp.data
  data = data.astype(numpy.float32)/255
  means = ()
  stdevs = ()
  for i in range(num_of_inp_channels):
      pixels = data[:,:,:,i].ravel()
      means = means +(round(numpy.mean(pixels)),)
      stdevs = stdevs +(numpy.std(pixels),)

  print("means: {}".format(means))
  print("stdevs: {}".format(stdevs))
  print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

  return means, stdevs

# visualize accuracy and loss graph
def visualize_graph(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def visualize_save_train_vs_test_graph(EPOCHS, dict_list, title, xlabel, ylabel, PATH, name="fig"):
  plt.figure(figsize=(20,10))
  #epochs = range(1,EPOCHS+1)
  for label, item in dict_list.items():
    x = numpy.linspace(1, EPOCHS+1, len(item))
    plt.plot(x, item, label=label)
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.savefig(PATH+"/"+name+".png")

def set_device():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  return device

# view and save comparison graph of cal accuracy and loss
def visualize_save_comparison_graph(EPOCHS, dict_list, title, xlabel, ylabel, PATH, name="fig"):
  plt.figure(figsize=(20,10))
  epochs = range(1,EPOCHS+1)
  for label, item in dict_list.items():
    plt.plot(epochs, item, label=label)
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.savefig(PATH+"/visualization/"+name+".png")

# view and save misclassified images
def classify_images(model, test_loader, device, max_imgs=25):
  misclassified_imgs = []
  correct_imgs = []
    
  with torch.no_grad():
    ind = 0
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

      misclassified_imgs_pred = pred[pred.eq(target.view_as(pred))==False]
      misclassified_imgs_indexes = (pred.eq(target.view_as(pred))==False).nonzero()[:,0]
      for mis_ind in misclassified_imgs_indexes:
        if len(misclassified_imgs) < max_imgs:
          misclassified_imgs.append({
              "target": target[mis_ind].cpu().numpy(),
              "pred": pred[mis_ind][0].cpu().numpy(),
              "img": data[mis_ind]
          })
    
	#for data, target in test_loader:
      correct_imgs_pred = pred[pred.eq(target.view_as(pred))==True]
      correct_imgs_indexes = (pred.eq(target.view_as(pred))==True).nonzero()[:,0]
      for ind in correct_imgs_indexes:
        if len(correct_imgs) < max_imgs:
          correct_imgs.append({
              "target": target[ind].cpu().numpy(),
              "pred": pred[ind][0].cpu().numpy(),
              "img": data[ind]
          })
      
  return misclassified_imgs, correct_imgs

def plot_images(images, PATH, name="fig", sub_folder_name="/visualization", is_cifar10 = True, labels_list=None):
  cols = 2
  rows = math.ceil(len(images) / cols)
  fig = plt.figure(figsize=(20,10))

  for i in range(len(images)):
    img = denormalize(images[i]["img"])
    plt.subplot(rows,cols,i+1)
    plt.tight_layout()
    plt.imshow(numpy.transpose(img.cpu().numpy(), (1, 2, 0)), cmap='gray', interpolation='none')
    if is_cifar10:
      CIFAR10_CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
      plt.title(f"{i+1}) Ground Truth: {CIFAR10_CLASS_LABELS[images[i]['target']]},\n Prediction: {CIFAR10_CLASS_LABELS[images[i]['pred']]}")
    elif labels_list is not None:
      plt.title(f"{i+1}) Ground Truth: {labels_list[images[i]['target']]},\n Prediction: {labels_list[images[i]['pred']]}")
    else:
      plt.title(f"{i+1}) Ground Truth: {images[i]['target']},\n Prediction: {images[i]['pred']}")
    plt.xticks([])
    plt.yticks([])
  plt.savefig(PATH+sub_folder_name+"/"+str(name)+".png")

def show_save_misclassified_images(model, test_loader, device, PATH, name="fig", max_misclassified_imgs=25, is_cifar10 = True, labels_list=None):
  misclassified_imgs, _ = classify_images(model, test_loader, device, max_misclassified_imgs)
  plot_images(misclassified_imgs, PATH, name, is_cifar10 = is_cifar10, labels_list=labels_list)

def show_save_correctly_classified_images(model, test_loader, device, PATH, name="fig", max_correctly_classified_images_imgs=25, is_cifar10 = True, labels_list=None):
  _, correctly_classified_images = classify_images(model, test_loader, device, max_correctly_classified_images_imgs)
  plot_images(correctly_classified_images, PATH, name, is_cifar10 = is_cifar10, labels_list=labels_list)

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
  single_img = False
  if tensor.ndimension() == 3:
    single_img = True
    tensor = tensor[None,:,:,:]

  if not tensor.ndimension() == 4:
    raise TypeError('tensor should be 4D')

  mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
  std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
  ret = tensor.mul(std).add(mean)
  return ret[0] if single_img else ret

def imshow(img):
	img = denormalize(img)
	npimg = img.numpy()
	plt.imshow(numpy.transpose(npimg, (1, 2, 0)))

def show_sample_images(train_loader, labels_list, num_imgs=5):
	# get some random training images
	dataiter = iter(train_loader)
	images, labels = dataiter.next()
	# show images
	imshow(torchvision.utils.make_grid(images[:num_imgs]))
	# print labels
	print(' '.join('%5s' % labels_list[labels[j]] for j in range(num_imgs)))

def class_to_label_mapping(DATA_PATH):
  # find class names
  train_paths = glob.glob(DATA_PATH+'/train_set/*')
  class_list = []
  for path in train_paths:
    folder = path.split('/')[-1].split('\\')[-1]
    class_list.append(folder)

  labels_list = []
  with open(DATA_PATH+'/words.txt', 'r') as f:
    data = f.read()

  for i in (data.splitlines()):
    ind = i.split('\t')[0]
    if ind in class_list:
      label = i.split('\t')[1]
      if ',' in label:
        label = label.split(',')[0] + ",etc"
      labels_list.append(label)
  
  return labels_list

def merge_split_data(imagenet_root):
  target_folder = imagenet_root+"/val/"
  dest_folder = imagenet_root+"/train/"

  val_dict = {}
  with open(imagenet_root+'/val/val_annotations.txt','r') as f:
      for line in f.readlines():
          split_line = line.split('\t')
          val_dict[split_line[0]] = split_line[1]

  paths = glob.glob(imagenet_root+'/val/images/*')

  for path in paths:
      file = path.split('/')[-1].split('\\')[-1]
      folder = val_dict[file]
      dest = dest_folder + str(folder) + '/images/' + str(file)
      move(path, dest)

  target_folder = imagenet_root+'/train/'
  train_folder = imagenet_root+'/train_set/'
  test_folder = imagenet_root+'/test_set/'

  os.mkdir(train_folder)
  os.mkdir(test_folder)

  paths = glob.glob(imagenet_root+'/train/*')

  for path in paths:
      folder = path.split('/')[-1].split('\\')[-1]
      source = target_folder + str(folder+'/images/')
      train_dest = train_folder + str(folder+'/')
      test_dest = test_folder + str(folder+'/')
      os.mkdir(train_dest)
      os.mkdir(test_dest)
      images = glob.glob(source+str('*'))
      
      # shuffle
      random.shuffle(images)
      
      test_imgs = images[:165].copy()
      train_imgs = images[165:].copy()
      
      for image in test_imgs:
          file = image.split('/')[-1].split('\\')[-1]
          dest = test_dest + str(file)
          move(image, dest)
      
      for image in train_imgs:
          file = image.split('/')[-1].split('\\')[-1]
          dest = train_dest + str(file)
          move(image, dest)

#########################################################

# def set_device():
#     """set device as with/without cuda based on availability

#     Returns:
#         string: returns "cuda" f cuda is available else "cpu"
#     """
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     return device

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


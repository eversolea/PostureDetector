#Resources used:
#Took inspiration from my homework assignment 2
#Also used ideas from the tutorial hosed on the website: https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/
#such as how to create my own custom dataset class and a convienent way to store labels, bounding box, and image data all in a single file that I can load into memory.
#And many thanks to the help I recieved from Prof. Lombardi via Slack



import argparse
from asyncio.windows_events import NULL
from collections import defaultdict

import numpy as np

from PIL import Image

import matplotlib
matplotlib.use('Agg') # use Agg backend so that we don't require an X server
from matplotlib import pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

import os
from imutils import paths

import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# We're not using the GPU.
use_gpu = False

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

default_train_transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize rescales and shifts the data so that it has a zero mean 
    # and unit variance. This reduces bias and makes it easier to learn!
    # The values here are the mean and variance of our inputs.
    # This will change the input images to be centered at 0 and be 
    # between -1 and 1.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

default_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])







class CustomTensorDataset(Dataset):
    # initialize the constructor
    def __init__(self, tensors, transforms):


        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
            # grab the image, label, and its bounding box coordinates
            image = self.tensors[0][index]
            label = self.tensors[1][index]
            bbox = self.tensors[2][index]
            # transpose the image such that its channel dimension becomes
            # the leading one
            image = image.permute(2, 0, 1)

            # check to see if we have any image transformations to apply
            # and if so, apply them
            if self.transforms:
                pilImg = ToPILImage()(image.to('cpu'))
                image = self.transforms(pilImg) #TODO: Message=pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>


            # return a tuple of the images, labels, and bounding
            # box coordinates
            return (image, label, bbox)

    def __len__(self):
            # return the size of the dataset
            return self.tensors[0].size(0)


def get_train_loader(batch_size, trainTensor, transform=default_train_transform):


    trainset = CustomTensorDataset(trainTensor, transform)


    return torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_test_loader(batch_size, testTensor, transform=default_test_transform):
    
    testset = CustomTensorDataset(testTensor, transform)

    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)


# The function we'll call to train the network each epoch
def train(net, loader, optimizer, criterion, epoch, use_gpu=False):
    running_loss = 0.0
    total_loss = 0.0

    # Send the network to the correct device
    if use_gpu:
        net = net.cuda()
    else:
        net = net.cpu()

    # tqdm is a useful package for adding a progress bar to your loops
    pbar = tqdm(loader)
    for i, data in enumerate(pbar):
        inputs, labels, bbox = data

        # If we're using the GPU, send the data to the GPU
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()  # Set the gradients of the parameters to zero.
        outputs = net(inputs)  # Forward pass (send the images through the network)
        loss = criterion(outputs, labels)  # Compute the loss w.r.t the labels.
        loss.backward()  # Backward pass (compute gradients).
        optimizer.step()  # Use the gradients to update the weights of the network.

        running_loss += loss.item()
        total_loss += loss.item()
        pbar.set_description(f"[epoch {epoch+1}] loss = {running_loss/(i+1):.03f}")
    
    average_loss = total_loss / (i + 1)
    tqdm.write(f"Epoch {epoch} summary -- loss = {average_loss:.03f}")
    
    return average_loss


def show_hard_negatives(hard_negatives, label, nrow=10):
    """Visualizes hard negatives"""
    grid = make_grid([(im+1)/2 for im, score in hard_negatives[label]], 
                     nrow=nrow, padding=1)
    grid = grid.permute(1, 2, 0).mul(255).byte().numpy()
    #ipd.display(Image.fromarray((grid)))


# The function we'll call to test the network
def test(net, loader, tag='', use_gpu=False, num_hard_negatives=10):
    correct = 0
    total = 0

    # Send the network to the correct device
    net = net.cuda() if use_gpu else net.cpu()

    # Compute the overall accuracy of the network
    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {tag}"):
            images, labels = data

            # If we're using the GPU, send the data to the GPU
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            # Forward pass (send the images through the network)
            outputs = net(images)

            # Take the output of the network, and extract the index 
            # of the largest prediction for each example
            _, predicted = torch.max(outputs.data, 1)

            # Count the number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    average_accuracy = correct/total
    tqdm.write(f'{tag} accuracy of the network: {100*average_accuracy:.02f}%')

    # Repeat above, but estimate the testing accuracy for each of the labels
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    hard_negatives = defaultdict(list)
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            predicted_scores, predicted_labels = torch.max(outputs, 1)
            correct_mask = (predicted_labels == labels).squeeze()
            incorrect_mask = ~correct_mask
            unique_labels, unique_counts = torch.unique(labels, return_counts=True)
            for l, c in zip(unique_labels, unique_counts):
                l = l.item()
                label_mask = (labels == l)
                predicted_mask = (predicted_labels == l)
                # This keeps track of the most hardest negatives
                # i.e. mistakes with the highest confidence.
                hard_negative_mask = (~correct_mask & predicted_mask)
                if hard_negative_mask.sum() > 0:
                    hard_negatives[l].extend([
                        (im, score.item()) 
                        for im, score in zip(images[hard_negative_mask], 
                                             predicted_scores[hard_negative_mask])])
                    hard_negatives[l].sort(key=lambda x: x[1], reverse=True)
                    hard_negatives[l] = hard_negatives[l][:num_hard_negatives]
                class_correct[l] += (correct_mask & label_mask).sum()
                class_total[l] += c


    for i in range(10):
        tqdm.write(f'{tag} accuracy of {classes[i]} = {100*class_correct[i]/class_total[i]:.02f}%')
        #if len(hard_negatives[i]) > 0:
        #    print(f'Hard negatives for {classes[i]}')
        #    show_hard_negatives(hard_negatives, i, nrow=10)
        #else:
        #    print("There were no hard negatives--perhaps the model got 0% accuracy?")

    
    return average_accuracy
  
def train_network(net, 
                  lr, 
                  epochs, 
                  batch_size, 
                  criterion=None,
                  optimizer=None,
                  lr_func=None,
                  train_transform=default_train_transform, 
                  eval_interval=10,
                  use_gpu=use_gpu): 
    assert optimizer is not None

    # Initialize the loss function
    if criterion is None:
        # Note that CrossEntropyLoss has the Softmax built in!
        # This is good for numerical stability. 
        # Read: https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

    # Initialize the data loaders

    print("[INFO] loading dataset...")
    data = []
    labels = []
    bboxes = []
    imagePaths = []

    # define the base path to the input dataset and then use it to derive
    # the path to the input images and annotation CSV files
    BASE_PATH = "dataset"
    IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
    ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])
    # define the path to the base output directory
    BASE_OUTPUT = "output"
    # define the path to the output model, label encoder, plots output
    # directory, and testing image paths
    MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
    LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
    PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
    TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

    # loop over all CSV files in the annotations directory
    for csvPath in paths.list_files(ANNOTS_PATH, validExts=(".csv")):
        # load the contents of the current CSV annotations file
        rows = open(csvPath).read().strip().split("\n")
        # loop over the rows
        for row in rows:
            # break the row into the filename, bounding box coordinates,
            # and class label
            row = row.split(",")
            (filename, startX, startY, endX, endY, label) = row
            # derive the path to the input image, load the image (in
            # OpenCV format), and grab its dimensions
            imagePath = os.path.sep.join([IMAGES_PATH, label,
                filename])
            image = cv2.imread(imagePath)
            (h, w) = image.shape[:2]
            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
            startX = float(startX) / w
            startY = float(startY) / h
            endX = float(endX) / w
            endY = float(endY) / h
            # load the image and preprocess it
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            # update our list of data, class labels, bounding boxes, and
            # image paths
            data.append(image)
            labels.append(label)
            bboxes.append((startX, startY, endX, endY))
            imagePaths.append(imagePath)

    # convert the data, class labels, bounding boxes, and image paths to
    # NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)
    # perform label encoding on the labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.20, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    (trainPaths, testPaths) = split[6:]

    # convert NumPy arrays to PyTorch tensors
    (trainImages, testImages) = torch.tensor(trainImages),\
        torch.tensor(testImages)
    (trainLabels, testLabels) = torch.tensor(trainLabels),\
        torch.tensor(testLabels)
    (trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes),\
        torch.tensor(testBBoxes)
        
    trainTensor = (trainImages, trainLabels, trainBBoxes)
    testTensor = (testImages, testLabels, testBBoxes)

    train_loader = get_train_loader(batch_size, trainTensor, transform=train_transform)
    test_loader = get_test_loader(batch_size, testTensor)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        if lr_func is not None:
            lr_func(optimizer, epoch, lr)

        train_loss = train(net, train_loader, optimizer, criterion, epoch, use_gpu=use_gpu)
        train_losses.append(train_loss)

        # Evaluate the model every `eval_interval` epochs.
        if (epoch + 1) % eval_interval == 0:
            print(f"Evaluating epoch {epoch+1}")
            train_accuracy = test(net, train_loader, 'Train', use_gpu=use_gpu)
            test_accuracy = test(net, test_loader, 'Test', use_gpu=use_gpu)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
    
    return train_losses, train_accuracies, test_accuracies
    

# A function to plot the losses over time
def plot_results(train_losses, train_accuracies, test_accuracies):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(train_accuracies)
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    
    axes[2].plot(test_accuracies)
    axes[2].set_title('Testing Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')


# Training a classifier using only one fully connected Layer
# Implement a model to classify the images from Cifar-10 into ten categories
# using just one fully connected layer (remember that fully connected layers
# are called Linear in PyTorch).
#
# If you are new to PyTorch you may want to check out the tutorial on MNIST
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
#
# Fill in the code for LinearNet here.
#
# Hints:
#
# Note that nn.CrossEntropyLoss has the Softmax built in for numerical
# stability. This means that the output layer of your network should be linear
# and not contain a Softmax. You can read more about it here
#
# You can use the view() function to flatten your input image to a vector e.g.,
# if x is a (100,3,4,4) tensor then x.view(-1, 3*4*4) will flatten it into a
# vector of size 48.
#
# The images in CIFAR-10 are 32x32.
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define model here
        self.dense = nn.Linear(32*32*3, 10 )  

    def forward(self, x):
        # TODO: Implement forward pass for LinearNet
        # The size of the matx in this relu actiation function is:
        #      [32 * 32 * 32 * 3, 32]. I think this is 32 copies of 3 channels of 32x32 pixels?
        
        # x = nn.functional.relu(self.fc1(x))
        print(x.size())
        x = x.view(-1,32*32*3)
        x = self.dense(x)
        
        return x


# Training a classifier using multiple fully connected layers
# Implement a model for the same classification task using multiple fully
# connected layers.
#
# Start with a fully connected layer that maps the data from image size 
# (32 * 32 * 3) to a vector of size 120, followed by another fully connected 
# that reduces the size to 84 and finally a layer that maps the vector of size 
# 84 to 10 classes.
#
# Use any activation you want.
#
# Fill in the code for MLPNet below.
class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for MLPNet
        
        #MLP with activation functions - Architecture 1
        x = x.view(-1,32*32*3)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x) #No activation function is just x = self.fc3(x)
        
        #MLP with no activation functions - Architecture 2
        #x = x.view(-1,32*32*3)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x) #No activation function is just x = self.fc3(x)
        return x


# Training a classifier using convolutions
# Implement a model using convolutional, pooling and fully connected layers.
#
# You are free to choose any parameters for these layers (we would like you to 
# play around with some values).
#
# Fill in the code for ConvNet below. Explain why you have chosen these layers 
# and how they affected the performance. Analyze the behavior of your model.
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define model here

        #Conv blocks:
        self.convB1 = nn.Conv2d(3, 6, 5, padding=2) #perform local constrast norm with it
        self.poolB1 = nn.MaxPool2d(2)
        self.convB2 = nn.Conv2d(6, 16, 5, padding=2) #perform local constrast norm with it
        self.poolB2 = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(16, 20, 5, padding = 2)
        self.conv2 = nn.Conv2d(20, 30, 5, padding = 2)
        self.conv3 = nn.Conv2d(30, 34, 5, padding = 2)
        self.poolB3 = nn.MaxPool2d(2, padding=1)

        self.fc1 = nn.Linear(16*34, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        #Apply Dropout to reduce Overfitting
        self.dropout = nn.Dropout(0.2)

        self.convBaseline1 = nn.Conv2d(3, 6, 5) #perform local constrast norm with it
        self.convBaseline2 = nn.Conv2d(6, 16, 5) #perform local constrast norm with it
        self.convNoReLu = nn.Conv2d(16, 20, 5)
        self.linear1 = nn.Linear(20 * 20 * 20, 10)

    def forward(self, x):

        #Convnet Baseline - Architecture 1
        x = nn.functional.relu(self.convBaseline1(x))
        x = nn.functional.relu(self.convBaseline2(x))
        x = self.convNoReLu(x)
        x = x.view(-1,20 * 20 * 20)
        x = self.linear1(x)
        
        
        #Convnet with third convolution layer removed - Architecture 2
        #x = nn.functional.relu(self.convBaseline1(x))
        #x = nn.functional.relu(self.convBaseline2(x))
        #x = x.view(-1,24 * 24 * 24)
        #x = self.linear1(x)     



        #Attempted really interesting extra credit idea - Architecture 3
        #Implementing full image classification nn model
        #x =  nn.functional.max_pool2d(nn.functional.relu(self.convB1(x)), 2)
        #x =  nn.functional.max_pool2d(nn.functional.relu(self.convB2(x)), 2)
        #x = nn.functional.relu(self.conv1(x))
        #x = nn.functional.relu(self.conv2(x))
        #x =  nn.functional.max_pool2d(nn.functional.relu(self.conv3(x)), 2)

        #x = x.view(-1,16*34)
        #x = self.dropout(x)

        #x = nn.functional.relu(self.fc1(x))
        #x = self.dropout(x)
        #x = nn.functional.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = self.fc3(x)


        return x


# During training it is often useful to reduce learning rate as the training
# progresses.
#
# Fill in set_learning_rate below to scale the learning rate by 
# 0.1 (reduce by 90%) every 30 epochs and observe the behavior of network for
# 90 epochs.
def set_learning_rate(optimizer, epoch, base_lr):
    # TODO: adjust learning rate here
    ls = base_lr
    if epoch % 30 == 0 and epoch != 0:
        lr = base_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Most of the popular computer vision datasets have tens of thousands of images.
#
# Cifar-10 is a dataset of 60000 32x32 colour images in 10 classes, which can be
# relatively small in compare to ImageNet which has 1M images.
#
# The more the number of parameters is, the more likely our model is to overfit 
# to the small dataset. As you might have already faced this issue while 
# training the ConvNet, after some iterations the training accuracy reaches its 
# maximum (saturates) while the test accuracy is still relatively low.
#
# To solve this problem, we use the data augmentation to help the network avoid 
# overfitting.
#
# Add data transformations in to the class below and compare the results. You 
# are free to use any type and any number of data augmentation techniques.
# 
# Just be aware that data augmentation should just happen during training phase.
custom_train_transform = transforms.Compose([
    # TODO: Add data augmentations here

    # transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    transforms.Grayscale(3),
    # You can find a list of transforms here:
    # https://pytorch.org/vision/stable/transforms.html
    # https://pytorch.org/vision/stable/auto_examples/plot_transforms.html
    transforms.ToTensor(), 
    # Normalize rescales and shifts the data so that it has a zero mean 
    # and unit variance. This reduces bias and makes it easier to learn!
    # The values here are the mean and variance of our inputs.
    # This will change the input images to be centered at 0 and be 
    # between -1 and 1.
    
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class MSELossClassification(nn.Module):
    def forward(self, output, labels):
        one_hot_encoded_labels = \
            torch.nn.functional.one_hot(labels, num_classes=output.shape[1]).float()
        return nn.functional.mse_loss(output, one_hot_encoded_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains and evaluates a neural network.')
    parser.add_argument('output', type=str, help='Output file path')
    parser.add_argument('--optim', type=str, default='SGDmomentum', help='Optimizer used to train')
    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--mse', action='store_true', help='Use MSE loss')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Size of each minibatch')


    args = parser.parse_args()
    
    net = ConvNet()

        
    if args.optim == "SGDmomentum":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        print("Unknown optimizer")
        raise

    if args.aug:
        train_transform = custom_train_transform
    else:
        train_transform = default_train_transform

    if args.mse:
        criterion = MSELossClassification()
    else:
        criterion = nn.CrossEntropyLoss()
        
    lr_func = None
    
    train_losses, train_accuracies, test_accuracies = train_network(
        net,
        optimizer=optimizer,
        lr=args.lr, 
        lr_func=lr_func,
        criterion=criterion,
        train_transform=train_transform,
        epochs=args.epochs,
        batch_size=args.batch_size)
    
    plot_results(train_losses, train_accuracies, test_accuracies)
    plt.savefig(args.output)

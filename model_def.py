"""
The module contains:
the function used to define and set up the NN model to be used in train.py
the function to load a model saved into a checkpoint to be used in predict.py

"""

from collections import OrderedDict
from torchvision import models
from torch import nn
import torch

def define_model(arch, hidden_units):
    """
    The function defines the NN model to be used.
    It takes "arch" as an input for the pretrained model to be imported
    from torchvision.models
    (available choices are vgg for vgg16 or resnet for resnet18)
    and it substitutes the fully connected part with a neural netrwork of our
    choice (keeping the structure of one input layer, one hidden layer,
    one output layer with ReLU() as activation functions, making use of
    Dropout(p) and using LogSoftmax to generate the final log-probabilities.)
    Input: the architecture to be used (to be chosen among the
    torchvision.models), the number of nodes for the hidden layer
    Output: the pretrained model, with additional fully connected layer matching
    the needed output to be used in the train.py file

    """

    if arch == "vgg":
        model = models.vgg16(pretrained=True)
    elif arch == "resnet":
        model = models.resnet18(pretrained=True)
    else:
        print("Error in the selected architecture. Only possible choices are 'vgg' and 'resnet'")

    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    #Define classifier with the correct input and output number based on the
    #selected architecture and the number of classes (102)
    if arch == "vgg":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 1024)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(1024, hidden_units)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        model.classifier = classifier
    elif arch == "resnet":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, hidden_units)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))
        model.fc = classifier

    return model

def load_model(file_path):
    #file_path is the location of the file, let us store it into a variable (dictionary) checkpoint
    checkpoint = torch.load(file_path)
    #import the pretrained model
    model = define_model(checkpoint["arch"], checkpoint["n_hidden"])

    model.class_to_idx = checkpoint["class_to_idx"]
    #load the state of the trained model from checkpoint
    model.load_state_dict(checkpoint["state_dict"])

    return model

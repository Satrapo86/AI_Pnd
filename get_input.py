"""
Module used to import arguments used in the command line to launch train.py
or predict.py
"""
import argparse

def get_train_args():
    """
    Funtion to import arguments from a command line used to launch train.py
    selecting some extra options
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type = str, default = "flowers", help = "path to the folder where the images are saved")
    parser.add_argument("--save_dir", type = str, default = "", help = "path to the folder where to save a model checkpoint")
    parser.add_argument("--arch", type = str, default = "resnet", help = "the CNN model architecture to be used, possible vaules are vgg, resnet")
    parser.add_argument("--hidden_units", type = int, default = 256, help = "Number of nodes in the hidden layer")
    parser.add_argument("--learn_rate", type = float, default = 0.0005, help = "Learn rate value for the Adam optimizer")
    parser.add_argument("--epochs", type = int, default = 15, help = "Number of epochs used to train the NN")
    parser.add_argument("--gpu", action = "store_true", default = False, help = "Run the training on the gpu")
    parser.add_argument("--test", action = "store_true", default = False, help = "Run the model on a test set")

    return parser.parse_args()

def get_predict_args():
    """
    Funtion to import arguments from a command line used to launch predict.py
    selecting some extra options
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type = str, default = "test_flower.jpg", help = "path to the image on which to be made a prediction")
    parser.add_argument("--checkpoint", type = str, default = "checkpoint.pth", help = "path to the checkpoint file containing the trained model to be used for inference")
    parser.add_argument("--topk", type = int, default = 3, help = "Number of top classes predicted by the NN")
    parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "Path to the file containing a dictionary to convert the classes to flower names")
    parser.add_argument("--gpu", action = "store_true", default = False, help = "Run the prediction on the gpu")

    return parser.parse_args()

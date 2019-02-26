"""
The module contains functions to load data, and to process images into the
right format to be used by Pytorch or to print them with pyplot
"""
import torch
import numpy as np
from torchvision import transforms, datasets
from PIL import Image

def load_data(data_dir):
    """
    Loads the data into three different loaders, applying different
    transformations. The function assumes a specific folder structure:
    "/data_dir/train/", "/data_dir/valid/", "/data_dir/test/"
    containing respectively the training data, the validation data
    and the testing data

    Input: the path of the data folder in string format

    Output: it returns three data loaders trainloader for training the NN,
    validloader to validate the NN, testloader to test the NN
    the data loaders contain batches of images in a torch.tensor format
    trainloader contains data-augmented images (random rotations, cropping,
    flipping are applied for training purposes)
    validloader images are only resized, cropped, converted to tensors and
    normalized to match the format expected by the pre-trained NN models
    testloader contains the same type of images as validloader
    The funtion also returns train_data to be able to calculate model.class_to_idx
    when saving the checkpoint
    """

    # Define data directories paths
    data_path = data_dir
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'


    # Define transformations to be applied to the different data sets
    train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load datasets using ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define dataloaders as described above
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_data, trainloader, validloader, testloader


def process_image(image_path):
    """
    Takes a .jpg image as input and processes it to be used as an input for a
    forward pass in the NN model
    Input: path to the image to be processed
    Output: a numpy array containing the processed image
    """

    img = Image.open(image_path)

    width, height = img.size
    if width >= height:
        img.thumbnail((256*width/height, 256))
    elif height > width:
        img.thumbnail((256, (height/width)*256))

    width2, height2 = img.size

    # Crop
    crop_val = 224
    left = int((width2 - crop_val)/2)
    upper = int((height2 - crop_val)/2)
    right = int(left + crop_val)
    lower = int(upper + crop_val)

    img = img.crop((left, upper, right, lower))


    # Convert to array and scale
    np_image = np.array(img)/255

    # Normalize color channel values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std

    # Channel as first dimension
    np_image = np_image.transpose(2, 0, 1)

    return np_image

def predict(image_path, model, device, topk):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    #Invert the dictionary to get the class from the ID

    class_dict = dict()
    for key in model.class_to_idx.keys():
        class_dict[str(model.class_to_idx[key])] = int(key)

    #Process image and turn it into torch tensor
    img = process_image(image_path)
    imgt = torch.from_numpy(img).unsqueeze_(0).type(torch.FloatTensor).to(device)
    #Run the image through the trained model
    model.to(device)
    model.eval()
    with torch.no_grad():
        log_ps = model(imgt)
        ps = torch.exp(log_ps)
        #Top probabilities and top ids
        top_ps, top_ids = ps.topk(topk, dim=1)

        #Convert top
        top_probs = []
        top_classes = []

        for idx in top_ids[0,:]:
            top_classes.append(str(class_dict[str(idx.item())]))
        for p in top_ps[0,:]:
            top_probs.append(p.item())

    return top_probs, top_classes

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

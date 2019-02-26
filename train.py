"""
The file is used to train a nural nework on a specific set of files.
The neural network uses a pre-trained model imported from torchvision.
"""

import torch
from torch import optim, nn
from model_def import define_model
from data_utils import load_data
from get_input import get_train_args
#from workspace_utils import active_session

# Import arguments from command line
train_arg = get_train_args()

if train_arg.gpu:
    device = "cuda"
else:
    device = "cpu"

# Load data
train_data, trainloader, validloader, testloader = load_data(train_arg.dir)

# Define model
model = define_model(train_arg.arch, train_arg.hidden_units)
model.to(device)

# Define loss and optimizer

criterion = nn.NLLLoss()
if train_arg.arch == "resnet":
    optimizer = optim.Adam(model.fc.parameters(), lr = train_arg.learn_rate)
elif train_arg.arch == "vgg":
    optimizer = optim.Adam(model.classifier.parameters(), lr = train_arg.learn_rate)

# Train the model
epochs = train_arg.epochs
check_every = 50
steps = 0
train_loss = 0

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if steps % check_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_accuracy = 0
                for im, lab in validloader:
                    im, lab = im.to(device), lab.to(device)
                    vlogps = model(im)
                    vloss = criterion(vlogps, lab)
                    valid_loss += vloss.item()

                    ps = torch.exp(vlogps)
                    top_ps, top_class = ps.topk(1,dim=1)
                    equals = top_class == lab.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Training loss: {train_loss/check_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
            train_loss = 0
            model.train()

# Save checkpoint after training
model.class_to_idx = train_data.class_to_idx
checkpoint = {"arch" : train_arg.arch,
              "n_hidden" : train_arg.hidden_units,
              "state_dict" : model.state_dict(),
              "class_to_idx" : model.class_to_idx,
              "epochs" : epochs,
              "optimizer_state" : optimizer.state_dict}
file = train_arg.save_dir + "checkpoint.pth"
torch.save(checkpoint, file)

# Test to be run only if --test added to command line
if train_arg.test:
    print("Testing the model on Test data: \n")
    model.eval()
    with torch.no_grad():
        test_accuracy = 0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {test_accuracy/len(testloader):.3f}")

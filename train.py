''' Name: Viraj S. Shah.
    Date: 1-24-2020.
    This is the code for training the model.
'''
# All the import statements.
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import json
from new_model import newModel

# Defining the arg parser for train.py.
parser = argparse.ArgumentParser(description='Customize training your model.')
parser.add_argument('data_dir', action='store')
parser.add_argument('--save', action="store", default='save_directory', dest='save_directory')
parser.add_argument('--arch', action="store", default='vgg13', dest='arch')
parser.add_argument('--learning_rate', action="store", default=0.0003, type=int, dest='learning_rate')
parser.add_argument('--hidden_units', action="store", default=5000, type=int, dest='hidden_units')
parser.add_argument('--epochs', action="store", default=4, type=int, dest='epochs')
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()
#print(args.arch)

# Loading the data.
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining the transforms for the training, validation, and testing sets.
data_transform_train = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 
# This transform is for both testing and validation.
data_transform_test = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 

# Load the datasets with ImageFolder, we have 3 datasets and dataloaders.
train_data = datasets.ImageFolder(train_dir, transform=data_transform_train)
test_data = datasets.ImageFolder(test_dir, transform=data_transform_test)
validate_data = datasets.ImageFolder(valid_dir, transform=data_transform_test)

# Using the image datasets and the trainforms to define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
validate_dataloader = torch.utils.data.DataLoader(validate_data, batch_size=32, shuffle=True)

# Label mapping.
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Checking for GPU.
device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu')

# define our model, criterion, optimizer.
new_model = newModel(args.arch, args.hidden_units, train_data)
model = new_model[0]
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Actual code for training the model.
model.to(device)
epochs = args.epochs
steps = 0
running_loss = 0

for e in range(epochs):
    running_loss = 0
    for images, labels in train_dataloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        #clearing the gradients.
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % 10 == 0:
            # validation step.
            validate_accuracy = 0
            validate_loss = 0
            model.eval()
            with torch.no_grad():
                for image, label in validate_dataloader:
                    image, label = image.to(device), label.to(device)
                    logps = model.forward(image)
                    ps = torch.exp(logps) 
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    validate_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    validate_loss += criterion(logps, labels).item()
            print(f"Epoch {e+1}/{epochs}    "
                  f"Train loss: {running_loss/10:.3f}    "
                  f"Validation Loss: {validate_loss/len(validate_loader)}    "
                  f"Validation accuracy: {validate_accuracy/len(validate_dataloader):.3f}")
            running_loss = 0
            model.train()
            
# Testing the network.
testAccuracy = 0
for images, labels in test_dataloader: 
    images, labels = images.to(device), labels.to(device)
    logps = model(images)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    testAccuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(f'Final Test Accuracy is {testAccuracy/len(test_dataloader)*100:.3f} %.')

# Making a checkpoint, and saving it inside the desired folder (by default this is save_directory).
checkpoint = {'input_size': new_model[1],
              'output_size': 102,
              'hidden_layer1_size': new_model[2],
              'hidden_layer2_size': new_model[3],
              'hidden_layer3_size': new_model[4],
              'state_dict': model.state_dict(),
              'epochs': args.epochs}
torch.save(checkpoint, args.save_directory+'/'+args.arch+'_checkpoint.pth')
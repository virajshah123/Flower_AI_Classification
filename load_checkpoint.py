from new_model import newModel
import torch
from torchvision import datasets, transforms

train_dir = 'flowers/train'
# Defining the transforms for the training sets.
data_transform_train = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 


# Load the datasets with ImageFolder, we have 3 datasets and dataloaders.
train_data = datasets.ImageFolder(train_dir, transform=data_transform_train)

# Code for loading the checkpoint.
def load_checkpoint(filepath, modelString, gpu=False):
    device = torch.device('cuda:0' if (gpu) else 'cpu')
    checkpoint = torch.load(filepath)
    new_model = newModel(modelString, checkpoint['hidden_layer1_size'], train_data)
    model = new_model[0]
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    return model
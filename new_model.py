# importing statements.
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

# Over here I have only made the vgg, alexnet and resnet models.
model_dic = {'vgg11': [models.vgg11(pretrained=True), 25088],
             'vgg13': [models.vgg13(pretrained=True), 25088],
             'vgg16': [models.vgg16(pretrained=True), 25088],
             'vgg19': [models.vgg19(pretrained=True), 25088],
             'alexnet': [models.alexnet(pretrained=True), 9216],
             'densenet121': [models.densenet121(pretrained=True), 1024],
             'densenet161': [models.densenet121(pretrained=True), 2208],
             'densenet169': [models.densenet121(pretrained=True), 1664],
             'densenet201': [models.densenet121(pretrained=True), 1920],
            }

class Classifier(nn.Module):
    '''Defines the necessary classifier for the model.
    '''
    def __init__(self, input_size, hid1_size, hid2_size, hid3_size, output_size, train_data):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.class_to_idx = train_data.class_to_idx
    
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

def newModel(modelString, hiddenInput, train_data):
    input_size = model_dic[modelString][1] 
    hid1 = hiddenInput
    hid2 = (int) (hid1/4)
    hid3 = (int) (hid2/4)
    
    model = model_dic[modelString][0]
    
    # Freeze all the parameters in the model.
    for parameters in model.parameters():
        parameters.requires_grad = False
    
    model.classifier = Classifier(input_size, hid1, hid2, hid3, 102, train_data)
    
    return model, input_size, hid1, hid2, hid3
    

    
        

''' Name: Viraj S. Shah.
    Date: 1-24-2020.
    This is the code for predicting the model.
'''

# Importing statements.

import argparse
import json
import PIL
from PIL import Image
import numpy as np
from IPython.display import display
import torch
from load_checkpoint import load_checkpoint


# Making the positional and the optional arguments of the arg parser.
parser = argparse.ArgumentParser(description='Customize predicting your model.')
parser.add_argument('image', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument('--top_k', action="store", default=5, type=int, dest='top_k')
parser.add_argument('--category_names', action="store", default='cat_to_name.json', dest='category_names')
parser.add_argument('--gpu', action="store_true", default=False, dest='gpu')
args = parser.parse_args()

# Label mapping.
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
#Defining device.
device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu')

# Processing the given image so that it can be used.
def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model.
    if img.size[0] <= img.size[1]:
        aspectRatio = img.size[1]/img.size[0]
        new_size = (256, int(aspectRatio*256))
    else:
        aspectRatio = img.size[0]/img.size[1]
        new_size = (int(256*aspectRatio), 256)

    img = img.resize(new_size)

    # Cropping the image.
    widthLeft = int(img.size[0]/2) - 112
    widthRight = int(img.size[0]/2) + 112
    heightBottom = int(img.size[1]/2) - 112
    heightTop = int(img.size[1]/2) + 112
    img = img.crop((widthLeft, heightBottom, widthRight, heightTop))

    # Converting the encoded integers to expected floats between 0 and 1.
    np_image = np.array(img)
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image/255
    np_image = (np_image-mean)/std
    np_image = np_image.transpose(2, 0, 1)
    
    # Converting into a tensor.
    np_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    return np_image

# Actual function for predicting.
def predict(image_path, checkpoint, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Finding modelString. Over here we exploit the format with which we have saved each of the checkpoints.
    list_string = checkpoint.split('/')
    new_list_string = list_string[1].split('_')
    modelString = new_list_string[0]
    
    # Loading the checkpoint into the classifier's model. We are using vgg16 model.
    model = load_checkpoint(checkpoint, modelString, args.gpu)
    
    # Getting our image ready.
    img = Image.open(image_path, 'r')
    final_image = process_image(img).to(device)
    final_image = final_image.unsqueeze(0) # We have to do this since otherwise the convolutional dimensions aren't matched.
    
    # Create inverse dictionary (index to class).
    class_to_index = model.classifier.class_to_idx
    index_to_class = {}
    for c in class_to_index:
        # Swap the key and the value.
        index_to_class[class_to_index[c]] = c
    
    # Now we have our trained model ready. Return the top-k values.
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(final_image))
        top_p, top_classes = ps.topk(topk, dim=1)
        top_p, top_classes = top_p[0].tolist(), top_classes[0].tolist()
        top_classes = [index_to_class[index] for index in top_classes]
    return top_p, top_classes

# Function for displaying the top_k values.
def display(image_path, checkpoint, topk=args.top_k):
    
    top_p, top_classes = predict(image_path, checkpoint)
    top_class_names = [cat_to_name[c] for c in top_classes]
    print(top_class_names)
    
# Showing to the user.    
display(args.image, args.checkpoint)


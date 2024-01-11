import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import json
import input_args_predict as args

with open(args.result.cat_to_name.name, 'r') as f:
    cat_to_name = json.load(f, strict=False)

class restored_model:


    def __init__(self,filepath):
    
        
        self.filepath=filepath
        self.checkpoint=torch.load(self.filepath)
        self.model=self.checkpoint['base_model']
        self.classifier=nn.Sequential(nn.Linear(self.checkpoint['input_size'],self.checkpoint['hidden_layer1'][0]),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.checkpoint['hidden_layer1'][0],self.checkpoint['hidden_layer1'][1]),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.checkpoint['hidden_layer1'][1],self.checkpoint['output_size']),
                                nn.LogSoftmax(dim=1)
                                )
        
        self.model.classifier=self.classifier
        self.model.load_state_dict(self.checkpoint['state_dict'])
        
    
    def return_model(self):
        return self.model

    

#process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    min_ = 256
    new_=224
    
    width,height = image.size
    a = max(min_/width, min_/height)    
    image.thumbnail((width*a,height*a))
    
    
    width1,height1 = image.size
    left = (width1 - new_)//2
    top = (height1 - new_)//2
    right = (width1 + new_)//2
    bottom = (height1 + new_)//2
    image=image.crop((left, top, right, bottom))
    
    

    np_image = np.array(image)
    np_image=(np_image-np_image.min())/(np_image.max()-np_image.min())
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std
    np_image=np_image.transpose(2,0,1)
    torch_image=torch.from_numpy(np_image.astype(np.float32))
    
    
    return torch_image


#predict image
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    if args.result.gpu==True:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    
    #print(image_path)
    image = Image.open(image_path)
    image= process_image(image).unsqueeze(0) 
    if args.result.gpu==True:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image.to(device)
    
    with torch.no_grad():
        log_ps=model(image)
    ps=torch.exp(log_ps)
    
    
    top_p , top_class=ps.topk(topk,dim=1)

    if cat_to_name:
        top_class=[cat_to_name[str(number)] if str(number) in cat_to_name else number for number in top_class.numpy().flatten() ]
    
        
    print(f"the predicted class of this image is {top_class[0]} with a probability of {top_p.numpy().flatten()[0] }\n")
    for a,i in enumerate(range(topk), 1):
        print(f"the top {a} predicted class of this image is {top_class[i]} with a probability of {top_p.numpy().flatten()[i] }")
    
    return top_p.numpy().flatten(),top_class

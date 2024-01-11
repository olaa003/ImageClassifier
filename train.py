# Imports here
import torch
import train_data_transforms as trans_data
import input_args_train as args
import torchvision
from torchvision import models
from torch import nn 
from torch import optim
import numpy as np


if __name__=='__main__':


    # Build and train your network
    resnet18 = models.resnet18(pretrained=True)
    vgg11 = models.vgg11(pretrained=True)

    models = {'resnet': resnet18,  'vgg': vgg11}

    model=models[args.result.model]

    for param in model.parameters():
        param.requires_grad = False

    if args.result.model=='vgg': 
        classifier=nn.Sequential(nn.Linear(model.classifier[0].in_features,args.result.hidden[0]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(args.result.hidden[0],args.result.hidden[1]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(args.result.hidden[1],102),
                                 nn.LogSoftmax(dim=1))
    else:
            classifier=nn.Sequential(nn.Linear(model.fc[0].in_features,args.result.hidden[0]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(args.result.hidden[0],args.result.hidden[1]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(args.result.hidden[1],102),
                                 nn.LogSoftmax(dim=1))
    if args.result.model=='vgg':
        model.classifier=classifier
    else:
        model.fc=classifier

    criterion=nn.NLLLoss()

    if args.result.model=='vgg':
        optimizer=optim.Adam(model.classifier.parameters(),lr=args.result.learning_rate)
    else:
        optimizer=optim.Adam(model.fc.parameters(),lr=args.result.learning_rate)


    if args.result.gpu==True:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    epoch=args.result.epoch

    for e in range(1,epoch):
        train_loss=0

        for image,label in trans_data.train_dataloader:
            if args.result.gpu==True:
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                image, label= image.to(device),label.to(device)

            optimizer.zero_grad()

            log_ps=model(image)
            loss=criterion(log_ps, label)
            loss.backward()
            optimizer.step()
            train_loss+=loss

        else:
            model.eval()
            accuracy = 0
            test_loss=0

            with torch.no_grad():

                for image,label in trans_data.valid_dataloader:
                    if args.result.gpu==True:
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        image, label= image.to(device),label.to(device)

                    log_ps=model(image)
                    loss=criterion(log_ps, label)
                    test_loss+=loss

                    # Calculate accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()

        print(f'ephoch : {e}, Training loss: {train_loss/len(trans_data.train_dataloader)}')   
        print(f'ephoch : {e}, Validation loss: {test_loss/len(trans_data.valid_dataloader)}')
        print(f'ephoch : {e}, Validation accuracy: {(accuracy/len(trans_data.valid_dataloader))*100}')

    # To save model for later use
    model.to('cpu') #moving the model back to cpu so it can be unserialized without a gpu

    checkpoint={'input_size':model.classifier[0].in_features,
                'output_size':102,
                'hidden_layer1':args.result.hidden,
                'state_dict':model.state_dict(),
                'epoch':epoch,
                'optimizer_state':optimizer.state_dict,  
                'base_model':models[args.result.model]
    }

    torch.save(checkpoint, args.result.checkpoint_dir)


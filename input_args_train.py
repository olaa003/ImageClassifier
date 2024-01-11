import argparse

parser=argparse.ArgumentParser(
    description='this is to receive the data and other parameters to be used in training the model')

parser.add_argument('training_data', action='store',help='this directory contains two folders one for training dataset and the other for validation data set')
parser.add_argument('--arch', action='store', dest='model', help='Store the model architecture to used in training',choices=('VGG', 'resnet'),default='vgg')
parser.add_argument('--save_dir', action='store', dest='checkpoint_dir', help='Store the trained model',default='./checkpoint.pth')
parser.add_argument('--learning_rate', action='store', dest='learning_rate', help='Store the learning rate', type=float, default=0.003)
parser.add_argument('--hidden_units', action='store', dest='hidden', type=list, help='Store a list of 2 hidden units nodes',default=[256,128])
parser.add_argument('--epochs', action='store', dest='epoch', type=int, help='Store the number of training epochs',default=3)
parser.add_argument('--gpu', action='store_true', dest='gpu', default=False, help= 'Set a switch to true')


try:
    result = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))
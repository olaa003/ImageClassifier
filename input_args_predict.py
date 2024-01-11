import argparse


parser=argparse.ArgumentParser(
    description='this is to receive an image and a trained model and in turn return the predicted class of the image')

parser.add_argument('image', action='store',help='this directory to the image to be predicted', type=argparse.FileType('rt'))
parser.add_argument('checkpoint', action='store',help='this directory to the saved model', type=argparse.FileType('rt'))
parser.add_argument('--gpu', action='store_true', dest='gpu', default=False, help= 'Set a switch to true')
parser.add_argument('--top_k', action='store', dest='topk', help='Store the top k model predictions', type=int, default=1)
parser.add_argument('--cat_to_name', action='store',help='this directory to the category to name json for the predicted labels', type=argparse.FileType('rt'),dest='cat_to_name', default='cat_to_name.json')

try:
    result = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

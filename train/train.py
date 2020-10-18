import sys
import argparse
import conf
sys.path.insert(0, conf.path2code)
from utils import standard_loader, standard_saver
import train

parser = argparse.ArgumentParser(description='Train copula vine model')
parser.add_argument('-exp', default='', help='Experiment name')
parser.add_argument('-layers', default=-1, help='How many layers? (-1 = all possible)', type=int)
# TODO paths to exps

args = parser.parse_args()

gpu_list = range(2)

X,Y = standard_loader(f"{conf.path2data}/{args.exp}_layer0.pkl")

# figure out how many trees to train
if args.layers == -1:
	layers = Y.shape[-1]-1
else:
	layers = args.layers

all_models = []
for layer in range(layers):
	print(f'Starting {args.exp} layer {layer}/{layers}')
	models, Y = train.train_next_layer(X, Y, args.exp, layer, gpu_list)
	all_models.append(models)
	# save to disk (unnecessary)
	path = f"{conf.path2data}/{args.exp}_layer{layer+1}.pkl"
	standard_saver(path,X,Y)

import pickle as pkl
with open("allmodels.pkl","wb") as f:
	pkl.dump(all_models,f)
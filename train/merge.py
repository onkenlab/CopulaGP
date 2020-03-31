import glob
import os
import pickle as pkl

def merge_results(path_models, layer):
	list_files = glob.glob(f"{path_models}/layer{layer}*.pkl")

	with open(list_files[0],"rb") as f:
	    res = pkl.load(f)
	if len(list_files)>1:
		print('Not implemented')
		exit()
	    # for file in list_files[1:]:
	    #     with open(file,"rb") as f:
	    #         res_add = pkl.load(f)
	
	with open(f"{path_models}/layer{layer}_models.pkl","wb") as f:
	    pkl.dump(res,f)

	# for file in list_files:
	# 	os.remove(file)
import time
import pickle as pkl
from torch import device
import numpy as np
import sys
import multiprocessing

import utils
import select_copula

import traceback
import warnings
import os

gpu_id_list = [1]
unique_id_list = np.random.randint(0,10000,len(gpu_id_list)) #TODO: make truely unique
#[i//2 for i in range(8*2)]  # 2 workers on each GPU

animal = 'ST260'
dayN = 1
day_name = 'Day{}'.format(dayN)
path2data = '/home/nina/VRData/Processing/pkls'

exp_pref = '{}_{}'.format(animal,day_name)

repeats = 10

out_dir = '../test_standard/'+exp_pref
try:
	os.mkdir(out_dir)
except FileExistsError as error:
	print(error)

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def worker(X, Y0, Ys, idxs, NN, progress):
	# get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
	gpu_id = gpu_id_list[cpu_id]

	device_str = f'cuda:{gpu_id}'

	print(f'Start a new batch ({progress}) on {device_str}')

	unique_id = unique_id_list[cpu_id]

	for n,Y1 in zip(idxs,Ys.T):

		Y = np.stack([Y0,Y1]).T

		print(f'Selecting 0-{n} on {device_str}')
		try:
			t_start = time.time()
			(likelihoods, waic) = select_copula.select_with_heuristics(X,Y,device(device_str),exp_pref,out_dir,0,n)
			t_end = time.time()
			print('Selection took {} min'.format(int((t_end-t_start)/60)))
		except RuntimeError as error:
			print(error)
		finally:
			with open(out_dir+'_model_list.txt','a') as f:
				f.write("{}-{} {}\t{:.0f}\t{}\n".format(0,n,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)))
			
			results_file = f"{out_dir}_{unique_id}_models.pkl"
			if os.path.exists(results_file):
				with open(results_file,'rb') as f:
					results = pkl.load(f)  
			else:
				results = np.empty(NN,dtype=object)

			assert (results[n]==None)
			results[n] = [likelihoods,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)]

			with open(results_file,'wb') as f:
				pkl.dump(results,f)   

	return 0

if __name__ == '__main__':

	warnings.showwarning = warn_with_traceback

	pool = multiprocessing.Pool(len(gpu_id_list))

	X,Y = utils.standard_loader(f"{path2data}/{exp_pref}_standard.pkl")
	NN = Y.shape[-1]-1

	batch = int(np.ceil(NN/len(gpu_id_list)/repeats))

	print(f"Batch size: {batch}")

	list_idx = np.arange(1,NN+1)
	resid = len(list_idx)%batch
	if resid!=0:
		list_idx = np.concatenate([list_idx,np.zeros(batch-resid)]).astype('int')
	batches = np.reshape(list_idx,(batch,-1)).T

	for i,b in enumerate(batches):
		res = pool.apply_async(worker, (X, Y[:,0], Y[:,b[b!=0]], b[b!=0], NN, f"{i+1}/{len(batches)}", ))

	pool.close()
	pool.join()  # block at this line until all processes are done
	print("completed")


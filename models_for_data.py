import time
import os
import pickle as pkl
from torch import device
import numpy as np
import sys
import multiprocessing

import utils
import select_copula

gpu_id_list = [1,2,3,4,5,6,7]
#[i//2 for i in range(8*2)]  # 2 workers on each GPU
unique_id_list = np.random.randint(0,10000,len(gpu_id_list)) #TODO: make truely unique

animal = 'ST260'
dayN = 2
day_name = 'Day{}'.format(dayN)
path2data = '/disk/scratch_fast/ninas_dataset'

exp_pref = '{}_{}'.format(animal,day_name)

out_dir = '../out_newlogic/'+exp_pref
try:
	os.mkdir(out_dir)
except FileExistsError as error:
	print(error)

d = {
    'ST260': 104,
    'ST262': 61,
    'ST263': 23,
    'ST264': 34
}

NN = d[animal] #number of neurons
beh = 5

def worker(n1,n2):
	#get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
	gpu_id = gpu_id_list[cpu_id]
	os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"

	device_str = 'cuda:{}'.format(gpu_id)

	unique_id = unique_id_list[cpu_id]
	results = np.empty((NN+beh,NN+beh),dtype=object)

	X,Y = utils.load_experimental_data(path2data, animal, day_name, n1, n2, t1=0, t2=1)

	print('Selecting {}-{} on {}'.format(n1,n2,device_str))
	try:
		t_start = time.time()
		(likelihoods, waic) = select_copula.select_copula_model(X,Y,device(device_str),'',out_dir,n1,n2)
		t_end = time.time()
		print('Selection took {} min'.format(int((t_end-t_start)/60)))
	except RuntimeError as error:
		print(error)
	finally:
		path2model = "{}/{}-{}.pkl".format(out_dir,n1,n2)   
		with open(path2model,'wb') as f:
			pkl.dump(likelihoods,f)

		with open(out_dir+'_model_list.txt','a') as f:
			f.write("{}-{} {}\t{:.0f}\t{}\n".format(n1,n2,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)))
		results_file = f"{out_dir}_{unique_id}_models.pkl"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)
		
		assert(results[beh+n1,beh+n2]==None)
		results[beh+n1,beh+n2] = [likelihoods,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)]

		with open(results_file,'wb') as f:
			pkl.dump(results,f)   

	return 0

if __name__ == '__main__':
	pool = multiprocessing.Pool(len(gpu_id_list))

	for n1 in range(28,NN-1):
		for n2 in range(n1+1,NN):
			if (n1>28) | (n2>=32):
				res = pool.apply_async(worker, (n1,n2,))
	pool.close()
	pool.join()  # block at this line until all processes are done
	print("completed")
		



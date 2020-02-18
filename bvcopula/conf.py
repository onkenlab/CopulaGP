#learning rates
grid_size = 128
base_lr = 1e-2
hyper_lr = 1e-3
iter_print = 100
max_num_iter = 3000
loss_tol = 0.0001 #the minimal change in loss that indicates convergence
loss_tol2check_waic = 0.005
min_waic = -0.01

# copula's theta ranges
# here thetas are mainly constrained by the summation of probabilities in mixture model,
# which should not become +inf
Gauss_Safe_Theta = 0.9999	# (-safe,+safe), for safe mode, otherwise (-1,1)
Frank_Theta_Max = 16.8 		# (-max, max)
Frank_Theta_Flip = 9.0
Clayton_Theta_Max = 9.4 	# (0, max)
Gumbel_Theta_Max = 10.0		# (1, max)
# looser limits for sample generation
Frank_Theta_Sampling_Max = 88.0 	# (-max, max)
Clayton_Theta_Sampling_Max = 22.5 	# (0, max)
Gumbel_Theta_Sampling_Max = 16.0 	# (1, max) #no clear critical value here, it is around 16

#Gaussian full dependence
Gauss_diag = 1e-5 # how far from diagonal the point can be to be considered as u==v

# waic parameters
waic_samples = 1000

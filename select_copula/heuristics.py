import torch
from . import conf
import logging
import bvcopula
from utils import get_copula_name_string, Plot_Fit

def cov(x,y):
    return torch.mean((x-x.mean())*(y-y.mean()))
    
def pearson(x,y):
    return cov(x,y)/(x.std()*y.std())

def evaluate(model,device):
    # define uniform test set (optionally on GPU)
    test_x = torch.linspace(0,1,100).cuda(device=device)
    with torch.no_grad():
        output = model(test_x)
    gplink = model.likelihood.gplink_function
    thetas, mixes = gplink(output.mean, normalized_thetas=False)

    return thetas, mixes

def important_copulas(model, device):
    thetas, mixes = evaluate(model,device)
    which = torch.torch.any(mixes>0.1,dim=1) # if higher than 10% somewhere -> significant
    return which
   
def select_with_heuristics(X: torch.Tensor, Y: torch.Tensor, device: torch.device,
    exp_pref: str, path_output: str, name_x: str, name_y: str,
    train_x = None, train_y = None):

    exp_name = f'{exp_pref}_{name_x}-{name_y}'
    log_name = f'{path_output}/log_{device}_{exp_name}.txt'
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(filename=log_name, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

    #convert numpy data to tensors (optionally on GPU)
    if train_x is None:
        train_x = torch.tensor(X).float().to(device=device)
    if train_y is None:
        train_y = torch.tensor(Y).float().to(device=device)

    waic, _ = bvcopula.infer([bvcopula.FrankCopula_Likelihood()],train_x,train_y,device=device)

    if waic<conf.waic_threshold*X.shape[0]:
        return ([bvcopula.IndependenceCopula_Likelihood()], 0)
    else:
        (waic_gumbels, model_gumbels) = bvcopula.infer(conf.gumbel_likelihoods,train_x,train_y,device=device)
        (waic_claytons, model_claytons) = bvcopula.infer(conf.clayton_likelihoods,train_x,train_y,device=device)

        print(waic_claytons,waic_gumbels)
        
        if waic_claytons>waic_gumbels:
            which_leader = important_copulas(model_claytons,device)
            which_follow = important_copulas(model_gumbels,device)
            likelihoods_leader = conf.clayton_likelihoods[2:]
            likelihoods_follow = conf.gumbel_likelihoods[2:]
        else:
            which_leader = important_copulas(model_gumbels,device)
            which_follow = important_copulas(model_claytons,device)
            likelihoods_leader = conf.gumbel_likelihoods[2:]
            likelihoods_follow = conf.clayton_likelihoods[2:]

        print(which_leader)
        print(which_follow)
        
        symmetric_part = which_leader[:2] + which_follow[:2] # + = elementwise_or
        assymetric_part = which_leader[2:] + which_follow[2:]

        def reduce(likelihoods,which):
            assert len(likelihoods)==len(which)
            idx = torch.arange(0,len(which))[which]
            return [likelihoods[i] for i in idx]

        waic_max = max(waic_claytons,waic_gumbels)
        symmetric_likelihoods = reduce(conf.clayton_likelihoods[:2],symmetric_part)
        print("Symmetric: "+get_copula_name_string(symmetric_likelihoods))
        best_likelihoods = symmetric_likelihoods + reduce(likelihoods_leader,which_leader[2:])
        count_swaps=0
        for i in torch.arange(4)[assymetric_part]:
            likelihoods = symmetric_likelihoods.copy()
            if (count_swaps==0) and (i==torch.sum(assymetric_part)-1):
                print('No need to swap the last one, as we already tried that model')
            else:
                for j in torch.arange(4)[assymetric_part]:
                    if i!=j:
                        likelihoods.append(likelihoods_leader[j])
                    else:
                        likelihoods.append(likelihoods_follow[j])
                print(get_copula_name_string(likelihoods))
                (waic, model) = bvcopula.infer(likelihoods,train_x,train_y,device=device)
                print(waic)
                if waic>waic_max:
                    print("Swap "+get_copula_name_string([likelihoods_leader[i]])+"->"+get_copula_name_string([likelihoods_follow[i]]))
                    likelihoods_leader[i] = likelihoods_follow[i]
                    count_swaps+=1
                    waic=waic_max
                    best_likelihoods = likelihoods.copy()
                    which_leader = important_copulas(model, device)
                    # plot results
                    plot_res = '{}/res_{}.png'.format(path_output,name)
                    Plot_Fit(model, X, Y, name_x, name_y, plot_res, device=device)
                    # save weights
                    name = '{}_{}'.format(exp_name,get_copula_name_string(best_likelihoods))
                    weights_filename = '{}/w_{}.pth'.format(path_output,name)
                    torch.save(model.state_dict(),weights_filename)

        print("Assymetric: "+get_copula_name_string(likelihoods_leader))
   
        if torch.any(which_leader==False):
            best_likelihoods = reduce(best_likelihoods,which_leader)
            print("Re-running reduced model...")
            (waic, model) = bvcopula.infer(best_likelihoods,train_x,train_y,device=device)
            waic_max = waic
        else:
            print('Nothing to reduce')

        # Finally, check if Gaussian Copula is better than Frank
        if symmetric_part[1]==True:
            with_gauss = best_likelihoods.copy()
            with_gauss[torch.sum(symmetric_part)-1] = bvcopula.GaussianCopula_Likelihood()
            print('Trying Gauss: '+get_copula_name_string(with_gauss))
            (waic, model) = bvcopula.infer(with_gauss,train_x,train_y,device=device)
            if waic>waic_max:
                print('Gauss is better than Frank')
                waic_max = waic
                best_likelihoods = with_gauss
    
        print("Final model: "+get_copula_name_string(best_likelihoods))

        return best_likelihoods, waic_max
    
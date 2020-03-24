import torch
from . import conf

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

# def which_corners(model, device):

#     thetas, mixes = evaluate(model,device)

#     which = torch.torch.any(mixes>0.1,dim=1) # if higher than 10% somewhere -> significant
    
#     opposite_together = False
    
#     if pearson(thetas[1],thetas[3])>0.7:
#         opposite_together = True
        
#     if pearson(thetas[2],thetas[4])>0.7:
#         opposite_together = True
 
#     return (which,opposite_together) 

def important_copulas(model, device):
    thetas, mixes = evaluate(model,device)
    which = torch.torch.any(mixes>0.1,dim=1) # if higher than 10% somewhere -> significant
    return which

def models_to_try(which,opposite_together):
    likelihoods_list = []
    idx = torch.arange(0,len(which))[which]
    claytons = [conf.clayton_likelihoods[i] for i in idx]
    gumbels = [conf.gumbel_likelihoods[i] for i in idx]
    likelihoods_list.append(claytons)
    likelihoods_list.append(gumbels)
    if opposite_together:
        for sym in conf.symmetric_likelihoods:
            likelihoods_list.append([sym]+claytons)
            likelihoods_list.append([sym]+gumbels)
    return likelihoods_list
   
def select_with_heuristics():
    pass

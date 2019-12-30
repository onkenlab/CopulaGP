import torch
from torch.distributions.distribution import Distribution
from torch.distributions import constraints, normal, studentT
from . import conf

class SingleParamCopulaBase(Distribution):
    '''
    This abstract class represents a copula with a single parameter.
    Parameters
    ----------
    theta : float
        Parameter of the copula.
    rotation : string, optional
        Clockwise rotation of the copula.  Can be one of the elements of
        `Copula.rotation_options` or `None`.  (Default: `None`)
    Attributes
    ----------
    theta : float
        Parameter of the copula.
    rotation : string
        Clockwise rotation of the copula.
    Methods
    -------
    log_prob(samples)
        Log of the probability density function.
    rsample(size=1)
        Generate random variates.
    ppcf(samples)
        Inverse conditional cdf on the copula. 
        Required for sample generation using Rosenblatt transform.
    expand(batch_shape)
        Expends the batch space: adds extra dimension for MCMC sampling
        corresponding to sampling particles.
    '''
    has_rsample = True
    num_thetas = 1
    rotation_options = ['0°', '90°', '180°', '270°']
    
    def __init__(self, theta, rotation=None, validate_args=None):
        self.theta = theta
        self.__check_rotation(rotation)
        self.rotation = rotation
        batch_shape, event_shape = self.theta.shape, torch.Size([2])
        super(SingleParamCopulaBase, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @classmethod
    def __check_rotation(cls, rotation):
        '''
        Checks the `rotation` parameter.
        Parameters
        ----------
        rotation : string
            Rotation of the copula.  Can be one of the elements of
            `Copula.rotation_options` or `None`.
        '''
        if rotation is not None and rotation not in cls.rotation_options:
            raise ValueError("rotation '" + rotation + "' not supported")

    def __rotate_input(self, samples):
        '''
        Preprocesses the input to account for the copula rotation.  The input
        is changed and a reference to the input is returned.
        Parameters
        ----------
        samples : array_like
            [batch_dims, 2] tensor of samples.
        Returns
        -------
        samples : array_like
            [batch_dims, 2] tensor of rotated samples.
        '''
        if self.rotation == '90°':
            samples[..., 1] = 1 - samples[..., 1]
        elif self.rotation == '180°':
            samples[...] = 1 - samples[...]
        elif self.rotation == '270°':
            samples[..., 0] = 1 - samples[..., 0]
        return samples
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SingleParamCopulaBase, _instance)
        batch_shape = torch.Size(batch_shape)
        if batch_shape == torch.Size([]):
            batch_shape = torch.Size([1])
        theta_shape = batch_shape + torch.Size(self.event_shape[:-1])
        new.theta = self.theta.expand(theta_shape) 
        super(SingleParamCopulaBase, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new.rotation = self.rotation
        new._validate_args = self._validate_args
        return new
    
    def ppcf(self, samples):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size([])):
        shape = self._extended_shape(sample_shape) # now it is theta_size (batch) x sample_size x 2 (event)
        
        if sample_shape == torch.Size([]):   # not sure what to do with 1 sample
            shape = torch.Size([1]) + shape
            
        samples = torch.empty(size=shape).uniform_(1e-4, 1. - 1e-4) #torch.rand(shape) torch.rand in (0,1]
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            samples = samples.cuda(device=get_cuda_device)
        samples[...,0] = self.ppcf(samples)
        samples = self._SingleParamCopulaBase__rotate_input(samples)
        return samples

    def log_prob(self, value):
        raise NotImplementedError

class IndependenceCopula(Distribution):
    '''
    This class represents a copula from the Independence Copula.
    '''
    has_rsample = True
    num_thetas = 0 
    
    def __init__(self, theta, rotation=None, validate_args=None):
        if theta is None:
            raise ValueError("Theta has to be provided anyway. Should be empty for independence copula, but will indentify the device where samples must be stored.")
        elif theta.shape != torch.Size([0]):
            raise ValueError("Theta should be empty for independence copula.")
        else:
            if theta.is_cuda:
                self.cuda_device = theta.get_device()
            else:
                self.cuda_device = None
        batch_shape, event_shape = torch.Size([]), torch.Size([2])
        super(IndependenceCopula, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size([])):
        shape = self._extended_shape(sample_shape) # now it is theta_size (batch) x sample_size x 2 (event)
            
        samples = torch.empty(size=shape).uniform_(1e-4, 1. - 1e-4) #torch.rand(shape) torch.rand in (0,1]
        if self.cuda_device is not None:
            samples = samples.cuda(device=self.cuda_device)

        return samples

    def log_prob(self, value):
        return torch.zeros_like(value[...,0])

class GaussianCopula(SingleParamCopulaBase):
    '''
    This class represents a copula from the Gaussian family.
    '''
    arg_constraints = {"theta": constraints.interval(-1,1)}
    support = constraints.interval(0,1) # [0,1]
    
    def ppcf(self, samples):
        assert torch.all(self.theta.abs()<=1.0)
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            nrvs = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(samples)
            vals = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(nrvs[..., 0] * torch.sqrt(1 - self.theta**2) + 
                                 nrvs[..., 1] * self.theta)
        else:    
            nrvs = normal.Normal(0,1).icdf(samples)
            vals = normal.Normal(0,1).cdf(nrvs[..., 0] * torch.sqrt(1 - self.theta**2) + 
                                 nrvs[..., 1] * self.theta) 
        return vals

    def log_prob(self, value, safe=False):
        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros_like(self.theta) # by default 0 and already on a correct device

        if safe==True:
            thetas = self.theta*conf.Gauss_Safe_Theta
        else:
            thetas = self.theta

        # Check CUDA and make a Normal distribution
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            nrvs = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(value)
        else:
            nrvs = normal.Normal(0,1).icdf(value)
        
        mask_theta = (thetas < 1.) & (thetas > -1.)
        m = 1e-6
        mask_samples = (value[..., 0] > m) & (value[..., 1] > m) & \
                (value[..., 0] < 1.-m) & (value[..., 1] < 1.-m)
        mask = mask_theta & mask_samples

        log_prob[..., mask] = (2 * thetas * nrvs[..., 0] * nrvs[..., 1] - thetas**2 \
            * (nrvs[..., 0]**2 + nrvs[..., 1]**2))[..., mask]
        log_prob[..., mask] /= 2 * (1 - thetas**2)[..., mask]
        log_prob[..., mask] -= torch.log(1 - thetas**2)[..., mask] / 2

        #check that formulas were computed correctly (without Nan or inf)
        assert torch.all(log_prob.abs()!=float("Inf"))
        assert torch.all(log_prob==log_prob)

        #now add inf were it is appropriate (will be ignored in integration anyway)
        log_prob[(thetas >= 1.)  & ((value[..., 0] - value[..., 1]).abs() < conf.Gauss_diag)]      = float("Inf") # u==v
        log_prob[(thetas <= -1.) & ((value[..., 0] - 1 + value[..., 1]).abs() < conf.Gauss_diag)]  = float("Inf") # u==1-v

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask_theta & ~mask_samples] = -float("Inf") 

        assert torch.all(log_prob==log_prob)
        
        return log_prob

class FrankCopula(SingleParamCopulaBase):
    '''
    This class represents a copula from the Frank family.
    '''
    arg_constraints = {"theta": constraints.interval(-conf.Frank_Theta_Max,conf.Frank_Theta_Max)}
    support = constraints.interval(0,1) # [0,1]
    
    def ppcf(self, samples):
        vals = samples[..., 0] #will stay this for self.theta == 0
        theta_ = self.theta.clone()#.abs() # generate everything for small or negative thetas, then flip
        theta_[self.theta > conf.Frank_Theta_Flip] = -self.theta[self.theta > conf.Frank_Theta_Flip] 
        u = samples[...,0]
        v = samples[...,1]
        vals = (-torch.log1p(u * torch.expm1(-theta_) \
                / (torch.exp(-theta_ * v) \
                - u * torch.expm1(-theta_ * v))) \
                / theta_)
        vals[..., self.theta==0] = samples[..., 0][...,self.theta==0]
        vals[..., self.theta > conf.Frank_Theta_Flip] = 1. - vals[..., self.theta > conf.Frank_Theta_Flip] # flip for highly positive thetas here
        return torch.clamp(vals,0.,1.) # could be slightly higher than 1 due to numerical errors

    def log_prob(self, value, safe=True):

        value[torch.isnan(value)] = 0 # log_prob = -inf

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros_like(self.theta) # by default 0 and already on a correct device
        
        mask = (self.theta.abs() > 1e-2) & (self.theta.abs() < conf.Frank_Theta_Max)
        theta_ = self.theta[mask]
        value_ = value.expand(self.theta.shape + torch.Size([2]))[mask]
        log_prob[..., mask] = torch.log(-theta_ * torch.expm1(-theta_)) \
                            - (theta_ * (value_[..., 0] + value_[..., 1])) \
                            - 2*torch.log(torch.abs(torch.expm1(-theta_)
                             + torch.expm1(-theta_ * value_[..., 0])
                             * torch.expm1(-theta_ * value_[..., 1])))

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask & ((value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1))] = -float("Inf") 

        assert torch.all(log_prob==log_prob)
        assert torch.all(log_prob!=float("Inf"))
        
        return log_prob

class ClaytonCopula(SingleParamCopulaBase):
    '''
    This class represents a copula from the Clayton family.
    '''
    arg_constraints = {"theta": constraints.interval(0.,conf.Clayton_Theta_Max)}
    support = constraints.interval(0.,1.) # [0,1]
    
    def ppcf(self, samples):
        min_lim = 0 #min value for accurate calculation of logpdf. Below -- independence copula
        thetas_ = self.theta.expand(samples.shape[:-1])
        vals = torch.zeros_like(thetas_)
        vals[thetas_<=min_lim] = samples[thetas_<=min_lim][..., 0] #for self.theta == 0
        nonzero_theta = thetas_[thetas_>min_lim]
        unstable_part = torch.empty_like(vals)
        unstable_part[thetas_>min_lim] = (samples[thetas_>min_lim][..., 0] * (samples[thetas_>min_lim][..., 1]**(1 + nonzero_theta))) \
                ** (-nonzero_theta / (1 + nonzero_theta))
        if unstable_part.numel() > 0:
            unstable_part = unstable_part.reshape(*samples.shape[:-1])
            mask = (thetas_>min_lim) & (unstable_part != float("Inf"))
            vals[mask] = (1 - samples[mask][..., 1]**(-thetas_[mask]) + unstable_part[mask])** (-1 / thetas_[mask])
            mask = (thetas_>min_lim) & (unstable_part == float("Inf"))
            vals[mask] = 0. # (inf)^(-1/nonzero_theta) is still something very small
        assert torch.all(vals==vals)
        return vals

    def log_prob(self, value, safe=True):

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        value = self._SingleParamCopulaBase__rotate_input(value.clone())
        log_prob = torch.zeros_like(self.theta) # by default

        value_ = value.expand(self.theta.shape + torch.Size([2]))
        
        #log_base = -torch.min(value[...,0],value[...,1]).log() # max_theta depends on the coordinate of the value
        #mask = (self.theta > 0) & (self.theta < conf.Clayton_Theta_Max)
        self.theta = torch.clamp(self.theta,0.,conf.Clayton_Theta_Max)
        log_prob = (torch.log(1 + self.theta) + (-1 - self.theta) \
                       * torch.log(value).sum(dim=-1) \
                       + (-1 / self.theta - 2) \
                       * torch.log(value_[...,0].pow(-self.theta) + value_[...,1].pow(-self.theta) - 1))
        log_prob[self.theta<1e-4] = 0.

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[..., (value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1)] = -float("Inf") 

        assert torch.all(log_prob==log_prob)
        assert torch.all(log_prob!=float("Inf"))

        #log_prob[(self.theta<1e-2) | (self.theta>16.)] = -float("Inf") 
        
        return log_prob

class GumbelCopula(SingleParamCopulaBase):
    '''
    This class represents a copula from the Gumbel family.
    '''
    arg_constraints = {"theta": constraints.interval(1.,conf.Gumbel_Theta_Max)}
    support = constraints.interval(0,1) # [0,1]
    
    def ppcf(self, samples):

        def h(z,samples):
            x = -samples[...,1].log()
            return z+(thetas-1)*z.log() - (x + (thetas-1)*x.log()-samples[...,0].log())

        def hd(z):
            return 1+(self.theta-1)*z.pow(-1)

        thetas = torch.clamp(self.theta,1.0,conf.Gumbel_Theta_Max)

        x = -samples[...,1].log()
        z = x
        thetas_ = thetas.expand_as(z)
        for _ in range(3):             #increase number of Newton-Raphson iteration if sampling fails
            z = z - h(z,samples)/hd(z)
            y = (z.pow(thetas) - x.pow(thetas)).pow(1/thetas)

        v = torch.exp(-y)
        # assert torch.all(v>0)
        # assert torch.all(v<1)
        v = torch.clamp(v,0,1) # for theta>10 v is sometimes >1
        return v

    def log_prob(self, value, safe=True):

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        value = self._SingleParamCopulaBase__rotate_input(value.clone())
        log_prob = torch.zeros_like(self.theta) # by default

        self.theta = torch.clamp(self.theta,1.,conf.Gumbel_Theta_Max)

        h1 = self.theta - 1.0
        h2 = (1.0 - 2.0 * self.theta) / self.theta
        h3 = 1.0 / self.theta

        h4 = -value[...,0].log()
        h5 = -value[...,1].log()
        h6 = (torch.pow(h4, self.theta) + torch.pow(h5, self.theta))
        h7 = torch.pow(h6, h3)

        log_prob = -h7+h4+h5 + h1*h4.log() + h1 * h5.log() + h2 * h6.log().clamp(-1e38,float("Inf")) + (h1+h7).log()

        # # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[..., (value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1)] = -float("Inf") 

        assert torch.all(log_prob==log_prob)
        assert torch.all(log_prob!=float("Inf"))
        
        return log_prob

class StudentTCopula(SingleParamCopulaBase):
    '''
    This class represents a copula from the Student T family.
    '''    
    arg_constraints = {"theta": constraints.interval(-1,1)} 
    # here theta is actually Kendall's tau
    # it does not make sence to transform it through pi/2 * sin(tau) to get traditional theta
    support = constraints.interval(0,1)

    @staticmethod
    def cdf_approx_2(x):
        return 0.5 + 0.5 * x / torch.sqrt(x**2 + 2.)

    @staticmethod
    def icdf_approx_2(y):
        return torch.sign(y-0.5) * torch.sqrt(2/(1/(2*y-1)**2 - 1))

    @staticmethod
    def icdf_approx_3(y):
        x = torch.sign(y-0.5) * torch.sqrt(2/(1/(2*y-1)**2 - 1)) #icdf_approx_2
        ar = torch.atan(x/torch.sqrt(torch.tensor([3.])))/2 # initial seed in form trigonometric
        PI=torch.acos(torch.Tensor([-1]))
        for _ in range(3):
            ar = ar - (ar+ 0.5*torch.sin(2*ar) - PI * (y-0.5))/(1+torch.cos(2*ar))
        return torch.sqrt(torch.tensor([3.]))*torch.tan(ar)
    
    def ppcf(self, samples):
        '''
        all packages actually invert cdf, like R:
        https://github.com/SurajGupta/r-source/blob/master/src/nmath/qt.c
        which is not an option on GPU;
        There are many approximations for particular dfs, described in:
        William T. Shaw Sampling Student’s T distribution – use of the
        inverse cumulative distribution function
        He also describes a sampling method that oes not have to use icdf at all
        (Bailey’s method), which is also, unfortunatelly, iterative,
        meaning that it is not GPU-friendly.

        For now, we'll simply use an pproximation for df=2.0:
        cdf = 1/2 + x/2/sqrt(x**2+2)
        icdf = sign(x-0.5) * sqrt(2/(1/(2y-1)**2 - 1))
        '''
        kTM = torch.zeros_like(self.theta)
        kTM[self.theta<0] = 1
        h1 = 1.0 - torch.pow(self.theta, 2.0)
        df = 2.0
        nu1 = df + 1.0 # nu1 = theta[1] + 1.0
        dist1 = studentT.StudentT(df=df, scale=1.0, loc=0.0)
        #dist2 = studentT.StudentT(df=nu1, scale=1.0, loc=0.0)

        samples[...,0] = kTM + torch.sign(self.theta) * samples[...,0]  # TODO: check input bounds

        # inverse CDF yields quantiles
        x = self.icdf_approx_3(samples[...,0])   #dist2
        y = self.icdf_approx_2(samples[...,1])   #dist1

        # eval H function
        vals = self.cdf_approx_2(x * torch.sqrt((df + torch.pow(y, 2.0)) * h1 / nu1) + self.theta * y)
        return vals

    def log_prob(self, value, safe=True):

        self.exp_thr = torch.tensor(torch.finfo(torch.float32).max).log()

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros(self.theta.shape) # by default

        value_ = value.expand(self.theta.shape + torch.Size([2]))

        rho2 = torch.pow(self.theta, 2.0)
        df = 2.0
        h1 = 1.0 - rho2
        h2 = df / 2.0
        h3 = h2 + 0.5
        h4 = h2 + 1.0
        h5 = 1.0 / df
        h6 = h5 / h1

        PI=torch.acos(torch.Tensor([-1]))
        
        y = self.icdf_approx_2(value)
        
        log_prob = torch.lgamma(torch.tensor(h4))+torch.lgamma(torch.tensor(h2)) \
            - 0.5*h1 - 2.*torch.lgamma(torch.tensor(h3)) \
            + h3*torch.log(1+h5*y[...,0]**2) + h3*torch.log(1+h5*y[...,1]**2) \
            - h4*torch.log(1+h6*(y[...,0]**2+y[...,1]**2 - 2*self.theta*y[...,0]*y[...,1]))

        # log_prob = -0.5*(1-self.theta**2).log() + torch.lgamma(torch.tensor((self.df + 2.)/2.)) \
        #         - torch.lgamma(torch.tensor(self.df/2.)) - torch.log(PI*self.df) \
        #         - (self.df + 2.)/2. * torch.log(1. + (torch.sum(value**2,dim=-1)
        #             - 2*self.theta*value[...,0]*value[...,1])/self.df/(1.-self.theta**2))

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[..., (value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1)] = -float("Inf") 

        assert torch.all(log_prob==log_prob)
        #assert torch.all(log_prob!=float("Inf"))
        print(value_[log_prob!=float("Inf")])
        print(self.theta[log_prob!=float("Inf")])
        
        return log_prob

class MixtureCopula(Distribution):
    '''
    This class represents Mixture copula
    '''
    has_rsample = True
    arg_constraints = {"theta": constraints.real, 
                       "mix": constraints.interval(0,1)} #TODO:write simplex constraint for leftmost dimention
    support = constraints.interval(0,1) # [0,1]
    
    def __init__(self, theta, mix, copulas, rotations=None, theta_sharing = None, validate_args=None):
        self.theta = theta
        if theta_sharing is not None:
            self.theta_sharing = theta_sharing
        else:
            self.theta_sharing = torch.arange(0,len(copulas)).long()
        self.mix = mix
        self.copulas = copulas
        if rotations:
            self.rotations = rotations
        else:
            self.rotations = [None for _ in copulas]
        #TODO Check theta when there will be more than 1 param. Now it is checked by gpytorch
        batch_shape, event_shape = self.theta.shape, torch.Size([2])
        super(MixtureCopula, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def _entropy_given_params(self, alpha=0.05, sem_tol=1e-3):
        '''
            TODO: write docstr
        '''
        # Gaussian confidence interval for sem_tol and level alpha
        conf = torch.erfinv(torch.tensor([1. - alpha]))
        sem = torch.ones(self.theta.shape[1])*float('inf')
        ent = torch.zeros(self.theta.shape[1])
        var_sum = torch.zeros(self.theta.shape[1])
        log2 = torch.tensor([2.]).log()
        k = 0
        mc_size = self.theta.shape[-1]
        with torch.no_grad():
            while torch.all(sem >= sem_tol):
                # Generate samples
                samples = self.rsample()
                assert samples.dim()==3 # [conditioning_variable, samples, 2]
                logp = self.log_prob(samples)
                assert torch.all(logp==logp)
                assert torch.all(logp.abs()!=float("inf")) #otherwise make masked tensor below
                log2p = logp / log2 #maybe should check for inf 2 lines earlier
                k += 1
                # Monte-Carlo estimate of entropy
                ent += (-log2p.mean(dim=(-1)) - ent) / k
                # Estimate standard error
                var_sum += ((-log2p.t() - ent) ** 2).sum(dim=0)
                sem = conf * (var_sum / (k * mc_size * (k * mc_size - 1))).pow(.5)
        return ent#, sem
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MixtureCopula, _instance)
        batch_shape = torch.Size(batch_shape)
        if batch_shape == torch.Size([]):
            batch_shape = torch.Size([1])
        theta_shape = batch_shape + torch.Size(self.event_shape[:-1])
        new.theta = self.theta.expand(theta_shape) 
        new.mix = self.mix.expand(theta_shape)
        super(MixtureCopula, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new.copulas = self.copulas
        new.rotations = self.rotations
        new.theta_sharing = self.theta_sharing
        new._validate_args = self._validate_args
        return new
    
    def rsample(self, sample_shape=torch.Size([])):
        shape = self._extended_shape(sample_shape) # now it is copulas x sample_size x 2 (event)    
        samples = torch.zeros(size=shape[1:])
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            samples = samples.cuda(device=get_cuda_device)

        num_thetas = self.theta.shape[0]
        
        assert self.mix.shape[0]==len(self.copulas)
        #assert self.theta_sharing.shape[0]==len(self.copulas)
        #gplink already returns thetas for each copula
        #TODO: avoid returning identical thetas
        
        onehot = torch.distributions.one_hot_categorical.OneHotCategorical(
            probs=torch.einsum('i...->...i', self.mix)).sample()
        onehot = torch.einsum('...i->i...', onehot)
        onehot = onehot.type(torch.bool)
        for i,c in enumerate(self.copulas):
            if c.num_thetas == 0:
                samples[onehot[i],...] = c(self.theta[self.theta!=self.theta]).sample(self.theta[i,onehot[i]].shape)
            else:
                samples[onehot[i],...] = c(self.theta[i,onehot[i]], rotation=self.rotations[i]).sample()
        
        return samples

    def log_prob(self, value, safe=False):

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        prob = torch.zeros_like(self.theta[0]) # by default
        
        assert self.mix.shape[0]==len(self.copulas)
        #assert self.theta_sharing.shape[0]==len(self.copulas)

        if self._validate_args:
            sum_mixes = self.mix.sum(dim=0)
            assert torch.allclose(sum_mixes,torch.ones_like(sum_mixes),atol=0.01)
        
        for i, c in enumerate(self.copulas):
            if c.num_thetas == 0:
                add = c(self.theta[self.theta!=self.theta]).log_prob(value)
            else:
                add = c(self.theta[i], rotation=self.rotations[i]).log_prob(value,safe=safe).clamp(-100.,88.)
                #-100 to 88 is a range of x such that torch.exp(x).log()!=+-inf
            #print(add.min(),add.max(),self.mix[i])
            prob += self.mix[i]*torch.exp(add)
            #TODO is it possible to vectorize this part?
        log_prob = prob.log()
    
        assert torch.all(log_prob==log_prob)
        #assert torch.all(log_prob!=float("inf")) #can be -inf though

        return log_prob

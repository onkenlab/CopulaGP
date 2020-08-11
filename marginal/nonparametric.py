import numpy as np
from fastkde import fastKDE
import scipy.interpolate as intrp

def interpolate2d_masked_array(masked_array2d):
    m = ~masked_array2d.mask.flatten()
    X,Y = masked_array2d.shape
    xx,yy = np.mgrid[:X,:Y]
    x = np.unique(xx.flatten()[m])
    y = np.unique(yy.flatten()[m])
    sel = np.ix_(x,y)
    if any(masked_array2d.mask[sel].flatten()):
        raise ValueError("Points are not on the grid")
    return intrp.RectBivariateSpline(x,y,masked_array2d.data[sel], kx=1, ky=1)

def transform_coord(x,y,axes):
    f_x = np.clip(np.sum(axes[0]<x),1,len(axes[0])-1)
    f_y = np.clip(np.sum(axes[1]<y),1,len(axes[1])-1)
    wx = (axes[0][f_x]-x)/(axes[0][f_x]-axes[0][f_x-1])
    wy = (axes[1][f_y]-y)/(axes[1][f_y]-axes[1][f_y-1])
    return (f_y - wy),(f_x - wx)

def fast_signal2uniform(Y,X,Y_=None,X_=None,numPointsPerSigma=20):
    '''
    Transforms the data Y with fast conditional KDE.
    :math:`Y' = CDF_{Y|X_}^{-1} (Y_)`
    By default, Y_ = Y, X_ = X.
    The CDF is always estimated with KDE using (X,Y).

    Parameters:
    ----------
    :attr:`Y` (:class:`np.array`)
            Values of :math:`Y`.
    :attr:`X` (:class:`np.array`)
            Values of :math:`X`.
    :attr:`Y_` (:class:`np.array`, optional)
            Values of :math:`Y_` to be transformed.
            If not provided, Y_ = Y.
    :attr:`X_` (:class:`np.array`, optional)
            Values of :math:`X_` to be transformed.
            If not provided, X_ = X.
    Returns:
    ----------
        `Y'` (:class:`np.array`)
            Transformed values :math:`Y'`.
    '''
    if Y_ is None:
        assert X_ is None, "If Y_ is None, why X_ is not?"
        X_ = X
        Y_ = Y

    pOfYGivenX,axes = fastKDE.conditional(Y,X,numPointsPerSigma=numPointsPerSigma)

    #make CDF from PDF
    cOfYGivenX = np.empty_like(pOfYGivenX)
    for i, p in enumerate(pOfYGivenX.T):
        cOfYGivenX[:,i] = np.cumsum(p)/np.sum(p)
        
    f = interpolate2d_masked_array(cOfYGivenX)
    #transform data points as CDF(x)
    s_tr = np.zeros_like(Y_)
    for i, (x,y) in enumerate(zip(X_,Y_)):
        s_tr[i] = f(*transform_coord(x,y,axes))

    return s_tr

def zeroinflated_signal2uniform(Y,X,Y_=None,X_=None,numPointsPerSigma=50):
    '''
    Transforms the zero-inflated data Y with fast conditional KDE.
    For :math:`Y_>0`:
        :math:`Y' = CDF_{Y|X_}^{-1} (Y_) (1-p0) + p0`
    else:
        :math:`Y' ~ U[0,p0]`.
    In other words, zeros are transformed to U[0,p0] and the rest of the data
    is transformed to U(p0,1]. As a result, the transformed data is U[0,1]. 
    By default, Y_ = Y, X_ = X. The CDF is always estimated with KDE using (X,Y).

    Parameters:
    ----------
    :attr:`Y` (:class:`np.array`)
            Values of :math:`Y`.
    :attr:`X` (:class:`np.array`)
            Values of :math:`X`.
    :attr:`Y_` (:class:`np.array`, optional)
            Values of :math:`Y_` to be transformed.
            If not provided, Y_ = Y.
    :attr:`X_` (:class:`np.array`, optional)
            Values of :math:`X_` to be transformed.
            If not provided, X_ = X.
    Returns:
    ----------
        `Y'` (:class:`np.array`)
            Transformed values :math:`Y'`.
    '''
    if Y_ is None:
        assert X_ is None, "If Y_ is None, why X_ is not?"
        X_ = X
        Y_ = Y

    transformed = np.empty_like(Y_)
    zeros = len(Y_[Y_==0])
    part_zero = zeros/len(Y_)
    transformed[Y_!=0] = part_zero + (1-part_zero)* \
        fast_signal2uniform(Y[Y!=0],X[Y!=0],Y_[Y_!=0],X_[Y_!=0],numPointsPerSigma=numPointsPerSigma)
    # ^ here we are using nonzero part of (X,Y) to estimate CDF, and transform the nonzero Y_
    transformed[Y_==0] = np.random.rand(zeros)*part_zero

    return transformed

def single_unit_MI(response,stimulus):
    '''
    Estimates mutual information (MI) between a single variable and a conditioning variable:
    .. math::
        I(S;R) = \sum\limits_{r,s} P(r|s) P(s)\, \mathrm{log_2}\frac{P(r|s)}{P(r)}
    Args:
        :attr:`response` (:class:`np.array`)
            Values of :math:`R`.
        :attr:`stimulus` (:class:`np.array`)
            Values of :math:`S`.
        :attr:`kwargs`
    Returns
        `real` (mutual information in bits)
    '''
    pOfYGivenX,axes = fastKDE.conditional(response,stimulus,numPointsPerSigma=30)
    working_range = [np.sum(axes[0]<np.min(stimulus)),np.sum(axes[0]<np.max(stimulus))]
    assert np.all(pOfYGivenX[:,working_range[0]:working_range[1]].mask==False)
    Ns = pOfYGivenX[:,working_range[0]:working_range[1]].shape[1]
    pRS = pOfYGivenX[:,working_range[0]:working_range[1]]
    pR = np.sum(pRS,axis=1)*1/Ns
    I=0
    for i in range(Ns):
        I+=np.sum(pRS[:,i]*1/Ns*np.log(pRS[:,i]/pR)/np.log(2))
    return I

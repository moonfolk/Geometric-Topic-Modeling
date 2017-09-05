import numpy as np
from sklearn.preprocessing import normalize

from scipy.sparse.linalg import norm
from numpy.linalg import norm as np_norm
from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator, ClusterMixin

import pylab
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

## Functions for operations on csr sparse matrices
def sparse_norm(data, cent):
    cent_norm = np_norm(cent)
    dot_p = data.dot(cent)
    raw_norms = norm(data, axis=1)
    norms = np.sqrt(raw_norms**2 - 2*dot_p + cent_norm**2)
    return norms

def sparse_cos_sim(data, t, cent, norms):
    t_cent = np.dot(t, cent)
    data_dot = data.dot(t)
    cos_dist = 1 - (data_dot-t_cent)/(np_norm(t) * norms)
    return cos_dist

## Plotting function. Used when toy=True
def plot_clust(data, centers, clusters):
    fig, ax = plt.subplots()
    if 'blue' in set(clusters):
        ax.scatter(data[:,0], data[:,1], color=clusters, alpha=1, s=0.05)
    else:
        ax.scatter(data[:,0], data[:,1], c=clusters, s=0.05)
    y_data = ax.get_ylim()
    x_data = ax.get_xlim()
    all_x = centers[:,0].tolist() + list(x_data)
    all_y = centers[:,1].tolist() + list(y_data)
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    for i in range(centers.shape[0]):
        ax.annotate('', xytext=(0,0), xy=(centers[i,0], centers[i,1]),
                    arrowprops=dict(facecolor='black'))
    plt.show()

## Mean Shifting procedure
def angle_update(data, cent, norms, min_cos, algo0 = False, plot=False):
    cur = np.argmax(norms)
    new_t = data[cur,:].A.flatten() - cent
    if algo0:
        cos_dist = sparse_cos_sim(data, new_t, cent, norms)
        if plot:
            clust = np.array(np.repeat('blue', len(norms)), dtype='|S5')
            clust[cos_dist<min_cos] = 'red'
            plot_clust(data.toarray()-cent, np.array([new_t]), clust)
        return new_t, cos_dist
    alpha_old = 1.
    alpha_new = 0.
    it = 0
    while np.abs(alpha_old-alpha_new)>1e-03  and it<50:
        it += 1
        alpha_old = alpha_new
        old_t = new_t
        cos_dist = sparse_cos_sim(data, old_t, cent, norms)
        near_doc = cos_dist < min_cos
        new_t = data[near_doc,:].mean(axis=0).A.flatten() - cent
        alpha_new = cosine(old_t, new_t)
    new_t = new_t/np_norm(new_t)
    new_t *= max(data[near_doc,:].dot(new_t) - np.dot(new_t, cent))
    cos_dist = sparse_cos_sim(data, new_t, cent, norms)
    if plot:
        clust = np.array(np.repeat('blue', len(norms)), dtype='|S5')
        clust[cos_dist<min_cos] = 'red'
        plot_clust(data.toarray()-cent, np.array([new_t]), clust)
    return new_t, cos_dist

## SPARSE Conic Scan algorithm
def angle_raw(wdfn, cent, max_discard=100, delta=0.4, prop_discard=0.5, prop_n = 0.01, verbose=False, algo0 = False, toy=False):

    beta_angle = []
    M = wdfn.shape[0]
    ind_remain = np.arange(M)
    norms = sparse_norm(wdfn, cent)
    remain_bool = [M, 0]
    cut_r = np.percentile(norms, 100*prop_discard)
    min_neighbor = prop_n * M
    big_norms = norms>cut_r
    disc_count = 0
    while sum(remain_bool)>0:
        new_t, cos_dist = angle_update(wdfn, cent, norms, delta, algo0, toy)
        remain_bool = cos_dist > delta
        if len(remain_bool) - sum(remain_bool) > min_neighbor:
            if verbose:
                print 'Keeping'
            beta_angle += [cent + new_t]
        else:
            if verbose:
                print 'Discarding'
            disc_count += 1
        if verbose:
            ## Plot
            cos_sort = np.argsort(cos_dist)
            color = np.array(np.repeat('blue', len(remain_bool)), dtype='|S5')
            color[remain_bool[cos_sort]] = 'red'
            color[big_norms[cos_sort]] = 'green'
            plt.figure()
            plt.scatter(cos_dist[cos_sort], norms[cos_sort], color=color, alpha=1)
            plt.show()
            
        # Remove processed data
        norms = norms[remain_bool]
        wdfn = wdfn[remain_bool,:]
        ind_remain = ind_remain[remain_bool]
        big_norms = big_norms[remain_bool]
        if sum(big_norms)==0:
            if verbose:
                print 'Only small norms remain'
            break
        if disc_count == max_discard:
            if verbose:
                print 'Discarded %d times ... stopping' % max_discard
            break
    # Renormalizing topics
    betas = np.array(beta_angle)
    betas[betas<0] = 0
    betas = normalize(betas, 'l1')
    return betas

## Spherical k-means
def sph_means(data, cent, init=None, K=0, it=1000, toy=False, only_direct=True):
    M, V = data.shape
    if init is None:
        ctopics = data[np.random.choice(range(M), K, replace=False),:].toarray()- cent
    else:
        ctopics = np.copy(init) - cent
        K = ctopics.shape[0]
    
    norms = sparse_norm(data, cent)
    
    for i in range(it):
        D = []
        ## Get clustering
        for k in range(K):
            D.append(sparse_cos_sim(data, ctopics[k], cent, norms))
        clusters = np.argmin(D, axis=0)
    
        ## Update centers
        for k in range(K):
            c = np.where(clusters == k)[0]
            if len(c) > 0:
                ctopics[k] = data[c,:].mean(axis=0).A.flatten() - cent
                if i == it-1 and not only_direct:
                    ctopics[k] = ctopics[k]/np_norm(ctopics[k])
                    ctopics[k] *= max(data[c,:].dot(ctopics[k]) - np.dot(ctopics[k], cent))
        if toy:
            plot_clust(data.toarray()-cent, ctopics, clusters)
    
    ctopics = ctopics + cent
    ctopics[ctopics<0] = 0
    ctopics = normalize(ctopics, 'l1')
    return ctopics, clusters

## Minimum mathcing distance
def min_match(beta, beta_t):
    b_to_t = np.apply_along_axis(lambda x: np.sqrt(((beta_t-x)**2).sum(axis=1)), 1, beta)
    return max([max(np.min(b_to_t, axis=0)), max(np.min(b_to_t, axis=1))])

## Class wrapper for the algorithms
class geom_tm(BaseEstimator, ClusterMixin):  
    def __init__(self, max_discard=100, delta=0.4, prop_discard=0.5, prop_n=0.01, verbose=False, algo0 = False, toy=False):

        self.max_discard = max_discard # number of discards before algorithm stops
        self.delta = delta # cone angle (\omega in the paper)
        self.prop_discard = prop_discard # quantile to compute \mathcal{R}
        self.prop_n = prop_n # proportion of data to be used as outlier threshold - \lambda
        self.verbose = verbose # whether to plot cosine-norm plots
        self.algo0 = algo0 # whether to use Algorithm 1 - i.e. without mean shifting
        self.toy = toy # only set to True if V=K=3. It will plot triangles then

    def fit_a(self, data, cent):
        
        self.a_betas_ = angle_raw(data, cent, self.max_discard, self.delta, self.prop_discard, self.prop_n, self.verbose, self.algo0, self.toy)
        self.K_ = self.a_betas_.shape[0]
        return self
    
    def fit_sph(self, data, cent, init=None, K=0, it = 10, only_direct=False):
        if init is None and hasattr(self, 'a_betas_'):
            init = self.a_betas_
            K = self.K_
        
        self.sph_betas_, self.sph_clust_ = sph_means(data, cent, init, K, it, self.toy, only_direct)
        return self
    
    def fit_all(self, data, cent, it=5):
        self = self.fit_a(data, cent)
        self = self.fit_sph(data, cent, it=it)
#        self = self.fit_m(data, cent)
        return self
    def score_beta(self, beta_t):
        score = []
        if hasattr(self, 'a_betas_'):
            score.append(min_match(self.a_betas_, beta_t))
        if hasattr(self, 'sph_betas_'):
            score.append(min_match(self.sph_betas_, beta_t))
        if hasattr(self, 'm_betas_'):
            score.append(min_match(self.m_betas_, beta_t))
        return score
        
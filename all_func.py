import numpy as np
from numpy.linalg import lstsq, norm

from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize

import logging
logging.disable(logging.INFO)

import time

import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

## Data simul
def gen_data(V=1000, K=10, M=1000, Nm=100, eta=0.1, alpha=0.1, M_test=500, anchor=False, equi=False):
    beta_t = np.random.dirichlet(np.ones(V)*eta, K)
    if equi:
        if K!=V:
            print 'DONT DO IT'
        beta_t = np.eye(K)
    theta_t = np.random.dirichlet(np.ones(K)*alpha, M)
    theta_test = np.random.dirichlet(np.ones(K)*alpha, M_test)
    anchor_set = []
    if anchor:
        for i in range(K):
            cand = np.argmax(beta_t[i,:])
            anchor_set.append(cand)
            beta_t[np.arange(K)!=i,cand] = 0
        beta_t = np.apply_along_axis(lambda x: x/x.sum(), 1, beta_t)
    simplex = np.dot(theta_t, beta_t)
    wdf = np.apply_along_axis(lambda x: np.random.multinomial(Nm, x), 1, simplex).astype('float')
#    sum(wdf.sum(axis=0)==0)
    full_set = wdf.sum(axis=0)>0
    new_V = [i for i in range(V) if full_set[i]]
    anchor_set = [new_V.index(i) for i in anchor_set]
    wdf = wdf[:,full_set]
    beta_t = normalize(beta_t[:,full_set], 'l1')
    simplex_test = np.dot(theta_test, beta_t)
    wdf_test = np.apply_along_axis(lambda x: np.random.multinomial(Nm, x), 1, simplex_test).astype('float')
    return wdf, beta_t, theta_t, simplex[:,full_set], anchor_set, theta_test, wdf_test

## GDM algorithms
def get_beta(cent, centers, m):
    betas = np.array([cent + m[x]*(centers[x,:] - cent) for x in range(centers.shape[0])])
    betas[betas<0] = 0
    betas = normalize(betas, 'l1')
    return betas

def gdm(wdfn, K, ncores=-1):
    glob_cent = np.mean(wdfn, axis=0)
    kmeans = KMeans(n_clusters=K, n_jobs=ncores, max_iter=1000).fit(wdfn)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    m = []
    for k in range(K):
        k_dist = euclidean(glob_cent, centers[k])
        Rk = max(np.apply_along_axis(lambda x: euclidean(glob_cent, x), 1, wdfn[labels==k,:]))
        m.append(Rk/k_dist)
    
    beta_means = get_beta(glob_cent, centers, m)
    
    return beta_means

## Geometric Theta
def proj_on_s(beta, doc, K, ind_remain=[], first=True, distance=False):
    if first:
        ind_remain = np.arange(K)
    s_0 = beta[0,:]
    if beta.shape[0]==1:
        if distance:
            return norm(doc-s_0)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = 1.
            return theta
    beta_0 = beta[1:,:]
    alpha = lstsq((beta_0-s_0).T, doc-s_0)[0]
    if np.all(alpha>=0) and alpha.sum()<=1:
        if distance:
            p_prime = (alpha*(beta_0-s_0).T).sum(axis=1)
            return norm(doc-s_0-p_prime)
        else:
            theta = np.zeros(K)
            theta[ind_remain] = np.append(1-alpha.sum(), alpha)
            return theta
    elif np.any(alpha<0):
        ind_remain = np.append(ind_remain[0], ind_remain[1:][alpha>0])
        return proj_on_s(np.vstack([s_0, beta_0[alpha>0,:]]), doc, K, ind_remain, False, distance)
    else:
        return proj_on_s(beta_0, doc, K, ind_remain[1:], False, distance)

## Evaluation
def min_match(beta, beta_t):
    b_to_t = np.apply_along_axis(lambda x: np.sqrt(((beta_t-x)**2).sum(axis=1)), 1, beta)
    return max([max(np.min(b_to_t, axis=0)), max(np.min(b_to_t, axis=1))])

def perplexity(docs, beta, theta='geom', scale=True):
  if type(theta)==str:
      theta = np.apply_along_axis(lambda x: proj_on_s(beta, x, beta.shape[0]), 1, normalize(docs, 'l1'))  
      scale = True
  est = np.dot(theta, beta)
  if scale:
      est = np.log(normalize(np.apply_along_axis(lambda x: x + x[x>0].min(), 1, est), 'l1'))
      mtx = docs * est
  else:
      est = np.log(est)
      mtx = docs * est
      mtx[np.isnan(mtx)] = 0.
  return np.exp(-mtx.sum()/docs.sum())

def get_stat(t_s, beta_est, beta_t, data_test, theta_test, hdp_gibbs = False):
    mm = min_match(beta_est, beta_t)
    if type(hdp_gibbs)!=bool:
        pp = perplexity(data_test, hdp_gibbs, theta=theta_test, scale=False)
        if pp == np.inf:
            pp = perplexity(data_test, hdp_gibbs, theta=theta_test, scale=True)
    else:
        pp = perplexity(data_test, beta_est, theta=theta_test, scale=False)
        if pp == np.inf:
            pp = perplexity(data_test, beta_est, theta=theta_test, scale=True)
    t = time.time() - t_s
    return [mm, pp, t]

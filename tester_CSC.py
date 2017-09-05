import sys
import time
import numpy as np
from sklearn.preprocessing import normalize
import os
from scipy.sparse import csr_matrix

cur_dir =  os.path.dirname(os.path.realpath('tester_CSC.py'))
sys.path.insert(0, cur_dir)
from all_func import gen_data, gdm, get_stat
from geom_tm import geom_tm
np.random.seed(1)

V = 2000 # vocabulary size
K = 15 # number of topics
M = 5000 # number of documents
Nm = 500 # words per document
eta = 0.1 # Dirichlet topic parameter
M_test = M/4 # size of test set for perplexity comparison
alpha = np.repeat(0.1, K) # Dirichlet topic proportions parameter
gen_param = [V, K, M, Nm, eta, alpha, M_test]

wdf, beta_t, theta_t, simplex, anchor_set, theta_test, wdf_test = gen_data(*gen_param)

wdf_s = csr_matrix(wdf)

V = wdf.shape[1]
wdfn_s = normalize(wdf_s, 'l1')
wdfn = normalize(wdf, 'l1')
cent = wdfn.mean(axis=0)

results = {}


## Raw angle
t_s = time.time()
tm = geom_tm(verbose=False, delta=0.6, prop_n=0.001, prop_discard=0.5)
tm.fit_a(wdfn_s, cent) # fit Cone Scan wihout spherical k-means
print 'CS learned %d topics; true K is %d' % (tm.K_, K) # number of topics learned
tm.fit_sph(wdfn_s, cent, it=30) # Peform spherical k-means with it iterations using Cone Scan as initialization
results['Angle'] = get_stat(t_s, tm.sph_betas_, beta_t, wdf_test, 'geom')
print 'Angle method Min-Match distance is %f, Perplexity is %f, took %d sec' % tuple(results['Angle'])

## GDM method
t_s = time.time()
gdm_t = gdm(wdfn, K)
results['GDM'] = get_stat(t_s, gdm_t, beta_t, wdf_test, 'geom')
print 'GDM Min-Match distance is %f, Perplexity is %f, took %d sec' % tuple(results['GDM'])

## Gibbs
import lda
t_s = time.time()
model_g = lda.LDA(n_topics=K, n_iter=300, refresh=1500)
model_g.fit(wdf_s.astype(int))
gibbs_t = model_g.topic_word_
#gibbs_theta = model_g.doc_topic_
gibbs_theta_test = model_g.transform(wdf_test.astype(int), max_iter=20)
results['LDA_Gibbs'] = get_stat(t_s, gibbs_t, beta_t, wdf_test, gibbs_theta_test)
print 'Collapsed Gibbs sampler Min-Match distance is %f, Perplexity is %f, took %d sec' % tuple(results['LDA_Gibbs'])

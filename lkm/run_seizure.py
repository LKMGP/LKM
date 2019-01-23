import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from tensorflow.train import AdamOptimizer
from gpflowSlim import Param, transforms
from gpflowSlim.kernels import RBF, Periodic, Product
from lkm.lkm_slim import LKM
import scipy.io as sio

import scipy.io as sio

exp = 2
data_file = "../data/seizure/get_50_exp_{}.mat".format(exp)

iterations = 300

data = sio.loadmat(data_file)

N = data['X'].shape[0]

additive_kernels = []

with tf.variable_scope("gp_hyperparameters"):
    # SE
    for i in range(2):
        k = RBF(1, variance=np.random.uniform(0.5, 1.5),
                lengthscales=np.random.uniform(0.5, 1.5),
                name="SE_{}".format(i))
        additive_kernels.append(k)

    # PER
    for i in range(2):
        k = Periodic(1, variance=np.random.uniform(0.5, 1.5),
                     lengthscales=np.random.uniform(0.5, 1.5),
                     period=np.random.uniform(5., 20.),
                     name="PER_{}".format(i))
        additive_kernels.append(k)

    # PER x SE
    for i in range(2):
        per = Periodic(1, variance=np.random.uniform(0.5, 1.5),
                     lengthscales=np.random.uniform(0.5, 1.5),
                     period=np.random.uniform(5., 20.),
                     name="Prod_PER_{}".format(i))

        se = RBF(1, variance=np.random.uniform(0.5, 1.5),
                       lengthscales=np.random.uniform(0.5, 1.5),
                       name="Prod_SE_{}".format(i))
        k = Product([per,se], name="Prod_{}".format(i))
        additive_kernels.append(k)

    likelihoods = [Param(0.01, transforms.positive, name="gaussian_noise_{}".format(n)) for n in range(N)]

model = LKM(data, additive_kernels, likelihoods)

gp_train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gp_hyperparameters")

ibp_train_vars = list(set(tf.global_variables()) - set(gp_train_vars))
update_tau = model.closed_form_update_tau()
elbo = model.build_marginal_loglikelihood()

z, nll_gp_refined = model.refine()

t_test, K, K_star, K_star_star, noise = model.prepare_for_postprocess()

# train IBP parameters with Adam
adam = AdamOptimizer(0.01)
# train_ibp = adam.minimize(-elbo, var_list=ibp_train_vars)
train_ibp = adam.minimize(-elbo, var_list=ibp_train_vars)

train_gp = ScipyOptimizerInterface(-elbo,
                                   var_list=gp_train_vars,
                                   method='L-BFGS-B',
                                   options={"maxiter": 10})

# refined train
train_gp_refine = ScipyOptimizerInterface(nll_gp_refined,
                                          var_list=gp_train_vars,
                                          method='L-BFGS-B',
                                          options={"maxiter": 300}
                                          )

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(iterations):
        # coordinate descent
        sess.run([update_tau])

        [sess.run([train_ibp]) for _ in range(5)]

        train_gp.minimize(sess)

        e_llike = sess.run(elbo)
        print('Iter {} : EBLO = {:.2f}'.format(i, e_llike))

    train_gp_refine.minimize(sess)

    z_eval = sess.run(z)
    z_eval = np.concatenate(z_eval)
    K_eval = sess.run(K)
    K_star_eval = sess.run(K_star)
    K_star_star_eval = sess.run(K_star_star)
    noise_eval = sess.run(noise)

    sio.savemat("./seizure_exp_{}.mat".format(exp),
                {
                    "Z": z_eval,
                    "t": data["t"],
                    "t_test": t_test,
                    "K": K_eval,
                    "K_star": K_star_eval,
                    "K_star_star": K_star_star_eval,
                    "noise": noise_eval
                })

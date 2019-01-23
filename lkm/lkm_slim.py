import numpy as np
import tensorflow as tf

from tensorflow.distributions import Bernoulli
from lkm.util import gumbel_softmax_v2, log_det
from gpflowSlim import Param, transforms
import matplotlib.pyplot as plt

float_type = tf.float64
jitter = 1.e-6


class IBP(object):

    """
    Base IBP contains initial variational parameters, compute KL w.r.t IBP variational parameters, entropy
    """

    def __init__(self, num_ibp_features, num_ibp_objects, alpha = 1., q_tau = None, q_nu = None, finite=True):
        self.num_ibp_features = num_ibp_features
        self.num_ibp_objects = num_ibp_objects
        self.alpha = alpha
        self.finite=finite

        self._init_ibp_variational_parameters(q_tau, q_nu, alpha, self.num_ibp_features, self.num_ibp_objects)

        self._ibp_parameters = [self._q_tau] + [self.q_nu]


    def _init_ibp_variational_parameters(self, q_tau, q_nu, alpha, num_features, num_objects, rescale=100.):

        # init tau
        if q_tau is None:
            q_tau = np.ones((2, num_features))
            alpha_K = alpha / num_features
            q_tau[0,:] = alpha_K * np.ones(shape=(1, num_features))
            q_tau = q_tau + 1.5 * min(1, alpha_K)* (np.random.rand(2, num_features) - 1.5)
            self._q_tau = Param(q_tau, name="q_tau", transform=transforms.positive)
        else:
            self._q_tau = Param(q_tau, name="q_tau", transform=transforms.positive)

        # init nu
        if q_nu is None:
            q_nu = np.random.uniform(0., 1., (num_objects, num_features))
            in_range = transforms.Chain(transforms.Logistic(a=0., b=1.), transforms.Rescale(rescale))
            self._q_nu = Param(q_nu, transform=in_range, trainable=True, name="q_nu")

        self.q_nu_un = tf.unstack(self.q_nu)
        self.q_nu_un = [tf.reshape(nu_n, shape=(1, self.num_ibp_features)) for nu_n in self.q_nu_un]
        self.sampler = [Bernoulli(probs=nu_n, validate_args=True) for nu_n in self.q_nu_un]

    @property
    def q_nu(self):
        return self._q_nu.value

    @property
    def q_tau(self):
        return self._q_tau.value

    def _build_KL_ibp(self):

        self.expect_pi = self.compute_expect_pi()
        self.expect_z = self.compute_expect_z()
        self.entropy = self.compute_entropy()

        return self.expect_pi + self.expect_z + self.entropy

    def get_digamma_sum(self, tau_1):
        sum_tau = tf.reduce_sum(self.q_tau, 0)
        return tf.reduce_sum(tf.digamma(tau_1) - tf.digamma(sum_tau), name='digama_sum')

    def compute_expect_pi(self):
        tau = tf.unstack(self.q_tau)
        tau_1 = tf.reshape(tau[0], (self.num_ibp_features, 1))
        digama_sum = self.get_digamma_sum(tau_1)
        K = tf.cast(self.num_ibp_features, dtype=float_type)

        expect_pi = K * tf.log(self.alpha / K) + (self.alpha / K - 1.0) * digama_sum
        return expect_pi

    def compute_expect_z(self):
        tau = tf.unstack(self.q_tau)
        tau_1 = tf.reshape(tau[0], (self.num_ibp_features, 1))
        tau_2 = tf.reshape(tau[1], (self.num_ibp_features, 1))

        psi_1 = tf.digamma(tau_1)
        psi_2 = tf.digamma(tau_2)
        nu = self.q_nu
        nu_psi_1 = tf.reduce_sum(tf.matmul(psi_1, nu, transpose_a=True, transpose_b=True))
        nu_psi_2 = tf.reduce_sum(tf.matmul(psi_2, 1 - nu, transpose_a=True, transpose_b=True))
        nu_psi_1_2 = self.num_ibp_objects * tf.reduce_sum(tf.digamma(tau_1 + tau_2))

        temp = nu_psi_2 - nu_psi_1_2
        return tf.add(nu_psi_1, temp, name='expect_z')

    def compute_entropy(self):
        tau = tf.unstack(self.q_tau)
        tau_1 = tf.reshape(tau[0], (self.num_ibp_features, 1))
        tau_2 = tf.reshape(tau[1], (self.num_ibp_features, 1))

        # compute entropy w.r.t variable pi
        sum_tau = tf.reduce_sum(self.q_tau, axis=0)
        temp = tf.lgamma(tau_1) + tf.lgamma(tau_2) - tf.lgamma(sum_tau) - \
               tf.multiply(tau_1 - 1.0, tf.digamma(tau_1)) - tf.multiply(tau_2 - 1.0, tf.digamma(tau_2)) + tf.multiply(
            tau_1 + tau_2 - 2.0, tf.digamma(sum_tau))

        entropy_pi = tf.reduce_sum(temp)

        nu = self.q_nu
        entropy_z = tf.reduce_sum(
            -tf.multiply(nu, tf.log(nu)) - tf.multiply(1.0 - nu, tf.log(1.0 - nu)))

        return tf.add(entropy_pi, entropy_z, name='entropy')

    def closed_form_update_tau(self):

        nu = self.q_nu
        sum_nu = tf.reduce_sum(nu, axis=0)
        K = tf.cast(self.num_ibp_features, float_type)
        tau_1 = self.alpha / K + sum_nu
        tau_2 = 1. + tf.reduce_sum(1.0 - nu, axis=0)

        new_tau = tf.stack([tau_1, tau_2], axis=0)
        return tf.assign(self._q_tau.vf_val, new_tau)


class LKM(IBP):

    def __init__(self, data, additive_kernels, likelihoods, alpha=1.0, beta=1.0):

        """
        :param data:
        :param additive_kernels: list of kernels
        :param likelihoods: list of Gaussian noise variances
        :param tau: IBP variational param
        :param nu: IBP variational param
        :param alpha: IBP param
        :param beta: IBP param
        """

        self.K = len(additive_kernels)
        self.N = data['X'].shape[0]
        self.D = data['X'].shape[1]
        self.data = data


        super().__init__(num_ibp_features=self.K, num_ibp_objects=self.N, alpha=alpha)

        # let's keep data as constant, can be place holders but this is okay
        self.t = tf.constant(data['t'], dtype=float_type)
        self.X = tf.constant(data['X'], dtype=float_type)

        self.alpha, self.beta = alpha, beta

        # since each time series has different noise
        assert self.N == len(likelihoods)

        self.kernels = additive_kernels
        self.likelihoods = likelihoods

        self.need_precompute = True

        self.initialize()


    def initialize(self):

        # separate different timeries
        self.X_un = tf.unstack(self.X)
        self.X_un = [tf.reshape(X, shape=(self.D,1)) for X in self.X_un]

        # collect GP hyperparameters
        self._gp_hyperparameters = []
        for k in self.kernels:
            self._gp_hyperparameters.extend(k._parameters)


    def precompute(self):

        if self.need_precompute:
            self.Ks = tf.stack([kernel.K(self.t) for kernel in self.kernels])
            self.need_precompute = False

    def sample_gumble(self):
        """list of samples for each time series"""
        z = [gumbel_softmax_v2(nu_n, temperature=0.01, hard=False) for nu_n in self.q_nu_un]
        return z

    def compute_expect_gp(self, n_gumbles=10):

        self.precompute()
        store = []
        for _ in range(n_gumbles):
            z = self.sample_gumble()
            for n in range(self.N):

                X_n = self.X_un[n]
                z_n = z[n]
                z_C = tf.multiply(tf.reshape(z_n, shape=(self.K, 1, 1)), self.Ks)
                sum_z_C = tf.reduce_sum(z_C, axis=0)
                sum_z_C_I = sum_z_C + tf.eye(self.D, dtype=float_type) * self.likelihoods[n].value
                chol = tf.cholesky(sum_z_C_I + jitter * tf.eye(self.D, dtype=float_type))
                alpha = tf.matrix_triangular_solve(chol, X_n, lower=True)
                nll = 0.5 * tf.reduce_sum(tf.square(alpha)) + 0.5 * log_det(chol)

                store.append(nll)

        return -tf.add_n(store) / n_gumbles


    def build_marginal_loglikelihood(self):

        """Gather all variational expectations"""

        self.expect_pi = self.compute_entropy()
        self.expect_z = self.compute_expect_z()
        self.expect_gp = self.compute_expect_gp()
        self.entropy = self.compute_entropy()

        llike = self.expect_pi + self.expect_z + self.expect_gp + self.entropy

        return llike

    def refine(self):
        z = [gumbel_softmax_v2(nu_n, temperature=0.01, hard=True) for nu_n in self.q_nu_un]
        store = []
        for n in range(self.N):
            X_n = self.X_un[n]
            z_n = z[n]
            z_C = tf.multiply(tf.reshape(z_n, shape=(self.K, 1, 1)), self.Ks)
            sum_z_C = tf.reduce_sum(z_C, axis=0)
            sum_z_C_I = sum_z_C + tf.eye(self.D, dtype=float_type) * self.likelihoods[n].value
            chol = tf.cholesky(sum_z_C_I + jitter * tf.eye(self.D, dtype=float_type))
            alpha = tf.matrix_triangular_solve(chol, X_n, lower=True)
            nll = 0.5 * tf.reduce_sum(tf.square(alpha)) + 0.5 * log_det(chol)

            store.append(nll)

        return z, tf.add_n(store)

    def prepare_for_postprocess(self, num_interpolation=300):
        t = self.data['t']
        t_test = np.linspace(np.min(t), np.max(t), num_interpolation).reshape(num_interpolation, 1)
        K = []
        K_star = []
        K_star_star = []
        for kernel in self.kernels:
            K.append(kernel.K(t))
            K_star.append(kernel.K(t, t_test))
            K_star_star.append(kernel.K(t_test, t_test))

        noise = []
        for l in self.likelihoods:
            noise.append(l.value)

        K = tf.stack(K)
        K_star = tf.stack(K_star)
        K_star_star = tf.stack(K_star_star)
        noise = tf.stack(noise)

        return t_test, K, K_star, K_star_star, noise





    def predict(self, t_test, full_cov=False):
        """
        Predict new data t_test
        :param t_test:
        :param full_cov:
        :return:
        """

        self.precompute()

        K_train_test = [kernel.K(self.t, t_test) for kernel in self.kernels]
        K_train_test = tf.stack(K_train_test)
        K_test = [kernel.K(t_test) for kernel in self.kernels] if full_cov else [kernel.K(t_test) for kernel in
                                                                                 self.kernels]
        K_test = tf.stack(K_test)

        z = [s.sample() for s in self.sampler]
        z = [tf.cast(temp, dtype=float_type) for temp in z]
        pred_means, pred_vars = [], []

        for n in range(self.N):

            noise = self.likelihoods[n].value
            X_n = self.X_un[n]
            z_n = tf.reshape(z[n], shape=(self.K,1,1))

            z_C_train = tf.multiply(z_n, self.Ks)
            z_C_train = tf.reduce_sum(z_C_train, axis=0) + tf.eye(self.D, dtype=float_type) * noise
            z_C_train_test = tf.multiply(z_n, K_train_test)
            z_C_train_test = tf.reduce_sum(z_C_train_test, axis=0)
            # z_C_test = tf.multiply(tf.reshape(z_n, shape=(self.K, 1)), K_test)
            z_C_test = tf.multiply(z_n, K_test)
            z_C_test = tf.reduce_sum(z_C_test, axis=0)

            L = tf.cholesky(z_C_train)
            A = tf.matrix_triangular_solve(L, z_C_train_test, lower=True)
            V = tf.matrix_triangular_solve(L, X_n)
            f_mean = tf.matmul(A, V, transpose_a=True)
            if full_cov:
                f_var = z_C_test - tf.matmul(A, A, transpose_a=True)
                f_var = tf.tile(f_var[None, :, :], [1, 1, 1])
            else:
                f_var = tf.matrix_diag_part(z_C_test) - tf.reduce_sum(tf.square(A), axis=0)
                # f_var = tf.tile(f_var[None, :], [1, 1])
                f_var = tf.tile(tf.reshape(f_var, (-1, 1)), [1, tf.shape(X_n)[1]])

            pred_means.append(f_mean)
            pred_vars.append(f_var) # only work if full_cov=False

        return pred_means, pred_vars

    def plot_gp(self, session, num_expolation=300):

        """Plot utility"""

        min_t = np.min(self.data['t'])
        max_t = np.max(self.data['t'])
        xx = np.linspace(min_t, max_t, num_expolation).reshape(num_expolation, 1)

        pred_means, pred_vars = self.predict(xx)
        pred_means = session.run(pred_means)
        pred_vars = session.run(pred_vars)
        fig, axes = plt.subplots(ncols=1, nrows=self.N)

        for n in range(self.N):
            mean, var = pred_means[n], pred_vars[n]
            if isinstance(axes, np.ndarray):
                ax = axes[n]
            else:
                ax = axes
            ax.plot(self.data['t'], self.data['X'][n, :].transpose(), 'kx', lw=2)
            ax.plot(xx, mean, 'C0', lw=2)
            ax.fill_between(xx[:, 0],
                            mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                            mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                            color='C0', alpha=0.2)

        return fig






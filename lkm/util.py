from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from gpflow import settings
from gpflow.transforms import Transform
import jinja2
import os


class Chain(Transform):
    """
    Chain two transformations together:
    .. math::
       y = t_1(t_2(x))
    where y is the natural parameter and x is the free state
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def forward_tensor(self, x):
        return self.t1.forward_tensor(self.t2.forward_tensor(x))

    def backward_tensor(self, y):
        return self.t2.backward_tensor(self.t1.backward_tensor(y))

    def forward(self, x):
        return self.t1.forward(self.t2.forward(x))

    def backward(self, y):
        return self.t2.backward(self.t1.backward(y))

    def log_jacobian_tensor(self, x):
        return self.t1.log_jacobian_tensor(self.t2.forward_tensor(x)) +\
               self.t2.log_jacobian_tensor(x)

    def __str__(self):
        return "{} {}".format(self.t1.__str__(), self.t2.__str__())

ft = settings.tf_float

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1, dtype=ft)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample_v2(logits, temperature):
    y = tf.log(logits) + sample_gumbel(tf.shape(logits))
    z = tf.log(1. - logits) + sample_gumbel(tf.shape(logits))
    stacked = tf.stack([y, z])
    softmax = tf.nn.softmax(stacked/temperature, axis=0)
    return tf.unstack(softmax)[0]

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temperature)

def gumbel_softmax(logits, temperature, hard=False):

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def gumbel_softmax_v2(logits, temperature, hard=False):

    y = gumbel_softmax_sample_v2(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def reset_tensorflow():
    tf.reset_default_graph()

def get_session():

    global _SESSION

    if tf.get_default_session() is None:
        _SESSION = tf.InteractiveSession()
    else:
        _SESSION = tf.get_default_session()

    return _SESSION


def log_det(chol):
    return 2.*tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)))

def pdf_possion_binomial(p, k, recursive = False):

    """
    Get the probability of k-time success of Possion Binomial dist.

    :param p:
    :param k:
    :return:
    """
    if not recursive:

        p_complex = tf.complex(p,
                               tf.constant(0.0, shape=[1,k], dtype=tf.float64))

        i_complex = tf.complex(tf.constant(0.0, dtype=tf.float64),
                               tf.constant(1.0, dtype=tf.float64))
        C = tf.exp(2.0*np.pi*i_complex / (k+1))
        C_l = tf.pow(
            tf.tile(tf.reshape(C, shape=[1]), [k + 1]),
            tf.complex(tf.convert_to_tensor(np.arange(k+1), dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
        )

        one = tf.complex(tf.constant(1.0, dtype=tf.float64),
                         tf.constant(0.0, dtype=tf.float64))
        prod = tf.reduce_prod(one - tf.matmul(tf.reshape(one - C_l, shape=[k + 1, 1]), tf.reshape(p_complex, shape=[1,k])), 1)

        prob = [None] * (k + 1)
        for i in range(0, k+1):
            prob[i] = one/(k+1) * tf.matmul(tf.reshape(tf.pow(one / C_l, i), shape=[1, k+1]), tf.reshape(prod, shape=[k+1, 1]))

        return tf.reshape(tf.real(tf.stack(prob)), shape=(k+1,))
    else:
        # TODO: remove p=0 and p=1
        def T(i):
            temp = tf.divide(p, 1 - p)
            return tf.reduce_sum(tf.pow(temp, i))

        persist_T = [None]*k
        for i in range(k):
            persist_T[i] = T(i+1)

        previous = tf.reduce_prod(1 - p)
        if k == 0:
            return previous

        #create list of k probabilities
        prob = [None]*(k+1)
        prob[0] = previous
        # bottom-up compute probability
        for i in range(1,k + 1):
            temp = 0.0
            for j in range(1, i + 1):
                if (j - 1) % 2 == 0:
                    temp = temp + prob[i - j]*persist_T[j-1]
                else:
                    temp = temp - prob[i - j]*persist_T[j-1]

            prob[i] = temp / i

        return tf.stack(prob)

def generate_mask(k):

    masks = {}
    for i in range(0, k):
        masks[i] = np.full((k), True, dtype=bool)
        masks[i][i] = False

    return masks

def expect_given_discrete(dist, func, K):
    # type: (list, func, int) -> tf.Tensor
    """
    Expect for discrete value from 1 to K
    :param dist:
    :param func:
    :return:
    """
    expect = tf.Variable(0.0, dtype=tf.float64)
    for k in range(K+1):
        expect = expect + dist[k]*func(k)

    return expect

def load_kernel(from_file):
    with open(from_file, 'r') as f:
        kernel_strs = f.readlines()

    kernel_strs = [line.strip() for line in kernel_strs]
    return kernel_strs

def move_first_to_last(nu):
    new_nu = np.zeros(nu.shape)
    new_nu[:,:-1] = nu[:,1:]
    new_nu[:,-1] = nu[:, 0]
    return new_nu


def get_template(template_file):

    latex_jinja_env = jinja2.Environment(
        block_start_string='\BLOCK{',
        block_end_string='}',
        variable_start_string='\VAR{',
        variable_end_string='}',
        comment_start_string='\#{',
        comment_end_string='}',
        line_statement_prefix='%%',
        line_comment_prefix='%#',
        trim_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(os.path.abspath('/'))
    )

    template = latex_jinja_env.get_template(os.path.realpath(template_file))
    return template




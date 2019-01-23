import numpy as np
import tensorflow as tf
import lkm.flexible_function as ff
import gpflowSlim.kernels as gpf_kernel
from gpflowSlim import Param
from gpflowSlim import transforms

float_type = tf.float64

class Linear(gpf_kernel.Kernel):

    """What difference: introduce `shift' parameter and variance is squared"""

    def __init__(self, input_dim, variance=1.0, shift=0.0, active_dims=None, ARD=False, name='LIN'):

        super().__init__(input_dim, active_dims, name=name)
        self.ARD = ARD

        with tf.variable_scope(name):
            variance = np.ones(self.input_dim, float_type)*variance if ARD else variance
            self._variance = Param(variance, transform=transforms.positive, name="variance")
            shift =np.ones(self.input_dim, float_type)*shift if ARD else shift
            self._shift = Param(shift, name="shift")

        self._parameters = self._parameters + [self._variance] + [self._shift]


    @property
    def variance(self):
        return self._variance.value

    @property
    def shift(self):
        return self._shift.value


    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
            if X2 is None:
                shifted_X = X - tf.squeeze(self.shift)
                return tf.matmul(shifted_X*self.variance, shifted_X*self.variance, transpose_b=True)
            else:
                shifted_X = X - tf.ones_like(X)*self.shift
                shifted_X2 = X2 - tf.ones_like(X2)*self.shift
                return tf.matmul(shifted_X*self.variance, shifted_X2*self.variance, transpose_b=True)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        shifted_X = X - tf.ones_like(X) * self.shift
        return tf.reduce_sum(tf.square(shifted_X)*self.variance**2, 1)


class ChangePoint(gpf_kernel.Combination):
    """
    Change point kernel
    """

    def __init__(self, kern_list, location=0.0, steepness=0.1):

        assert len(kern_list) == 2

        gpf_kernel.Combination.__init__(self, kern_list)

        self._location = Param(location)
        self._steepness = Param(steepness, transforms.positive)

    @property
    def location(self):
        return self._location.value

    @property
    def steepness(self):
        return self._steepness.value

    def K(self, X, X2=None, presliced=False):
        # see https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/gpml/cov/covChangePointMultiD.m

        if not presliced:
            X, X2 = self._slice(X, X2)

        tx = tf.tanh((X - self.location)*self.steepness)
        ax = 0.5 + 0.5*tx
        if X2 is not None:
            tz = tf.tanh((X2 - self.location)*self.steepness)
            az = 0.5 + 0.5*tz
        else:
            az = ax

        K0 = self.kern_list[0].K(X, X2)
        K1 = self.kern_list[1].K(X, X2)

        K = tf.multiply(K0, ax)
        K = tf.multiply(K, tf.transpose(az))

        temp = tf.multiply(K1, 1 - ax)
        temp = tf.multiply(temp, tf.transpose(1 - az))

        K = K + temp

        return K

    def Kdiag(self, X, presliced=False):

        if not presliced:
            X,_ = self._slice(X, None)

        sigma_x = 0.5 + tf.scalar_mul(0.5,
                                      tf.tanh(tf.scalar_mul(tf.squeeze(self.steepness), X - self.location))
                                      )
        sigma_x = 1 - sigma_x

        K0 = self.kern_list[0]
        K1 = self.kern_list[1]

        K = tf.multiply(tf.squeeze(sigma_x), K0.Kdiag(X))
        K = tf.multiply(K, tf.squeeze(sigma_x))

        temp = tf.multiply(1 - tf.squeeze(sigma_x), K1.Kdiag(X))
        temp = tf.multiply(temp, 1 - tf.squeeze(sigma_x))

        K = K + temp
        return K

class ChangeWindow(gpf_kernel.Combination):
    """
    Change-window kernel
    """
    def __init__(self, kern_list, location=0.0, steepness=1.0, width=1.0):
        assert len(kern_list) == 2
        gpf_kernel.Combination.__init__(self, kern_list)

        self._location = Param(location)
        self._steepness = Param(steepness, transforms.positive)
        self._width = Param(width, transforms.positive)

    @property
    def location(self):
        return self._location.value

    @property
    def steepness(self):
        return self._steepness.value

    @property
    def width(self):
        return self._width.value

    def K(self, X, X2=None, presliced=False):
        # see https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/gpml/cov/covChangeWindowMultiD.m

        if not presliced:
            X, X2 = self._slice(X, X2)

        tx1 = tf.tanh((X - self.location + 0.5*self.width)*self.steepness)
        tx2 = tf.tanh((-X + self.location + 0.5*self.width)*self.steepness)
        ax = tf.multiply(0.5 + 0.5* tx1, 0.5 + 0.5* tx2)

        if X2 is not None:
            tz1 = tf.tanh((X2 - self.location + 0.5 * self.width) * self.steepness)
            tz2 = tf.tanh((-X2 + self.location + 0.5 * self.width) * self.steepness)
            az = tf.multiply(0.5 + 0.5 * tz1, 0.5 + 0.5 * tz2)
        else:
            az = ax

        # see the order to apply ax.*K0.*az in covChangeWindowMultiD.m
        K0 = self.kern_list[1].K(X, X2)
        K1 = self.kern_list[0].K(X, X2)

        K = tf.multiply(ax, K0)
        K = tf.multiply(K, tf.transpose(az))

        temp = tf.multiply(1 - ax, K1)
        temp = tf.multiply(temp, tf.transpose(1- az))

        return K + temp

    def Kdiag(self, X, presliced=False):

        if not presliced:
            X, _ = self._slice(X, None)

        tx1 = tf.tanh((X - self.location + 0.5 * self.width) * self.steepness)
        tx2 = tf.tanh((-X + self.location + 0.5 * self.width) * self.steepness)

        ax = tf.multiply(0.5 + tf.scalar_mul(0.5, tx1),0.5 + tf.scalar_mul(0.5, tx2))

        # see the order to apply ax.*K0.*az in covChangeWindowMultiD.m
        K0 = self.kern_list[1].Kdiag(X)
        K1 = self.kern_list[0].Kdiag(X)

        ax = tf.reshape(ax, shape=tf.shape(K0))

        K = tf.multiply(ax, K0)
        K = tf.multiply(K, ax)

        temp = tf.multiply(1 - ax, K1)
        temp = tf.multiply(temp, 1 - ax)

        return K + temp


def to_gpflow_kernel(k):
    """
    Convert kernels in flexible function to GPflow kernels
    :param k: input kernel
    :type k: ff.Kernel
    :return: feature-wrapped GPflow kernel
    """
    # Note that: sf (scale factor) in GPModel kernel is just the squared root of variance in GPflow kernel
    # Note that: some hyperparameter in GPModel kernel is in the log scale. we have to scale back to normal when using GPflow kernel
    # Periodic Kernel
    if isinstance(k, ff.PeriodicKernel):
        return gpf_kernel.PeriodicKernel(input_dim=1,
                                            # period=np.exp(k.period),
                                            variance=np.exp(2.0 * k.sf),
                                            lengthscales=np.exp(k.lengthscale)
                                            )

    # Linear Kernel
    elif isinstance(k, ff.LinearKernel):
        return Linear(input_dim=1,
                                    variance=np.exp(k.sf),
                                    shift=k.location)

    # Squared Exponential
    elif isinstance(k, ff.SqExpKernel):
        return gpf_kernel.RBF(input_dim=1,
                                 variance=np.exp(k.sf*2.0),
                                 lengthscales=np.exp(k.lengthscale))

    # Sum
    elif isinstance(k, ff.SumKernel):
        kern_list = [to_gpflow_kernel(op) for op in k.operands]
        return gpf_kernel.Add(kern_list=kern_list)

    # Product
    elif isinstance(k, ff.ProductKernel):
        kern_list = [to_gpflow_kernel(op) for op in k.operands]
        return gpf_kernel.Prod(kern_list=kern_list)

    # ChangePoint
    elif isinstance(k, ff.ChangePointKernel):
        kern_list = [to_gpflow_kernel(op) for op in k.operands]
        return ChangePoint(kern_list=kern_list,
                                         location=k.location,
                                         steepness=np.exp(k.steepness))

    elif isinstance(k, ff.ChangeWindowKernel):
        kern_list = [to_gpflow_kernel(op) for op in k.operands]
        return ChangeWindow(kern_list=kern_list,
                                          location=k.location,
                                          steepness=np.exp(k.steepness),
                                          width=np.exp(k.width)
                                          )
    elif isinstance(k, ff.ConstKernel):
        # No dimension is defined in ff.ConstKernel. Let set default input = 1
        return gpf_kernel.Constant(input_dim=1,
                                      variance=np.exp(k.sf*2.0)
                                      )

    elif isinstance(k, ff.NoiseKernel):
        # No dimension is defined in ff.NoiseKernel. Let set default input = 1
        return gpf_kernel.White(input_dim=1,
                                   variance=np.exp(k.sf * 2.0)
                                   )
    else:
        # raise unknown kernel error
        raise RuntimeError("Unknown kernel %s !" % k.__class__)


def to_gpml_kernel(k, sess):
    """
    Convert GPflow kernel to gpml kernel
    :param k:
    :return:
    """

    if isinstance(k, gpf_kernel.Periodic):
        variance, lengthscales, period = sess.run([k.variance, k.lengthscales, k.period])
        return ff.PeriodicKernel(dimension=0,
                                 period=np.log(period),
                                 sf=np.log(variance) / 2.0,
                                 lengthscale=np.log(lengthscales))
    elif isinstance(k, Linear):
        variance, shift = sess.run([k.variance, k.shift])
        return ff.LinearKernel(dimension=0,
                               sf=np.log(variance) / 2.0,
                               location=shift)
    elif isinstance(k, gpf_kernel.RBF):
        variance, lengthscales = sess.run([k.variance, k.lengthscales])
        return ff.SqExpKernel(dimension=0,
                              lengthscale=np.log(lengthscales),
                              sf=np.log(variance) / 2.0)
    elif isinstance(k, gpf_kernel.Add):
        operands = [to_gpml_kernel(op) for op in k.kern_list]
        return ff.SumKernel(operands=operands)
    elif isinstance(k, gpf_kernel.Prod):
        operands = [to_gpml_kernel(op) for op in k.kern_list]
        return ff.ProductKernel(operands=operands)
    elif isinstance(k, ChangePoint):
        operands = [to_gpml_kernel(op) for op in k.kern_list]
        location, steepness = sess.run([k.location, k.steepness])
        return ff.ChangePointKernel(dimension=0,
                                    location=location,
                                    steepness=np.log(steepness),
                                    operands=operands)
    elif isinstance(k, ChangeWindow):
        operands = [to_gpml_kernel(op) for op in k.kern_list]
        location, steepness, width = sess.run([k.location, k.steepness, k.width])
        return ff.ChangeWindowKernel(dimension=0,
                                     location=location,
                                     steepness=np.log(steepness),
                                     width=np.log(width),
                                     operands=operands)
    elif isinstance(k, gpf_kernel.Constant):
        variance = sess.run(k.variance)
        return ff.ConstKernel(sf=np.log(variance) / 2.0)
    elif isinstance(k, gpf_kernel.White):
        variance = sess.run(k.variance)
        return ff.NoiseKernel(sf=np.log(variance) / 2.0)
    else:
        raise RuntimeError("Unknown kernel %s !" % type(k).__name__)

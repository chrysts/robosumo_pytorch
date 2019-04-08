"""
A variety of utilities.
"""
import copy
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict


class RunningMeanStd(nn.Module):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        #with tf.variable_scope(scope, reuse=reuse):
        super(RunningMeanStd, self).__init__()
        self._sum = nn.Parameter(torch.zeros(shape, dtype=torch.float64), requires_grad=False)
        self._sumsq = nn.Parameter(torch.zeros(shape, dtype=torch.float64), requires_grad=False)
        self._count = nn.Parameter(torch.zeros(1, dtype=torch.float64), requires_grad=False)

        self.shape = shape

    def init_calc(self):
        self.mean = (self._sum / self._count).double()
        var_est = (self._sumsq / self._count) - (self.mean)**2
        var_est = torch.clamp(var_est, min=1e-2)
        # if var_est < 1e-2:
        #     var_est = 1e-2
        self.std = torch.sqrt(var_est).double()


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean.double()
        self.logstd = logstd.double()
        self.std = torch.exp(logstd.double())
        #print("a")
        #self.distribution =

    def sample(self):
        randomized = torch.zeros_like(self.mean).normal_(0, 1).double()
        return self.mean + self.std * randomized #tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean


# def dense(x, size, name, weight_init=None, bias=True):
#     w = tf.get_variable(name + "/w", [x.get_shape()[1], size],
#                         initializer=weight_init)
#     ret = tf.matmul(x, w)
#     if bias:
#         b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
#         return ret + b
#     else:
#         return ret

#
def switch(condition, if_exp, else_exp):
    x_shape = if_exp.shape
    if condition == True:
        x = if_exp
    else :
        x = else_exp
    # if_exp_mask = if_exp * condition
    # neg_condition = condition + 1
    # neg_condition[neg_condition > 1] = 0
    # else_exp_mask = else_exp * neg_condition
    # x = if_exp_mask + else_exp_mask
    # x = tf.cond(tf.cast(condition, 'bool'),
    #             lambda: if_exp,
    #             lambda: else_exp)
    x.reshape(x_shape)
    return x


def load_params(path):
    return np.load(path)


def set_from_flat(model, flat_params):
    parameters = OrderedDict(model.named_parameters())
    parameters_cad = OrderedDict(model.named_parameters()).popitem(last=False)

    parameters.pop("logstd")
    parameters[parameters_cad[0]] = parameters_cad[1]


    counter = 0
    fweights = dict()
    for key, val in parameters.items():

        name = key#parameters[ii].name
        shape = val.shape
        shape_flat = int(np.prod(val.shape))
        start = counter
        end = counter + shape_flat
        #vals = torch.from_numpy(flat_params[start:end])#.reshape(list(shape))

        counter = end

        if len(shape) > 1:
            vals = torch.from_numpy(flat_params[start:end]).reshape(shape[1], shape[0])
            vals = vals.transpose(1, 0)
        else:
            vals = torch.from_numpy(flat_params[start:end]).reshape(list(shape))

        fweights[name] = vals

    return model.load_state_dict({name : fweights[name] for name in parameters})


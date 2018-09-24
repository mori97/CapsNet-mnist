import chainer
from chainer import initializers, variable
from chainer.backends import cuda
import chainer.functions as F


def squash(x):
    x_l2_norm_squared = F.sum(x * x, axis=2)
    x_l2_norm = F.sqrt(x_l2_norm_squared)

    ret = x_l2_norm_squared / (1 + x_l2_norm_squared) / x_l2_norm
    ret = F.broadcast_to(F.expand_dims(ret, axis=2), x.shape) * x
    return ret


class Capsules(chainer.Link):

    def __init__(self, in_capsules, in_size, out_capsules, out_size,
                 initial_weight=None, n_dynamic_routing_iter=3):
        super(Capsules, self).__init__()

        self._in_capsules = in_capsules
        self._out_capsules = out_capsules
        self.n_dynamic_routing_iter = n_dynamic_routing_iter

        with self.init_scope():
            weight_initializer = initializers._get_initializer(initial_weight)
            self.weight = variable.Parameter(weight_initializer)
            self.weight.initialize((out_capsules, in_capsules, out_size, in_size))

    def __call__(self, x):
        batch_size = x.shape[0]
        xp = cuda.get_array_module(x)

        u_hat = F.einsum('jiyx,bix->bjiy', self.weight, x)

        # Routing
        b = xp.zeros((batch_size, self._in_capsules, self._out_capsules),
                     dtype=xp.float32)
        c = F.softmax(b, axis=2)
        s = F.einsum('bij,bjiy->bjy', c, u_hat)
        v = squash(s)

        for _ in range(self.n_dynamic_routing_iter - 1):
            b = b + F.einsum('bjiy,bjy->bij', u_hat, v)
            c = F.softmax(b, axis=2)
            s = F.einsum('bij,bjiy->bjy', c, u_hat)
            v = squash(s)

        return v

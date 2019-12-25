import numpy as np
from keras.backend import get_session
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, CarliniWagnerL2, \
    MadryEtAl, SaliencyMapMethod, FastFeatureAdversaries


# # DEBUG
# def debug(s):
#   if __debug__:
#     print(s)


# GLOBAL VARIABLES
BS = 128

class ABC_Attack:

    def __init__(self, attack, attack_params):
        self._attack = attack
        self._attack_params = attack_params

    def _generate(self, model, X, y=None, bs=BS):

        targeted_attack = y is not None

        # debug("============= ATTACK GRAPH BUILDING ===============")
        # Retrieve needs
        _in_shape, _n_classes = model.input_shape[1:], model.output_shape[1]

        # Convert keras model in tf format
        wrap = KerasModelWrapper(model)
        cleverhans_attack = eval(self._attack)(wrap, sess=get_session())

        # Build graph
        x = tf.placeholder(tf.float32, shape=[None, *_in_shape])
        if targeted_attack:
            y_tgt = tf.placeholder(tf.float32, shape=[None, _n_classes])
            self._attack_params['y_target'] = y_tgt

        # Attack tensor
        x_adv = cleverhans_attack.generate(x, **self._attack_params)
        x_adv = tf.stop_gradient(x_adv)

        # debug("============= ADVEX GENERATION ===============")
        # Generate attack in batches
        X_adv = np.zeros_like(X)
        n_samples = X.shape[0]
        nb_batches = n_samples // bs

        sess = get_session()
        for i in range(nb_batches + 1):
            _start, _end = i * bs, min(n_samples, (i + 1) * bs)
            if _start < _end:       # TODO: HACK to handle the case in which n_samples is multiple of bs
                _X = X[_start:_end]

                feeds = {
                    x: _X
                }

                # Targeted attack?
                if targeted_attack:
                    _y = y[_start:_end]
                    feeds[y_tgt] = _y

                X_adv[_start:_end] = sess.run(x_adv, feeds)

        return X_adv


class FGSM(ABC_Attack):

    def __init__(self, eps=0.3, ord=np.inf, clip_min=0.0, clip_max=1.0):
        params = {
            'eps': eps,
            'ord': ord,
            'clip_min': clip_min,
            'clip_max': clip_max,
        }
        super(FGSM, self).__init__('FastGradientMethod', params)

    def generate(self, model, X, y=None):
        return super(FGSM, self)._generate(model, X, y)


class BIM(ABC_Attack):

    def __init__(self, eps=0.3, eps_iter=0.05, ord=np.inf, clip_min=0.0, clip_max=1.0, nb_iter=10):
        params = {
            'eps': eps,
            'eps_iter': eps_iter,
            'ord': ord,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'nb_iter': nb_iter
        }
        super(BIM, self).__init__('BasicIterativeMethod', params)

    def generate(self, model, X, y=None):
        return super(BIM, self)._generate(model, X, y)


class CW_L2(ABC_Attack):

    def __init__(self, confidence=0, batch_size=64, clip_min=0.0, clip_max=1.0):

        params = {'confidence': confidence,
                  'batch_size': batch_size,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
        super(CW_L2, self).__init__('CarliniWagnerL2', params)

    def generate(self, model, X, y=None):
        return super(CW_L2, self)._generate(model, X, y, bs=self._attack_params['batch_size'])


class PGD(ABC_Attack):

    def __init__(self, eps=0.3, eps_iter=0.01, ord=np.inf, clip_min=0.0, clip_max=1.0, nb_iter=40):
        params = {
            'eps': eps,
            'eps_iter': eps_iter,
            'ord': ord,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'nb_iter': nb_iter
        }
        super(PGD, self).__init__('MadryEtAl', params)

    def generate(self, model, X, y=None):
        return super(PGD, self)._generate(model, X, y)


class JSMA(ABC_Attack):

    def __init__(self, theta=1.0, gamma=1.0, clip_min=0.0, clip_max=1.0):
        '''
        Jacobian Saliency Map Attack
        :param theta: Perturbation induced to modified components (can be positive or negative)
        :param gamma: Maximum percentage of perturbed features
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        '''
        params = {
            'theta': theta,
            'gamma': gamma,
            'clip_min': clip_min,
            'clip_max': clip_max
        }
        super(JSMA, self).__init__('SaliencyMapMethod', params)

    def generate(self, model, X, y=None):
        return super(JSMA, self)._generate(model, X, y, bs=64)


class Features_Attack(ABC_Attack):

    def __init__(self, layer, eps=0.3, eps_iter=0.01, ord=np.inf, clip_min=0.0, clip_max=1.0, nb_iter=100):
        params = {
            'layer': layer,
            'eps': eps,
            'eps_iter': eps_iter,
            'ord': ord,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'nb_iter': nb_iter
        }
        super(Features_Attack, self).__init__('FastFeatureAdversaries', params)

    def generate(self, model, X_src, X_guide):
        wrap = KerasModelWrapper(model)
        cleverhans_attack = FastFeatureAdversaries(wrap, sess=get_session())

        # ============= ATTACK GRAPH BUILDING ===============
        assert X_src.shape == X_guide.shape, "Inconsistency between X_src and X_guide shapes!"
        x_src = tf.placeholder(tf.float32, shape=[None, *(X_src.shape[1:])])
        x_guide = tf.placeholder(tf.float32, shape=[None, *(X_src.shape[1:])])
        # Attack tensor
        x_adv = cleverhans_attack.generate(x_src, x_guide, **self._attack_params)
        x_adv = tf.stop_gradient(x_adv)
        # ============= ADVEX GENERATION ===============
        sess = get_session()
        X_adv = sess.run(x_adv, feed_dict={
                x_src: X_src,
                x_guide: X_guide
            }
        )

        return X_adv

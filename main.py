from __future__ import division
import argparse
import os

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import numpy as np
SEED = 692019
np.random.seed(SEED)

import setGPU

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from sklearn.model_selection import train_test_split

from datasets import read_data
from models import get_basic_cnn, get_cnn
from detectors import AE_Detector
from utils import compute_security_curves, plot_security_curves, mkdirs


def get_class_idxs(Y):
    class_idxs = {}
    for i in range(Y.shape[1]):
        class_idxs[i] = np.where(Y.argmax(1) == i)[0]

    return class_idxs


def get_AE_Detector(model, layers, X_nat, Y_nat, X_adv):

    class_idxs = get_class_idxs(Y_nat)

    # Create dataset
    data = {}
    for i in range(Y_nat.shape[1]):

        X_i = X_nat[class_idxs[i]]

        m = min(X_i.shape[0], X_adv[i].shape[0])
        data[i] = {
            'natural': X_i[:m],
            'adversarial': X_adv[i][:m]
        }

    # Training
    return AE_Detector(model, layers).fit(data)


def security_evaluation(model, X_nat, Y_nat, X_adv, k, n_samples, title="", fname=None):

    attacks = np.array(list(X_adv.keys()))
    eps = np.array(sorted(X_adv[attacks[0]].keys()))
    accs = np.zeros([attacks.shape[0], eps.shape[0], k])

    for i in range(k):
        accs[:, :, i] = compute_security_curves(model, X_nat, Y_nat, X_adv, attacks, eps, n_samples)

    plot_security_curves(attacks, eps, accs.mean(axis=2), title, fname)

    return accs


def run(dataset, model_type, n_samples, k, output, layers):

    model_dir = os.path.join(output, 'models')

    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = read_data(dataset)
    (X_train, X_val, Y_train, Y_val) = train_test_split(X_train, Y_train, test_size=.4, stratify=Y_train.argmax(1), random_state=SEED)

    # Create a CNN
    _get_cnn = get_basic_cnn if model_type == "basic_cnn" else get_cnn
    classifier = _get_cnn(X_train, Y_train, fname=os.path.join(model_dir, model_type + "_weights.h5"))

    # Compute performance on test data
    acc = classifier.evaluate(X_test, Y_test, verbose=0)[1]
    logging.debug('1. Test accuracy: {0:.2f}'.format(acc))

    # ================================== Create a detector  ==================================

    logging.debug("2. AE Detector fit")

    attacks_dir = os.path.join(output, "attacks")
    detector_dir = os.path.join(output, "detector")

    if not os.path.exists(detector_dir):
        # Load CW adversarial samples
        X_CW = np.load(os.path.join(attacks_dir, 'X_CW.npy'), allow_pickle=True).item()
        # Create a MLD detector
        detector = get_AE_Detector(classifier, layers, X_val, Y_val, X_CW)
        detector.save(detector_dir)
        # TODO: CHECK THIS HACK!
        detector.detector.gatekeeper.to_fit = False
    else:
        detector = AE_Detector(classifier, layers)
        detector.restore(detector_dir)

    # ================================== Compute security curves  ==================================

    logging.debug("3. Security Evaluation")

    sec_eval_dir = os.path.join(output, 'security_evaluation')

    # Load natural test data
    fname = os.path.join(attacks_dir, 'X_NAT.npy')
    _X = np.load(fname, allow_pickle=True).item()['_X']

    # Load prepared adversarial data
    fname = os.path.join(attacks_dir, 'X_ADV.npy')
    X_adv = np.load(fname, allow_pickle=True).item()

    # Sec curves
    _Y = classifier.predict(_X)
    classifier_accs = security_evaluation(classifier, _X, _Y, X_adv, k, n_samples,
                                     title=dataset.upper() + " Security Curves - Classifier",
                                     fname=mkdirs(os.path.join(sec_eval_dir, 'classifier.png')))

    detector_accs = security_evaluation(detector, _X, _Y, X_adv, k, n_samples,
                                          title=dataset.upper() + " Security Curves - Detector",
                                          fname=mkdirs(os.path.join(sec_eval_dir, 'detector.png')))

    # Save data
    attacks = np.array(list(X_adv.keys()))
    eps = np.array(list(X_adv[attacks[0]].keys()))

    d = {
        'attacks': attacks,
        'eps': eps,
        'k': k,
        '_X': _X,
        'classifier_accs': classifier_accs,
        'detector_accs': detector_accs
    }
    np.save(mkdirs(os.path.join(sec_eval_dir, 'sec_eval.npy')), d)


if __name__ == '__main__':

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("dataset", help="Dataset", type=str, choices=['mnist', 'cifar10'])
    parser.add_argument("model", help="Model type", type=str, choices=['basic_cnn', 'cnn'])
    parser.add_argument("layers", help="Layers to attach the detector", type=str, nargs='+')
    parser.add_argument("output", help="Output dir", type=str)
    parser.add_argument("-n", help="Number of samples to use for evaluation", type=int, default=100)
    parser.add_argument("-k", help="Number of evaluations", type=int, default=3)

    args = parser.parse_args()

    # Logging
    logging.getLogger().addHandler(logging.FileHandler(mkdirs(os.path.join(args.output, 'experiment.log'))))

    # Run
    run(args.dataset, args.model, args.n, args.k, args.output, args.layers)

    logging.debug("done.")


# TODO:
# - Parametrizzare i layers a cui attaccare il detector [x]
# - Restore parametrico di alcune parti del detector []
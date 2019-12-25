from __future__ import division

import argparse
import os
import numpy as np
np.random.seed(692019)

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import setGPU

from datasets import read_data
from models import get_basic_cnn, get_cnn
from attacks import FGSM, BIM, CW_L2, PGD, JSMA
from utils import generate_attack_samples, mkdirs

EPS = np.around(np.arange(0, 0.5, 0.05), 3)

if __name__ == '__main__':

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("dataset", help="Dataset", type=str, choices=['mnist', 'cifar10'])
    parser.add_argument("model", help="Model type", type=str, choices=['basic_cnn', 'cnn'])
    parser.add_argument("attack", help="Attack to perform", type=str, choices=['FGSM', 'BIM', 'CW_L2', 'PGD', 'JSMA'])
    parser.add_argument("output", help="Output dir", type=str, default=".")
    parser.add_argument("-n", help="Number of adversarial samples to craft", type=int, default=500)
    args = parser.parse_args()

    # Logging
    logging.getLogger().addHandler(logging.FileHandler(mkdirs(os.path.join(args.output, 'experiment.log'))))

    # Read data
    (X_train, Y_train), (X_test, Y_test) = read_data(args.dataset)

    # Train the model on training set
    model_dir = os.path.join(args.output, 'models')
    _get_cnn = get_basic_cnn if args.model == "basic_cnn" else get_cnn
    model = _get_cnn(X_train, Y_train, fname=os.path.join(model_dir, args.model + "_weights.h5"))

    # Compute performance on test data
    logging.debug("Evaluate model on test data")
    acc = model.evaluate(X_test, Y_test)[1]
    logging.debug('Test accuracy: {0:.2f}'.format(acc))

    # And store them
    attacks_dir = os.path.join(args.output, "attacks")
    fname = os.path.join(attacks_dir, "X_NAT.npy")
    if not os.path.exists(fname):
        # Select a subset of data
        p_idxs = np.random.permutation(X_test.shape[0])[:args.n]
        _X, _y = X_test[p_idxs], Y_test[p_idxs]
        X_nat = {
            '_X': _X,
            '_y': _y
        }
        np.save(mkdirs(fname), X_nat)
    else:
        d = np.load(fname, allow_pickle=True).item()
        _X, _y = d['_X'], d['_y']

    logging.debug("Generate attack samples")
    generate_attack_samples(model, [eval(args.attack)], EPS, _X, fname=mkdirs(os.path.join(attacks_dir, "X_ADV.npy")))

    logging.debug("done.")

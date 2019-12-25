from __future__ import division

import os

import argparse
import numpy as np

from utils import mkdirs

np.random.seed(692019)

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import setGPU
from attacks import CW_L2
from datasets import read_data
from models import get_basic_cnn, get_cnn

N = 10 ** 4
N_SAMPLES = 1000

if __name__ == '__main__':
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("dataset", help="Dataset", type=str, choices=['mnist', 'cifar10'])
    parser.add_argument("model", help="Model type", type=str, choices=['basic_cnn', 'cnn'])
    parser.add_argument("output", help="Output dir", type=str, default=".")
    args = parser.parse_args()

    # Logging
    logging.getLogger().addHandler(logging.FileHandler(mkdirs(os.path.join(args.output, 'experiment.log'))))

    # Load dataset
    (X_train, Y_train), (X_test, Y_test) = read_data(args.dataset)

    # Train the model on training set
    model_dir = os.path.join(args.output, 'models')
    _get_cnn = get_basic_cnn if args.model == "basic_cnn" else get_cnn
    model = _get_cnn(X_train, Y_train, fname=os.path.join(model_dir, args.model+"_weights.h5"))

    # Compute performance on test data
    logging.debug("Evaluate model on test data")
    acc = model.evaluate(X_test, Y_test)[1]
    logging.debug('Test accuracy: {0:.2f}'.format(acc))

    # Restrict to a subset of data
    _X, _Y = X_train[:N], Y_train[:N]
    class_idxs = {}
    # Index them
    for i in range(_Y.shape[1]):
        class_idxs[i] = np.where(_Y.argmax(axis=1) == i)[0]

    # CW attack
    X_adv = {}

    logging.debug("Generating adversarial samples:")
    N_CLASSES = _Y.shape[1]
    for T in range(N_CLASSES):
        logging.debug("- Target label: ", T)
        # Selecting N_SAMPLES from the OTHER classes
        _X_subs = np.zeros_like(X_train[:N_SAMPLES])
        OTHERS = [i for i in range(N_CLASSES) if i != T]
        SZ = N_SAMPLES // len(OTHERS)
        for i, o in enumerate(OTHERS):
            _start, _end = i * SZ, min(N_SAMPLES, (i+1) * SZ)
            _X_subs[_start:_end] = np.random.permutation(_X[class_idxs[o]])[:SZ]

        # Setting target labels
        Y_TGT = np.zeros((N_SAMPLES, N_CLASSES))
        Y_TGT[:, T] = 1
        # Crafting adversarial samples
        BS = int(0.1 * N_SAMPLES)                           # TODO: ACHTUNG! CW requires N_SAMPLES / batch_size == 0
        X_adv[T] = CW_L2(batch_size=BS).generate(model, _X_subs, y=Y_TGT)

        # Update dict
        attacks_dir = os.path.join(args.output, "attacks")
        np.save(mkdirs(os.path.join(attacks_dir, "X_CW.npy")), X_adv)

    logging.debug("done.")

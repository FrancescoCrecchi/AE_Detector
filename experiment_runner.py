import os
import argparse
import subprocess
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import setGPU

from utils import mkdirs
from attacks import FGSM, BIM, PGD, JSMA
attacks = [FGSM, BIM, PGD]              # JSMA


if __name__ == '__main__':

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter)
    parser.add_argument("dataset", help="Dataset", type=str, choices=['mnist', 'cifar10'])
    parser.add_argument("model", help="Model type", type=str, choices=['basic_cnn', 'cnn'])
    parser.add_argument("layers", help="Layers to attach the detector", type=str, nargs='+')
    args = parser.parse_args()

    output = os.path.join("Experiments", datetime.now().strftime("%d.%m.%y-%H:%M:%S"))
    logging.getLogger().addHandler(logging.FileHandler(mkdirs(os.path.join(output, 'experiment.log'))))

    logging.debug(" +++ START EXPERIMENT +++ ")

    logging.debug("\n === CONFIG === ")
    logging.debug("- Dataset: {0}".format(args.dataset))
    logging.debug("- Model: {0}".format(args.model))
    logging.debug("- Output: {0}".format(output))

    logging.debug("\n === GENERATE CW SAMPLES === ")
    subprocess.run(["python", "generate_CW.py", args.dataset, args.model, output])

    logging.debug("\n === GENERATE ATTACK SAMPLES === ")
    for attack in attacks:
        subprocess.run(["python", "generate_attack_samples.py", args.dataset, args.model, attack.__name__, output])

    logging.debug("\n === COMPUTE SECURITY CURVES === ")
    subprocess.run(["python", "main.py", args.dataset, args.model, args.layers, output])

    logging.debug("\n +++ END EXPERIMENT +++ ")

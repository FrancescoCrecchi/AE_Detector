import os
from sklearn.externals import joblib

from sklearn.neighbors import LocalOutlierFactor, KernelDensity
import numpy as np
from utils import isempty, mkdirs
# Grid search for kernel bandwidth
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from manifold import P_tSNE
from tqdm import tqdm
from sklearn.svm import SVC
from keras.utils import to_categorical


class KDE():

    def __init__(self):
        """
            Scikit-learn KernelDensity wrapper
        """
        self.kde = None
        self.mean_score = -np.inf
        self.scaler = None

        # To fit?
        self.to_fit = True

    def _auto_tune(self, X):
        # Auto-tune bandwidth using cross-validation
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths}, cv=3)
        grid.fit(X)
        return grid.best_estimator_

    def fit(self, X):
        if self.to_fit:
            # KDE
            self.kde = self._auto_tune(X)
            # Scaler
            _scores = self._score_samples(X)
            self.mean_score = np.mean(_scores)
            self.scaler = MinMaxScaler(feature_range=(0, 1)).fit((_scores/self.mean_score).reshape(-1, 1))

            # To fit?
            self.to_fit = False

        return self

    def _score_samples(self, X):
        return np.exp(self.kde.score_samples(X))

    def score_samples(self, X):
        assert not self.to_fit, "KDE has to be fit before being used in prediction!"
        scores = self.scaler.transform((self._score_samples(X)/self.mean_score).reshape(-1, 1))
        scores = np.maximum(scores, 0)
        # assert (0 <= scores).all() & (scores <= 1).all()
        return scores           # in [0,1]

    # Save & Restore model
    def save(self, dir_path):
        # Save KDE
        joblib.dump(self.kde, mkdirs(os.path.join(dir_path, "kde.pkl")))
        # Save scaler and mean_score
        np.save(mkdirs(os.path.join(dir_path, "mean_score.npy")), self.mean_score)
        joblib.dump(self.scaler, mkdirs(os.path.join(dir_path, "scaler.pkl")))

    def restore(self, dir_path):
        # Save KDE
        self.kde = joblib.load(os.path.join(dir_path, "kde.pkl"))
        # Save scaler and mean_score
        self.mean_score = np.load(os.path.join(dir_path, "mean_score.npy"), allow_pickle=True)
        self.scaler = joblib.load(os.path.join(dir_path, "scaler.pkl"))
        # Mark as fit
        self.to_fit = False


class LayerDetector():

    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.num_classes = model.output_shape[1]
        # Used for store single-class outlier detectors
        self.mappers = {}
        self.estimators = {}
        for i in range(self.num_classes):
            self.mappers[i] = P_tSNE(self.model, self.layer)
            self.estimators[i] = KDE()

    @property
    def to_fit(self):
        return all([m.to_fit for m in self.mappers.values()]) & all([e.to_fit for e in self.estimators.values()])

    @to_fit.setter
    def to_fit(self, value):
        # Going recursively on its components
        for m in self.mappers.values():
            m.to_fit = value
        for e in self.estimators.values():
            e.to_fit = value

    def fit(self, data):
        '''
        Fit a mapper and an estimator per class
        :param data: dictionary containing pairs of good and adversarial samples for each class like {0: {'natural': X, 'adversarial': X_adv}}
        :return: self
        '''
        if self.to_fit:
            for i in tqdm(range(self.num_classes)):
                if not isempty(data[i]):
                    # Retrieve class data
                    X_nat = data[i]['natural']
                    X_adv = data[i]['adversarial']
                    _X = np.concatenate((X_nat, X_adv))
                    # Fit a P_tSNE model (if needed)
                    self.mappers[i].fit(_X)
                    # Fit estimator ONLY ON NATURAL DATA! ;-)
                    self.estimators[i].fit(self.mappers[i].project(X_nat))

        return self

    def score_samples(self, X):
        '''
        Compute p_value scores for given input samples.
        :param X: New samples
        :return: P_value scores for each class and sample.
        '''
        n_samples = X.shape[0]
        p_value_scores = -1 * np.ones((n_samples, self.num_classes))

        assert not self.to_fit, "LayerDetector has to be fit before being used for project!"
        # Going through each embedding to perform adversarial detection
        for i in range(self.num_classes):
            # todo: BETTER HANDLING OF TRAINED MAPPER/ESTIMATOR FOR A CERTAIN CLASS!
            if (i in self.mappers) and (i in self.estimators):
                # Map through mapper
                e = self.mappers[i].project(X)
                # Pass through estimator
                s = self.estimators[i].score_samples(e)
                # Store results
                p_value_scores[:, i] = s.squeeze()

        return p_value_scores

    def save(self, dir_path):
        for i in range(self.num_classes):
            # Save Mapper
            cl_dir = os.path.join(dir_path, str(i))
            self.mappers[i].save(os.path.join(cl_dir, "mapper"))
            # Save Estimator
            self.estimators[i].save(os.path.join(cl_dir, "estimator"))

    def restore(self, dir_path, restore_dict):
        '''
        Restore from disk (partially, in case)
        :param dir_path: files path
        :param restore_dict: dictionary of the form
            restore_dict = {
                'mapper': True/False,
                'estimator': True/False
            }
            to (partially) restore detector's components.
        '''
        for i in range(self.num_classes):
            cl_dir = os.path.join(dir_path, str(i))
            # Restore Mapper (one for each class)
            if restore_dict['mapper']:
                self.mappers[i].restore(os.path.join(cl_dir, "mapper"))
            # Restore Estimator (one for each class)
            if restore_dict['estimator']:
                self.estimators[i].restore(os.path.join(cl_dir, "estimator"))


class MultilayerDetector:

    class GateKeeper:

        def __init__(self):
            self.model = None
            # To fit?
            self.to_fit = True

        def fit(self, X, y):
            if self.to_fit:
                # Auto-tune via cross-validation
                grid = GridSearchCV(SVC(kernel='linear'), {'C': [1, 10, 100, 1000]}, cv=3).fit(X, y)
                self.model = grid.best_estimator_

                # Mark as fit
                self.to_fit = False

            return self

        def predict(self, X):
            assert not self.to_fit, "Gatekeeper has to be fit before being used for project!"
            return self.model.predict(X)

        def save(self, dir_path):
            joblib.dump(self.model, mkdirs(os.path.join(dir_path, 'gatekeeper.pkl')))

        def restore(self, dir_path):
            self.model = joblib.load(mkdirs(os.path.join(dir_path, 'gatekeeper.pkl')))
            # Mark as fit
            self.to_fit = False

    def __init__(self, model, layers):
        # We save parameters
        self.model = model
        self.layers = layers
        self.num_classes = model.output_shape[1]
        # We need a LayerDetector for each layer
        self.layer_detectors = {}
        for i in range(len(layers)):
            self.layer_detectors[i] = LayerDetector(self.model, self.layers[i])

        # We need a binary classifier to detect if the samples is adversarial or not
        self.gatekeeper = self.GateKeeper()


    @property
    def to_fit(self):
        return all([ld.to_fit for ld in self.layer_detectors.values()]) & self.gatekeeper.to_fit

    @to_fit.setter
    def to_fit(self, value):
        # Going recursively of its components
        for l in self.layer_detectors.values():
            l.to_fit = value
        self.gatekeeper.to_fit = True

    def _score_samples(self, X):
        '''
        Compute p_value scores for given input samples.
        :param X: New samples
        :return: P_value scores as a tensor of shape [n_samples, n_classes, n_layers]
        '''

        n_samples = X.shape[0]
        p_scores_series = np.zeros((n_samples, self.num_classes, len(self.layers)))

        for i in self.layer_detectors.keys():
            p_scores_series[:, :, i] = self.layer_detectors[i].score_samples(X)

        return p_scores_series

    def fit(self, data):
        '''
        Fit a LayerDetector for each layer
        :param data: dictionary containing pairs of good and adversarial samples for each class like {0: {'natural': X, 'adversarial': X_adv}}
        '''
        # For each layer
        for i, l in tqdm(enumerate(self.layers)):
            # Fit a layer detector
            self.layer_detectors[i].fit(data)

        # ============ GATEKEEPER FIT ==========
        if self.gatekeeper.to_fit:      # TODO: Redundant check but creating the dataset is costly!
            # Dataset creation phase
            dset, lbls = [], []
            for i in data.keys():
                if not isempty(data[i]):
                    nat, adv = data[i]['natural'], data[i]['adversarial']
                    n_nat, n_adv = nat.shape[0], adv.shape[0]
                    _pscores = self._score_samples(np.concatenate((nat, adv)))
                    _lbls = np.concatenate((-1 * np.ones(n_nat), np.ones(n_adv)))
                    dset.append(_pscores.reshape(_pscores.shape[0], -1))
                    lbls.append(_lbls)
            dset, lbls = np.concatenate(dset), np.concatenate(lbls)
            # Fit
            self.gatekeeper.fit(dset, lbls)

        return self

    def predict(self, X):
        assert not self.to_fit, "MultilayerDetector (and all of its components!) has to be fit before being used for project!"
        # Compute p_scores
        _pscores = self._score_samples(X)
        preds = self.gatekeeper.predict(_pscores.reshape(_pscores.shape[0], -1))

        return preds

    # Save & Restore model
    def save(self, dir_path):
        '''
        Save the detector to disk and a dictionary describing its parts
        :param dir_path: directory used for model saving
        '''
        restore_dict = {'LD': {}, 'MLD': {}}
        # Save LayerDetectors
        for i, l in enumerate(self.layers):
            ld_dir = os.path.join(dir_path, "LD", l)
            self.layer_detectors[i].save(ld_dir)
            restore_dict['LD'][l] = {
                'mapper': True,
                'estimator': True
            }
        # Save Gatekeeper
        gkpr_dir = os.path.join(dir_path, "MLD")
        self.gatekeeper.save(gkpr_dir)
        restore_dict['MLD']['Gatekeeper'] = True
        # Dump restore_dict to file
        np.save(os.path.join(dir_path, 'restore_dict.npy'), restore_dict)

    def restore(self, dir_path, restore_dict):
        '''
        Restore a detector from disk
        :param dir_path: saved detector folder
        :param restore_dict: if not None, restore parts of the detector specified as:
            restore dict = {
                'LD': {
                    ... ,
                    'fc1': {
                        'mapper': True,
                        'estimator': False
                    }
                    ...
                }
                'MLD': {
                    'Gatekeeper': True
                }
            }
        :return: (Partially) restored estimator for inference.
        '''
        # Check if is a total or partial restore
        if restore_dict is None:
            # Use the default restore configuration (aka restore all components)
            restore_dict = np.load(os.path.join(dir_path, 'restore_dict.npy'), allow_pickle=True).item()

        assert sorted(list(restore_dict.keys())) == ['LD', 'MLD'], ValueError("Incorrect 'restore_dict' format passed!")
        # Restore LayerDetectors
        for i, l in enumerate(self.layers):
            ld_dir = os.path.join(dir_path, "LD", l)
            self.layer_detectors[i].restore(ld_dir, restore_dict['LD'][l])
        # Save Gatekeeper
        gkpr_dir = os.path.join(dir_path, "MLD")
        if restore_dict['MLD']['Gatekeeper']:
            self.gatekeeper.restore(gkpr_dir)


class AE_Detector():

    def __init__(self, model, layers):
        # Params
        self.model = model
        self.layers = layers
        # Multilayer Detector
        self.detector = MultilayerDetector(self.model, self.layers)         # <-- Outputs 0/1 we need a N+1 classifier!

    @property
    def to_fit(self):
        return self.detector.to_fit

    @to_fit.setter
    def to_fit(self, value):
        # Going recursively on its components
        self.detector.to_fit = value

    def fit(self, data):
        # Security check
        assert self.to_fit, "Trying to fit an ALREADY FIT detector, " \
                            "if this is what you really want set is 'to_fit' property as True"
        # Fit the detector
        self.detector.fit(data)

        # Mark as fit
        self.to_fit = self.detector.to_fit

        return self

    def predict(self, X):
        assert not self.to_fit, "AE_Detector has to be fit before being used for project!"
        y_pred = self.model.predict(X).argmax(axis=1)       # in [0, N-1]
        y_gkpr = self.detector.predict(X)                   # in [0, 1] -> [natural, adversarial]

        N = self.model.output_shape[1]
        y_pred[y_gkpr == 1] = N                            # N is the AdvEx label

        return to_categorical(y_pred)

    # Save & Restore model
    def save(self, dir_path):
        self.detector.save(dir_path)

    def restore(self, dir_path, restore_dict=None):
        self.detector.restore(dir_path, restore_dict)

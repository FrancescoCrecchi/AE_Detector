import os
import numpy as np

from sklearn.manifold import TSNE

from keras.models import Model, Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from utils import mkdirs


class P_tSNE:

    def __init__(self, model, layer='fc1'):

        # Creating a feature extractor from the original model
        try:
            feat_extractor = Model(inputs=model.input,
                               outputs=model.get_layer(layer).output)
        except:
            raise ValueError("Layer '{}' to attach LD not found!".format(layer))
        self.feat_extractor = feat_extractor

        # Creating an encoder
        encoder = Sequential()
        encoder.add(Dense(500, activation='relu', input_shape=(np.prod(feat_extractor.output_shape[1:]),)))
        encoder.add(Dense(500, activation='relu'))
        encoder.add(Dense(2000, activation='relu'))
        encoder.add(Dense(2))
        encoder.compile(loss='mse', optimizer='adam')
        self.encoder = encoder

        # To fit?
        self.to_fit = True

    def fit(self, X, verbose=0):
        if self.to_fit:
            # Computing embeddings using T-SNE
            features = self.feat_extractor.predict(X, batch_size=128).reshape(X.shape[0], -1)
            embeddings = TSNE(verbose=verbose).fit_transform(features)

            # Fit the encoder on these embeddings
            self.encoder.fit(features, embeddings, epochs=100, verbose=verbose, callbacks=[
                EarlyStopping(monitor='loss', patience=10)
            ])

            # Mark as fit
            self.to_fit = False

        return self

    def project(self, X):
        assert not self.to_fit, "Mapper has to be fit before being used for project!"
        return self.encoder.predict(self.feat_extractor.predict(X).reshape(X.shape[0], -1))

    # Save & Restore model
    def save(self, dir_path):
        # Save feature-extractor
        self.feat_extractor.save(mkdirs(os.path.join(dir_path, 'feat_extractor.h5')))
        # Save encoder
        self.encoder.save(mkdirs(os.path.join(dir_path, 'encoder.h5')))

    def restore(self, dir_path):
        # Restore feature-extractor
        self.feat_extractor = load_model(os.path.join(dir_path, 'feat_extractor.h5'))
        # Restore encoder
        self.encoder = load_model(os.path.join(dir_path, 'encoder.h5'))
        # Mark as fit
        self.to_fit = False

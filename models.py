import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from utils import mkdirs

# GLOBAL PARAMETERS

VERBOSE = 1
BS = 128
EPOCHS = 100
PATIENCE = 10


def get_basic_cnn(X_train, Y_train, fname=None):
    model = Sequential()

    _in_shape = X_train.shape[1:]
    n_classes = Y_train.shape[1]

    model.add(Conv2D(64, (8, 8), strides=(2, 2), padding='same', input_shape=_in_shape, name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (6, 6), strides=(2, 2), padding='same', input_shape=_in_shape, name='conv2'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='same', input_shape=_in_shape, name='conv3'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dense(n_classes, name='output'))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Load or save weights?
    if fname is not None and os.path.exists(fname):
        # Load weights
        model.load_weights(fname)

    else:
        # Train and validation split
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, Y_train, test_size=0.3, stratify=Y_train.argmax(axis=1))

        model.fit(X_tr, y_tr, batch_size=BS, epochs=EPOCHS, verbose=VERBOSE,
                  validation_data=(X_vl, y_vl),
                  callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])

        # Save weigths
        if fname is not None:
            model.save_weights(mkdirs(fname))

    return model


def get_cnn(X_train, Y_train, fname=None):

    model = Sequential()

    _in_shape = X_train.shape[1:]
    _num_classes = Y_train.shape[1]

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=_in_shape, name='conv1'))
    model.add(BatchNormalization(axis=3, name='bn_conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name='conv2'))
    model.add(BatchNormalization(axis=3, name='bn_conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization(axis=3, name='bn_conv3'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), name='conv4'))
    model.add(BatchNormalization(axis=3, name='bn_conv4'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, name='fc1'))
    model.add(BatchNormalization(axis=1, name='bn_fc1'))
    model.add(Activation('relu'))
    model.add(Dense(_num_classes, name='output'))
    model.add(BatchNormalization(axis=1, name='bn_outptut'))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Load or save weights?
    if fname is not None and os.path.exists(fname):
        # Load weights
        model.load_weights(fname)

    else:
        # Train the model
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        # Train and validation split
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, Y_train, test_size=0.3, stratify=Y_train.argmax(axis=1))

        model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=BS),
                            steps_per_epoch=X_tr.shape[0] // BS,
                            epochs=EPOCHS,
                            validation_data=(X_vl, y_vl),
                            callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])

        # Save weigths
        if fname is not None:
            model.save_weights(mkdirs(fname))

    return model
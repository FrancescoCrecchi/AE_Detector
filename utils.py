import os
import numpy as np
import itertools

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

import setGPU


def plot_with_labels(lowDWeights, labels, size=10, title="", ticksize=None, classes=['natural', 'adversarial'],
                     figsize=(10, 8), fig=None, ax=None):
    assert lowDWeights.shape[0] >= len(labels), "More labels than weights"

    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(figsize=figsize)  # in inches


    # Tick size
    if ticksize is not None:
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    markers = ['o', 'X', '+', 'v']
    colors = ['blue', 'red', 'green', 'white']
    for i, l in enumerate(np.unique(labels)):
        ix = np.where(labels == l)
        ax.scatter(lowDWeights[ix, 0], lowDWeights[ix, 1], c=colors[i], label=classes[i], s=size, marker=markers[i],
                   alpha=1.0 / (i + 1))

    ax.legend()
    ax.set_title(title)

    # plt.show()

#
# def plot_level_sets(clf, embeds, lbls, title="Local Outlier Factor (LOF)", cmap='jet'):
#     # Computing grid
#     (x_min, y_min), (x_max, y_max) = embeds.min(axis=0), embeds.max(axis=0)
#     offset = 10
#     xx, yy = np.meshgrid(np.linspace(x_min - offset, x_max + offset), np.linspace(y_min - offset, y_max + offset))
#
#     foo = []
#     # Compute level sets
#     for i in range(len(clf)):
#         _Z = clf[i]._decision_function(np.c_[xx.ravel(), yy.ravel()])
#         _Z = _Z.reshape(xx.shape)
#
#         foo.append(_Z)
#     # Stack among the new dimension and compute the max
#     Z = np.stack(foo, 2).max(axis=2)
#
#     fig = plt.figure()
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#     plt.title(title)
#
#     good_embeds = embeds[lbls == 1]
#     bad_embeds = embeds[lbls == -1]
#     plt.scatter(good_embeds[:, 0], good_embeds[:, 1], c='blue', label='normal', marker="o", s=1, edgecolor='')
#     plt.scatter(bad_embeds[:, 0], bad_embeds[:, 1], c='red', label='adversarial', marker="x", s=3, edgecolor='')
#     plt.legend()
#     fig.tight_layout()
#
#     plt.show()
#
#
# def compute_statistics(y_true, y_pred):
#     # Computing confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#
#     # Getting values from confusion matrix
#     TN, FP, FN, TP = cm.ravel()
#
#     # Sensitivity, hit rate, recall, or true positive rate
#     TPR = TP / (TP + FN)
#     # Specificity or true negative rate
#     TNR = TN / (TN + FP)
#     # Fall out or false positive rate
#     FPR = FP / (FP + TN)
#     # False negative rate
#     FNR = FN / (TP + FN)
#
#     # Overall accuracy
#     ACC = (TP + TN) / (TP + FP + FN + TN)
#
#     # Collecting
#     s = {
#         'FP': FP,
#         'FN': FN,
#         'TP': TP,
#         'TN': TN,
#         'TPR': TPR,
#         'TNR': TNR,
#         'FPR': FPR,
#         'FNR': FNR,
#         'ACC': ACC
#     }
#
#     return cm, s


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(8,6)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum()
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.grid(0)
    plt.show()


def mkdirs(fname):
    dir_path = fname.rsplit("/", 1)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return fname


def show_image_delta(orig_img, orig_preds, adv_img, adv_preds, ord, classes=None,
                     cmap=mpl.rcParams['image.cmap'], title="", figsize=(8, 6), show_cbar=True):

    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    delta = adv_img - orig_img
    d = compute_distance(adv_img, orig_img, ord)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    _ = ax1.imshow(orig_img.squeeze(), cmap=cmap)
    # Remove tick labels
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # Set original label
    _y = orig_preds.argmax()
    ax1.set_title("{0} ({1:.2f})".format(_y if classes is None else classes[_y], orig_preds[_y]))
    _ = ax2.imshow(adv_img.squeeze(), cmap=cmap)
    # Remove tick labels
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    _y = adv_preds.argmax()
    ax2.set_title("{0} ({1:.2f})".format(_y if classes is None else classes[_y], adv_preds[_y]))
    img3 = ax3.imshow(np.sum(delta, axis=2), cmap=cmap)
    # Remove tick labels
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_title(d)

    if show_cbar:
        colorbar(img3)
    plt.tight_layout(h_pad=1)

    fig.suptitle(title)
    fig.subplots_adjust(top=0.88)

    plt.show()


def compute_distance(x_adv, x, ord):
    return np.linalg.norm((x_adv - x).ravel(), ord)


def visualize_density(f, fig=None, ax=None, c=(0, 0), span=(100, 100), n_points=100, title="", pad=5):

    def colorbar(mappable, fig, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label('p_values', rotation=270, labelpad=20)
        return cbar

    # Obtaining a 2d grid
    x_axis = np.linspace(c[0]-span[0]/2+pad, c[0]+span[0]/2-pad, num=n_points)
    y_axis = np.linspace(c[1]-span[1]/2+pad, c[1]+span[1]/2-pad, num=n_points)
    x, y = np.meshgrid(x_axis, y_axis)
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    # Evaluate function
    z = f(xy).reshape(n_points, n_points)

    # Plot
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots()

    cs = ax.contourf(x, y, z)
    ax.set_title(title)
    colorbar(cs, fig, ax)

    # plt.show()


def plot_class_hist(Y, ax=None, classes=None, title=""):
    if ax is None:
        ax = plt.subplot()

    n_classes = Y.shape[1]
    _ = ax.hist(Y.argmax(axis=1), bins=np.arange(n_classes+1) - 0.5)
    _ = ax.set_xticks(range(n_classes))
    if classes is not None:
        _ = ax.set_xticklabels(classes, rotation=45)
    _ = ax.set_title(title)

    plt.show()


def plot_natural_vs_adversarial_labels(Y_nat, Y_adv, classes=None, figsize=(10, 4)):
    n_classes = Y_nat.shape[1]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

    _ = ax1.hist(Y_nat.argmax(axis=1), bins=np.arange(n_classes+1) - 0.5)
    _ = ax1.set_xticks(range(n_classes))
    if classes is not None:
        _ = ax1.set_xticklabels(classes, rotation=45)
    _ = ax1.set_title('Original labels')

    _ = ax2.hist(Y_adv, bins=np.arange(n_classes+1) - 0.5)
    _ = ax2.set_xticks(range(n_classes))
    if classes is not None:
        _ = ax2.set_xticklabels(classes, rotation=45)
    _ = ax2.set_title('Adversarial labels')

    plt.show()


def plot_image_with_pvalues(img, lbl, p_scores, top_k=10, layers=None, classes=None, figsize=(15, 6)):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Show image
    ax1.imshow(img.squeeze(), aspect='auto')
    l = lbl
    if classes is not None:
        l = classes[l]
    ax1.set_title(l)

    # Show p_values
    ax2.set_xticks(range(p_scores.shape[0]))

    _n_classes = p_scores.shape[0]
    if classes is None:
        classes = range(_n_classes)

    # Select top-K p_scores
    sum_scores = np.sum(p_scores, axis=1)
    top_k_idxs = np.argsort(sum_scores)[::-1][:top_k]
    for i in top_k_idxs:
        ax2.plot(p_scores[i, :], label=classes[i])
    ax2.legend()
    ax2.set_ylabel('p_value_scores')

    if layers is not None:
        ax2.set_xticklabels(layers, rotation=45)

    l = p_scores[:, -1].argmax(axis=0)      # Todo: It's not always that last layer for detector is 'output'!
    if classes is not None:
        l = classes[l]
    ax2.set_title(l)

    plt.show()


def plot_pvalues_per_layer(p_scores, layers, classes=None, figsize=(15, 6)):
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize, squeeze=False)

    for i in range(n_layers):
        # title
        _ = axes[0, i].set_title(layers[i])
        # hist
        lbls = p_scores[:, :, i]
        n_classes = lbls.shape[1]
        _ = axes[0, i].hist(lbls.argmax(axis=1), bins=np.arange(n_classes + 1) - 0.5)
        # x-ticks
        _ = axes[0, i].set_xticks(range(n_classes))
        if classes is not None:
            _ = axes[0, i].set_xticklabels(classes, rotation=45)

    plt.show()


def plot_sec_curve(ax, eps, acc, attack):
    # Plot curve
    ax.plot(eps, acc, label=attack)
    ax.set_xticks(eps)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend()


def plot_security_curves(attacks, eps, accuracies, title, fname=None):
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(attacks.shape[0]):
        # Plot
        plot_sec_curve(ax, eps, accuracies[i, :], attacks[i])

    plt.title(title)
    if fname is not None:
        plt.savefig(mkdirs(fname))
    else:
        plt.show()


def compute_security_curves(classifier, X_nat, Y_nat, X_adv, attacks, eps, n_samples):
    # Accuracy scores
    attack_accuracies = np.zeros([attacks.shape[0], eps.shape[0]])

    # Select a subset of data
    idxs = np.random.permutation(X_nat.shape[0])[:n_samples]
    X_nat = X_nat[idxs]
    Y_nat = Y_nat[idxs]

    n_classes = Y_nat.shape[1]
    Y_nat_c = Y_nat.argmax(1)

    # Attack the model with increasing confidence/distance
    for i in range(attacks.shape[0]):
        attack = attacks[i]
        for j in range(eps.shape[0]):
            _eps = eps[j]

            # Fixed adversarial data
            _X_adv = X_adv[attack][_eps][idxs]
            assert _X_adv.shape[0] == X_nat.shape[0]

            # Compute adversarial predictions
            Y_adv_c = classifier.predict(_X_adv).argmax(1)

            # Set to N adversarial examples: induced class label as marker!
            _Y = Y_nat_c.copy()
            _Y[_Y != Y_adv_c] = n_classes  # AdvEX

            attack_accuracies[i, j] = accuracy_score(_Y, Y_adv_c)

    return attack_accuracies


def generate_attack_samples(model, attacks, eps, X, fname=None):
    if fname is not None and os.path.exists(fname):
        X_adv = np.load(fname, allow_pickle=True).item()
    else:
        X_adv = {}

    for attack in attacks:
        # Make space for new attack samples
        X_adv[attack.__name__] = {}
        print("- " + attack.__name__ + ":")

        # For each epsilon
        for _eps in tqdm(eps):
            if attack.__name__ == "CW_L2":
                X_adv[attack.__name__][_eps] = attack(_eps * 100).generate(model, X)  # TODO: Remove this HACK!
            else:
                X_adv[attack.__name__][_eps] = attack(_eps).generate(model, X)

            # Checkpointing
            if fname is not None:
                np.save(fname, X_adv)

    return X_adv


def isempty(d):
    return not bool(d)

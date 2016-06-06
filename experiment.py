import csv
import random
from gensim.models import Word2Vec

w2v = Word2Vec.load_word2vec_format("/Users/joelkuiper/Desktop/cui.embeddings.bin", binary=True)

def csv_as_dict(file, ref_header, delimiter=","):
    reader = csv.DictReader(open(file), delimiter=delimiter)
    result = {}
    for row in reader:
        key = row.pop(ref_header)
        if key in result:
            pass
        result[key] = row
    return result

mapping_cui = csv_as_dict("/Users/joelkuiper/Desktop/CUI_LABEL_TYPE_RESTRICTED.csv", ref_header="cui")
mapping_label = csv_as_dict("/Users/joelkuiper/Desktop/CUI_LABEL_TYPE_RESTRICTED.csv", ref_header="label")


##### from https://github.com/oreillymedia/t-SNE-tutorial

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def scatter(x, labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    colors = le.transform(labels)

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels
    txts = []
    for i, label in enumerate(le.classes_):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, label, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def embedding(model, key):
    if model.vocab.has_key(key):
        return np.array(model[key])
    else:
        #return np.zeros(model.layer1_size,  dtype='float32')
        return None


def get_X_y(model, index, field, length=1000):
    vocab = model.vocab.keys()
    sample = random.sample(index.keys(), length)

    embeddings = [(k, embedding(model, k)) for k in sample]
    embeddings = [e for e in embeddings if e[1] is not None]

    # the asfarray comes from this bug https://github.com/scikit-learn/scikit-learn/issues/4124
    X = np.vstack([e[1] for e in embeddings])

    y = np.vstack([index.get(e[0])[field] for e in embeddings])

    return X, y

from sklearn.covariance import EllipticEnvelope
def plot(X, y):
    proj = TSNE().fit_transform(X)
    e = EllipticEnvelope(assume_centered=True, contamination=.25) # Outlier detection
    e.fit(X)

    good = np.where(e.predict(X) == 1)
    X = X[good]
    y = y[good]

    scatter(proj, y)

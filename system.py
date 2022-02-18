"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def reduce_dimensions(feature_vectors_full, model):
    """Performs PCA on training data

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    EIG_RANGE = 40
    # Check if test or train
    if "fvectors_train" in model.keys():
        # If test, get v and mean from model
        v = np.array(model["train_v"])
        mean = model["train_mean"]
        # Center data
        pca_feature = np.dot(feature_vectors_full - mean, v)
    else:
        # Calculating the covariance matrix
        covariance = np.cov(feature_vectors_full, rowvar=0)
        N = covariance.shape[0]
        # Calculating the eigenvectors
        w, v = scipy.linalg.eigh(covariance, eigvals=(N - EIG_RANGE, N - 1))
        v = np.fliplr(v)
        # Find mean
        mean = np.mean(feature_vectors_full)
        # Add v and mean to model
        model["train_v"] = v.tolist()
        model["train_mean"] = mean

    # Center data
    pca_feature = np.dot(feature_vectors_full - mean, v)

    # Check for features
    if "features" in model.keys():
        features = model["features"]
    # Find divergence
    else:
        # Get classes
        train_label = np.array(model["labels_train"])
        classes = np.unique(train_label)

        # Define divergence array
        d = np.zeros(EIG_RANGE)
        # Loop through classes to find divergence
        for i in range(classes.size):
            for j in range(i + 1, classes.size):
                c1 = pca_feature[train_label == classes[i], :]
                c2 = pca_feature[train_label == classes[j], :]
                if c1.size >= 10 and c2.size >= 10:
                    d += divergence(c1, c2)

        # Sort by divergence
        sorted_d = np.argsort(-d)
        # Remove first PCA
        d_remaining = np.delete(d, sorted_d[0])

        corr = np.corrcoef(pca_feature, rowvar=0)
        features = []
        # Find 10 best features
        for i in range(10):
            # Loop through previous added features
            for j in range(i):
                # Penalise correlation
                for feat in d_remaining:
                    feat = feat * (1 - corr[features[j]] * 1e17)
            # Add feature
            sorted_d = np.argsort(-d_remaining)
            index = int(np.argwhere(d == d_remaining[sorted_d[0]])[0][0])
            features.append(index)
            # Remove from remaining
            d_remaining = np.delete(d_remaining, sorted_d[0])

        # Add features to model
        model["features"] = features

    # Return features
    return pca_feature[:, features]

def divergence(class1, class2):
    """compute a vector of 1-D divergences

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2

    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)
    # Variance floors
    floor = 1e-16
    f = np.vectorize(lambda x: floor if x < floor else x)
    v1 = f(v1)
    v2 = f(v2)

    # Find 1-D divergence
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)
    return d12


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.


def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print("Reading data")
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print("Extracting features from training data")
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    DATA = fvectors_train_full.shape
    NOISE_DATA = round(DATA[0] / 2)

    # Add gaussion noise to some of train data
    mean = np.mean(fvectors_train_full)
    noisy_fvectors_train = fvectors_train_full[NOISE_DATA, :] + np.random.normal(-mean, 75, (NOISE_DATA, DATA[1]))
    # Setting bounds on values below 0 or above 255
    noisy_fvectors_train[noisy_fvectors_train < 0] = 0
    noisy_fvectors_train[noisy_fvectors_train > 255] = 255

    # Append noisy data to training data
    fvectors_train_full = np.append(fvectors_train_full[:DATA[0] - NOISE_DATA, :], noisy_fvectors_train, axis = 0)

    model_data = dict()
    model_data["labels_train"] = labels_train.tolist()
    model_data["bbox_size"] = bbox_size
    print("Reducing to 10 dimensions")
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    # Add word dictionary to model
    print("Adding word dictionary")
    with open("words.txt", "r") as file:
        words = file.readlines()

    # Add to dict while stripping new line
    model_data["dictionary"] = list(map(lambda x: x.strip(), words))

    model_data["fvectors_train"] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model["bbox_size"]
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """Nearest Neighbour classification

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    # Defining train data and labels
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    # Define page matrix as array
    test = np.array(page)

    # Nearest neighbour implementation
    x = np.dot(test, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance

    # Sort array by distances
    dist = np.argsort(-dist)
    # Find k nearest neighbour
    K = 10
    W = 3
    k_nearest = dist[:,:K]
    k_label = labels_train[k_nearest]

    # Create weights
    weight = W * np.flip(np.arange(K))

    # Convert char to ord value
    f = np.vectorize(lambda x: ord(x))
    label_code = f(k_label)
    # Find most frequent label ord and convert back to chr
    label = np.apply_along_axis(lambda x, y: chr(np.bincount(x, y).argmax()), 1, label_code, weight)

    return label


def correct_errors(page, labels, bboxes, model):
    """Error correction

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    # Get word dictionary from model
    dictionary = np.array(model["dictionary"])
    words = np.array([])
    w = ""
    punct = ['.', ',', '!', '?', ';']
    # Loop through characters and construct words
    for i in range(labels.size):
        # Check for punctuation
        if labels[i] in punct:
            # Add word and punctuation to array
            words = np.append(words, [check_word(w, dictionary), str (labels[i])])
            w = ""
        else:
            # Add label to string
            w += labels[i]

            if i == labels.size - 1:
                words = np.append(words, check_word(w, dictionary))
            else:
                # Find space between letters
                space = bboxes[i+1][0] - bboxes[i][2]

                # Add word when space found
                if space > 5:
                    words = np.append(words, check_word(w, dictionary))
                    w = ""
    labels = np.fromiter(''.join(words), (np.unicode, 1))
    return labels

def check_word(word, dictionary):
    """Check if word is in dictionary

    parameters:

    dictionary - array of words
    word - to check
    """
    # Convert word to lower case
    word_lower = word.lower()

    # Check if word is valid
    if word_lower in dictionary:
        return word


    LENGTH = len(word)
    # Check for word length 0 or 1
    if LENGTH < 2:
        return word

    # Find all words of length
    #correction = [x for x in dictionary if compare_word(word, x) == 1]
    diff1 = np.array([])
    diff2 = np.array([])
    for w in dictionary:
        score = compare_word(word, w)
        if score == 1:
            diff1 = np.append(diff1, w)
        if LENGTH > 4 and score == 2:
            diff2 = np.append(diff1, w)

    # Add to array in order of difference, with original word for if no correction
    correction = np.concatenate((diff1, diff2, [word]))

    if word_lower == word:
        return correction[0]
    return correction[0].capitalize()


def compare_word(word, comparing):
    """Compare two words based on their similar characters
    parameters:

    word - word to be corrected
    comparing - comparing word to be scored
    """
    LENGTH = len(word)
    # Check word length
    if LENGTH == len(comparing):
        score = 0
        # Loop through characters to compare difference
        for i in range(LENGTH):
            if word[i] != comparing[i]:
                score += 1
                # Exit function if score is too large
                if score > 2:
                    return -1
        return score
    return -1
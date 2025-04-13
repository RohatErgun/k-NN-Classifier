##  Mini Project: k-NN Classifier

This project implements a simple **k-Nearest Neighbors (k-NN)** classifier.

## Description

The program takes 3 command-line arguments:
`<k> <train-set> <test-set>`
- `k`: A positive integer (k-NN hyperparameter).
- `train-set`: CSV file with training data.
- `test-set`: CSV file with test data.

## Features

- Applies the k-NN algorithm to classify each vector in the test set.
- Outputs **accuracy** (correctly classified examples / total).
- Allows manual input of vectors for real-time classification.
- Supports any dataset in a similar format to `iris.data` (any number of dimensions and classes).

## Testing

You can test the program using:

- `iris.data` – training set
- `iris.test.data` – test set

---


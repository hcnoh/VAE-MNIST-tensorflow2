import os
import pickle

import numpy as np

import config as conf


class Mnist(object):
    def __init__(self, dataset_path=conf.DATASET_PATH):
        if not os.path.exists(
            os.path.join(dataset_path, "train_labels.pickle")
        ) or \
            not os.path.exists(
                os.path.join(dataset_path, "train_features.pickle")
            ) or \
                not os.path.exists(
                    os.path.join(dataset_path, "test_labels.pickle")
                ) or \
                    not os.path.exists(
                        os.path.join(dataset_path, "test_features.pickle")
                    ):
            train_csv_data = np.loadtxt(
                os.path.join(dataset_path, "mnist_train.csv"),
                dtype=np.float64,
                delimiter=",",
                skiprows=1
            )
            test_csv_data = np.loadtxt(
                os.path.join(dataset_path, "mnist_test.csv"),
                dtype=np.float64,
                delimiter=",",
                skiprows=1
            )

            self.train_labels = train_csv_data[:, 0] / 255 # [None,]
            self.train_features = train_csv_data[:, 1:] / 255 # [None, 784]
            self.test_labels = test_csv_data[:, 0] / 255 # [None,]
            self.test_features = test_csv_data[:, 1:] / 255 # [None, 784]

            with open(os.path.join(dataset_path, "train_labels.pickle"), "wb") as f:
                pickle.dump(self.train_labels, f)
            with open(os.path.join(dataset_path, "train_features.pickle"), "wb") as f:
                pickle.dump(self.train_features, f)
            with open(os.path.join(dataset_path, "test_labels.pickle"), "wb") as f:
                pickle.dump(self.test_labels, f)
            with open(os.path.join(dataset_path, "test_features.pickle"), "wb") as f:
                pickle.dump(self.test_features, f)
        
        else:
            with open(os.path.join(dataset_path, "train_labels.pickle"), "rb") as f:
                self.train_labels = pickle.load(f)
            with open(os.path.join(dataset_path, "train_features.pickle"), "rb") as f:
                self.train_features = pickle.load(f)
            with open(os.path.join(dataset_path, "test_labels.pickle"), "rb") as f:
                self.test_labels = pickle.load(f)
            with open(os.path.join(dataset_path, "test_features.pickle"), "rb") as f:
                self.test_features = pickle.load(f)
        
        self.num_train_sets = self.train_labels.shape[0]
        self.num_test_sets = self.test_labels.shape[0]
        self.feature_depth = self.train_features.shape[-1]
        self.feature_shape = [int(np.sqrt(self.feature_depth)), int(np.sqrt(self.feature_depth))]
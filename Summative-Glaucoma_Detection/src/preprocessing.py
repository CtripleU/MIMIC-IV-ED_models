import os
import pandas as pd
from sklearn.model_selection import train_test_split

class GlaucomaDataset:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir

    def get_train_data_paths(self):
        filepaths = []
        labels = []
        for label_name in ['glaucoma', 'normal']:
            label_dir = os.path.join(self.train_dir, label_name)
            for filename in os.listdir(label_dir):
                filepath = os.path.join(label_dir, filename)
                filepaths.append(filepath)
                labels.append(label_name)
        return filepaths, labels

    def get_test_data_paths(self):
        filepaths = []
        labels = []
        for label_name in ['glaucoma', 'normal']:
            label_dir = os.path.join(self.test_dir, label_name)
            for filename in os.listdir(label_dir):
                filepath = os.path.join(label_dir, filename)
                filepaths.append(filepath)
                labels.append(label_name)
        return filepaths, labels

    def create_dataframe(self, filepaths, labels):
        filepaths_series = pd.Series(filepaths, name='filepaths')
        labels_series = pd.Series(labels, name='labels')
        return pd.concat([filepaths_series, labels_series], axis=1)

    def split_train_data(self, val_size=0.2, random_state=42):
        filepaths, labels = self.get_train_data_paths()
        data = self.create_dataframe(filepaths, labels)
        train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state, stratify=data['labels'])
        return train_data, val_data

    def get_test_data(self):
        filepaths, labels = self.get_test_data_paths()
        return self.create_dataframe(filepaths, labels)
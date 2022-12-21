"""
Created on Thu May 13 2021

@author Name Redacted Surname Redacted
"""


import numpy as np


class EarlyStopping():

    def __init__(self, patience):
        self.patience = patience
        self.accuracy_history = np.array([-np.inf])
        self.metrics_history = [None]
        self.epochs = 0

    def stop(self, metrics):

        accuracy = metrics["FH@1"]
        assert accuracy >= 0

        self.epochs += 1
        self.metrics_history.append(metrics)
        self.accuracy_history = np.append(self.accuracy_history, accuracy)
        return True if self.epochs-np.argmax(self.accuracy_history) == self.patience else False

    def best(self):
        epochs = np.argmax(self.accuracy_history)
        metrics = self.metrics_history[epochs]
        return epochs, metrics

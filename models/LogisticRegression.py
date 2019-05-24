import torch
from torch.nn import *
from models.GeneralClassifier import GeneralClassifier


class LogisticRegressionModel(GeneralClassifier):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.linear = Linear(input_size, num_classes)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.5)
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        return self.linear(x)

import torch
from torch.nn import *
from models.GeneralClassifier import GeneralClassifier


class SimpleMLPClassifier(GeneralClassifier):
    def __init__(self, input_size, hidden_units, num_classes):
        super(SimpleMLPClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_units = hidden_units

        self.input_to_hidden = Linear(input_size, hidden_units)
        torch.nn.init.normal_(self.input_to_hidden.weight, mean=0, std=0.1)
        self.hidden_to_output = Linear(hidden_units, num_classes)
        torch.nn.init.normal_(self.hidden_to_output.weight, mean=0, std=0.1)
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        return self.hidden_to_output(torch.relu(self.input_to_hidden(x)))

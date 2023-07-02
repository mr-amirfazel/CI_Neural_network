from layers.fullyconnected import FC
from activations import get_activation
from optimizers.gradientdescent import GD
from optimizers.adam import Adam
from losses.binarycrossentropy import BinaryCrossEntropy
from losses.meansquarederror import MeanSquaredError
import matplotlib.pyplot as plt
from model import Model
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('datasets/california_houses_price/california_housing_train.csv')

scaler = StandardScaler()

X_train = scaler.fit_transform(data[data.columns[0:8]])
y_train = data[data.columns[:-1]].to_numpy().reshape(-1, 1)

layer1 = FC(8, 64, "layer1")
activation1 = get_activation("sigmoid")()
layer2 = FC(64, 64, "layer2")
activation2 = get_activation("sigmoid")()
layer3 = FC(64, 1, "layer3")
activation3 = get_activation("relu")()

learning_rate = 0.01
epochs = 100
architecture = {"layer1": layer1, "activation1": activation1, "layer3": layer3, "activation3": activation3}

layers_list = {"layer1": layer1, "layer3": layer3}

criterion = MeanSquaredError()

optimizer = GD(architecture, learning_rate)
optimizer1 = Adam(layers_list)
model1 = Model(architecture, criterion, optimizer1)

model1.train(X_train.T, y_train.T, epochs=2000, batch_size=100, shuffling=True, verbose=10)

test_data = pd.read_csv('datasets/california_houses_price/california_housing_test.csv')

X_test = scaler.fit_transform(test_data[test_data.columns[0:8]])
y_test = test_data[test_data.columns[:-1]].to_numpy().reshape(-1, 1)

y_pred = model1.predict(X_test[:, :].reshape(8, -1))

print(y_pred[-1])

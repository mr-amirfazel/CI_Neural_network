import os
from layers.fullyconnected import FC
from activations import *
from model import Model
from optimizers.gradientdescent import GD
from PIL import Image
from losses.binarycrossentropy import BinaryCrossEntropy

pics_2 = ['datasets/MNIST/2/' + file_name for file_name in os.listdir('datasets/MNIST/2/')]
pics_5 = ['datasets/MNIST/5/' + file_name for file_name in os.listdir('datasets/MNIST/5/')]

data_2 = []
data_5 = []
for pic in pics_2:
    data_2.append(np.array(Image.open(pic)))

for pic in pics_5:
    data_5.append(np.array(Image.open(pic)))

data_2 = np.array(data_2) / 255
data_5 = np.array(data_5) / 255

print(data_2.shape)
print(data_5.shape)

data_2 = data_2.reshape(1000, -1).T
data_5 = data_5.reshape(1000, -1).T

print(data_2.shape)
print(data_5.shape)

input_train = np.concatenate((data_2, data_5), axis=1)
output_train = np.concatenate((np.zeros((1, 1000)), np.ones((1, 1000))), axis=1)

output_valid = np.concatenate((np.zeros((1, 300)), np.ones((1, 300))), axis=1)
input_valid = np.concatenate((data_2, data_5), axis=1)

architecture = {
    'FC1': FC(784, 32, 'FC1'),
    'ACTIVE1': ReLU(),
    'FC2': FC(32, 16, 'FC2'),
    'ACTIVE2': ReLU(),
    'FC3': FC(16, 1, 'FC3'),
    'ACTIVE3': Sigmoid()
}

criterion = BinaryCrossEntropy()
optimizer = GD(architecture, learning_rate=0.3)
model = Model(architecture, criterion, optimizer)

model.train(input_train, output_train, 1000, batch_size=30, shuffling=False, verbose=50)

print(model.predict(input_valid))

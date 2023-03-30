from train1 import tokenize, stem, bag_of_words
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('NLP/ChatBot/con.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)  # not using appened to avoid array of arrays.
        xy.append((w, tag))
ignore_words = [",", "?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = (sorted(set(all_words)))
tags = sorted(set(tags))
# print(tags)
# print(all_words)
# print(xy)

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # print(pattern_sentence, tag)
    bag = bag_of_words(pattern_sentence, all_words)

    # print(bag)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
# print(x_train)
# print(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        # print(self.x_data)
        # print("hi.........", self.y_data)

    def __getitem__(self, index):
        # super(ChatDataset, self).__init__()
        # print(self.x_data[index])
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
# in numpy array, for len(), first convert the float into string.
input_size = len(x_train[1])
# input_size = len(all_words)
# print("hello", input_size)
# print("hi", len(str(x_train[0])))
# print("hi", len(x_train[1]))
# print("hi", len(x_train[2]))
learning_rate = 0.001
num_epochs = 1000
# print(len(str(x_train[0])))
# print(len(str(y_train[0])))
# print(len(str(x_train)))
# print(len(str(y_train)))
# print(len(tags))
# print(input_size, hidden_size, num_classes)...................


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# actual training loop
# n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        # in pytorch optimization, first step is empty the gradiant
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags

}
file = "./data.pth"  # extension for pytorch
torch.save(data, file)
print("my file", file)

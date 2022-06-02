from cgi import test
from collections import OrderedDict
from unittest import TestLoader

import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from cnn_model import CNN
import data_handler as dh

trainloader, testloader = dh.load_data('MNIST_data/')



model = CNN()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



epochs = 30
train_loss = []
test_loss = []
score_loss = []
best_score = 0.95 
print_every = 40
optimizer = optim.SGD(model.parameters(), lr=0.001)
for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e+1}/{epochs}")

    for i, (images, labels) in enumerate(iter(trainloader)):

        
        
        optimizer.zero_grad()
        
        output = model.forward(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        running_loss += loss.item()
        
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
    train_loss.append(running_loss/len(trainloader))   
      

    model.eval()

    with torch.no_grad():
        test_score_loss = 0
        running_loss = 0
        for i, (images_test, labels_test) in enumerate(iter(testloader)):

            test_output = model(images_test)
            # test_score_loss += accuracy_score(labels, test_output)
            test_preds = torch.argmax(test_output, dim=1)

            # running accuracy
            test_score_loss += accuracy_score(labels_test, test_preds)


            test_losses = criterion(test_output, labels_test)
            running_loss += test_losses.item()

        score_loss.append(test_score_loss/len(testloader))
        test_loss.append(running_loss/len(testloader))

        if score_loss[-1] > best_score:
        # save model
            torch.save(model.state_dict(), 'my_model.pth')

        # update benckmark
            best_score = score_loss[-1]

      

    model.train()  

x_epochs = list(range(epochs))
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(x_epochs, train_loss, marker='o', label='train')
plt.plot(x_epochs, test_loss, marker='o', label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_epochs, score_loss, marker='o',
         c='black', label='accuracy_score')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.axhline(best_score, c='green', ls='--',
            label=f'benchmark_score({best_score})')
plt.legend()

plt.savefig('accuracy_score_losses1.jpg')
plt.show()   
# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import warnings
warnings.filterwarnings('ignore')

import torchvision
import torchvision.models as model
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

# Perform transformation to the image
image_size = 64
resize = transforms.Resize(image_size)
crop = transforms.CenterCrop(image_size)
normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

# Read training dataset
bs_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/bs_train.h5')
eb_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/eb_train.h5')
h_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/h_train.h5')
lb_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/lb_train.h5')
lm_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/lm_train.h5')
mv_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/mv_train.h5')
s_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/s_train.h5')
sm_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/sm_train.h5')
ts_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/ts_train.h5')
ylc_train = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Train/ylc_train.h5')

# Read fake samples for each classes
bs_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/bs_gen.h5').cpu().detach()
eb_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/eb_gen.h5').cpu().detach()
h_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/h_gen.h5').cpu().detach()
lb_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/lb_gen.h5').cpu().detach()
lm_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/lm_gen.h5').cpu().detach()
mv_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/mv_gen.h5').cpu().detach()
s_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/s_gen.h5').cpu().detach()
sm_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/sm_gen.h5').cpu().detach()
ts_train_fake = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Fake Samples/ts_gen.h5').cpu().detach()

# Read testing dataset
bs_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/bs_test.h5')
eb_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/eb_test.h5')
h_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/h_test.h5')
lb_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/lb_test.h5')
lm_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/lm_test.h5')
mv_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/mv_test.h5')
s_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/s_test.h5')
sm_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/sm_test.h5')
ts_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/ts_test.h5')
ylc_test = torch.load('/content/drive/MyDrive/Colab Notebooks/CS 284 Project/Test/ylc_test.h5')

# Combine all training samples
X_subset = torch.zeros((22553,3,image_size,image_size))
X_subset[0:1808,:,:,:] = bs_train
X_subset[1808:2658,:,:,:] = eb_train
X_subset[2658:4010,:,:,:] = h_train
X_subset[4010:5633,:,:,:] = lb_train
X_subset[5633:6442,:,:,:] = lm_train
X_subset[6442:6759,:,:,:] = mv_train
X_subset[6759:8264,:,:,:] = s_train
X_subset[8264:9689,:,:,:] = sm_train
X_subset[9689:10882,:,:,:] = ts_train
X_subset[10882:15435,:,:,:] = ylc_train
X_subset[15435:15627,:,:,:] = bs_train_fake
X_subset[15627:16777,:,:,:] = eb_train_fake
X_subset[16777:17425,:,:,:] = h_train_fake
X_subset[17425:17802,:,:,:] = lb_train_fake
X_subset[17802:18993,:,:,:] = lm_train_fake
X_subset[18993:20676,:,:,:] = mv_train_fake
X_subset[20676:21171,:,:,:] = s_train_fake
X_subset[21171:21746,:,:,:] = sm_train_fake
X_subset[21746:22553,:,:,:] = ts_train_fake

# Create labels of training samples
y_subset = torch.zeros((22553,))
y_subset[0:1808] = 0
y_subset[1808:2658] = 1
y_subset[2658:4010] = 2
y_subset[4010:5633] = 3
y_subset[5633:6442] = 4
y_subset[6442:6759] = 5
y_subset[6759:8264] = 6
y_subset[8264:9689] = 7
y_subset[9689:10882] = 8
y_subset[10882:15435] = 9
y_subset[15435:15627] = 0
y_subset[15627:16777] = 1
y_subset[16777:17425] = 2
y_subset[17425:17802] = 3
y_subset[17802:18993] = 4
y_subset[18993:20676] = 5
y_subset[20676:21171] = 6
y_subset[21171:21746] = 7
y_subset[21746:22553] = 8

# Combine all testing samples
X_test = torch.zeros((2725,3,image_size,image_size))
X_test[0:319,:,:,:] = bs_test
X_test[319:469,:,:,:] = eb_test
X_test[469:708,:,:,:] = h_test
X_test[708:994,:,:,:] = lb_test
X_test[994:1137,:,:,:] = lm_test
X_test[1137:1193,:,:,:] = mv_test
X_test[1193:1459,:,:,:] = s_test
X_test[1459:1710,:,:,:] = sm_test
X_test[1710:1921,:,:,:] = ts_test
X_test[1921:2725,:,:,:] = ylc_test

# Create labels of testing samples
y_test = torch.zeros((2725,))
y_test[0:319] = 0
y_test[319:469] = 1
y_test[469:708] = 2
y_test[708:994] = 3
y_test[994:1137] = 4
y_test[1137:1193] = 5
y_test[1193:1459] = 6
y_test[1459:1710] = 7
y_test[1710:1921] = 8
y_test[1921:2725] = 9

# Randomly shuffle the dataset
idx = np.random.permutation(22553)
X_subset,y_subset = X_subset[idx], y_subset[idx]
idx = np.random.permutation(2725)
X_test,y_test = X_test[idx], y_test[idx]

# Divide dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, test_size = 0.15, random_state = 0, stratify = y_subset)

# VGG16 Classifier
vgg16 = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
vgg16.classifier[6] = nn.Linear(in_features = 4096, out_features = 10, bias = True)
vgg16 = vgg16.to(device)

# Freezing and Unfreezing Layers of VGG16
for p in vgg16.features.parameters():
  p.requires_grad = False
for p in vgg16.classifier.parameters():
  p.requires_grad = True

# ResNet50 Classifier
resNet50 = torchvision.models.resnet50(pretrained = True)

for param in resNet50.parameters():
  param.requires_grad = False

resNet50.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace = True), nn.Linear(128, 10))
resNet50.to(device)

# Choose classifier
backbone = 0
if backbone == 0:
  model = vgg16
  optimizer = optim.Adam(model.parameters(), lr = 0.0007)
  loss = nn.CrossEntropyLoss()
  epochs = 15
  batch_size = 64
else:
  model = resNet50
  optimizer = optim.Adam(model.parameters(), lr = 0.0001)
  loss = nn.CrossEntropyLoss()
  epochs = 30
  batch_size = 64

# Metrics for training and validation
train_loss = []
val_loss = []
train_acc = []
val_acc = []

for epoch in range(epochs):
  loss_train = 0
  acc_train = 0
  loss_val = 0
  acc_val = 0

  model.train()

  for i in range(0,X_train.shape[0], batch_size):
    batch_x = F.interpolate(X_train[i:i+batch_size,:,:,:], size = 224)
    batch_y = y_train[i:i+batch_size]
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()
    output_x = model(batch_x)
    pred = np.argmax(output_x.cpu().detach().numpy(),1)
    loss_x = loss(output_x, batch_y.long())

    loss_train += loss_x
    loss_x.backward()
    optimizer.step()


    print("train  " + str(epoch) + " " + str(i))

    acc_train += np.sum(np.argmax(output_x.cpu().detach().numpy(),1) == batch_y.cpu().detach().numpy().astype(int))

  train_loss.append(loss_train.cpu().detach().numpy()/int(X_train.shape[0]/batch_size))
  train_acc.append(acc_train/(i + batch_size))

  for i in range(0,X_val.shape[0], batch_size):
    batch_x_val = F.interpolate(X_val[i:i+batch_size,:,:,:], size = 224)
    batch_y_val = y_val[i:i+batch_size]
    batch_x_val = batch_x_val.to(device)
    batch_y_val = batch_y_val.to(device)

    output_x_val = model(batch_x_val)
    loss_x_val = loss(output_x_val, batch_y_val.long())

    loss_val += loss_x_val
    acc_val += np.sum(np.argmax(output_x_val.cpu().detach().numpy(),1) == batch_y_val.cpu().detach().numpy().astype(int))

  val_loss.append(loss_val.cpu().detach().numpy()/int(X_val.shape[0]/batch_size))
  val_acc.append(acc_val/(i+batch_size))
  
 ## Plot of Training and Validation Accuracies

# Plot the Accuracies
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.plot(range(epochs), train_acc, label = "Train Accuracy")
plt.plot(range(epochs), val_acc, label = "Val Accuracy")
plt.legend()

# Plot the Losses
plt.subplot(1,2,2)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.plot(range(epochs), train_loss, label = "Train Loss")
plt.plot(range(epochs), val_loss, label = "Val Loss")
plt.legend()

plt.suptitle('Accuracy and Loss at Learning Rate = 0.0007')
plt.show()

# Determine test metrics
test_pred = np.zeros((2725,))
acc_test = 0

for i in range(0,X_test.shape[0], batch_size):
  batch_x_test = F.interpolate(X_test[i:i+batch_size,:,:,:], size = 224)
  batch_y_test = y_test[i:i+batch_size]
  batch_x_test = batch_x_test.to(device)
  batch_y_test = batch_y_test.to(device)

  output_x = model(batch_x_test)
  acc_test += np.sum(np.argmax(output_x.cpu().detach().numpy(),1) == batch_y_test.cpu().detach().numpy().astype(int))
  test_pred[i:i+batch_size] = np.argmax(output_x.cpu().detach().numpy(),1)

test_acc = acc_test/(i+batch_size)
test_f1 = f1_score(y_test.cpu().detach().numpy(), test_pred, average = "macro")
test_accu = accuracy_score(y_test.cpu().detach().numpy(), test_pred)
test_precision = precision_score(y_test.cpu().detach().numpy(), test_pred, average = "macro")
test_recall = recall_score(y_test.cpu().detach().numpy(), test_pred, average = "macro")


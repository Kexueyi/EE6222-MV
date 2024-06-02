import torch

train_features = torch.load('train_features.pt')
train_labels = torch.load('train_labels.pt')

val_features = torch.load('val_features.pt')
val_labels = torch.load('val_labels.pt')

print("Dimensions of features:", train_features.shape)
print("Dimensions of labels:", train_labels.shape)
print("Dimensions of features:", val_features.shape)
print("Dimensions of labels:", val_labels.shape)
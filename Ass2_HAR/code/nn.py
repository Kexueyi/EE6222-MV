from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load features and labels
train_features = torch.load('train_features.pt')
train_labels = torch.load('train_labels.pt')

# Load validation features and labels
val_features = torch.load('val_features.pt')
val_labels = torch.load('val_labels.pt')

# Create the network
input_size = train_features.size(1)
hidden_size = 256
num_classes = torch.max(train_labels) + 1
net = Classifier(input_size, hidden_size, num_classes)

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = net.to(device)
train_features = train_features.to(device)
train_labels = train_labels.to(device)

val_features = val_features.to(device)
val_labels = val_labels.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(10):
    net.train()
    # Forward pass
    outputs = net(train_features)
    loss = criterion(outputs, train_labels)
    train_losses.append(loss.item())

    # Calculate training accuracy
    _, train_predicted = torch.max(outputs, 1)
    train_accuracy = (train_predicted == train_labels).float().mean()
    train_accuracies.append(train_accuracy.item())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    net.eval()  # Set the model to evaluation mode
    val_outputs = net(val_features)
    val_loss = criterion(val_outputs, val_labels)
    val_losses.append(val_loss.item())

    # Calculate validation accuracy
    _, val_predicted = torch.max(val_outputs, 1)
    val_accuracy = (val_predicted == val_labels).float().mean()
    val_accuracies.append(val_accuracy.item())

    print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# After training, plot the training and validation losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot the training and validation accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('pics/nn_loss_acc.png')
plt.close()

# Evaluate the model using validation set
net.eval() 
with torch.no_grad():
    val_outputs = net(val_features)
    _, predicted = torch.max(val_outputs.data, 1)
    total = val_labels.size(0)
    correct = (predicted == val_labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total}%')

val_labels_cpu = val_labels.cpu().numpy()
predicted_cpu = predicted.cpu().numpy()

# confusion matrix
conf_mat = confusion_matrix(val_labels_cpu, predicted_cpu)
plt.figure()
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('pics/nn_evalu.png')
plt.close()

# report
report = classification_report(val_labels_cpu, predicted_cpu)
with open('results/nn/report.txt', 'w') as file:
    file.write(report)
print(report)

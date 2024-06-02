import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# define the LSTM classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # last time step
        out = self.fc(lstm_out)
        return out

input_dim = 1000
hidden_dim = 128
output_dim = 6  # number of classes
batch_size = 16
num_epochs = 10


train_features = torch.load('train_features.pt')
train_labels = torch.load('train_labels.pt')
val_features = torch.load('val_features.pt')
val_labels = torch.load('val_labels.pt')

train_labels = train_labels.long()
val_labels = val_labels.long()

def check_labels(labels, num_classes):
    if not all(0 <= label < num_classes for label in labels):
        raise ValueError("Found label index out of range")

check_labels(train_labels, 6)
check_labels(val_labels, 6)


train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_features, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = LSTMClassifier(input_dim, hidden_dim, output_dim)
model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    model = nn.DataParallel(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

# evaluate
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device) 
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / len(val_loader.dataset)
    return avg_loss, accuracy


train_model(model, train_loader, criterion, optimizer, num_epochs)
val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

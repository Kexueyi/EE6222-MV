from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class EndToEndClassifier(nn.Module):
    def __init__(self, num_classes, num_frames_to_sample):
        super(EndToEndClassifier, self).__init__()
        # Load pre-trained ResNet50 for spatial feature extraction
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2])) # Remove the last classification layer
        
        last_block = list(self.resnet.children())[-1]
        resnet_feature_size = last_block[-1].bn3.num_features

        # Normalization transform for ResNet
        self.normalize = transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.1, 0.09, 0.08])
        self.num_frames_to_sample = num_frames_to_sample

        # Classifier's fully connected layers
        self.fc1 = nn.Linear(resnet_feature_size, 256) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: [batch_size, num_frames, height, width, channels]
        print(x.size())
        batch_size, num_frames, _, h, w, c = x.size()

        # Process each frame and store features
        frame_features_list = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            frame = self.preprocess_frame(frame)
            frame_features = self.resnet(frame)
            frame_features = frame_features.view(frame_features.size(0), -1)  # Flatten
            frame_features_list.append(frame_features)

        # Late Fusion: Averaging features from all frames
        avg_features = torch.stack(frame_features_list).mean(dim=0)

        # Pass averaged features through fully connected layers
        out = self.fc1(avg_features)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    
    def preprocess_frame(self, frame):
        # Convert frame to PIL image and apply transformations
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToPILImage()(frame)
        frame = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            self.normalize
        ])(frame)

        # Add an extra batch dimension and send to the same device as the model
        frame = frame.unsqueeze(0).to(next(self.parameters()).device)
        return frame


class VideoDataset(Dataset):
    def __init__(self, file_path, base_dir, num_frames_to_sample):
        super(VideoDataset, self).__init__()
        self.data = []
        self.num_frames_to_sample = num_frames_to_sample
        self.normalize = transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.1, 0.09, 0.08])

        with open(file_path, 'r') as file:
            for line in file:
                index, label, video_rel_path = line.strip().split()
                video_path = os.path.join(base_dir, video_rel_path)
                self.data.append((video_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.uniform_sampling(video_path)
        return torch.stack(frames), label

    def uniform_sampling(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampling_interval = max(1, total_frames // self.num_frames_to_sample)

        frames = []
        for i in range(0, total_frames, sampling_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = self.preprocess_frame(frame)
                frames.append(frame)
            if len(frames) == self.num_frames_to_sample:
                break

        cap.release()
        return frames

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToPILImage()(frame)
        frame = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            self.normalize
        ])(frame).unsqueeze(0)
        frame = frame.to(device)
        return frame

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        # Validation phase
        model.eval()  
        val_loss = 0.0
        val_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_corrects.double() / total
        print(f'Validation - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}')


    print('Finished Training')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 6
num_frames_to_sample = 15

# Prepare Dataset and DataLoader
train_dataset = VideoDataset('data/train.txt', 'data/train', num_frames_to_sample)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

# Validation Dataset and DataLoader
val_dataset = VideoDataset('data/validate.txt', 'data/validate', num_frames_to_sample)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  

# Model, loss function, optimizer
model = EndToEndClassifier(num_classes, num_frames_to_sample).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import cv2
import torch.nn as nn
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet model with updated syntax
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

model = model.to(device)
model.eval()

# Normalization transform
normalize = transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.1, 0.09, 0.08])

def extract_features(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transforms.ToPILImage()(frame)
    frame = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        normalize
    ])(frame).unsqueeze(0)

    frame = frame.to(device)

    with torch.no_grad():
        features = model(frame)

    return features

def late_fusion_average(features):
    return torch.mean(features, dim=0).squeeze(0)

def uniform_sampling(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling_interval = max(1, total_frames // num_frames)

    frames = []
    for i in range(0, total_frames, sampling_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            if len(frames) == num_frames:
                break

    cap.release()
    return frames


num_frames_to_sample = 15  # number of frames to sample from each video

def process_videos(file_path, base_dir, desc):
    features = []
    labels = []

    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=desc):
            index, label, video_rel_path = line.strip().split()
            video_path = os.path.join(base_dir, video_rel_path) 
                
            sampled_frames = uniform_sampling(video_path, num_frames_to_sample)

            video_features = [extract_features(frame) for frame in sampled_frames]
            averaged_features = late_fusion_average(torch.stack(video_features))

            features.append(averaged_features)
            labels.append(int(label))

    return torch.stack(features), torch.tensor(labels)

train_features, train_labels = process_videos('data/train.txt', 'data/train', "Processing Train Videos")
val_features, val_labels = process_videos('data/validate.txt', 'data/validate', "Processing Validation Videos")


torch.save(train_features, 'train_features.pt')
torch.save(train_labels, 'train_labels.pt')
torch.save(val_features, 'val_features.pt')
torch.save(val_labels, 'val_labels.pt')

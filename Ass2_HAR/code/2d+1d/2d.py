import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
import torchvision.transforms as transforms
import cv2
import os
from tqdm import tqdm
from PIL import Image

class R2Plus1DFeatureExtractor(nn.Module):
    def __init__(self):
        super(R2Plus1DFeatureExtractor, self).__init__()
        original_model = r2plus1d_18(pretrained=False)
        # remove the last classification layer in order to extract features
        self.feature_extractor = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        print(x.shape)
        x = self.feature_extractor(x)
        # late fusion
        x = torch.mean(x, dim=2)  # average features across time
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = R2Plus1DFeatureExtractor().to(device)
feature_extractor.eval()

normalize = transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.1, 0.09, 0.08])
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    normalize
])

from PIL import Image

def extract_features(frames):
    processed_frames = [preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames]
    video_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)  #  (1, T, C, H, W)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  #  (1, C, T, H, W)
    with torch.no_grad():
        features = feature_extractor(video_tensor)
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

def process_videos(file_path, base_dir, desc):
    features = []
    labels = []

    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=desc):
            index, label, video_rel_path = line.strip().split()
            video_path = os.path.join(base_dir, video_rel_path)

            sampled_frames = uniform_sampling(video_path, num_frames_to_sample)
            video_features = extract_features(sampled_frames)
            averaged_features = late_fusion_average(video_features)

            features.append(averaged_features)
            labels.append(int(label))

    return torch.stack(features), torch.tensor(labels)

num_frames_to_sample = 15  # number of frames to sample from each video
num_classes = 6 

# Process videos
train_features, train_labels = process_videos('data/train.txt', 'data/train', "Processing Train Videos")
val_features, val_labels = process_videos('data/validate.txt', 'data/validate', "Processing Validation Videos")

# Save features and labels
torch.save(train_features, 'train_features.pt')
torch.save(train_labels, 'train_labels.pt')
torch.save(val_features, 'val_features.pt')
torch.save(val_labels, 'val_labels.pt')

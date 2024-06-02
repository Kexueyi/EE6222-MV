import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video file
video_path = '/media/user/volume2/students/s123md214_01/mv/data/train/Jump/Jump_11_1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Function to sample frames uniformly
def uniform_sampling(video, sample_rate):
    frames = []
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frames.append(frame)
        count += 1
    return frames

# Function to sample frames randomly
def random_sampling(video, num_samples):
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_frames = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    sample_indices = np.random.choice(max_frames, num_samples, replace=False)
    sample_indices.sort()
    frames = []
    for idx in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    return frames

# Set the sample rate and number of samples
sample_rate = 5
num_samples = 10

# Reset video capture for uniform sampling
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
uniform_frames = uniform_sampling(cap, sample_rate)

# Reset video capture for random sampling
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
random_frames = random_sampling(cap, num_samples)

# Closing video file
cap.release()

# Adjusting the visualization to handle the case when the number of sampled frames is less than expected

# Determine the minimum number of frames obtained from both methods
min_num_frames = min(len(uniform_frames), len(random_frames))

# Update the subplot dimensions to match the minimum number of frames
fig, axs = plt.subplots(2, min_num_frames, figsize=(15, 6))

for i in range(min_num_frames):
    axs[0, i].imshow(cv2.cvtColor(uniform_frames[i], cv2.COLOR_BGR2RGB))
    axs[0, i].set_title(f"Uniform {i+1}")
    axs[0, i].axis('off')

    axs[1, i].imshow(cv2.cvtColor(random_frames[i], cv2.COLOR_BGR2RGB))
    axs[1, i].set_title(f"Random {i+1}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()


plt.savefig('pics/sample.png')
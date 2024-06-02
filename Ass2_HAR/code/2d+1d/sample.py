import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

# Load the video
video_path = '/mnt/data/Jump_11_1.mp4'
video = VideoFileClip(video_path)

# Parameters
num_frames = 10

# Total number of frames in the video
total_frames = int(video.fps * video.duration)

# Uniform sampling
uniform_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
uniform_frames = [video.get_frame(i / video.fps) for i in uniform_indices]

# Random sampling
random_indices = np.sort(np.random.choice(range(total_frames), num_frames, replace=False))
random_frames = [video.get_frame(i / video.fps) for i in random_indices]

# Plotting
fig, axs = plt.subplots(3, num_frames, figsize=(20, 6))  # Create a grid of subplots

# Set titles for rows
row_titles = ['Original Video', 'Uniform Sampling', 'Random Sampling']

# Plot the original video frames
for i in range(num_frames):
    axs[0, i].imshow(video.get_frame(i / video.fps))
    axs[0, i].set_title(f"Frame {i}")
    axs[0, i].axis('off')

# Plot the uniformly sampled frames
for i, idx in enumerate(uniform_indices):
    axs[1, i].imshow(uniform_frames[i])
    axs[1, i].set_title(f"Frame {idx}")
    axs[1, i].axis('off')

# Plot the randomly sampled frames
for i, idx in enumerate(random_indices):
    axs[2, i].imshow(random_frames[i])
    axs[2, i].set_title(f"Frame {idx}")
    axs[2, i].axis('off')

# Set the row titles
for ax, row_title in zip(axs[:,0], row_titles):
    ax.set_ylabel(row_title, rotation=90, size='large')

plt.tight_layout()
plt.show()

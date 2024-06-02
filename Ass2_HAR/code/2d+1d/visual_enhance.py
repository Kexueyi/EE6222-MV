import cv2
import matplotlib.pyplot as plt

# Define the path to the video file
video_path = 'data/train/Sit/Sit_2_4.mp4'

# Define the image enhancement function
def enhance_image_histogram_equalization(image):
    # Convert to grayscale for histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    enhanced = cv2.equalizeHist(gray)
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr

def extract_and_compare_frames_custom(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    comparison_images = []

    for idx in frame_indices:
        if idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                enhanced_frame = enhance_image_histogram_equalization(frame)
                comparison_images.append((frame, enhanced_frame))
        else:
            print(f"Frame index {idx} is out of range. Total frames: {total_frames}")

    cap.release()
    return comparison_images

# Update frame indices as needed
frame_indices = [10, 33, 60]  # Adjust these indices as necessary

# Extract and compare frames
comparison_images_custom = extract_and_compare_frames_custom(video_path, frame_indices)

# Check if we have valid comparison images
if comparison_images_custom:
    # Plotting the images for comparison
    fig, axs = plt.subplots(len(comparison_images_custom), 2, figsize=(10, len(comparison_images_custom) * 5))

    for i, (original, enhanced) in enumerate(comparison_images_custom):
        axs[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title(f"Original Frame at index {frame_indices[i]}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axs[i, 1].set_title(f"Enhanced Frame at index {frame_indices[i]}")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('pics/enhance_sit_2_4.png')

else:
    print("No valid frames were found for comparison.")




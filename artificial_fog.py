import cv2
import numpy as np

def add_heavy_fog_with_gray_overlay(frame, fog_intensity=5, overlay_intensity=0.6):
    """
    Adds a heavy gradient fog effect combined with a dense gray overlay to the frame.
    
    Parameters:
    - frame: The original video frame.
    - fog_intensity: Base intensity of the gradient fog effect (higher values increase fog).
    - overlay_intensity: Intensity of the uniform gray overlay (0 to 1).

    Returns:
    - foggy_frame: Frame with combined heavy fog effects applied.
    """
    height, width, _ = frame.shape

    # Create a gradient mask for fog intensity
    gradient = np.linspace(0, 1, height).reshape(height, 1).astype(np.float32)
    gradient = np.repeat(gradient, width, axis=1)

    # Exponentially increase fog density for the background
    gradient = np.power(gradient, 2)  # Squaring for stronger fall-off effect
    fog_mask = (gradient * fog_intensity * 255).clip(0, 255).astype(np.uint8)
    fog_mask = cv2.merge([fog_mask, fog_mask, fog_mask])  # Convert to 3 channels

    # Create a uniform gray overlay
    gray_overlay = np.full((height, width, 3), 220, dtype=np.uint8)  # Lighter gray (220, 220, 220)

    # Blend the original frame with the gray overlay
    overlay_frame = cv2.addWeighted(frame, 1 - overlay_intensity, gray_overlay, overlay_intensity, 0)

    # Blend the overlay frame with the gradient fog mask
    foggy_frame = cv2.addWeighted(overlay_frame, 1 - fog_intensity, fog_mask, fog_intensity, 0)

    # Further reduce contrast and brightness for more realism
    alpha = 0.5  # Stronger reduction in contrast
    beta = -15   # Dim brightness further
    foggy_frame = cv2.convertScaleAbs(foggy_frame, alpha=alpha, beta=beta)

    return foggy_frame

input_video_path = "sunny_road.mp4"
output_video_path = "heavy_foggy_road.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Initialize the video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Processing video...")

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Add fog effect to the frame
    foggy_frame = add_heavy_fog_with_gray_overlay(frame, fog_intensity=0.6)
    
    # Write the modified frame to the output video
    out.write(foggy_frame)
    
    # Display the frame (optional, for debugging)
    # cv2.imshow('Foggy Frame', foggy_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Foggy video saved to: {output_video_path}")

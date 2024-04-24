import mediapipe as mp
import cv2
import numpy as np
import os

def draw_landmarks_on_image(rgb_image, pose_landmarks):
    annotated_image = np.copy(rgb_image)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    return annotated_image

# Setup MediaPipe Pose
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)

# Setup input and output directories
input_dir = '/Users/divyatadimeti/Desktop/golf/data/videos_160'
output_dir = '/Users/divyatadimeti/Desktop/golf/data/'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# List of specific files to process
files_to_process = ['1108.mp4', '1376.mp4']

# Process each specified video file
for filename in files_to_process:
    video_path = os.path.join(input_dir, filename)
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate of the input video to use for the output video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    output_video_path = os.path.join(output_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            annotated_image = draw_landmarks_on_image(rgb_frame, results.pose_landmarks)
        else:
            annotated_image = rgb_frame

        final_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        out.write(final_image)

    # Release resources for this video
    cap.release()
    out.release()

cv2.destroyAllWindows()
pose.close()

print("Processing complete. Specified videos have been processed and saved.")


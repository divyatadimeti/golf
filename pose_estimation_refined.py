import mediapipe as mp
import cv2
import numpy as np
import os

def draw_landmarks_on_image(rgb_image, pose_landmarks, joints_of_interest):
    annotated_image = np.copy(rgb_image)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if pose_landmarks:
        # Drawing only the specified joints and the connections between them
        for connection in mp_pose.POSE_CONNECTIONS:
            if connection[0] in joints_of_interest and connection[1] in joints_of_interest:
                start_point = pose_landmarks.landmark[connection[0]]
                end_point = pose_landmarks.landmark[connection[1]]
                cv2.line(annotated_image, 
                         (int(start_point.x * rgb_image.shape[1]), int(start_point.y * rgb_image.shape[0])),
                         (int(end_point.x * rgb_image.shape[1]), int(end_point.y * rgb_image.shape[0])),
                         (0, 0, 255), 2)
                cv2.circle(annotated_image, 
                           (int(start_point.x * rgb_image.shape[1]), int(start_point.y * rgb_image.shape[0])),
                           2, (255, 0, 0), -1)
                cv2.circle(annotated_image, 
                           (int(end_point.x * rgb_image.shape[1]), int(end_point.y * rgb_image.shape[0])),
                           2, (255, 0, 0), -1)

    return annotated_image

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)

# Define joints of interest
joints_of_interest = [23, 24, 25, 26, 27, 28]  # Example: hips, shoulders, elbows, wrists

# Setup input and output directories
input_dir = '/home/dt2760/golf/data/videos_160'
output_dir = '/home/dt2760/golf/data/videos_160_pose_estimation'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Process each video file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):  # Check for video files
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
                annotated_image = draw_landmarks_on_image(rgb_frame, results.pose_landmarks, joints_of_interest)
            else:
                annotated_image = rgb_frame

            final_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            out.write(final_image)

        # Release resources for this video
        cap.release()
        out.release()

cv2.destroyAllWindows()
pose.close()

print("Processing complete. All videos have been processed and saved.")
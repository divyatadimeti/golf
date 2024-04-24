!pip install -q mediapipe
!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np

def draw_landmarks_on_image(rgb_image, pose_landmarks):
    annotated_image = np.copy(rgb_image)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Draw the pose landmarks.
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    return annotated_image

# Setup mediapipe instance
with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:

    # Initialize video reader and writer
    video_path = '987.mp4'
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the frame.
        annotated_image = draw_landmarks_on_image(rgb_frame, results.pose_landmarks)

        # Convert RGB back to BGR for final output
        final_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        out.write(final_image)

        # Optionally display the frame (disabled by default to speed up processing)
        # cv2.imshow('Pose Estimation Video', final_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
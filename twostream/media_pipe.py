import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Pose model.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# Specify the joints you are interested in.
joints_of_interest = [0, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]

def extract_joints_from_frame(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        joint_info = {f'joint_{joint}_{axis}': getattr(landmarks[joint], axis)
                      for joint in joints_of_interest
                      for axis in ['x', 'y', 'z', 'visibility']}
        return joint_info, results
    return {}, None

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    joint_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        joint_info, results = extract_joints_from_frame(frame)
        if joint_info:
            joint_info['frame'] = frame_count
            joint_info['video_id'] = os.path.basename(video_path)
            joint_data.append(joint_info)

        #if results:
            # Optional visualization
         #   mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          #  cv2.imshow('Pose', frame)  # Changed from cv2_imshow to cv2.imshow
           # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return joint_data

def process_directory(directory_path):
    joint_data_all_videos = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp4"):  # Assuming all videos are in .mp4 format
            video_path = os.path.join(directory_path, filename)
            joint_data = process_video(video_path)
            joint_data_all_videos.extend(joint_data)
    return joint_data_all_videos

# Specify the directory containing golf swing videos
directory_path = 'data/videos_160'  # Update this path as necessary
joint_data_all_videos = process_directory(directory_path)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(joint_data_all_videos)
print(df.head())

# Optionally save the DataFrame to a CSV file
output_csv_path = 'pose_estimation_data.csv'  # Update this path as necessary
df.to_csv(output_csv_path, index=False)


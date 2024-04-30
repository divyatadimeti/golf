import os
import subprocess

def check_moov_atom(directory):
    # List all MP4 files in the directory
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            filepath = os.path.join(directory, file)
            # Use ffmpeg to probe the file
            result = subprocess.run(['ffmpeg', '-i', filepath, '-hide_banner'],
                                    text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # Check if 'moov atom not found' is in the output
            if 'moov atom not found' in result.stderr:
                print(f"Moov atom not found in file: {file}")

# Example usage: adjust the path to the directory containing your MP4 files
directory_path = '/home/dt2760/golf/data/videos_160_pose_estimation'
check_moov_atom(directory_path)


# Golf Swing Phase Detection

This repository extends the original GolfDB with modifications and human pose estimation to enhance golf swing phase detection. It includes experiments for baseline models, dimming, and pose estimation.
Much of the dataloading, training, and model code is adapted from McNally et. al's GolfDB repository. 

McNally, W., Vats, K., Pinto, T., Dulhanty, C., McPhee, J., & Wong, A. (2019). GolfDB: A Video Database for Golf Swing Sequencing. In *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops* (pp. pages). IEEE.



## Repository Structure

- **experiments/**: Contains scripts for dataloader configurations, training, and evaluation for different experiments (baseline, dimming, pose estimation).
- **twostream/**: In-progress code to adapt SwingNet to incorporate human pose information as an additional input feature. 
- **data/**: Contains GolfDB pickle files for training and validation splits.

## Working with the repo

### Prerequisites

Ensure you have Python and the necessary libraries from requiements.txt. 

### Setup

1. **Clone the repository:**
2. **Download the golf swing videos from GolfDB's official repository.**
3. **Generate Data Splits:** by running generate_splits.py

### Running Experiments

#### Pose Estimation

1. **Generate Pose Estimation Data:** by running pose_estimation_refined.py. Specify key points for the pose estimation skeleton as needed (see MediaPipe Pose Landmark website). This script will generate a new directory with the processed videos.
2. **Train and Evaluate the Pose Estimation Model:** using the scripts within experiments/pose_estimation

#### Dimming Experiments

1. **Run Training and Evaluation for Dimming:** using the scripts within experiments/dimming. Adjust the pixel dimming specifications in the dataloader as required.

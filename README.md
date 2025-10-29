Here is the updated README.

-----

# Pedometer Ground Truth Pipeline

## 1\. Description

This Python library analyzes video files of a person walking to extract ground truth data for pedometer algorithm validation. It identifies individual steps and tracks the 3D movement of specified joints (like a wrist or ankle) over time.

The primary purpose is to create a reliable dataset from a video, which can then be used to verify the accuracy of data collected from a wearable sensor (like a Fitbit or another accelerometer-based device).

-----

## 2\. Requirements

To use this library, you must have Python 3 installed, along with the following packages. You can install them using `pip`:

```bash
pip install opencv-python mediapipe numpy scipy
```

  * **OpenCV:** Used for reading, processing, and displaying video files.
  * **MediaPipe:** Used for detecting the human pose (skeleton).
  * **NumPy:** Used for numerical calculations on coordinate data.
  * **SciPy:** Used for signal processing to identify steps.

-----

## 3\. How to Use

### 3.1. Video Recording Protocol

For the pipeline to work correctly, your videos **must** follow a specific protocol:

1.  Start recording with the subject clearly visible.
2.  The subject must perform a **clear clap** with both hands.
3.  The subject then performs the walking motion to be analyzed.
4.  At the end of the motion, the subject must perform another **clear clap**.
5.  Stop the recording.

These two claps are used as synchronization markers. You will manually select the exact frame for each clap in the interactive viewer.

### 3.2. Project Structure

Organize your files in a single folder as follows:

```
my_project/
├── ped_pipeline.py         # The library file
├── test_analysis.py        # Your script to run the analysis
└── videos/
    └── subject_01.mp4      # Your video file(s)
```

### 3.3. Running an Analysis (Example)

Run the `pipeline_test.py` The analysis is now a **three-step process**: preprocessing, interactive selection, and final analysis.

-----

## 4\. How It Works: The Two-Pass System

The library uses a two-pass system to separate slow processing from fast interaction.

### Pass 1: Non-Interactive Processing

When you call `run_preprocessing()`, the library performs the following steps:

1.  **Video Loading:** The script loads the video file, resizes it, and determines the frame rate (FPS).
2.  **Skeleton Tracking:** It processes the *entire* video frame by frame (this is the slowest part). For each frame, it runs MediaPipe Pose to find the skeleton.
3.  **Data Caching:** It saves all 33 body joint locations for every frame into memory. It saves both the 3D "world" coordinates (for analysis) and the 2D "screen" coordinates (for drawing).

### Pass 2: Interactive Clap Selection

When you call `find_sync_events()`, a new OpenCV window opens. This window allows you to find the exact frames for the start and end claps.

  * This pass is **fast** because it simply reads the video frames and overlays the skeleton data that was already computed in Pass 1.
  * You use keyboard commands to navigate:
      * **'d'**: Next Frame
      * **'a'**: Previous Frame
      * **'w'**: Fast-Forward 10 Frames
      * **'s'**: Rewind 10 Frames
      * **Spacebar**: Play/Pause
      * **'c'**: Mark the current frame as a "clap"
      * **'q'**: Quit the interactive session

The script stores the frame number of every press of the 'c' key. The first one is used as the start marker and the last one is used as the end marker.

### Final Analysis

When you call `analyze()`, the script performs the final computations only on the data *between* your selected start and end frames:

1.  **Step Counting:** It analyzes the 3D vertical (Y-coordinate) position of the left and right ankles. A "step" is detected at the lowest point of each ankle's movement cycle (representing the foot striking the ground).
2.  **Data Extraction:** The script gathers the 3D (x, y, z) coordinates for any joints you specified (e.g., "left\_wrist") for every single frame between the start and end claps.

-----

## 5\. Output Data

The `analyze()` method returns two Python objects:

### 1\. `step_report` (A Dictionary)

This object contains the summary of the step count.

  * `start_time_sec`: The timestamp (in seconds) of the first marked clap.
  * `end_time_sec`: The timestamp (in seconds) of the final marked clap.
  * `total_steps`: The total number of steps counted between the claps.
  * `step_timestamps`: A list containing the exact timestamp (in seconds) for every detected step.

### 2\. `joint_data` (A Dictionary)

This object contains the raw time-series data for the joints you requested.

  * The **keys** of the dictionary are the joint names (e.g., `joint_data['left_wrist']`).
  * The **value** for each key is a list of tuples.
  * Each tuple contains `(timestamp, x, y, z)`, representing the precise 3D coordinates of that joint at that moment in time. This data can be used to calculate velocity or acceleration.
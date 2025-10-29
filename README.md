# Pedometer Ground Truth Pipeline

# IMPORTANT!
Mediapipe only supports python 3.12 or lower!

## 1\. Description

This Python library analyzes video files of a person walking to extract ground truth data for pedometer algorithm validation. It identifies individual steps and tracks the 3D movement of specified joints (like a wrist or ankle) over time.

The primary purpose is to create a reliable dataset from a video, which can then be used to verify the accuracy of data collected from a wearable sensor (like a Fitbit or another accelerometer-based device).

-----

## 2\. Requirements

To use this library, you must have Python 3 installed, along with the following packages. You can install them using `pip`:

```bash
pip install opencv-python mediapipe numpy scipy
```

  * **OpenCV:** Used for reading and processing video files.
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

These two claps (at the beginning and end) are used as synchronization markers. The pipeline automatically finds them to determine the exact start and end time of the analysis.

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

Create a new Python file (e.g., `test_analysis.py`) to import and use the library.

```python
# test_analysis.py

import ped_pipeline

def run_full_analysis():
    # 1. Find all video files in the 'videos' directory
    video_files = ped_pipeline.prep_videos("./videos")

    if not video_files:
        print("No videos found in the './videos' folder.")
        return

    # 2. Process each video
    for video_file in video_files:
        print(f"\n--- Analyzing file: {video_file} ---")
        
        try:
            # 3. Create a pipeline object for the video
            pipeline = ped_pipeline.PedometerGroundTruth(video_file)
            
            # 4. Run preprocessing: finds the skeleton for the whole video
            pipeline.run_preprocessing()
            
            # 5. Find sync events: locates the start and end claps
            # You may need to adjust the clap_threshold (smaller = hands closer)
            pipeline.find_sync_events(clap_threshold=0.08)
            
            # 6. Run the main analysis
            # Specify the joints you want to get 3D data for
            ax6_locations = ["left_wrist", "right_ankle"]
            
            step_report, joint_data = pipeline.analyze(ax6_locations)
            
            # 7. Print the results
            print("--- Analysis Complete ---")
            print(f"Analysis Start Time: {step_report['start_time_sec']:.2f}s")
            print(f"Analysis End Time: {step_report['end_time_sec']:.2f}s")
            print(f"Total Steps Detected: {step_report['total_steps']}")
            
            # You can now save 'step_report' and 'joint_data'
            # to a .csv or .json file for further use.

        except Exception as e:
            print(f"Failed to process {video_file}. Error: {e}")

# This makes the script runnable
if __name__ == "__main__":
    run_full_analysis()
```

-----

## 4\. The Analysis Pipeline (How It Works)

The library follows a multi-step process to generate the data:

1.  **Video Loading:** The script first loads the video file, resizes it for faster processing, and determines the video's frame rate (FPS).

2.  **Skeleton Tracking:** It processes the entire video frame by frame. For each frame, it uses MediaPipe Pose to place a 3D skeleton on the person. The 3D "world coordinates" (in meters, relative to the person's hips) of all 33 body joints are stored.

3.  **Sync Event Detection:** The script analyzes the stored 3D coordinate data. It calculates the distance between the left and right wrists for every frame. A "clap" is identified when this distance briefly drops below a specific threshold (e.g., 8 centimeters). The first and last clap events found are set as the `start_frame` and `end_frame`.

4.  **Step Counting:** The analysis is now limited to the frames *between* the two claps. The script tracks the vertical (Y-coordinate) position of the left and right ankles. A "step" is detected at the lowest point of each ankle's movement cycle (representing the foot striking the ground).

5.  **Data Extraction:** Finally, the script gathers the 3D (x, y, z) coordinates for any joints you specified (e.g., "left\_wrist") for every single frame between the start and end claps.

-----

## 5\. Output Data

The `analyze()` method returns two Python objects:

### 1\. `step_report` (A Dictionary)

This object contains the summary of the step count.

  * `start_time_sec`: The timestamp (in seconds) of the first clap.
  * `end_time_sec`: The timestamp (in seconds) of the final clap.
  * `total_steps`: The total number of steps counted between the claps.
  * `step_timestamps`: A list containing the exact timestamp (in seconds) for every detected step.

### 2\. `joint_data` (A Dictionary)

This object contains the raw time-series data for the joints you requested.

  * The **keys** of the dictionary are the joint names (e.g., `joint_data['left_wrist']`).
  * The **value** for each key is a list of tuples.
  * Each tuple contains `(timestamp, x, y, z)`, representing the precise 3D coordinates of that joint at that moment in time. This data can be used to calculate velocity or acceleration.
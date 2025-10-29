"""
Media pipe to get from a video of a person walking to a list of steps and corresponding timestamps
for the pipeline to process the videos properly the subject needs to do the following:
    - clap
    - t-pose towards the camera (?)
    - walk
    - clap
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

import glob

mp_pose = mp.solutions.pose

# MediaPipe Landmark names (for clap detection)
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST

# MediaPipe Landmark names (for step detection)
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE

# A map from your string names to MediaPipe's objects
JOINT_MAP = {
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    # Add other joints as needed
}

def process_video_and_detect_skeleton(video_path, resize_height=720):
    """
    Combines prep_videos and detect_skeleton.
    
    Reads a video, resizes it, and runs MediaPipe Pose estimation
    on every frame.
    
    Returns:
        list: A list of all pose landmark results.
        float: The frames-per-second (FPS) of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get original dimensions for resizing
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new width to maintain aspect ratio
    scale = resize_height / orig_height
    resize_width = int(orig_width * scale)

    all_landmarks = []
    
    # Initialize MediaPipe Pose
    # Using 'world_landmarks' gives you 3D coordinates in meters,
    # which is MUCH better for analysis than 2D screen pixels.
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break # End of video

            # 1. Resize (your prep_videos step)
            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            
            # 2. Convert BGR (OpenCV) to RGB (MediaPipe)
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # 3. Process the frame to find the skeleton
            results = pose.process(image_rgb)
            
            # 4. Store the landmarks
            # We store 'pose_world_landmarks'
            if results.pose_world_landmarks:
                all_landmarks.append(results.pose_world_landmarks)
            else:
                # Append None if no person was detected in the frame
                all_landmarks.append(None)

    cap.release()
    print(f"Processed video, extracted {len(all_landmarks)} frames.")
    return all_landmarks, fps

def find_clap_frames(all_landmarks, threshold=0.1):
    """
    Finds the frame numbers where a clap occurs.
    A clap is defined as the distance between wrists being below a threshold.
    
    Args:
        all_landmarks (list): List of MediaPipe landmarks from process_video.
        threshold (float): Max distance (in meters) between wrists to count as a clap.
                           (You'll need to tune this.)
                           
    Returns:
        tuple: (start_frame, end_frame) or (None, None) if not found.
    """
    wrist_distances = []
    for frame_landmarks in all_landmarks:
        if frame_landmarks:
            # Get the 3D coordinates
            left_wrist_pos = frame_landmarks.landmark[LEFT_WRIST]
            right_wrist_pos = frame_landmarks.landmark[RIGHT_WRIST]
            
            # Calculate 3D Euclidean distance
            distance = np.linalg.norm([
                left_wrist_pos.x - right_wrist_pos.x,
                left_wrist_pos.y - right_wrist_pos.y,
                left_wrist_pos.z - right_wrist_pos.z
            ])
            wrist_distances.append(distance)
        else:
            wrist_distances.append(np.inf) # No person, so distance is "infinite"

    # Find where the distance drops below the threshold
    is_clapping = np.array(wrist_distances) < threshold
    
    clap_events = []
    # Detect the *start* of a clap (when False -> True)
    for i in range(1, len(is_clapping)):
        if is_clapping[i] and not is_clapping[i-1]:
            clap_events.append(i) # 'i' is the frame number

    if len(clap_events) < 2:
        print("Error: Did not find at least two clap events.")
        return None, None
        
    # Your `find_video_start_and_end` logic is essentially this:
    start_frame = clap_events[0]
    end_frame = clap_events[-1]
    
    print(f"Found start clap at frame {start_frame}, end clap at frame {end_frame}.")
    return start_frame, end_frame

def detect_steps(all_landmarks, start_frame, end_frame, fps):
    """
    Counts steps by finding the local minima in the ankle's Y-coordinate.
    
    Args:
        all_landmarks (list): Full landmark data.
        start_frame (int): Frame number to start analysis.
        end_frame (int): Frame number to end analysis.
        fps (float): Video FPS to convert frames to timestamps.
        
    Returns:
        list: A list of timestamps (in seconds) for each detected step.
    """
    # 1. Extract the Y-coordinates for both ankles in the trimmed segment
    analysis_landmarks = all_landmarks[start_frame:end_frame]
    
    left_ankle_y = []
    right_ankle_y = []
    
    for frame_landmarks in analysis_landmarks:
        if frame_landmarks:
            left_ankle_y.append(frame_landmarks.landmark[LEFT_ANKLE].y)
            right_ankle_y.append(frame_landmarks.landmark[RIGHT_ANKLE].y)
        else:
            # Handle missing data (e.g., person out of frame)
            left_ankle_y.append(np.nan)
            right_ankle_y.append(np.nan)
            
    # Simple interpolation to fill small gaps (optional but helpful)
    left_ankle_y = np.interp(np.arange(len(left_ankle_y)), 
                             np.flatnonzero(np.isfinite(left_ankle_y)), 
                             np.array(left_ankle_y)[np.isfinite(left_ankle_y)])
    right_ankle_y = np.interp(np.arange(len(right_ankle_y)), 
                              np.flatnonzero(np.isfinite(right_ankle_y)), 
                              np.array(right_ankle_y)[np.isfinite(right_ankle_y)])

    # 2. Find peaks (local minima)
    # find_peaks finds maxima, so we find peaks on the *negative* Y-data.
    # 'prominence' is key: it's how much a peak stands out. 
    # It filters out small jitters. You *must* tune this value.
    # 'distance' sets the minimum frames between steps (e.g., 0.25 sec * fps)
    min_step_distance = int(fps * 0.25) # Minimum 1/4 second between steps

    left_step_frames, _ = find_peaks(-left_ankle_y, prominence=0.01, distance=min_step_distance)
    right_step_frames, _ = find_peaks(-right_ankle_y, prominence=0.01, distance=min_step_distance)
    
    # 3. Combine and sort all detected steps
    all_step_frames = np.concatenate((left_step_frames, right_step_frames))
    all_step_frames.sort()
    
    # 4. Convert frame numbers to timestamps
    # Remember to add the 'start_frame' offset!
    step_timestamps = (all_step_frames + start_frame) / fps
    
    print(f"Detected {len(step_timestamps)} steps.")
    return list(step_timestamps)

def locate_ax6(all_landmarks, ax6_locations, start_frame, end_frame, fps):
    """
    Extracts the full 3D time-series data for specified joints.
    This is the "ground truth" acceleration/position data.
    
    Args:
        all_landmarks (list): Full landmark data.
        ax6_locations (list): List of strings (e.g., ["left_wrist", "right_ankle"]).
        start_frame (int): Frame to start.
        end_frame (int): Frame to end.
        fps (float): Video FPS.
        
    Returns:
        dict: A dictionary where keys are joint names and values are
              lists of (timestamp, x, y, z) tuples.
    """
    ground_truth_data = {loc: [] for loc in ax6_locations}
    
    for i, frame_landmarks in enumerate(all_landmarks):
        # Only analyze the trimmed section
        if i < start_frame or i > end_frame:
            continue
            
        timestamp = i / fps
        
        if frame_landmarks:
            for loc in ax6_locations:
                if loc not in JOINT_MAP:
                    print(f"Warning: Location '{loc}' not recognized.")
                    continue
                    
                # Get the landmark object
                landmark = frame_landmarks.landmark[JOINT_MAP[loc]]
                ground_truth_data[loc].append((
                    timestamp,
                    landmark.x,
                    landmark.y,
                    landmark.z
                ))
        else:
            # Append None if no person was detected
            for loc in ax6_locations:
                ground_truth_data[loc].append((timestamp, None, None, None))
                
    return ground_truth_data

def compile_all_steps(step_timestamps, start_frame, end_frame, fps):
    """
    Your `compile_all_steps` function.
    Formats the final step data.
    
    Returns:
        dict: A summary dictionary with key timestamps.
    """
    return {
        "start_time_sec": start_frame / fps,
        "end_time_sec": end_frame / fps,
        "total_steps": len(step_timestamps),
        "step_timestamps": step_timestamps
    }

def prep_videos(dir_path):
    return glob.glob(f"{dir_path}/*.mp4")

class PedometerGroundTruth:
    def __init__(self, video_path):
        print(f"Initializing pipeline for {video_path}")
        self.video_path = video_path
        self.all_landmarks = None
        self.fps = None
        self.start_frame = None
        self.end_frame = None

    def run_preprocessing(self):
        """Runs skeleton detection on the whole video."""
        self.all_landmarks, self.fps = process_video_and_detect_skeleton(self.video_path)
        if self.all_landmarks is None:
            raise Exception("Video processing failed.")
            
    def find_sync_events(self, clap_threshold=0.1):
        """Finds start/end claps."""
        if self.all_landmarks is None:
            print("Run run_preprocessing() first.")
            return
        self.start_frame, self.end_frame = find_clap_frames(self.all_landmarks, clap_threshold)
        if self.start_frame is None:
            raise Exception("Could not find two clap events.")
            
    def analyze(self, ax6_locations=["left_wrist"]):
        """Runs the main analysis (step counting and joint data extraction)."""
        if self.start_frame is None:
            print("Run find_sync_events() first.")
            return
        
        # 1. Get step counts
        step_timestamps = detect_steps(
            self.all_landmarks, 
            self.start_frame, 
            self.end_frame, 
            self.fps
        )
        
        # 2. Compile step report
        step_report = compile_all_steps(
            step_timestamps, 
            self.start_frame, 
            self.end_frame, 
            self.fps
        )
        
        # 3. Get joint-specific data
        joint_data = locate_ax6(
            self.all_landmarks, 
            ax6_locations, 
            self.start_frame, 
            self.end_frame, 
            self.fps
        )
        
        return step_report, joint_data


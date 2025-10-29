import cv2
import mediapipe as mp
import numpy as np
import glob
from scipy.signal import find_peaks

# --- Constants ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing the skeleton
LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
LEFT_ANKLE = mp.solutions.pose.PoseLandmark.LEFT_ANKLE
RIGHT_ANKLE = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE


def prep_videos(dir_path):
    """Finds all .mp4 files in a directory."""
    return glob.glob(f"{dir_path}/*.mp4")

def process_video_and_detect_skeleton(video_path, resize_height=720):
    """
    PASS 1: Processes the entire video non-interactively.
    This is the slow part.
    
    Returns:
        world_landmarks_list (for 3D analysis)
        screen_landmarks_list (for 2D drawing)
        fps (frames per second)
    """
    print("Starting Pass 1: Processing video and detecting skeletons...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get original dimensions for resizing
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = resize_height / orig_height
    resize_width = int(orig_width * scale)

    world_landmarks_list = []
    screen_landmarks_list = []
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        
        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Print progress
            frame_num += 1
            if frame_num % 100 == 0:
                print(f"  Processing frame {frame_num} / {total_frames}")

            frame_resized = cv2.resize(frame, (resize_width, resize_height))
            image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # Store BOTH world (3D) and normalized (2D) landmarks
            world_landmarks_list.append(results.pose_world_landmarks)
            screen_landmarks_list.append(results.pose_landmarks)

    cap.release()
    print("Pass 1 Complete: All skeletons processed.")
    return world_landmarks_list, screen_landmarks_list, fps

def get_clap_frames_interactively(video_path, screen_landmarks_list, resize_height=720):
    """
    PASS 2: Opens an interactive frame-scrubbing window.
    
    Controls:
        'a' : Previous Frame
        'd' : Next Frame
        's' : Rewind 10 Frames
        'w' : Fast-Forward 10 Frames
        ' ' : (Spacebar) Play/Pause
        'c' : Mark current frame as a Clap
        'q' : Quit
    """
    print("Starting Pass 2: Interactive clap selection...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = resize_height / orig_height
    resize_width = int(orig_width * scale)

    current_frame = 0
    clap_frames = []
    is_paused = True # Start paused

    window_name = "Interactive Frame Selector"
    cv2.namedWindow(window_name)

    while True:
        # Set the video capture to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()
        
        if not success:
            current_frame = total_frames - 1 # Go to last frame if we go too far
            continue

        frame_resized = cv2.resize(frame, (resize_width, resize_height))

        # Get the PRE-COMPUTED skeleton for this frame
        screen_landmarks = screen_landmarks_list[current_frame]
        
        # Draw the skeleton
        if screen_landmarks:
            mp_drawing.draw_landmarks(
                frame_resized,
                screen_landmarks,
                mp_pose.POSE_CONNECTIONS)

        # --- Add text overlays ---
        info_text = f"Frame: {current_frame} / {total_frames} | Claps Marked: {len(clap_frames)}"
        cv2.putText(frame_resized, info_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        
        if is_paused:
            cv2.putText(frame_resized, "PAUSED", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            controls_text = "'d':Fwd, 'a':Back, 'w':+10, 's':-10, 'c':Mark, 'q':Quit"
            cv2.putText(frame_resized, controls_text, (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, frame_resized)

        # --- Key-press Handling ---
        # Wait 1ms if playing, wait indefinitely (0) if paused
        key = cv2.waitKey(1 if not is_paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '): # Spacebar
            is_paused = not is_paused
        elif key == ord('c'):
            print(f"*** Manual Clap Marked at frame: {current_frame} ***")
            clap_frames.append(current_frame)
        
        if is_paused:
            if key == ord('d'):   # Next frame
                current_frame += 1
            elif key == ord('a'): # Previous frame
                current_frame -= 1
            elif key == ord('w'): # Fast-forward 10
                current_frame += 10
            elif key == ord('s'): # Rewind 10
                current_frame -= 10
        elif not is_paused:
            current_frame += 1 # Advance frame if playing

        # Ensure frame number is in bounds
        current_frame = max(0, min(current_frame, total_frames - 1))

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Pass 2 Complete: Found {len(clap_frames)} claps.")
    return clap_frames

def detect_steps(world_landmarks, start_frame, end_frame, fps):
    """
    Analyzes the 3D 'world_landmarks' to find steps.
    (Note: This function is mostly unchanged, but now uses world_landmarks)
    """
    analysis_landmarks = world_landmarks[start_frame:end_frame]
    
    left_ankle_y = []
    right_ankle_y = []
    
    for frame_landmarks in analysis_landmarks:
        if frame_landmarks:
            left_ankle_y.append(frame_landmarks.landmark[LEFT_ANKLE].y)
            right_ankle_y.append(frame_landmarks.landmark[RIGHT_ANKLE].y)
        else:
            left_ankle_y.append(np.nan)
            right_ankle_y.append(np.nan)
            
    # Interpolate missing frames
    left_ankle_y = np.interp(np.arange(len(left_ankle_y)), 
                             np.flatnonzero(np.isfinite(left_ankle_y)), 
                             np.array(left_ankle_y)[np.isfinite(left_ankle_y)])
    right_ankle_y = np.interp(np.arange(len(right_ankle_y)), 
                              np.flatnonzero(np.isfinite(right_ankle_y)), 
                              np.array(right_ankle_y)[np.isfinite(right_ankle_y)])

    # Find peaks (minima of Y-coordinate)
    min_step_distance = int(fps * 0.25) # Min 1/4 sec between steps
    prominence_threshold = 0.05 # --- TUNE THIS VALUE ---
                                # Higher value = less sensitive (fewer steps)
                                # Lower value = more sensitive (more steps)

    left_step_frames, _ = find_peaks(-left_ankle_y, prominence=prominence_threshold, distance=min_step_distance)
    right_step_frames, _ = find_peaks(-right_ankle_y, prominence=prominence_threshold, distance=min_step_distance)
    
    all_step_frames = np.concatenate((left_step_frames, right_step_frames))
    all_step_frames.sort()
    
    step_timestamps = (all_step_frames + start_frame) / fps
    
    print(f"Detected {len(step_timestamps)} steps.")
    return list(step_timestamps)

def locate_ax6(world_landmarks, ax6_locations, start_frame, end_frame, fps):
    """
    Extracts 3D ground truth data for specified joints.
    (Note: This function is mostly unchanged, but now uses world_landmarks)
    """
    JOINT_MAP = {
        "left_wrist": LEFT_WRIST, "right_wrist": RIGHT_WRIST,
        "left_ankle": LEFT_ANKLE, "right_ankle": RIGHT_ANKLE,
    }
    
    ground_truth_data = {loc: [] for loc in ax6_locations}
    
    for i, frame_landmarks in enumerate(world_landmarks):
        if i < start_frame or i > end_frame:
            continue
        
        timestamp = i / fps
        
        if frame_landmarks:
            for loc in ax6_locations:
                if loc not in JOINT_MAP:
                    print(f"Warning: Location '{loc}' not recognized.")
                    continue
                
                landmark = frame_landmarks.landmark[JOINT_MAP[loc]]
                ground_truth_data[loc].append((
                    timestamp, landmark.x, landmark.y, landmark.z
                ))
        else:
            for loc in ax6_locations:
                ground_truth_data[loc].append((timestamp, None, None, None))
                
    return ground_truth_data

def compile_all_steps(step_timestamps, start_frame, end_frame, fps):
    """(This function is unchanged)"""
    return {
        "start_time_sec": start_frame / fps,
        "end_time_sec": end_frame / fps,
        "total_steps": len(step_timestamps),
        "step_timestamps": step_timestamps
    }


class PedometerGroundTruth:
    def __init__(self, video_path):
        print(f"Initializing pipeline for {video_path}")
        self.video_path = video_path
        self.world_landmarks = None # For 3D analysis
        self.screen_landmarks = None # For 2D drawing
        self.fps = None
        self.start_frame = None
        self.end_frame = None

    def run_preprocessing(self):
        """
        Runs the non-interactive Pass 1 to process all skeleton data.
        """
        self.world_landmarks, self.screen_landmarks, self.fps = \
            process_video_and_detect_skeleton(self.video_path)
        
        if self.world_landmarks is None:
            raise Exception("Video processing failed.")

    def find_sync_events(self):
        """
        Runs the interactive Pass 2 to find claps.
        You must run run_preprocessing() first.
        """
        if self.screen_landmarks is None:
            raise Exception("Error: 'run_preprocessing()' must be called first.")
            
        claps = get_clap_frames_interactively(self.video_path, self.screen_landmarks)
        
        if len(claps) < 2:
            raise Exception(f"Error: Only {len(claps)} claps were marked. "
                            "Please mark at least two (start and end).")

        self.start_frame = claps[0]
        self.end_frame = claps[-1] # Use the very last clap marked
        
        print(f"Using Start Frame: {self.start_frame} and End Frame: {self.end_frame}")

    def analyze(self, ax6_locations=["left_wrist"]):
        """
        Runs the main analysis (step counting and joint data extraction).
        """
        if self.start_frame is None:
            raise Exception("Error: 'find_sync_events()' must be called first.")
        
        # 1. Get step counts (uses 3D world_landmarks)
        step_timestamps = detect_steps(
            self.world_landmarks, 
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
        
        # 3. Get joint-specific data (uses 3D world_landmarks)
        joint_data = locate_ax6(
            self.world_landmarks, 
            ax6_locations, 
            self.start_frame, 
            self.end_frame, 
            self.fps
        )
        
        return step_report, joint_data
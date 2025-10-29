import analyzer_pipeline
import glob

debug = True

def main():
    """Runs the full test process."""
    
    print("Starting pipeline test...")
    
    # 2. Use the 'prep_videos' function FROM your library
    video_files = analyzer_pipeline.prep_videos("./walking_videos")
    
    if not video_files:
        print("No videos found in './walking_videos' folder. Aborting.")
        return

    # 3. Loop through the files and use the main class
    for video_file in video_files:
        print(f"\n--- Processing video: {video_file} ---")
        try:
            # 4. Create an instance of the class FROM your library
            pipeline = analyzer_pipeline.PedometerGroundTruth(video_file)
            
            # Step 1 & 2: Process video, get skeleton
            pipeline.run_preprocessing() 
            
            # Step 3 & 4: Find claps
            pipeline.find_sync_events(clap_threshold=0.08) # Tune this!
            
            # Analysis Step: Count steps and get wrist data
            step_report, joint_data = pipeline.analyze(
                ax6_locations=["left_wrist", "right_ankle"]
            )
            
            print("\n--- Analysis Complete ---")
            print(f"Video: {video_file}")
            print(f"Start Time: {step_report['start_time_sec']:.2f}s")
            print(f"End Time: {step_report['end_time_sec']:.2f}s")
            print(f"Total Steps: {step_report['total_steps']}")
            print(f"Extracted data for: {list(joint_data.keys())}")
            
        except Exception as e:
            print(f"Failed to process {video_file}: {e}")

# This __name__ == "__main__" block is standard for a script.
# It means "when I run 'python test_my_library.py', run the main() function."
if __name__ == "__main__":
    main()
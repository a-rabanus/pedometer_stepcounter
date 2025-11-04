import analyzer_pipeline

def main():
    """Runs the full test process."""
    
    print("Starting pipeline test...")
    
    video_files = analyzer_pipeline.prep_videos("./walking_videos") # Or your "./walking_videos"
    
    if not video_files:
        print("No videos found in './walking_videos' folder. Aborting.")
        return

    for video_file in video_files:
        print(f"\n--- Processing video: {video_file} ---")
        try:
            pipeline = analyzer_pipeline.PedometerGroundTruth(video_file)
            
            # --- NEW WORKFLOW ---
            
            # Step 1: Run the slow, non-interactive processing
            pipeline.run_preprocessing() 
            
            # Step 2: Run the fast, interactive clap selection
            pipeline.find_sync_events()
            
            # Step 3: Run the final analysis (no change here)
            step_report, joint_data = pipeline.analyze(
                ax6_locations=["left_wrist", "right_ankle"]
            )
            # ---------------------
            
            if step_report is None:
                continue # Skip if analysis failed

            print("\n--- Analysis Complete ---")
            print(f"Video: {video_file}")
            print(f"Start Time: {step_report['start_time_sec']:.2f}s")
            print(f"End Time: {step_report['end_time_sec']:.2f}s")
            print(f"Total Steps: {step_report['total_steps']}")
            print(f"Extracted data for: {list(joint_data.keys())}")
            
        except Exception as e:
            print(f"Failed to process {video_file}. Error: {e}")

if __name__ == "__main__":
    main()
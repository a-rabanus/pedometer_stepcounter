# scripts/00_run_video_pipeline.py
# Läuft deine Pipeline einmal durch, speichert step_report.json
# und CSVs für die gewünschten Joints (t,x,y,z) nach ./out

import os, json, pandas as pd
from pathlib import Path
from analyzer_pipeline import PedometerGroundTruth

VIDEO = "Video/Vidvv2.mp4"  
JOINTS = ["left_ankle","right_ankle","left_wrist","right_wrist"]

def main():
    Path("out").mkdir(exist_ok=True)
    pipe = PedometerGroundTruth(VIDEO)

    print("Pass 1: preprocess …")
    pipe.run_preprocessing()

    print("Pass 2: claps markieren …")
    pipe.find_sync_events()  # Fenster: d/a=±1, w/s=±10, Space=Play/Pause, c=Clap, q=Quit

    print("Analyse …")
    step_report, joint_data = pipe.analyze(ax6_locations=JOINTS)

    with open("out/step_report.json", "w") as f:
        json.dump(step_report, f, indent=2)

    for name, series in joint_data.items():
        df = pd.DataFrame(series, columns=["t","x","y","z"])
        df.to_csv(f"out/{name}.csv", index=False)

    print("Fertig. Dateien liegen in ./out:")
    print(" - out/step_report.json")
    for name in joint_data.keys():
        print(f" - out/{name}.csv")

if __name__ == "__main__":
    main()
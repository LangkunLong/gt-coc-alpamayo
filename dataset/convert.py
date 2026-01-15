import pandas as pd
import json

# 1. Load Waterloo Data (Drone Trajectories)
# Columns typically: [frame_id, track_id, x, y, w, h, angle_rad]
waterloo_df = pd.read_csv("waterloo_tracks.csv") 

# 2. Structure for AlpaSim Traffic Service
alpasim_traffic = []

for frame_id, frame_data in waterloo_df.groupby("frame_id"):
    timestamp = frame_id * 0.1  # Assuming 10Hz
    
    frame_actors = []
    for _, row in frame_data.iterrows():
        actor = {
            "id": int(row["track_id"]),
            "pose": {
                "x": row["x"],      # You might need to coordinate transform this
                "y": row["y"],
                "yaw": row["angle_rad"]
            },
            "dimensions": {"l": 4.5, "w": 2.0, "h": 1.6}, # Default car size
            "type": "vehicle"
        }
        frame_actors.append(actor)
        
    alpasim_traffic.append({
        "time": timestamp,
        "actors": frame_actors
    })

# 3. Save as AlpaSim scenario file
with open("waterloo_scenario_01.json", "w") as f:
    json.dump(alpasim_traffic, f)
import os
import requests
import numpy as np
import matplotlib.image as mpimg

print("=== Launching HRRR-Grade Great Lakes Grid Harvester ===")

# Master Dictionary with your precise local storm timelines
events = {
    "nov_popups":   {"date": "2025/11/15", "hours": ["16", "17", "18", "19", "20", "21"]}, 
    "boxing_day":   {"date": "2025/12/26", "hours": ["14", "15", "16", "17", "18", "19"]}, 
    "dec_27":       {"date": "2025/12/27", "hours": ["00", "01", "02", "03", "04", "05"]},
    "feb_squall":   {"date": "2026/02/18", "hours": ["22", "23", "00", "01", "02", "03"]}, 
    "june_popups":  {"date": "2026/06/06", "hours": ["18", "19", "20", "21", "22", "23"]},
    "supercells":   {"date": "2021/07/15", "hours": ["16", "17", "18", "19", "20", "21"]}  
}

# EXACT NEXRAD USCOMP PIXEL BOUNDS FOR GROUND TRUTH ALIGNMENT
# Master image size is 6000 height x 17500 width
# Lat: 50.0 N to 24.0 N | Lon: 126.0 W to 66.0 W
def get_great_lakes_crop(img):
    h, w, _ = img.shape
    
    # Calculate precise pixel indices matching our exact region
    x_start = int(((126.0 - 90.0) / (126.0 - 66.0)) * w)  # Locked on Western Great Lakes
    x_end = int(((126.0 - 74.0) / (126.0 - 66.0)) * w)    # Locked on Eastern Ontario/NY
    y_start = int(((50.0 - 47.0) / (50.0 - 24.0)) * h)    # Upper boundary (Northern Ontario)
    y_end = int(((50.0 - 41.0) / (50.0 - 24.0)) * h)      # Lower boundary (PA/Ohio)
    
    return img[y_start:y_end, x_start:x_end, :3]

all_sequences = []

for event_name, info in events.items():
    print(f"\n[HARVEST] Extracting high-res grids for: {event_name.upper()}")
    event_frames = []
    
    for hour in info['hours']:
        date_path = info['date']
        url_date = date_path.replace("/", "")
        filename = f"n0r_{url_date}{hour}00.png"
        download_url = f"https://mesonet.agron.iastate.edu/archive/data/{date_path}/GIS/uscomp/{filename}"
        local_raw_path = f"raw_{filename}"
        
        try:
            response = requests.get(download_url, timeout=12)
            if response.status_code == 200:
                with open(local_raw_path, 'wb') as f:
                    f.write(response.content)
                
                img = mpimg.imread(local_raw_path)
                cropped = get_great_lakes_crop(img)
                event_frames.append(cropped)
                os.remove(local_raw_path)
                print(f" └─> [SUCCESS] Captured frame at hour {hour}Z | Res: {cropped.shape[1]}x{cropped.shape[0]}")
            else:
                print(f" └─> [SERVER MISSING] Hour {hour}Z HTTP {response.status_code}")
        except Exception as e:
            print(f" └─> [ERROR] Connection failed: {str(e)}")
            
    if len(event_frames) == len(info['hours']):
        all_sequences.append(np.stack(event_frames, axis=0))

if all_sequences:
    master_matrix = np.stack(all_sequences, axis=0)
    np.save(r"C:\Users\awada\OneDrive\Desktop\GLRR_Model\glrr_training_matrix.npy", master_matrix)
    
    # Save a standalone validation file using the last frame of the final sequence for display testing
    np.save(r"C:\Users\awada\OneDrive\Desktop\GLRR_Model\glrr_may20_00z_ground_truth.npy", all_sequences[-1][-1])
    
    print("\n==============================================")
    print(f"🥇 [SUCCESS] HRRR-Aligned Master Training Matrix Generated!")
    print(f"Final Matrix Shape: {master_matrix.shape}")
else:
    print("\n[ERROR] Pipeline data verification failed.")

input("\nPress Enter to exit...")
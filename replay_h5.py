import h5py
import cv2
import sys
import os
import glob
from PIL import Image

def replay_h5_to_gif(file_path=None):
    # 1. Auto-select the most recent file if none is provided
    if file_path is None:
        search_dir = "dataset_hdf5/sim_session"
        files = glob.glob(os.path.join(search_dir, "*.h5"))
        
        if not files:
            print(f"[ERROR] No .h5 files found in {search_dir}")
            return
            
        # Sort files by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        file_path = files[0]
        print(f"--> Auto-selected most recent episode: {file_path}")
    else:
        print(f"--> Opening {file_path}...")
    
    try:
        with h5py.File(file_path, 'r') as f:
            imgs = f['observations/images/top'][:]
            states = f['observations/state'][:]
            
            if len(imgs) == 0:
                print("No images found in HDF5.")
                return
            
            out_path = file_path.replace('.h5', '.gif')
            print(f"Loaded {len(imgs)} frames. Exporting to {out_path}...")
            
            gif_frames = []
            
            for i, img in enumerate(imgs):
                # Make a writable copy of the RGB image for OpenCV to draw on
                img_np = img.copy()
                
                # Overlay the frame number and suction state
                grip_state = states[i][6]
                grip_txt = "ON" if grip_state > 0.5 else "OFF"
                cv2.putText(img_np, f"Frame: {i:03d} | Suction: {grip_txt}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert the numpy array to a PIL Image and add to our list
                pil_img = Image.fromarray(img_np)
                gif_frames.append(pil_img)
                
            # 2. Save the list of images as an animated GIF
            # duration=33ms gives roughly 30 FPS (1000ms / 30 = 33.3)
            gif_frames[0].save(
                out_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=33, 
                loop=0 # 0 means loop infinitely
            )
            print(f"--> Export complete! You can now share {out_path}")
                    
    except Exception as e:
        print(f"[ERROR] Could not process file: {e}")

if __name__ == "__main__":
    # Accept a specific path via command line, otherwise default to None (auto-select)
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    replay_h5_to_gif(target_file)
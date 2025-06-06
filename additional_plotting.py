import numpy as np
from dedalus import public as de
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import glob
import subprocess

# Settings
frame_rate = 3
input_pattern = "/home/chaddyfynn/scripts/social-dynamics/frames/frame_%04d.png"

# Output names
palette_path = "palette.png"
gif_path = "output.gif"


# snapshot_files = sorted(glob.glob('snapshots/snapshots_*.h5'))

# for i, snap_file in enumerate(snapshot_files):
#     with h5py.File(snap_file, 'r') as f:
#         # Get data for field n1
#         n1_data = f['tasks']['n1'][:]

#         # Make a frame plot
#         plt.clf()
#         plt.plot(n1_data)  # assuming 1D data for simplicity
#         # plt.title(f"Time: {f.attrs['sim_time']:.3f}")
#         plt.savefig(f'frames/frame_{i:04d}.png')

def get_sim_time(path):
    with h5py.File(path, 'r') as f:
        # Adjust if you saved sim_time as a field instead
        return f['tasks']['sim_time'][0, 0]

#snapshot_files = sorted(glob.glob('snapshots/snapshots_*.h5'))
snapshot_files = sorted(glob.glob('snapshots/snapshots_*.h5'), key=get_sim_time)

for i, snap_file in enumerate(snapshot_files):
    with h5py.File(snap_file, 'r') as f:
        # Load spatial grid (should now be saved properly)
        x = f['tasks']['x'][0]  # shape: (Nx,) or maybe (1, Nx)

        # Load field data for n1
        n1 = f['tasks']['n1'][0]  # first time slice
        n2 = f['tasks']['n2'][0]
        n3 = f['tasks']['n3'][0]

        time_val = f['tasks']['sim_time'][:][0,0]  # or slice accordingly

        plt.clf()
        plt.plot(x, n1, label=f'n1')
        plt.plot(x, n2, label=f'n2')
        plt.plot(x, n3, label=f'n3')
        plt.legend()
        sim_time_data = f['tasks']['sim_time'][:]
        # Usually 1 entry per snapshot: shape = (1, 1, 1, 1) for 0D field
        sim_time_val = sim_time_data[0, 0]
        plt.title(f"Time: {sim_time_val:.3f}")
        plt.xlabel('x')
        plt.ylabel('nᵢ(x)')
        plt.savefig(f'frames/frame_{i:04d}.png')


# Step 1: Generate palette (color optimization)
subprocess.run([
    "ffmpeg",
    "-framerate", str(frame_rate),
    "-i", input_pattern,
    "-vf", "palettegen",
    "-y",  # overwrite if exists
    palette_path
], check=True)

# Step 2: Create the GIF using the palette
subprocess.run([
    "ffmpeg",
    "-framerate", str(frame_rate),
    "-i", input_pattern,
    "-i", palette_path,
    "-lavfi", "paletteuse",
    "-an",  # no audio
    "-y",
    gif_path
], check=True)

print(f"GIF saved to {gif_path}")
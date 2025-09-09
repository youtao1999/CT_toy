import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
from PIL import Image

def extract_threshold_from_filename(filename):
    """Extract threshold value from the filename."""
    match = re.search(r'threshold(\d+\.\d+e[-+]\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def main():
    """
    Create an animation from the pre-existing loss manifold comparison plots 
    in the tmi_compare_results folder.
    """
    # Parameters
    p_fixed = 0.000
    p_fixed_name = 'pctrl'
    source_folder = 'tmi_compare_results'
    
    # Find all PNG files in the source folder with the right pattern
    plot_pattern = f'{source_folder}/loss_manifold_comparison_{p_fixed_name}{p_fixed:.3f}_threshold*.png'
    plot_files = glob.glob(plot_pattern)
    
    if not plot_files:
        print(f"Error: No plot files found matching pattern: {plot_pattern}")
        return
    
    # Extract thresholds and sort files by decreasing threshold (descending order)
    threshold_files = [(extract_threshold_from_filename(f), f) for f in plot_files]
    threshold_files = sorted(threshold_files, key=lambda x: x[0], reverse=True)
    
    thresholds = [t for t, _ in threshold_files]
    plot_files = [f for _, f in threshold_files]
    
    print(f"Found {len(plot_files)} plot files with thresholds (in descending order):")
    for i, (threshold, filename) in enumerate(threshold_files):
        print(f"  Plot {i}: {os.path.basename(filename)} (Threshold: {threshold:.1e})")
    
    # Create animations directory if it doesn't exist
    os.makedirs('animations', exist_ok=True)
    
    # Create figure to hold the animation
    fig = plt.figure(figsize=(15, 10))
    
    # Function to update each frame
    def update(frame_num):
        # Clear figure
        plt.clf()
        
        # Load image from file
        img = Image.open(plot_files[frame_num])
        
        # Display image
        plt.imshow(np.array(img))
        plt.axis('off')  # Hide axes
        
        # Add threshold information to title
        threshold = thresholds[frame_num]
        plt.title(f"Threshold: {threshold:.1e}", fontsize=16)
        
        return plt.gca(),
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(plot_files),
        interval=100,  # 0.1 second between frames
        blit=False,
        repeat=True
    )
    
    # Save animation
    output_file = f'animations/loss_contour_animation_decreasing_{p_fixed_name}{p_fixed:.3f}.gif'
    print(f"Saving animation to {output_file}...")
    
    # Use a writer that should be available
    writer = animation.PillowWriter(fps=1)
    ani.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")
    
    # Also show the animation in the plot window
    plt.show()

if __name__ == "__main__":
    main() 
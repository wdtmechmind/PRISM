"""
Trajectory analysis and visualization tool for post-processing.
Compares online and offline trajectories to diagnose collection issues.
"""

import os
import csv
import json
import numpy as np

# Use non-interactive backend before importing matplotlib
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for file output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from prism.common import console
from prism.reconstruction.realtime_reconstruction import COLOR_ORDER, COLOR_MPL


def load_trajectory_csv(csv_path):
    """Load trajectory from CSV file.
    
    CSV format: t_sec, color, x, y, z, mode, num_obs, max_err, cameras
    """
    data = {name: {'t': [], 'x': [], 'y': [], 'z': [], 'mode': [], 'num_obs': []} 
            for name in COLOR_ORDER}
    
    if not os.path.exists(csv_path):
        console.warning(f'trajectory file not found: {csv_path}')
        return data
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                t = float(row[0])
                color = row[1]
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                mode = row[5]
                num_obs = int(row[6]) if len(row) > 6 else 0
                
                if color in data:
                    data[color]['t'].append(t)
                    data[color]['x'].append(x)
                    data[color]['y'].append(y)
                    data[color]['z'].append(z)
                    data[color]['mode'].append(mode)
                    data[color]['num_obs'].append(num_obs)
            except (ValueError, IndexError):
                continue
    
    # Convert to numpy arrays
    for color in data:
        if data[color]['t']:
            data[color]['t'] = np.array(data[color]['t'])
            data[color]['x'] = np.array(data[color]['x'])
            data[color]['y'] = np.array(data[color]['y'])
            data[color]['z'] = np.array(data[color]['z'])
            data[color]['num_obs'] = np.array(data[color]['num_obs'])
        else:
            data[color]['t'] = np.array([])
            data[color]['x'] = np.array([])
            data[color]['y'] = np.array([])
            data[color]['z'] = np.array([])
            data[color]['num_obs'] = np.array([])
    
    return data


def compute_trajectory_statistics(traj_data):
    """Compute statistics for each LED trajectory."""
    stats = {}
    
    for color in COLOR_ORDER:
        data = traj_data[color]
        t = data['t']
        x, y, z = data['x'], data['y'], data['z']
        mode = data['mode']
        
        if len(t) == 0:
            stats[color] = {
                'frame_count': 0,
                'duration': 0.0,
                'measured_frames': 0,
                'predicted_frames': 0,
                'lost_frames': 0,
                'paused_frames': 0,
                'measured_ratio': 0.0,
                'spatial_range': {'x': [0, 0], 'y': [0, 0], 'z': [0, 0]},
            }
            continue
        
        # Filter out NaN values
        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        z_valid = z[valid_idx]
        mode_valid = [mode[i] for i in range(len(mode)) if valid_idx[i]]
        
        measured_count = sum(1 for m in mode_valid if m == 'measured')
        predicted_count = sum(1 for m in mode_valid if m == 'predicted')
        lost_count = sum(1 for m in mode_valid if m == 'lost')
        paused_count = sum(1 for m in mode_valid if m == 'paused')
        
        stats[color] = {
            'frame_count': len(t),
            'duration': t[-1] - t[0] if len(t) > 1 else 0.0,
            'measured_frames': measured_count,
            'predicted_frames': predicted_count,
            'lost_frames': lost_count,
            'paused_frames': paused_count,
            'measured_ratio': measured_count / len(t) if len(t) > 0 else 0.0,
            'spatial_range': {
                'x': [float(np.min(x_valid)), float(np.max(x_valid))] if len(x_valid) > 0 else [0, 0],
                'y': [float(np.min(y_valid)), float(np.max(y_valid))] if len(y_valid) > 0 else [0, 0],
                'z': [float(np.min(z_valid)), float(np.max(z_valid))] if len(z_valid) > 0 else [0, 0],
            }
        }
    
    return stats


def plot_trajectory_3d(traj_data, title='LED Trajectories (3D)', output_path=None):
    """Plot 3D trajectory visualization."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for color in COLOR_ORDER:
        data = traj_data[color]
        x, y, z = data['x'], data['y'], data['z']
        
        # Filter out NaN values
        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x_plot = x[valid_idx]
        y_plot = y[valid_idx]
        z_plot = z[valid_idx]
        mode_plot = [data['mode'][i] for i in range(len(data['mode'])) if valid_idx[i]]
        
        if len(x_plot) > 0:
            # Plot line
            ax.plot(x_plot, y_plot, z_plot, color=COLOR_MPL[color], linewidth=2, 
                   label=color, alpha=0.7)
            # Plot start and end points
            ax.scatter(x_plot[0], y_plot[0], z_plot[0], color=COLOR_MPL[color], 
                      s=100, marker='o', edgecolor='black', linewidth=2, zorder=10)
            ax.scatter(x_plot[-1], y_plot[-1], z_plot[-1], color=COLOR_MPL[color], 
                      s=100, marker='X', edgecolor='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-45)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.success(f'saved 3D trajectory plot: {output_path}')
    return fig


def plot_trajectory_2d_projections(traj_data, title='LED Trajectories (2D Projections)', output_path=None):
    """Plot 2D projections of trajectories (XY, XZ, YZ)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    projections = [
        (('x', 'y'), 'X (m)', 'Y (m)', 'XY Projection'),
        (('x', 'z'), 'X (m)', 'Z (m)', 'XZ Projection'),
        (('y', 'z'), 'Y (m)', 'Z (m)', 'YZ Projection'),
    ]
    
    for ax, (axes_pair, xlabel, ylabel, subplot_title) in zip(axes, projections):
        ax1_name, ax2_name = axes_pair
        
        for color in COLOR_ORDER:
            data = traj_data[color]
            ax1 = data[ax1_name]
            ax2 = data[ax2_name]
            
            # Filter out NaN values
            valid_idx = ~(np.isnan(ax1) | np.isnan(ax2))
            ax1_plot = ax1[valid_idx]
            ax2_plot = ax2[valid_idx]
            
            if len(ax1_plot) > 0:
                ax.plot(ax1_plot, ax2_plot, color=COLOR_MPL[color], linewidth=2, 
                       label=color, alpha=0.7)
                ax.scatter(ax1_plot[0], ax2_plot[0], color=COLOR_MPL[color], 
                          s=80, marker='o', edgecolor='black', linewidth=2, zorder=5)
                ax.scatter(ax1_plot[-1], ax2_plot[-1], color=COLOR_MPL[color], 
                          s=80, marker='X', edgecolor='black', linewidth=2, zorder=5)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(subplot_title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.success(f'saved 2D trajectory plot: {output_path}')
    return fig


def plot_trajectory_timeline(traj_data, title='LED Tracking Timeline', output_path=None):
    """Plot tracking status timeline for each LED."""
    fig, axes = plt.subplots(len(COLOR_ORDER), 1, figsize=(14, 2*len(COLOR_ORDER)))
    if len(COLOR_ORDER) == 1:
        axes = [axes]
    
    for ax, color in zip(axes, COLOR_ORDER):
        data = traj_data[color]
        t = data['t']
        mode = data['mode']
        
        if len(t) == 0:
            ax.text(0.5, 0.5, f'{color}: no data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-0.5, 1.5)
            continue
        
        # Color map for modes
        mode_colors = {
            'measured': 'green',
            'predicted': 'yellow',
            'lost': 'red',
            'paused': 'gray',
            'none': 'white',
        }
        
        # Plot mode timeline
        for i, (ti, mi) in enumerate(zip(t, mode)):
            color_val = mode_colors.get(mi, 'white')
            ax.barh(0, 1, left=ti, height=0.5, color=color_val, edgecolor='black', linewidth=0.5)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(t[0] if len(t) > 0 else 0, t[-1] if len(t) > 0 else 1)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_yticks([])
        ax.set_title(f'{color} - Tracking Status', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='black', label='Measured'),
        mpatches.Patch(facecolor='yellow', edgecolor='black', label='Predicted'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Lost'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='Paused'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.success(f'saved timeline plot: {output_path}')
    return fig


def compare_trajectories(online_data, offline_data=None, output_dir=None):
    """Compare online and offline trajectories side-by-side."""
    if offline_data is None:
        return plot_trajectory_3d(online_data, title='LED Trajectories (Online Collection)',
                                output_path=os.path.join(output_dir, 'trajectory_3d_online.png') if output_dir else None)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Online
    ax1 = fig.add_subplot(121, projection='3d')
    for color in COLOR_ORDER:
        data = online_data[color]
        x, y, z = data['x'], data['y'], data['z']
        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if len(x[valid_idx]) > 0:
            ax1.plot(x[valid_idx], y[valid_idx], z[valid_idx], color=COLOR_MPL[color], 
                    linewidth=2, label=color, alpha=0.7)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Online Collection', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=-45)
    
    # Offline
    ax2 = fig.add_subplot(122, projection='3d')
    for color in COLOR_ORDER:
        data = offline_data[color]
        x, y, z = data['x'], data['y'], data['z']
        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if len(x[valid_idx]) > 0:
            ax2.plot(x[valid_idx], y[valid_idx], z[valid_idx], color=COLOR_MPL[color], 
                    linewidth=2, label=color, alpha=0.7)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Offline Reconstruction', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=-45)
    
    fig.suptitle('Online vs Offline Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=150, bbox_inches='tight')
        console.success(f'saved comparison plot')
    return fig


def print_trajectory_stats(stats, title='Trajectory Statistics'):
    """Print trajectory statistics to console."""
    console.rule(title)
    
    for color in COLOR_ORDER:
        s = stats[color]
        console.info(f'\n{color.upper()}:')
        console.info(f'  Total frames:     {s["frame_count"]}')
        console.info(f'  Duration:         {s["duration"]:.3f} s')
        console.info(f'  Measured frames:  {s["measured_frames"]} ({s["measured_ratio"]*100:.1f}%)')
        console.info(f'  Predicted frames: {s["predicted_frames"]}')
        console.info(f'  Lost frames:      {s["lost_frames"]}')
        console.info(f'  Paused frames:    {s["paused_frames"]}')
        console.info(f'  X range: {s["spatial_range"]["x"][0]:.4f} ~ {s["spatial_range"]["x"][1]:.4f} m')
        console.info(f'  Y range: {s["spatial_range"]["y"][0]:.4f} ~ {s["spatial_range"]["y"][1]:.4f} m')
        console.info(f'  Z range: {s["spatial_range"]["z"][0]:.4f} ~ {s["spatial_range"]["z"][1]:.4f} m')


def analyze_task_directory(task_dir, output_dir=None):
    """Analyze a complete task directory with online and offline trajectories."""
    if output_dir is None:
        output_dir = task_dir
    
    console.rule(f'Trajectory Analysis: {task_dir}')
    
    # Load online trajectory
    online_nearest_path = os.path.join(task_dir, 'trajectory_led_nearest.csv')
    online_data = load_trajectory_csv(online_nearest_path)
    online_stats = compute_trajectory_statistics(online_data)
    
    print_trajectory_stats(online_stats, title='ONLINE COLLECTION STATISTICS')
    
    # Generate visualizations
    plot_trajectory_3d(online_data, title='Online Collection - 3D Trajectory',
                      output_path=os.path.join(output_dir, 'traj_3d_online.png'))
    plot_trajectory_2d_projections(online_data, title='Online Collection - 2D Projections',
                                  output_path=os.path.join(output_dir, 'traj_2d_online.png'))
    plot_trajectory_timeline(online_data, title='Online Collection - Tracking Timeline',
                            output_path=os.path.join(output_dir, 'traj_timeline_online.png'))
    
    # Check for offline data
    first_trial_dir = None
    for name in sorted(os.listdir(task_dir)):
        trial_path = os.path.join(task_dir, name)
        if os.path.isdir(trial_path) and name.startswith('trial_'):
            first_trial_dir = trial_path
            break
    
    if first_trial_dir:
        console.info(f'Found trial directory: {first_trial_dir}')
        aligned_dir = os.path.join(first_trial_dir, 'aligned_offline')
        if os.path.exists(aligned_dir):
            console.info('Offline reconstruction found - you can compare after running rebuild')
    
    console.success('Trajectory analysis complete!')
    console.info(f'Visualizations saved to: {output_dir}')


def main():
    """CLI entry point for trajectory analysis."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze and visualize LED trajectories from PRISM collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze task directory
  prism-analyze-trajectory data/raw/task_20260723_120000_grasp-demo
  
  # Specify output directory
  prism-analyze-trajectory data/raw/task_20260723_120000_grasp-demo --output-dir ./trajectory_plots
        '''
    )
    parser.add_argument('task_dir', nargs='?', help='Path to task directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for visualizations (default: task directory)')
    
    args = parser.parse_args()
    
    if not args.task_dir:
        parser.print_help()
        console.warning('task_dir argument required')
        return 1
    
    task_path = os.path.expanduser(args.task_dir)
    output_path = os.path.expanduser(args.output_dir) if args.output_dir else None
    
    if not os.path.exists(task_path):
        console.warning(f'task directory not found: {task_path}')
        return 1
    
    try:
        analyze_task_directory(task_path, output_path)
        return 0
    except Exception as e:
        console.warning(f'analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

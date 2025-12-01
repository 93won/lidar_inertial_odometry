#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Labels: Method (Sensor)
labels = [
    'FAST-LIO2\n(AVIA)', 'FASTER-LIO\n(AVIA)', 'Surfel-LIO\n(AVIA)',
    'FAST-LIO2\n(Mid360)', 'FASTER-LIO\n(Mid360)', 'Surfel-LIO\n(Mid360)'
]

# Bold indices for Surfel-LIO
bold_indices = [2, 5]

# Data (original RMSE values)
rmse = [0.397, 0.362, 0.365, 0.342, 0.352, 0.342]
fps = [125, 184, 531, 282, 353, 690]

# Colors: FAST-LIO2, FASTER-LIO, Surfel-LIO (repeated for 2 sensors)
colors = ['#4A90D9', '#F5A623', '#7ED321', '#4A90D9', '#F5A623', '#7ED321']

fig, ax1 = plt.subplots(figsize=(6.5, 5))
ax2 = ax1.twinx()

# Calculate bar width and spacing so that:
# left_margin = gap_between_bars = gap_to_center_line = right_margin
n_bars = len(labels)
width = 0.42  # bar width for each (RMSE or FPS)
pair_width = 2 * width  # width of RMSE+FPS bar pair
gap = 0.22  # gap between all elements (bar pairs, margins, center line)

# x positions: need extra gap at center (between AVIA and Mid360)
# AVIA group: 0, 1, 2 (indices)
# Mid360 group: 3, 4, 5 (indices)
# Add extra gap between index 2 and 3
x = np.array([0, 1, 2, 3 + gap/pair_width, 4 + gap/pair_width, 5 + gap/pair_width]) * (pair_width + gap)

# RMSE bars (left y-axis)
bars1 = ax1.bar(x - width/2, rmse, width, color=colors, alpha=0.7, edgecolor='black', label='APE RMSE (m)')
ax1.set_ylabel('APE RMSE (m)', fontsize=12)
ax1.set_ylim(0, 0.5)
ax1.set_yticks([])  # Remove y-axis ticks

# FPS bars (right y-axis)
bars2 = ax2.bar(x + width/2, fps, width, color=colors, hatch='///', edgecolor='black', label='FPS')
ax2.set_ylabel('FPS', fontsize=12)
ax2.set_ylim(0, 800)
ax2.set_yticks([])  # Remove y-axis ticks

# X-axis labels (horizontal)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)

# Make Surfel-LIO labels bold
for i, tick in enumerate(ax1.get_xticklabels()):
    if i in bold_indices:
        tick.set_fontweight('bold')

# Set xlim: margin = gap
ax1.set_xlim(x[0] - width - gap, x[-1] + width + gap)

# Center line position: exactly in the middle between bar 2 (right edge) and bar 3 (left edge)
center_line_x = (x[2] + width + x[3] - width) / 2

# Add vertical separator line between AVIA and Mid360
ax1.axvline(x=center_line_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Value labels on bars
for bar, val in zip(bars1, rmse):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
             ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, fps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val}', 
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='APE RMSE (m)'),
    Patch(facecolor='gray', hatch='///', edgecolor='black', label='FPS')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout(pad=0)
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.savefig('benchmark_comparison.pdf', bbox_inches='tight', pad_inches=0.03)
print("Saved: benchmark_comparison.png, benchmark_comparison.pdf")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io import savemat

# Step 1: Load the CSV and create navpoint objects
filename = 'airways_db.csv'
data = pd.read_csv(filename)

# Dictionary to hold navpoints, keyed by "Numbered Fix"
navpoints = {}

# Create navpoint objects
for _, row in data.iterrows():
    numbered_fix = row['Numbered Fix']
    if numbered_fix not in navpoints:
        navpoints[numbered_fix] = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'center': row['Center'],
            'airways': set()
        }
    # Add airways to the navpoint's list
    navpoints[numbered_fix]['airways'].update(row['Airways'].split())

# Step 2: Construct airways
airways = defaultdict(list)

# Find unique airway names and associate navpoints
for numbered_fix, details in navpoints.items():
    for airway in details['airways']:
        airways[airway].append({
            str(numbered_fix): navpoints[numbered_fix]
        })

# Sort each airway's navpoints by longitude
for airway, points in airways.items():
    airways[airway] = sorted(points, key=lambda x: list(x.values())[0]['longitude'])

# Convert airways to MATLAB-compatible format
matlab_airways = {}

for airway, points in airways.items():
    # Create a MATLAB structure for each airway
    airway_struct = {}
    for point in points:
        numbered_fix, navpoint_details = list(point.items())[0]
        # Use numbered_fix as the field name
        airway_struct["idx_" + str(numbered_fix)] = {
            'latitude': navpoint_details['latitude'],
            'longitude': navpoint_details['longitude'],
            'center': navpoint_details['center']
        }
    # Assign to the main airways structure
    matlab_airways[airway] = airway_struct

# Save to a .mat file
savemat('airways.mat', {'airways': matlab_airways})

exit(2)

# Step 3: Plot the airways
plt.figure(figsize=(12, 8))
for airway, points in airways.items():
    # Extract coordinates for plotting
    longitudes = [list(point.values())[0]['longitude'] for point in points]
    latitudes = [list(point.values())[0]['latitude'] for point in points]

    # Plot the airway
    plt.plot(longitudes, latitudes, marker='o', label=airway)

    # Add labels for navpoints
    for point in points:
        numbered_fix = list(point.keys())[0]
        navpoint = list(point.values())[0]
        plt.text(navpoint['longitude'], navpoint['latitude'], numbered_fix, fontsize=8)

# Customize the plot
plt.title("Airways and Navpoints")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Airways", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

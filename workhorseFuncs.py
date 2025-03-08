from fontTools.unicodedata import block
from scipy.ndimage import gaussian_filter1d
import numpy as np
import PySimpleGUI as sg
import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime, timedelta
import ast
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import time
matplotlib.use('TkAgg')

def load_prh():
    layout = [
        [sg.Text("Select a MATLAB file")],
        [sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("MATLAB Files", "*.mat"),))],
        [sg.Button("Load"), sg.Button("Cancel")]
    ]
    window = sg.Window("Load MATLAB File", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None, None
        if event == "Load":
            fpath = values["-FILE-"]
            if fpath:
                window.close()
                try:
                    # Change the current working directory to the location of the file
                    #os.chdir(os.path.dirname(fpath))
                    prhdir = os.path.dirname(fpath)
                    print(f"Changed working directory to: {os.getcwd()}")

                    # Load the MATLAB file
                    mat_contents = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)

                    # Extract the file name from the full path
                    filename = os.path.basename(fpath)
                    return mat_contents, filename, prhdir  # Return contents and filename
                except Exception as e:
                    sg.popup_error(f"Error loading file: {str(e)}")
                    return None, None
            else:
                sg.popup_error("Please select a file.")
    window.close()
def load_progressfile(fname):
    feed = pd.DataFrame(pd.read_csv(fname, usecols= [0,1,2,3]))
    jerksigs = pd.DataFrame(pd.read_csv(fname, usecols= [4]))
    feed = pd.concat([feed,jerksigs], axis=1)
    #print(feed.to_string())
    return feed

def make_jerk(dat):
    """
    Compute the norm-jerk from triaxial acceleration data and add it to the prh dictionary.

    Parameters:
    - prh: Dictionary containing acceleration data.

    If the 'jerk' key does not exist, it will be created in the prh dictionary.
    """

    # Check if acceleration data is present
    if 'Aw' not in dat:
        print("The dictionary does not contain the acceleration data under key 'Aw'.")
        return

    A = np.array(dat['Aw'])  # Ensure A is a NumPy array

    # Check if A is a 2D array with at least 2 rows
    if A.ndim != 2 or A.shape[0] < 2 or A.shape[1] != 3:
        print("Acceleration data must be a 2D array with at least 2 rows and 3 columns.")
        return

    # Check if sampling rate is provided
    fs = dat.get('fs')
    if fs is None:
        print("The sampling rate 'fs' must be defined in the dictionary.")
        return

    # Compute the norm-jerk
    jerk = np.sqrt(np.sum(np.diff(A, axis=0)**2, axis=1)) * fs
    jerk = np.concatenate((jerk, [0]))  # Append 0 at the end

    # Add the 'jerk' key to the prh dictionary
    dat['jerk'] = jerk
    return dat

def smooth_jerk(dat, method='moving_average', window_size=5, sigma=1):
    """
    Smooths the jerk data in the prh dictionary and adds it as a new key 'sjerk' for smoothed jerk.
    Plots both the original and smoothed jerk data for comparison.

    Parameters:
    - prh: Dictionary containing biologging data.
    - method: Smoothing method ('moving_average' or 'gaussian').
    - window_size: Window size for moving average (default: 5).
    - sigma: Standard deviation for Gaussian filter (default: 1).
    """
    # Check if jerk data is available
    if 'jerk' not in dat:
        print("Jerk data is not present in the prh dictionary.")
        return

    # Ensure jerk is an array (not a callable)
    jerk_data = np.array(dat['jerk'])

    if method == 'moving_average':
        # Apply a simple moving average
        smoothed_jerk = np.convolve(jerk_data, np.ones(window_size) / window_size, mode='same')
        print(f"Applied moving average with window size {window_size}.")
    elif method == 'gaussian':
        # Apply a Gaussian filter for smoothing
        smoothed_jerk = gaussian_filter1d(jerk_data, sigma=sigma)
        print(f"Applied Gaussian filter with sigma {sigma}.")
    else:
        print("Invalid smoothing method selected.")
        return

    # Plotting the original and smoothed jerk data
    time = np.arange(len(jerk_data)) / dat['fs']  # Time axis based on the sampling rate

    plt.figure(figsize=(10, 6))
    plt.plot(time, jerk_data, label='Original Jerk', color='red', alpha=0.6)
    plt.plot(time, smoothed_jerk, label='Smoothed Jerk', color='blue', alpha=0.8)

    plt.title('Comparison of Original and Smoothed Jerk Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk')
    plt.legend()
    plt.grid(True)
    #plt.savefig(os.path.join(saveloc, 'smoothedjerk.png'))
    #plt.show(block = False)
    #time.sleep(1)
    # Add the smoothed jerk data as a new key 'sjerk' in the prh dictionary
    dat['sjerk'] = smoothed_jerk
    return dat

def check_jerkindices(rolls, dat):
    for idx, event in rolls.iterrows():
        # Check if jSig is a string and convert it to a list of integers
        if isinstance(event['jSig'], str):
            if event['jSig'].strip() == "[]":  # Handle empty lists represented as "[]"
                rolls.at[idx, 'jSig'] = []
            else:
                # Convert string representation of list to actual list of integers
                try:
                    rolls.at[idx, 'jSig'] = [int(x) for x in event['jSig'].strip('[]').split(',')]
                except ValueError:
                    print(f"Warning: Could not convert jSig at index {idx} to list. Skipping.")
                    continue

    fig, (ax_depth, ax_roll, ax_jerk) = plt.subplots(3, 1, figsize=(12, 12))

    def plot_roll_data(roll):
        stime = roll['sI']
        etime = roll['eI']

        jerk_data = dat['sjerk'][stime:etime]
        depth_data = dat['p'][stime:etime]
        roll_data = dat['roll'][stime:etime]
        time_data = np.arange(stime, etime)

        ax_depth.clear()
        ax_depth.plot(time_data, depth_data, color='g')
        ax_depth.set_ylabel('Depth (m)', color='g')
        ax_depth.tick_params(axis='y', labelcolor='g')
        ax_depth.invert_yaxis()

        ax_roll.clear()
        ax_roll.plot(time_data, roll_data, color='b')
        ax_roll.set_ylabel('Roll (degrees)', color='b')
        ax_roll.tick_params(axis='y', labelcolor='b')

        ax_jerk.clear()
        ax_jerk.plot(time_data, jerk_data, label="Jerk")
        ax_jerk.set_title(f"Select jerk indices for feed event ({stime} - {etime})")
        ax_jerk.set_xlabel("Time")
        ax_jerk.set_ylim([0, 4])
        ax_jerk.set_ylabel("Jerk")

        # Plot existing jSig values within range
        if 'jSig' in roll and roll['jSig']:  # Skip if jSig doesn't exist or is empty
            for idx in roll['jSig']:
                if stime <= idx <= etime:
                    ax_jerk.plot(idx, jerk_data[idx - stime], 'rx')

        plt.draw()

    def on_click(event):
        if event.inaxes == ax_jerk:
            closest_index = int(event.xdata)
            current_roll = rolls.iloc[current_roll_index]
            if current_roll['sI'] <= closest_index <= current_roll['eI']:
                if closest_index not in current_roll['jSig']:
                    current_roll['jSig'].append(closest_index)
                    ax_jerk.plot(closest_index, dat['sjerk'][closest_index], 'rx')
                    plt.draw()
                else:
                    current_roll['jSig'].remove(closest_index)
                    plot_roll_data(current_roll)
                    print(f"Removed jerk index: {closest_index}")

    def on_key(event):
        nonlocal current_roll_index
        if event.key == 'enter':
            current_roll_index += 1
            if current_roll_index < len(rolls):
                # Ensure we're looking at valid jSig indices
                current_roll = rolls.iloc[current_roll_index]
                while current_roll_index < len(rolls) and not any(
                        idx < current_roll['sI'] or idx > current_roll['eI']
                        for idx in current_roll.get('jSig', [])
                ):
                    current_roll_index += 1
                    if current_roll_index < len(rolls):
                        current_roll = rolls.iloc[current_roll_index]  # Update current_roll after incrementing

                if current_roll_index < len(rolls):
                    plot_roll_data(current_roll)
            else:
                plt.close()
                print("All rolls processed.")
    # Check if any jSig value is outside the sI and eI range before displaying the plot
    def has_outside_jsig(roll):
        return any(idx < roll['sI'] or idx > roll['eI'] for idx in roll.get('jSig', []))

    # Connect event handlers
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    current_roll_index = 0

    # Start by finding the first roll with jSig values outside the sI-eI range
    while current_roll_index < len(rolls) and not has_outside_jsig(rolls.iloc[current_roll_index]):
        current_roll_index += 1
        #print(current_roll_index)

    if current_roll_index < len(rolls):
        plot_roll_data(rolls.iloc[current_roll_index])

    plt.show()
    # Disconnect event handlers
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    return rolls

def refine_jerkindices(rolls, dat):
    fs = dat['fs']  # Sampling frequency
    window_size = int(0.5 * fs)  # ±0.25 seconds window

    original_values = []
    updated_values = []

    # Convert jSig strings to lists (if necessary)
    for idx, event in rolls.iterrows():
        # Check if jSig is a string and convert it to a list of integers
        if isinstance(event['jSig'], str):
            if event['jSig'].strip() == "[]":  # Handle empty lists represented as "[]"
                rolls.at[idx, 'jSig'] = []
            else:
                # Convert string representation of list to actual list of integers
                try:
                    rolls.at[idx, 'jSig'] = [int(x) for x in event['jSig'].strip('[]').split(',')]
                except ValueError:
                    print(f"Warning: Could not convert jSig at index {idx} to list. Skipping.")
                    continue

        # Now process the jSig list for jerk refinement
        if isinstance(rolls.at[idx, 'jSig'], list) and len(rolls.at[idx, 'jSig']) > 0:
            updated_jSig = []
            for j_index in rolls.at[idx, 'jSig']:
                start_idx = max(0, j_index - window_size)
                end_idx = min(len(dat['sjerk']), j_index + window_size + 1)

                # Extract the smoothed jerk data within the window
                jerk_window = dat['sjerk'][start_idx:end_idx]

                if len(jerk_window) == 0:
                    print(f"Warning: Jerk window is empty for event index {idx}")
                    continue

                # Find the max value in this window
                max_idx_relative = np.argmax(jerk_window)
                max_idx = start_idx + max_idx_relative

                if dat['sjerk'][max_idx] > dat['sjerk'][j_index]:
                    updated_jSig.append(max_idx)
                    original_values.append(dat['sjerk'][j_index])
                    updated_values.append(dat['sjerk'][max_idx])
                else:
                    updated_jSig.append(j_index)
                    original_values.append(dat['sjerk'][j_index])
                    updated_values.append(dat['sjerk'][j_index])

            # Update the 'jSig' column in the dataframe
            rolls.at[idx, 'jSig'] = updated_jSig

    # Plotting original vs updated values
    plt.figure(figsize=(10, 5))
    for i in range(len(original_values)):
        plt.plot([i, i], [original_values[i], updated_values[i]], 'k-')
    plt.plot(original_values, 'ro', label='Original')
    plt.plot(updated_values, 'bx', label='Updated')

    plt.title('Original vs Updated Jerk Indices')
    plt.xlabel('Index')
    plt.ylabel('Jerk Value')
    plt.legend()
    plt.grid(True)
    #plt.savefig(os.path.join(sLoc, 'refinedjerksignals.png'))
    plt.show()

    return rolls

def load_progress(fname):
    try:
        # Read the file and skip the header row
        data = []
        with open(fname, 'r') as f:
            next(f)  # Skip the header row
            for line in f:
                line = line.strip()  # Remove whitespace and newline characters
                fields = line.split(',')

                # Parse the four main indices (sI, sFP, eFP, eI)
                sI = int(fields[0])
                sFP = int(fields[1])
                eFP = int(fields[2])
                eI = int(fields[3])

                # Parse jSig, removing brackets and splitting if there are multiple values
                jSig_str = fields[4].strip()[1:-1]  # Remove the brackets []
                if jSig_str:  # If there are jerk indices, split them into a list of ints
                    jSig = [int(x) for x in jSig_str.split(',')]
                else:
                    jSig = []  # Empty list if no jerk indices

                # Append data as a row (tuple) to the list
                data.append((sI, sFP, eFP, eI, jSig))

        # Convert the data list to a DataFrame
        df = pd.DataFrame(data, columns=['sI', 'sFP', 'eFP', 'eI', 'jSig'])
        print(f"Successfully loaded data into DataFrame from {fname}.")
    except Exception as e:
        print(f"Error reading progress: {e}")

    return df

def load_feeddat(fname):
    """Load feeding events from a text file."""
    rolls = []
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) >= 5:  # Check if the line has enough parts
                    newrolls = {
                        'sI': int(parts[0]),
                        'sFP': int(parts[1]),
                        'eFP': int(parts[2]),
                        'eI': int(parts[3]),
                        'jSig': [int(idx) for idx in parts[4].strip('[]').split()]  # Convert string list to integers
                    }
                    rolls.append(newrolls)
        print(f"Loaded {len(rolls)} feeding events from {fname}.")
    else:
        print(f"No feeding events file found at {fname}. Starting fresh.")
    return rolls

def save_progress(rolls, dID):
    """
    Save the rolls data to a text file.

    Parameters:
    - rolls: Dictionary containing feeding event data.
    - fname: The name of the file to save the data to (default is 'feedevents_progress.txt').
    """
    # Convert rolls dictionary to DataFrame
    fname = os.getcwd()+"//feed//progress-feedevents_"+dID+".csv"

    # Save the DataFrame to a text file
    rolls.to_csv(fname, index=False, sep=',')  # Use tab as the delimiter

    print(f"Progress saved to {fname}")

def do_feedaudit(dat, rolls, tagonind, tagoffind, dID, increment_minutes=5):
    """
    Plots roll, depth, and smoothed jerk data for feeding events at specified time increments with fixed y-axis ranges.

    Parameters:
    - dat: Dictionary containing 'roll', 'p', and optionally 'jerk' and 'sjerk' data streams.
    """
    # Constants
    sampling_rate = int(dat['fs'])
    increment_seconds = increment_minutes * 60
    time = np.arange(len(dat['roll'])) / sampling_rate

    # If feeding events exist, start after the last one
    is_empty = len(rolls['eI'])==0
    if is_empty:
        # if the dataframe is empty, you should start at the tagonind
        current_start_index = tagonind
    else:
        current_start_index = rolls['eI'].iloc[-1] + 1  # Get the last value from the list in the 'eI' key
        if current_start_index >= tagoffind:
            print("No new data to audit after the last feeding event. Exiting plot.")
            return



    # Variables for managing current index and zoom level
    selection_mode = False
    selected_indices = []
    jerk_selection_mode = False
    jerk_indices = []

    # Manual y-limits for each plot
    roll_ylim = (-10, 180)
    jerk_ylim = (0, 4)

    # Set up the figure and axes
    fig, (ax_roll, ax_depth, ax_jerk, ax_pitch) = plt.subplots(4, 1, figsize=(12, 12))

    def plot_data(start_index):
        end_index = min(start_index + increment_seconds * sampling_rate, len(dat['roll']))

        ax_roll.clear()
        ax_roll.plot(time[start_index:end_index], dat['roll'][start_index:end_index], label='Roll (degrees)', color='b')
        ax_roll.set_xlabel('Time (s)')
        ax_roll.set_ylabel('Roll (degrees)', color='b')
        ax_roll.tick_params(axis='y', labelcolor='b')
        ax_roll.set_ylim(roll_ylim)  # Apply manual y-limits for roll

        # Plot selected roll points
        if selected_indices:
            selected_times = [time[idx] for idx in selected_indices if start_index <= idx < end_index]
            ax_roll.plot(selected_times, dat['roll'][selected_indices], 'rx', markersize=10, label='Selected Points')

        ax_depth.clear()
        ax_depth.plot(time[start_index:end_index], dat['p'][start_index:end_index], label='Depth (m)', color='g')
        ax_depth.set_ylabel('Depth (m)', color='g')
        ax_depth.tick_params(axis='y', labelcolor='g')
        ax_depth.invert_yaxis()  # Invert y-axis for depth

        if 'sjerk' in dat or 'jerk' in dat:
            ax_jerk.clear()
            jerk_data = dat['sjerk'] if 'sjerk' in dat else dat['jerk']
            ax_jerk.plot(time[start_index:end_index], jerk_data[start_index:end_index], label='Smoothed Jerk', color='r')
            ax_jerk.set_ylabel('Smoothed Jerk', color='r')
            ax_jerk.tick_params(axis='y', labelcolor='r')
            ax_jerk.set_ylim(jerk_ylim)  # Apply manual y-limits for smoothed jerk

            # Plot selected jerk indices only if valid
            if jerk_indices:
                valid_jerk_indices = [idx for idx in jerk_indices if start_index <= idx < end_index]
                if valid_jerk_indices:
                    selected_jerk_times = [time[idx] for idx in valid_jerk_indices]
                    ax_jerk.plot(selected_jerk_times, jerk_data[valid_jerk_indices], 'yx', markersize=10,
                                 label='Selected Jerk Indices')
        ax_pitch.clear()
        ax_pitch.plot(time[start_index:end_index], dat['pitch'][start_index:end_index], label='Pitch (rad)', color='g')
        ax_pitch.set_ylabel('Pitch (rad)', color='r')
        ax_pitch.tick_params(axis='y', labelcolor='g')

        plt.suptitle(f'Biologging Data Inspection - Time {time[start_index]:.1f} to {time[end_index - 1]:.1f}')
        plt.legend()

        # Display keystroke instructions on the plot
        ax_roll.text(0.5, 1.05,
                     "Instructions: Press 'Enter' to select points, 'n' for next, 'b' for previous, "
                     "'Esc' to exit, '-' to decrease increment, '+' to increase increment, "
                     "'j' to select jerk indices, 'r' to reset increment, 'd' to delete previous selection",
                     ha='center', va='bottom', fontsize=10, color='red', transform=ax_roll.transAxes)
        plt.draw()

    def delete_last_selected_point():
        """Delete the last selected point from the selection lists."""
        if selected_indices:
            removed_index = selected_indices.pop()
            print(f"Deleted last selected roll index: {removed_index}")
        elif jerk_indices:
            removed_jerk_index = jerk_indices.pop()
            print(f"Deleted last selected jerk index: {removed_jerk_index}")

    # Function to capture click events for feeding positions
    def onclick(event):
        """Capture click events for feeding positions."""
        nonlocal rolls, selected_indices, jerk_indices
        if event.xdata is not None:
            ix = int(event.xdata * sampling_rate)

            if selection_mode:
                selected_indices.append(ix)
                print(f"Selected roll index: {ix}")
                plot_data(current_start_index)  # Update the plot to show selected points

                # Store feeding event if four points are selected
                if len(selected_indices) == 4:
                    # Create a dictionary to store the feeding event details
                    # feeding_event = {
                    #     'sI': selected_indices[0],
                    #     'sFP': selected_indices[1],
                    #     'eFP': selected_indices[2],
                    #     'eI': selected_indices[3],
                    #     'jSig': jerk_indices[:]  # Store jerk indices if selected
                    # }
                    feed_event = [{'sI':selected_indices[0], 'sFP': selected_indices[1], 'eFP': selected_indices[2], 'eI': selected_indices[3],'jSig': jerk_indices[:]}]
                    feed_event = pd.DataFrame(feed_event)

                    print("Feeding event captured")
                    rolls = pd.concat([rolls, feed_event], axis=0)
                    # Save progress after capturing feeding event
                    save_progress(rolls, dID)
                    print("Saved progress... Resetting...")
                    selected_indices.clear()  # Reset for the next event
                    jerk_indices.clear()  # Clear jerk selections for the next event

            elif jerk_selection_mode:
                jerk_indices.append(ix)
                print(f"Selected jerk index: {ix}")
                plot_data(current_start_index)  # Update the plot to show selected jerk indices

    # Event handler for key presses
    def on_key(event):
        nonlocal current_start_index, selection_mode, increment_seconds, jerk_selection_mode
        if event.key == 's':  # Save progress and exit
            plt.close()  # Close the plot
        elif event.key == 'enter':
            selection_mode = True
            jerk_selection_mode = False
            print("Entered point selection mode. Click on the plot to select roll points.")
            plot_data(current_start_index)
        elif event.key == 'j':  # Select jerk indices
            jerk_selection_mode = True
            selection_mode = False
            print("Entered jerk index selection mode. Click on the jerk plot to select indices.")
            plot_data(current_start_index)
        elif event.key == 'd':  # Delete last selected point
            delete_last_selected_point()  # Call the delete function
            plot_data(current_start_index)  # Update the plot to reflect changes
        elif event.key == 'n':  # Next increment
            current_start_index += increment_seconds * sampling_rate
            if current_start_index >= tagoffind:
                print("Reached the end of data.")
                plt.close()
            else:
                plot_data(current_start_index)
        elif event.key == 'b':  # Previous increment
            current_start_index -= increment_seconds * sampling_rate
            if current_start_index < 0:
                current_start_index = 0
            plot_data(current_start_index)
        elif event.key == '-':  # Decrease increment
            increment_seconds = max(10, increment_seconds - 10)
            plot_data(current_start_index)
        elif event.key == '+':  # Increase increment
            increment_seconds += 10
            plot_data(current_start_index)
        elif event.key == 'r':  # Reset increment
            increment_seconds = 10 * 60
            print("Increment reset to 10 minutes.")
            plot_data(current_start_index)
        elif event.key == 'escape':  # Exit the plot
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', onclick)
    print(current_start_index)
    # Initial plot
    plot_data(current_start_index)

    plt.show()
    return rolls

def check_dups(df, dat):
    def find_overlaps(df):
        overlaps = []
        sorted_df = df.sort_values('sI')
        for i in range(len(sorted_df) - 1):
            if sorted_df.iloc[i]['eI'] > sorted_df.iloc[i + 1]['sI']:
                overlaps.append(sorted_df.index[i:i + 2].tolist())
        return overlaps

    overlapping_groups = find_overlaps(df)

    if not overlapping_groups:
        print("No overlapping events found.")
        return df

    def on_key(event):
        nonlocal choice
        if event.key in ['1', '2', 'r', 'n']:
            choice = event.key
            plt.close()

    for group in overlapping_groups:
        choice = None

        # Create the figure and axes
        fig, axs = plt.subplots(len(group), 1, figsize=(12, 10), sharex=True)

        min_si = min(df.loc[group, 'sI'])
        max_ei = max(df.loc[group, 'eI'])

        for i, idx in enumerate(group):
            row = df.loc[idx]
            si, ei = int(row['sI']), int(row['eI'])
            si, ei = row['sI'], row['eI']

            roll_data = dat['roll'][si:ei + 1]

            x = np.arange(si, ei + 1)
            axs[i].plot(x, roll_data)
            axs[i].set_title(f"Event {i + 1} (Index: {idx})")
            axs[i].set_ylabel("Roll")

        axs[-1].set_xlabel("Index")
        plt.xlim(min_si, max_ei)
        fig.suptitle("Overlapping events")

        keystroke_text = (
            "Keystrokes:\n"
            "1: Keep Event 1\n"
            "2: Keep Event 2\n"
            "r: Remove both events\n"
            "n: Keep both events\n"
        )
        fig.text(0.98, 0.98, keystroke_text, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        while choice is None:
            plt.pause(0.1)

        if choice == 'r':
            df = df.drop(group)
        elif choice in ['1', '2']:
            to_keep = int(choice) - 1
            df = df.drop([idx for i, idx in enumerate(group) if i != to_keep])
        elif choice == 'n':
            print("keeping both")

    print("All overlapping events have been processed.")
    return df

def accept_events(rolls, dat, dID):
    # Create lists to store the indices of rows to keep and reject
    kept_indices = []
    rejected_indices = []

    for index, row in rolls.iterrows():

        # Extract start and end indices for the current event
        start_index = row['sI']
        end_index = row['eI']

        # Check if indices are within the bounds of the dat dictionary
        if start_index < 0 or end_index >= len(dat['p']):
            print(f"Invalid indices for row {index}: start={start_index}, end={end_index}")
            continue

        # Clear the previous plot
        #plt.clf()

        # Generate your plot for the current row using the dat dictionary
        # Plot Depth
        plt.subplot(3, 1, 1)
        plt.plot(dat['p'][start_index:end_index], label='Depth')
        plt.gca().invert_yaxis()  # Invert y-axis for depth
        plt.title('Depth')
        plt.ylabel('Depth (m)')

        # Plot Roll
        plt.subplot(3, 1, 2)
        plt.plot(dat['roll'][start_index:end_index], label='Roll', color='orange')
        plt.axvline(x=row['sFP']-start_index, color='r', linestyle='--', label='sFP')
        plt.axvline(x=row['eFP']-start_index, color='orange', linestyle='--', label='eFP')
        plt.legend()
        plt.title('Roll')
        plt.ylabel('Roll (degrees)')
        plt.legend()

        # Plot Jerk
        plt.subplot(3, 1, 3)
        plt.plot(dat['sjerk'][start_index:end_index], label='Jerk', color='green')

        jSig_indices = row['jSig']
        for j in jSig_indices:
            if start_index <= j < end_index:  # Ensure the index is within the valid range
                plt.plot(j-start_index, dat['sjerk'][j], 'ro', markersize=5)  # Plot red circles for jSig

        plt.title('Jerk')
        plt.ylabel('Jerk')
        plt.xlabel('Time')

        plt.suptitle(f'Row Index: {index}')
        plt.pause(0.005)  # Allow the plot to be updated

        # Get user input to keep or remove the row
        key = input("Press 1 to keep, 2 to remove, or r to skip: ")

        if key == '1':
            kept_indices.append(index)  # Keep the row
            # Save the plot for accepted events
            plt.savefig(os.getcwd()+f'\\feed\\plots\\accepted-events\\{dID}_accepted-row-{index}.png')
        elif key == '2':
            rejected_indices.append(index)  # Record the rejected index
            # Save the plot for rejected events
            plt.savefig(os.getcwd()+f'\\feed\\plots\\rejected-events\\{dID}_rejected-row-{index}.png')
        elif key == 'r':
            print(f"Row {index} skipped.")
        else:
            print("Invalid input. Please use 1, 2, or r.")

        plt.close()  # Close the plot for the next iteration

    # Save the rejected rolls as a text file
    rejected_rolls = rolls.loc[rejected_indices]
    rejected_rolls.to_csv((os.getcwd()+"\\feed\\rejected-events\\rejected-rolls_"+dID+".csv"), index=False)
    print(f"Rejected rolls saved...")

    # Return the kept DataFrame
    return rolls.loc[kept_indices]

def analyze_feed(rolls, dat, stides, depID):
    rolls = rolls.sort_values(by=['sI'], ascending=True)
    rolls['depID'] = depID
    rolls['CRCID'] = input('Enter the CRCID number:')
    rolls['whalename'] = input('Enter the whales name:')
    rolls['depInd'] = range(1, len(rolls) + 1)

    for index, row in rolls.iterrows():
        sI = row['sI']
        eI = row['eI']
        sFP = row['sFP']
        eFP = row['eFP']

        # Extract relevant data slices
        depth_data = dat['p'][sFP:eFP]
        roll_data = dat['roll'][sFP:eFP]
        pitch_data = dat['pitch'][sFP:eFP]
        head_data = dat['head'][sI:eI]


        # Calculate durations
        rolls.loc[index,'durEntFP_s'] = (sFP - sI) / dat['fs']  # Time to enter feeding position
        rolls.loc[index,'durExFP_s'] = (eI - eFP) / dat['fs']  # Time to leave feeding position
        rolls.loc[index,'durFE_s'] = (eI - sI) / dat['fs']  # Total feeding event duration
        rolls.loc[index,'durFP_s'] = (eFP - sFP) / dat['fs']  # Duration in feeding position

        # Last and next event durations
        if index > 0:
            rolls.loc[index,'lastFE_s'] = (sI - rolls['eI'].iloc[index - 1]) / dat['fs'] # Time since last event
        else:
            rolls.loc[index,'lastFE_s'] = np.nan  # First event

        if index < len(rolls) - 1:
            rolls.loc[index,'nextFE_s'] = (rolls['sI'].iloc[index + 1]- eI)/ dat['fs']  # Time to next event
        else:
            rolls.loc[index,'nextFE_s'] = np.nan  # Last event

        # Date number and date string
        rolls.loc[index,'startDN'] = dat['DN'][sI]
        rolls.loc[index,'startDT'] = pd.to_datetime(dat['DT'][sI])
        rolls.loc[index,'tidalheight_ft'] = dat['tidalheight'][sI]
        rolls.loc[index,'tidalrate'] = dat['tiderate'][sI]
        rolls.loc[index,'inout'] = dat['tidedirection'][sI]

        rollDT = pd.to_datetime(dat['DT'][sI])

        for i, r in stides.iterrows():
            if r['startTD'] < rollDT < r['endTD']:
                cInd = i
                break
            else:
                continue


        rolls.loc[index,'nTD'] = stides['nTD'][cInd]
        if rollDT < stides['startTP2'][cInd]:
            rolls.loc[index,'nTP'] = 1
        else:
            rolls.loc[index,'nTP'] = 2

        rolls.loc[index,'tideType'] = stides['tideType'][cInd]

        rolls.loc[index,'durTD'] = stides['durTD'][cInd]
        rolls.loc[index,'fullTD'] = stides['fullTD'][cInd]
        if stides['fullTP1'][cInd] == 'Y' or stides['fullTP2'][cInd] == 'Y':
            rolls.loc[index,'fullTP'] = 'Y'
        else:
            rolls.loc[index,'fullTP'] = 'N'

        rolls.loc[index,'durTP1'] = stides['durTP1'][cInd]
        rolls.loc[index,'durTP2'] = stides['durTP2'][cInd]


        # Calculate depth statistics
        if depth_data.size > 0:
            rolls.loc[index,'meanDepth_m'] = np.mean(depth_data)
            rolls.loc[index,'stdDepth_m'] = np.std(depth_data)
        else:
            rolls.loc[index,'meanDepth_m'] = np.nan
            rolls.loc[index,'stdDepth_m'] = np.nan

        # Calculate roll statistics
        if roll_data.size > 0:
            rolls.loc[index,'meanRoll_deg'] = np.mean(roll_data)
            rolls.loc[index,'minRoll_deg'] = np.min(roll_data)
            rolls.loc[index,'maxRoll_deg'] = np.max(roll_data)
            rolls.loc[index,'rngRoll_deg'] = np.max(roll_data) - np.min(roll_data)
            rolls.loc[index,'stdRoll_deg'] = np.std(roll_data)
        else:
            rolls.loc[index,'meanRoll_deg'] = np.nan
            rolls.loc[index,'minRoll_deg'] = np.nan
            rolls.loc[index,'maxRoll_deg'] = np.nan
            rolls.loc[index,'rngRoll_deg'] = np.nan
            rolls.loc[index,'stdRoll_deg'] = np.nan

        # Calculate pitch statistics
        if pitch_data.size > 0:
            rolls.loc[index,'meanPitchFP'] = np.mean(pitch_data)
            rolls.loc[index,'rangePitchFP'] = np.max(pitch_data) - np.min(pitch_data)
            rolls.loc[index,'stdPitchFP'] = np.std(pitch_data)
        else:
            rolls.loc[index,'meanPitchFP'] = np.nan
            rolls.loc[index,'rangePitchFP'] = np.nan
            rolls.loc[index,'stdPitchFP'] = np.nan

        # Calculate heading during the event
        rolls.loc[index,'meanHead'] = np.mean(head_data) if head_data.size > 0 else np.nan

        # Camera status
        rolls.loc[index,'camon'] = int(np.any(dat['camon'][sI:eI] == 1))

        vidDT_array = pd.to_datetime(dat['vidDT'])
        # Video index and name
        vidInd = np.where(vidDT_array < rolls.loc[index,'startDT'])[0]

        if vidInd.size > 0:
            # If there are valid indices, get the last valid video index
            rolls.loc[index,'vidInd'] = vidInd[-1]  # Index of the last valid video
            rolls.loc[index,'vidNam'] = dat['vidNam'][vidInd[-1]]
        else:
            # If no valid index found, assign NaN and None
            rolls.loc[index,'vidInd'] = np.nan
            rolls.loc[index,'vidNam'] = None

    return rolls

def calc_InOutRates(rolls):
    # Initialize lists to store the results
    feeding_rates_in = []
    feeding_rates_out = []
    num_events_in = []
    num_events_out = []

    # Group by tidal cycle number (numTC)
    grouped = rolls.groupby('numTC')

    for name, group in grouped:
        # Filter for feeding events in each tide direction
        in_events = group[group['inout'] == 'in']
        out_events = group[group['inout'] == 'out']

        # Get the duration of the current tidal cycle (in decimal days)
        duration_in = group['durIn'].iloc[0] if not in_events.empty else 0
        duration_out = group['durOut'].iloc[0] if not out_events.empty else 0

        # Count the number of events for each direction
        count_in = len(in_events)
        count_out = len(out_events)

        # Calculate the feeding rate for each tidal cycle
        rate_in = count_in / duration_in if duration_in > 0 else 0
        rate_out = count_out / duration_out if duration_out > 0 else 0

        # Store the results
        feeding_rates_in.append(rate_in)
        feeding_rates_out.append(rate_out)
        num_events_in.append(count_in)
        num_events_out.append(count_out)

    # Create a summary dataframe to store the results
    summary_df = pd.DataFrame({
        'numTC': grouped.groups.keys(),
        'events_in': num_events_in,
        'feeding_rate_in': feeding_rates_in,
        'events_out': num_events_out,
        'feeding_rate_out': feeding_rates_out
    })

    return summary_df

def add_tides(dat, tdat, sloc):
    # GET TIDAL DATA, ADD DIRECTION OF TIDES
    tdat['Date Time'] = pd.to_datetime(tdat['DT'])
    def get_inout2(tdata):
        # Extract the predicted heights
        predictions = tdata['Prediction']
        directions = []
        # Iterate through the predictions to determine tide direction
        for i in range(len(predictions) - 1):
            current_value = float(predictions[i])
            next_value = float(predictions[i+1])
            if current_value > next_value:
                directions.append('out')
            elif current_value < next_value:
                directions.append('in')
            else:
                directions.append(directions[-1])

        directions.append(directions[-1])
        tdata['direction'] = directions
        return tdata

    def get_inout(tdata):
        # Extract the predicted heights
        predictions = tdata['Prediction']
        directions = []

        # Iterate through the predictions to determine tide direction
        for i in range(len(predictions) - 1):
            current_value = float(predictions[i])
            next_value = float(predictions[i + 1])
            if current_value > next_value:
                directions.append('out')
            elif current_value < next_value:
                directions.append('in')
            else:
                # Check if directions list has any previous value, otherwise default to 'in'
                directions.append(directions[-1] if directions else 'in')

        # Append the last known direction for consistency
        directions.append(directions[-1] if directions else 'in')
        tdata['direction'] = directions
        return tdata

    def get_tiderate(tdata):
        predictions = tdata['Prediction']
        trates = abs(np.diff(predictions, prepend=predictions[0]))
        tdata['rate'] = trates
        return tdata
    tdat = get_tiderate(tdat)
    tdat = get_inout(tdat)

    # GET PRH DATETIMES AND MAKE SURE BOTH ARE IN CORRECT ORDER BEFORE MERGING
    prhDT = pd.DataFrame({'DT': pd.to_datetime(dat['DT'])})
    prhDT = prhDT.sort_values('DT')
    tdat = tdat.sort_values('Date Time')
    # MERGE TIDAL AND PRH DATA BASED ON PRH DATETIMES, SAVE THIS INTO A CSV
    merged = pd.merge_asof(prhDT, tdat[['Date Time', 'Prediction', 'direction', 'rate']], left_on='DT', right_on='Date Time',
                           direction='nearest')
    merged = merged.sort_values(by='DT').reset_index(drop=True)  # Ensure order
    merged = merged.drop(columns=['Date Time'])
    dat['tidalheight'] = merged['Prediction'].values
    dat['tidedirection'] = merged['direction'].values
    dat['tiderate'] = merged['rate'].values

    merged.to_csv(sloc, index=False)
    return dat

def make_DT(datenums):
    # Ensure the input is a NumPy array
    datenums = np.array(datenums)

    # MATLAB's datenum offset to Python's datetime
    datenum_offset = 366

    # Prepare lists for datetime conversion
    python_datetimes = []

    for matlab_datenum in datenums:
        if np.isnan(matlab_datenum):  # Check for MATLAB NaN (same as np.nan)
            python_datetimes.append(np.nan)  # Append np.nan for missing values
        else:
            whole_part = int(np.floor(matlab_datenum))
            fractional_part = matlab_datenum % 1
            # Convert using fromordinal and adjust for fractional part
            dt_obj = datetime.fromordinal(whole_part) + timedelta(days=fractional_part) - timedelta(days=datenum_offset)
            python_datetimes.append(dt_obj)

    return python_datetimes

def analyze_tides(dat, stides, tagonDT, tagoffDT):
    def get_TideDurs(stides,tagonDT, tagoffDT):
        stides['startTD'] = pd.to_datetime(stides['startTD'])
        stides['endTD'] = pd.to_datetime(stides['endTD'])
        stides['startTP2'] = pd.to_datetime(stides['startTP2'])
        stides['timeTP1HT'] = pd.to_datetime(stides['timeTP1HT'])
        stides['timeTP2HT'] = pd.to_datetime(stides['timeTP2HT'])

        dursTC = []
        dursTP1 = []
        dursTP2 = []

        for j in range(len(stides)):
            onoff = int(stides['tagOnOff'][j])

            # if the tag was put on during this tidal cycle
            if onoff == 1:
                durThis = stides['endTD'][j] - tagonDT
                dursTC.append(durThis)
                # find out which tidal period the tag was put on
                if tagonDT < stides['startTP2'][j]:
                    durP1 = stides['startTP2'][j] - tagonDT
                    durP2 = stides['endTD'][j] - stides['startTP2'][j]
                else:
                    durP1 = np.nan
                    durP2 = stides['endTD'][j] - tagonDT
                dursTP1.append(durP1)
                dursTP2.append(durP2)

            # if the tag was not put on or off during this cycle
            elif onoff == 0:
                durThis = stides['endTD'][j] - stides['startTD'][j]
                durP1 = stides['startTP2'][j] - stides['startTD'][j]
                durP2 = stides['endTD'][j] - stides['startTP2'][j]
                dursTC.append(durThis)
                dursTP1.append(durP1)
                dursTP2.append(durP2)

            elif onoff == 2:
                durThis = tagoffDT - stides['startTD'][j]
                dursTC.append(durThis)
                # if the tag came off before high tide
                if tagoffDT < stides['startTP2'][j]:
                    durP1 = tagoffDT - stides['startTD'][j]
                    durP2 = np.nan
                # if the tag came off after ht
                else:
                    durP1 = stides['startTP2'][j] - stides['startTD'][j]
                    durP2 = tagoffDT - stides['startTP2'][j]
                dursTP1.append(durP1)
                dursTP2.append(durP2)

            # tag was put on and came off during this cycle
            elif onoff == 3:
                durThis = tagoffDT - tagonDT
                dursTC.append(durThis)
                # if the tag was put on before ht and came off after...
                if tagonDT < stides['startTP2'][j] < tagoffDT:
                    durP1 = stides['startTP2'][j] - tagonDT
                    durP2 = tagoffDT - stides['startTP2'][j]
                # if the tag was put on and came off before ht
                elif tagonDT < stides['startTP2'][j] and tagoffDT < stides['startTP2'][j]:
                    durP1 = tagoffDT - tagonDT
                    durP2 = np.nan
                # if the tag was put on and came off after HT
                elif tagonDT > stides['startTP2'][j] and tagoffDT > stides['startTP2'][j]:
                    durP2 = tagoffDT - tagonDT
                    durP1 = np.nan

                dursTP1.append(durP1)
                dursTP2.append(durP2)

        stides['durTD'] = dursTC
        stides['durTP1'] = dursTP1
        stides['durTP2'] = dursTP2
        try:
            stides['durTD'] = stides['durTD'].apply(lambda td: td.total_seconds() / 86400)
            stides['durTP1'] = stides['durTP1'].apply(lambda td: td.total_seconds() / 86400)
            stides['durTP2'] = stides['durTP2'].apply(lambda td: td.total_seconds() / 86400)
        except AttributeError:
            dursTC = pd.to_timedelta(dursTC, unit='D')
            dursTP1 = pd.to_timedelta(dursTP1, unit='D')
            dursTP2 = pd.to_timedelta(dursTP2, unit='D')
            stides['durTD'] = dursTC
            stides['durTP1'] = dursTP1
            stides['durTP2'] = dursTP2

            # stides['durTD'] = [timedelta(days=day) for day in stides['durTD']]
            # stides['durTP1'] = [timedelta(days=day) for day in stides['durTP1']]
            # stides['durTP2'] = [timedelta(days=day) for day in stides['durTP2']]
        #except ValueError:


        return stides

    def get_prhIndices(dat,stides, tagonDT, tagoffDT):
        sTDind = []
        eTDind = []
        sTP2ind = []
        dat = {'DT': pd.to_datetime(dat['DT'])}
        for j in range(len(stides)):
            if stides['startTD'][j] < tagonDT:
                sTDind.append(None)
            else:
                cTime = stides['startTD'][j]
                diffs = abs(dat['DT'] - cTime)
                sTDind.append(diffs.argmin())
            # if high tide happens before the tag was put on or after it was off, there is no index
            if stides['startTP2'][j] < tagonDT or stides['startTP2'][j] > tagoffDT:
                sTP2ind.append(None)
            else:
                cTime = stides['startTP2'][j]
                diffs = abs(dat['DT'] - cTime)
                sTP2ind.append(diffs.argmin())

            if stides['endTD'][j] > tagoffDT:
                eTDind.append(None)
            else:
                cTime = stides['endTD'][j]
                diffs = abs(dat['DT'] - cTime)
                eTDind.append(diffs.argmin())

        stides['sindTD'] = sTDind
        stides['eindTD'] = eTDind
        stides['sindTP2'] = sTP2ind
        return stides

    stides = get_TideDurs(stides,tagonDT, tagoffDT)
    stides = get_prhIndices(dat, stides, tagonDT, tagoffDT)
    return stides

def plot_tides(tides, stides, dat, tagonI, tagoffI, sloc):
    # Convert date times to pandas datetime for plotting
    #prhDT = dat['DT'][tagonI:tagoffI]
    prhTHeight = tides['Prediction'][tagonI:tagoffI].tolist()
    prhRoll = dat['roll'][tagonI:tagoffI].tolist()
    prhDepth = dat['p'][tagonI:tagoffI].tolist()

    # Create a plot
    fig, (ax_tides, ax_roll, ax_depth) = plt.subplots(3, 1, figsize=(12, 12))
    ax_tides.plot(prhTHeight, marker='o', linestyle='-', color='black', label='Tidal Height')
    # Annotate high and low tides using their indices

    def check_na(value):
        try:
            # Attempt to check if the value is NaN using notna()
            return pd.notna(value)
        except AttributeError:
            # If an AttributeError occurs, handle the case where value is a float
            # You can still check for NaN using np.isnan
            if isinstance(value, float):
                return np.isnan(value)  # This will return True if value is NaN
            else:
                return False  # Not NaN if it's not a float/int

    for index, row in stides.iterrows():
        #print(int(index))
        isNAN = check_na(row['startTD'])
        #print(isNAN)
        if isNAN:
            ax_tides.annotate('new tidal day',
                              xy=(row['sindTD']-tagonI, row['lvlLT1']),
                              xytext=(row['sindTD']-tagonI, row['lvlLT1'] + 0.5),
                              arrowprops=dict(facecolor='red', shrink=0.05),
                              fontsize=10, color='red')

        isNAN = check_na(row['startTP2'])
        #print(isNAN)
        if isNAN:
            ax_tides.annotate('new tidal period',
                              xy=(row['sindTP2']-tagonI, row['lvlLT2']),
                              xytext=(row['sindTP2']-tagonI, row['lvlLT2'] + 0.5),
                              arrowprops=dict(facecolor='green', shrink=0.05),
                              fontsize=10, color='green')

    ax_tides.set_ylabel('Tidal Height')

    ax_roll.plot( prhRoll, color='r')
    ax_roll.set_ylabel('Roll (degrees)')

    ax_depth.plot(prhDepth, color='b')
    ax_depth.set_ylabel('Depth (m)')
    ax_depth.invert_yaxis()  # Invert y-axis for depth

    plt.grid()
    plt.legend()
    fig.savefig(sloc)
    plt.show()

def enumerate_events(rolls, invar):
    # Initialize an empty list to store the enumerated indices
    indexs = []
    # Initialize the counter for each tidal cycle
    count = 1
    # Loop through the dataframe rows
    for i in range(len(rolls)):
        # If this is the first row, or the numTC value has changed
        if i == 0 or rolls.loc[i, invar] != rolls.loc[i - 1, invar]:
            count = 1  # Reset the counter
        indexs.append(count)
        count += 1  # Increment the counter for the current tidal cycle
    # Add the new column to the dataframe
    return indexs

def count_hourlyfeed(rolls, tonDT, toffDT):
    rolls['startDT'] = pd.to_datetime(rolls['startDT'])
    rolls['hour'] = rolls['startDT'].dt.floor('h')

    # Group by the hour and count the number of events
    hourly_counts = rolls.groupby('hour').size().reset_index(name='eventCount')

    # Merge the counts back to the original dataframe
    rolls = rolls.merge(hourly_counts, on='hour', how='left')

    # Check if the entire hour was covered by the deployment
    rolls['fullHourCoverage'] = rolls.apply(
        lambda row: (tonDT <= row['hour'] + pd.Timedelta(hours=1)) and
                    (toffDT >= row['hour']), axis=1
    )

    # Save the hourly rate in a new column 'hRate'
    rolls['hRate'] = rolls['eventCount'] / 1  # Since it's per hour

    # Drop the eventCount column as it's no longer needed
    rolls.drop(columns=['eventCount'], inplace=True)

    return rolls

def remove_feed15mtagoff(rolls, tonDT):
    # Ensure 'startTime' and 'tonDT' are in datetime format
    rolls['startDT'] = pd.to_datetime(rolls['startDT'])
    # Create a mask for events that occurred within 15 minutes of tonDT
    mask = (rolls['startDT'] - tonDT).dt.total_seconds() > 15 * 60
    # Filter the rolls dataframe to keep only events outside the 15-minute window
    rolls = rolls[mask]
    return rolls

def load_file(prmpt, filetype):
    layout = [
        [sg.Text(prmpt)],
        [sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("Excel Files", filetype),))],
        [sg.Button("Load"), sg.Button("Cancel")]
    ]
    window = sg.Window("Load Excel File", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Cancel":
            window.close()
            return None, None
        if event == "Load":
            fpath = values["-FILE-"]
            if fpath:
                window.close()
                try:
                    cdir = os.path.dirname(fpath)

                    # Load the MATLAB file
                    dat = pd.read_excel(fpath)

                    return dat, cdir  # Return contents and filename
                except Exception as e:
                    sg.popup_error(f"Error loading file: {str(e)}")
                    return None, None
            else:
                sg.popup_error("Please select a file.")
    window.close()

def sum_feed(sumfeed, rolls):
    def countInOut(dat):
        filtdat = dat[dat['fullInOut'] == 'Y']
        in_count = (filtdat['inout'] == "in").sum()
        out_count = (filtdat['inout'] == "out").sum()
        if in_count == 0:
            in_count = np.nan
        if out_count == 0:
            out_count = np.nan

        return in_count, out_count

    sumfeed.loc[0,'depID'] = rolls.loc[0,'depID']
    sumfeed.loc[0,'CRCID'] = rolls.loc[0,'CRCID']
    sumfeed.loc[0,'whalename'] = rolls.loc[0,'whalename']
    sumfeed.loc[0,'locFP'] = rolls['locFP'].unique().tolist()
    sumfeed.loc[0,'nFeed'] = len(rolls)
    sumfeed.loc[0,'mDurEntFP_s'] = np.mean(rolls['durEntFP_s'])
    sumfeed.loc[0,'mDurExFP_s'] = np.mean(rolls['durExFP_s'])
    sumfeed.loc[0,'mStdDurEntFP_s'] = np.std(rolls['durEntFP_s'])
    sumfeed.loc[0,'mStdDurExFP_s'] = np.std(rolls['durExFP_s'])
    sumfeed.loc[0,'mDurFP_s'] = np.mean(rolls['durFP_s'])
    sumfeed.loc[0,'stdDurFP_s'] = np.std(rolls['durFP_s'])
    sumfeed.loc[0,'mLastFE_s'] = np.mean(rolls['lastFE_s'])
    sumfeed.loc[0,'stdLastFE_s'] = np.std(rolls['lastFE_s'])
    sumfeed.loc[0,'rngTidalHeight_ft'] = np.max(rolls['tidalheight_ft'])- np.min(rolls['tidalheight_ft'])
    sumfeed.loc[0,'minTidalHeight_ft'] = np.min(rolls['tidalheight_ft'])

    cntIn, cntOut = countInOut(rolls)
    sumfeed.loc[0,'cntInTide'] = cntIn
    sumfeed.loc[0,'cntOutTide'] = cntOut

    unique_values = rolls['rateTC'].unique()
    # Calculate the average of these unique values
    sumfeed.loc[0,'mRateTC'] = unique_values.mean()

    sumfeed.loc[0,'cntDoubleFeed'] = rolls['doubleFeed'].sum()

    sumfeed.loc[0,'mMeanDepthFP_m'] = np.mean(rolls['meanDepth_m'])
    sumfeed.loc[0,'stdMeanDepth_m'] = np.std(rolls['meanDepth_m'])
    sumfeed.loc[0,'mMeanRollFP_deg'] = np.mean(rolls['meanRoll_deg'])
    sumfeed.loc[0,'mRngRollFP_deg'] = np.mean(rolls['rngRoll_deg'])
    sumfeed.loc[0,'mStdRollFP_deg'] = np.mean(rolls['stdRoll_deg'])
    sumfeed.loc[0,'mMeanPitchFP'] = np.mean(rolls['meanPitchFP'])
    sumfeed.loc[0,'mRngPitch'] = np.mean(rolls['rangePitchFP'])
    sumfeed.loc[0,'mStdPitchFP'] = np.mean(rolls['stdPitchFP'])
    sumfeed.loc[0,'mMeanHead'] = np.mean(rolls['meanHead'])
    sumfeed.loc[0,'stdMeanHead'] = np.std(rolls['meanHead'])

    sumfeed.loc[0, 'numFP'] = np.max(rolls['numFP'])
    unique_values = rolls['rateFP'].unique()
    # Calculate the average of these unique values
    sumfeed.loc[0,'mRateFP'] = unique_values.mean()


    return sumfeed

def plot_sumfinal(rolls, sLoc):
    # plot duration entering and exiting feeding position
    plt.scatter(rolls['durEntFP_s'], rolls['durExFP_s'])
    plt.title('Duration (sec) Entering Feeding Position v. Exiting ')
    plt.xlabel('entering')
    plt.ylabel('exiting')
    plt.grid(True)
    plt.savefig(os.path.join(sLoc, 'enteringvexitingFP.png'))
    plt.show()

    # plot time spent in foraging position relative to deployment time
    plt.scatter(rolls['sI'], rolls['durFP_s'])
    plt.title('Duration (sec) in Feeding Position over Time ')
    plt.xlabel('index')
    plt.ylabel('duration')
    plt.grid(True)
    plt.savefig(os.path.join(sLoc, 'DurFPOverTime.png'))
    plt.show()

    plt.scatter(rolls['sI'], rolls['lastFE_s'])
    plt.title('Duration (sec) Since Last Feed over Time ')
    plt.xlabel('index')
    plt.ylabel('duration')
    plt.grid(True)
    plt.savefig(os.path.join(sLoc, 'DurLastFeedOverTime.png'))
    plt.show()

    plt.scatter(rolls['sI'], rolls['meanHead'])
    plt.title('Mean Heading of Feeding Events over Time ')
    plt.xlabel('index')
    plt.ylabel('heading')
    plt.grid(True)
    plt.savefig(os.path.join(sLoc, 'MeanHeadoverTime.png'))
    plt.show()

    plt.scatter(rolls['meanDepth_m'], rolls['tidalheight_ft'])
    plt.title('Mean Depth (m) of Feeding v. Tidal Height (ft) ')
    plt.xlabel('depth (m)')
    plt.ylabel('tidal height (ft)')
    plt.grid(True)
    plt.savefig(os.path.join(sLoc, 'DepthandTidalHeight.png'))
    plt.show()
def add_behave(rolls, behaves):
    behaves['Duration'] = behaves['Duration'].apply(lambda x: x.strftime('%H:%M:%S'))
    behaves['Duration'] = pd.to_timedelta(behaves['Duration'])
    behaves['durFP_dd'] = behaves['Duration'].apply(lambda x: x.total_seconds() / 86400)

    behaves['sTime'] = pd.to_datetime(behaves['sTime'])
    behaves['eTime'] = pd.to_datetime(behaves['eTime'])

    for index, row in rolls.iterrows():
        rollDT = pd.to_datetime(rolls.loc[index, 'startDT'])

        for j, value in behaves['sTime'].items():
            if value < rollDT < behaves['eTime'][j]:
                cInd = j
                #print(cInd)
                break
            else:
                continue
        try:
            rolls.loc[index, 'nFP'] = behaves['nFP'][cInd]
            rolls.loc[index, 'durFP_dd'] = behaves['durFP_dd'][cInd]
            rolls.loc[index, 'locFP'] = behaves['location'][cInd]
        except KeyError:
            print('Check column heading in behaves file')
            exit()

    return rolls

def do_boutanalysis(rolls):
    #fig = px.histogram(rolls, x="lastFE_s", marginal="rug", hover_data=rolls.columns, nbins=20)
    #fig.update_xaxes(range=[0, 360])
    #fig = go.Figure(data=go.Scatter(x=rolls['lastFE_s'], y=rolls['lastFE_s'], mode='markers'))
    #fig.show()

    sns.kdeplot(data = rolls['lastFE_s'], bw_adjust=0.05)
    plt.xlim(0,420)
    plt.xticks(np.arange(0, 420, 10))
    plt.show()
    time.sleep(1)
    # Initialize variables
    minThresh = int(input('Enter the minimum bout threshold:'))
    maxThresh = int(input('Enter the maximum bout threshold:'))

    rolls['minBoutThresh_s'] = minThresh
    rolls['maxBoutThresh_s'] = maxThresh

    bout_num = 1
    feeding_event_num = 1  # Start from 1 since the first row is always a feeding event
    rolls['nBout'] = 0
    rolls['nFEB'] = 0
    feed_ids = []

    # Assign the first row as the first feeding event in the first bout
    rolls['nBout'].iloc[0] = bout_num
    rolls['nFEB'].iloc[0] = feeding_event_num
    feed_id = f"{rolls['depID'].iloc[0]}_{rolls['nTD'].iloc[0]}_{rolls['nTP'].iloc[0]}_" \
              f"{rolls['nFP'].iloc[0]}_B{bout_num}_FE{feeding_event_num}"
    feed_ids.append(feed_id)

    # Loop through the remaining rows in the DataFrame
    for i in range(1, len(rolls)):
        last_fe = rolls['lastFE_s'].iloc[i]

        # Determine if a new bout starts or if it's part of the current bout
        #if last_fe > min_threshold and last_fe < max_threshold:
        if last_fe < minThresh:
            feeding_event_num += 1  # Increment feeding event number
            rolls['nBout'].iloc[i] = bout_num
            rolls['nFEB'].iloc[i] = feeding_event_num
            # Create feedID
            feed_id = f"{rolls['depID'].iloc[i]}_{rolls['nTD'].iloc[i]}_{rolls['nTP'].iloc[i]}_" \
                      f"{rolls['nFP'].iloc[i]}_B{bout_num}_FE{feeding_event_num}"
            feed_ids.append(feed_id)

        elif last_fe > maxThresh:
            feeding_event_num = 1
            bout_num = 1
            rolls['nBout'].iloc[i] = bout_num
            rolls['nFEB'].iloc[i] = feeding_event_num
            feed_id = f"{rolls['depID'].iloc[i]}_{rolls['nTD'].iloc[i]}_{rolls['nTP'].iloc[i]}_" \
                      f"{rolls['nFP'].iloc[i]}_B{bout_num}_FE{feeding_event_num}"
            feed_ids.append(feed_id)
        else:
            # Start a new bout
            bout_num += 1
            feeding_event_num = 1  # Reset feeding event number for the new bout
            rolls['nBout'].iloc[i] = bout_num
            rolls['nFEB'].iloc[i] = feeding_event_num

            # Create feedID
            feed_id = f"{rolls['depID'].iloc[i]}_TD{rolls['nTD'].iloc[i]}_{rolls['nTP'].iloc[i]}_FP" \
                      f"{rolls['nFP'].iloc[i]}_B{bout_num}_FE{feeding_event_num}"
            feed_ids.append(feed_id)

    # Assign the generated feedIDs to the DataFrame
    rolls['feedID'] = feed_ids
    return rolls
    # Calculate descriptive statistics
    #bouts_per_fp = rolls.groupby(['depID', 'numFP']).numBout.nunique()
    #bouts_per_tc = rolls.groupby(['depID', 'numTC']).numBout.nunique()
    #events_per_bout = rolls.groupby(['depID', 'numFP', 'numTC', 'numBout']).size().reset_index(name='event_count')

    #return rolls, bouts_per_fp, bouts_per_tc, events_per_bout

def get_rollratePRH(dat, var, fs=10):
        # Time interval between samples (1/fs)
        time_interval = 1 / fs

        # Get the roll data from the dictionary
        roll_data = dat[var]

        # Calculate the rate of rolling (angular velocity)
        roll_rate = [None]  # The first value is NaN, since there's no previous value to compute the rate
        for i in range(1, len(roll_data)):
            rate = (roll_data[i] - roll_data[i - 1]) / time_interval
            roll_rate.append(rate)
        dat['rollrate'] = roll_rate
        return dat

def get_feedPRH(dat, rolls):
    feRoll = []
    fePitch = []
    feHeading = []
    feDepth = []
    feJerk = []
    feTides = []
    feRollRate = []
    feID = []
    maindf = pd.DataFrame()

    for index, row in rolls.iterrows():
        sI = row['sI']
        eI = row['eI']
        feI = row['feedID']

        # Extract relevant data slices
        depth_data = dat['p'][sI:eI]
        roll_data = dat['roll'][sI:eI]
        pitch_data = dat['pitch'][sI:eI]
        head_data = dat['head'][sI:eI]
        jerk_data = dat['sjerk'][sI:eI]
        tide_data = dat['tidalheight'][sI:eI]
        rollrate_data = dat['rollrate'][sI:eI]

        id_data = [feI] * len(jerk_data)

        feDepth.append(depth_data)
        feRoll.append(roll_data)
        fePitch.append(pitch_data)
        feJerk.append(jerk_data)
        feID.append(id_data)
        feTides.append(tide_data)
        feRollRate.append(rollrate_data)
        feHeading.append(head_data)

    feDepth = [x for x in feDepth if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    feRoll = [x for x in feRoll if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    fePitch = [x for x in fePitch if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    feJerk = [x for x in feJerk if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    feID = [x for x in feID if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    feHeading = [x for x in feHeading if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    feTides = [x for x in feTides if isinstance(x, (list, np.ndarray)) and len(x) > 0]
    feRollRate = [x for x in feRollRate if isinstance(x, (list, np.ndarray)) and len(x) > 0]

    feDepth = np.concatenate(feDepth).tolist()
    feRoll = np.concatenate(feRoll).tolist()
    feRollRate = np.concatenate(feRollRate).tolist()
    feTides = np.concatenate(feTides).tolist()

    fePitch = np.concatenate(fePitch).tolist()
    feJerk = np.concatenate(feJerk).tolist()
    feID = np.concatenate(feID).tolist()
    feHeading = np.concatenate(feHeading).tolist()

    df = pd.DataFrame({'feedID':feID, 'depth': feDepth, 'roll': feRoll, 'rollrate': feRollRate, 'pitch': fePitch, 'head': feHeading,
                           'sjerk': feJerk, 'tidalheight': feTides })

    combined_df = pd.concat([maindf, df], ignore_index=True)
    return combined_df

def get_PitCount(rolls, dat):
    rolls['jSig'] = rolls['jSig'].apply(pd.Series)
    try:
        rolls['jSig'] = rolls['jSig'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        rolls['nJSig'] = rolls['jSig'].apply(len)
    except ValueError:
        print('Need to remove "np.int64" text from accepted feed events file...')
        exit()
    except UnboundLocalError:
        print('Roll event outside foraging period in behavioral file, check indices')
        exit()

    for i, v in rolls.iterrows():
        if v['minRoll_deg']>70 and v['nJSig'] <=1:
            rolls.at[i, 'pitcount'] = 1
        else:
            start_index = v['sI']
            end_index = v['eI']
            plt.figure()
            # plotting stuff
            plt.subplot(3, 1, 1)
            plt.plot(dat['p'][start_index:end_index], label='Depth')
            plt.gca().invert_yaxis()  # Invert y-axis for depth
            plt.title('Depth')
            plt.ylabel('Depth (m)')

            # Plot Roll
            plt.subplot(3, 1, 2)
            plt.plot(dat['roll'][start_index:end_index], label='Roll', color='orange')
            plt.axvline(x=v['sFP'] - start_index, color='r', linestyle='--', label='sFP')
            plt.axvline(x=v['eFP'] - start_index, color='orange', linestyle='--', label='eFP')
            plt.legend()
            plt.title('Roll')
            plt.ylabel('Roll (degrees)')
            plt.legend()

            # Plot Jerk
            plt.subplot(3, 1, 3)
            plt.plot(dat['sjerk'][start_index:end_index], label='Jerk', color='green')

            plt.show(block=False)
            plt.pause(0.005)  # Allow the plot to be updated

            val = input('Enter the number of feeding events...')
            plt.close()

            if val == 'r':
                rolls = rolls.drop(i)
            else:
                rolls.at[i, 'pitcount'] = int(val)

    return rolls

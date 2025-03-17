# Import libraries and modules
import scipy
from scipy.signal import welch

import numpy as np
import matplotlib.pyplot as plt
import time

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds

# Import the custom module
from brainflow_stream import BrainFlowBoardSetup

# Code for setting up gif

import tkinter as tk
from PIL import Image, ImageTk

 # Create main window
root = tk.Tk()
root.title("Animated GIF")

# Load the GIF
gif_path = "/Users/stell/Downloads/Neurohack-2025/real-time-bci-stream THIS IS YOU/example-scripts/sierpinski-zoom41.gif"
gif = Image.open(gif_path)
#print("Total Frames:", gif.n_frames)
NEW_WIDTH = 600  
NEW_HEIGHT = 600  

# Display GIF using a label
label = tk.Label(root)
label.pack()

# Extract and resize frames
frames = []
for i in range(gif.n_frames):
    gif.seek(i)  # Move to the next frame
    resized_frame = gif.copy().resize((NEW_WIDTH, NEW_HEIGHT), Image.LANCZOS)  # Resize with high quality
    frames.append(ImageTk.PhotoImage(resized_frame.convert("RGBA")))  # Convert for compatibility

# Function to animate GIF
def update(ind=0):
    label.config(image=frames[ind])
    root.after(100, update, (ind + 1) % len(frames))  # Adjust speed if needed

# Function to stop the animation and close the window
def stop_animation():
    global running
    running = False
    root.quit()  # Close the window

board_id = BoardIds.CYTON_BOARD.value # Set the board_id to match the Cyton board

# Connect to the board
cyton_board = BrainFlowBoardSetup(
                                board_id = board_id,
                                name = 'Board_1', # Optional name for the board. This is useful if you have multiple boards connected and want to distinguish between them.
                                serial_port = None # If the serial port is not specified, it will try to auto-detect the board. If this fails, you will have to assign the correct serial port. See https://docs.openbci.com/GettingStarted/Boards/CytonGS/ 
                                ) 

cyton_board.setup() # This will establish a connection to the board and start streaming data.

# get data
time.sleep(5) # Wait for 5 seconds to allow the board to build up some samples into the buffer
raw_data_500 = cyton_board.get_current_board_data(num_samples = 500) # Get the latest 500 samples from the buffer


# main processing function (Welch algorithm)
def compute_band_power(eeg_data, fs, bands):
    n_chans = eeg_data.shape[0]  # Number of channels
    n_bands = len(bands)  # Number of frequency bands
    band_powers = np.zeros((n_chans, n_bands))  # Correct shape for band powers
    f, psd = welch(eeg_data, fs=fs, nperseg=fs*2)  # nperseg = window size (2 sec recommended)

    for i, (band_name, (low, high)) in enumerate(bands.items()):
        idx_band = np.where((f >= low) & (f <= high))[0]
        band_powers[:, i] = np.trapezoid(psd[:, idx_band], f[idx_band])
    
    return band_powers

# Preloaded function for initial processing of data
def remove_dc_offset(data):
    return data[1:9, :] - np.mean(data[1:9, :], axis=1, keepdims=True)

# Data
fs = 250  # Sampling frequency in Hz
bands = {"alpha": (8, 13), "beta": (13, 30)}

initialAlpha = 0
initialBeta = 0

for j in range (120): # can tinker with this value as needed
    raw_eeg_data = cyton_board.get_current_board_data(num_samples = 1000) # can experiment with the number of samples you want
    #raw_eeg_data = np.random.randn(8, fs*10) # this is EXAMPLE DATA ONLY
    eeg_data = remove_dc_offset(raw_eeg_data)
    
    # Call the Welch algorithm
    band_power = compute_band_power(eeg_data, fs, bands)

    # compute average of each wave
    alphaAvg = 0
    for i in range(0, len(band_power)):
        alphaAvg += band_power[i][0]
    alphaAvg = alphaAvg / len(band_power)
    betaAvg = 0
    for i in range(0, len(band_power)):
        betaAvg += band_power[i][1]
    betaAvg = betaAvg / len(band_power)

    # Creating a baseline to use for determining stress
    if j == 0:
        initialAlpha = alphaAvg
        initialBeta = betaAvg

    threshold = 0 # this will need to be changed later

    # GIF creation instructions

    if j > 0:
        if initialAlpha / alphaAvg > 1 and initialBeta / betaAvg < 1:
            print("User is stressed!")
            # Start triangle animation
            update()  # Call the function once to start the loop
            root.mainloop()
        else: 
            print("User is relaxed!")

    # collect new data from eeg
    time.sleep(3) # Build up 3 seconds of samples
    raw_data_500 = cyton_board.get_current_board_data(num_samples = 500) # Get the latest 500 samples from the buffer
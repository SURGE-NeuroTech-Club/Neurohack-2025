import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
import mne
from scipy.signal import butter, lfilter

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds


# Import the custom module.
from brainflow_stream import BrainFlowBoardSetup

#########################
# board_id = BoardIds.CYTON_BOARD.value # Set the board_id to match the Cyton board

# # Lets quickly take a look at the specifications of the Cyton board
# for item1, item2 in BoardShim.get_board_descr(board_id).items():
#     print(f"{item1}: {item2}")
# ##########################
# cyton_board = BrainFlowBoardSetup(
#                                 board_id = board_id,
#                                 name = 'Board_1', # Optional name for the board. This is useful if you have multiple boards connected and want to distinguish between them.
#                                 serial_port = None # If the serial port is not specified, it will try to auto-detect the board. If this fails, you will have to assign the correct serial port. See https://docs.openbci.com/GettingStarted/Boards/CytonGS/ 
#                                 ) 

# cyton_board.setup() # This will establish a connection to the board and start streaming data.

# DM01HOSQA
################################

# Function to create a bandpass filter for beta waves (13-30 Hz)

def bandpass_filter(data, lowcut=13, highcut=30, fs=250, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)

    return y

def remove_dc_offset(data):
    return data[1:9, :] - np.mean(data[1:9, :], axis=1, keepdims=True)

# Function to convert EEG data to an MNE object
def convert_to_mne(data, sfreq):
    ch_names = [f'EEG {i+1}' for i in range(data.shape[0])]
    ch_types = ['eeg'] * data.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info)


# Function to compute power spectral density and extract beta waves
def extract_beta_power(raw):
    psd, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=raw.info['sfreq'], fmin=13, fmax=30, n_fft=512)
    psd = np.maximum(psd, np.finfo(float).eps)
    psd_db = 10 * np.log10(psd)  # Convert power to dB
    return mne.io.RawArray(psd_db, raw.info)


def process_eeg_beta (period_time, total_time, cyton_board):
    period_sum = []
    period_average = []

    for i in range(int(total_time / period_time)):
        period_data = cyton_board.get_current_board_data(num_samples = 250 * period_time)
        data_eeg = period_data[1:9, :]
        data_eeg = remove_dc_offset(period_data) # Remove DC offset

        data_mne = convert_to_mne(data_eeg, 250)
        beta_power = extract_beta_power(data_mne)

        print(data_eeg.shape)
        print(beta_power.get_data().shape)  # Correct way to check shape

        freqs = np.linspace(13, 30, beta_power.get_data().shape[1])  # Generate frequency bins
        power_values = beta_power.get_data()  # Get power values

        period_sum.append(np.sum(power_values))
        period_average.append(np.mean(power_values))

        # Plot power spectrum for each EEG channel
        plt.figure(figsize=(10, 6))
        for i in range(power_values.shape[0]):  # Loop through channels
            plt.plot(freqs, power_values[i], label=f'EEG {i+1}')

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title("Beta Wave Power Spectrum (13-30 Hz)")
        plt.legend()
        plt.grid(True)
        plt.show()   

    return period_sum, period_average



import brainflow
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes
from mne.channels import read_layout
from psychopy import visual, core, event, gui, sound

CytonDaisy_board_id = 2
serial_port = "COM4"
full_screen = True
use_EEG = True

# C:\Users\murph\Documents\GitHub\OpenBCI\beep_vs_cat_exp.py
def main():

    if use_EEG:
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        sfreq = BoardShim.get_sampling_rate(CytonDaisy_board_id)
        board = BoardShim(CytonDaisy_board_id, params)
        ch_names = BoardShim.get_eeg_names(CytonDaisy_board_id)
        board.prepare_session()

        # config board can only be called after prepare_session and before start_stream
        board.config_board('x1040110X')
        board.config_board('x2040110X')
        board.config_board('x3040110X')
        board.config_board('x4040110X')
        board.config_board('x5040110X')
        board.config_board('x6040110X')
        board.config_board('x7040110X')
        board.config_board('x8040110X')
        board.config_board('xQ40110X')
        board.config_board('xW040110X')
        board.config_board('xE040110X')
        board.config_board('xR040110X')
        board.config_board('xT40110X')
        board.config_board('xU040110X')
        board.config_board('xI040110X')

        board.start_stream()

    if full_screen:
        win = visual.Window(size=(1920,1080), fullscr=True)
    else:  
        win = visual.Window(size=(1000,700))

    msg = visual.TextStim(win, text="Press spacebar to begin...").draw()
    win.flip()
    event.waitKeys(keyList=["space"])  # wait for a spacebar press before continuing
    event.clearEvents()

    time.sleep(30)
    # End of presentation stage
    # Begin data writing stage
  
    if use_EEG:
        data = board.get_board_data()
        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(CytonDaisy_board_id)
        marker_channel = BoardShim.get_marker_channel(CytonDaisy_board_id)

        #for channel in eeg_channels:
            #DataFilter.perform_bandpass(data[channel], sfreq, 26.0, 25, 2, FilterTypes.BUTTERWORTH.value, 0)
            #DataFilter.perform_bandstop(data[channel], sfreq, 50.0, 4.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            #DataFilter.perform_wavelet_denoising(data[channel], 'coif3', 3)
            #DataFilter.remove_environmental_noise(data[channel], sfreq, NoiseTypes.FIFTY.value)

        eeg_data = data[eeg_channels, :]
        eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

        # Creating MNE objects from brainflow data arrays
        ch_types = ['eeg'] * len(eeg_channels)
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data, info)

        raw.save('raw_30s.fif', overwrite=True)
        #events = data[marker_channel, :]
        #np.save('beep_vs_cat.npy', events)

if __name__ == '__main__':
    main()
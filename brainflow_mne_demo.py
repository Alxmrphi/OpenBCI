import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import mne
from mne.channels import read_layout


def main():

    board_id = 2 # BoardIds.SYNTHETIC_BOARD.value
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = "COM4"

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    for i in range(35):
        time.sleep(1)
        marker = 1 if i % 2 == 0 else 2
        board.insert_marker(marker)
  
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    marker_channel = BoardShim.get_marker_channel(board_id)

    print('Marker channel = "', marker_channel)

    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(board_id)
    sfreq = BoardShim.get_sampling_rate(board_id)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    # its time to plot something!
    raw.plot_psd(average=True)
    raw.save('OpenBCI_MNE_demo_raw.fif', overwrite=True)

    # Now save events
    events = data[marker_channel, :]
    np.save('test_events.npy', events)

    plt.savefig('psd.png')


if __name__ == '__main__':
    main()
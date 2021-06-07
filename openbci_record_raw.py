import time
import numpy as np
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

#import mne
#from mne.channels import read_layout

def main():

    board_id = 2 # BoardIds.SYNTHETIC_BOARD.value
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = "COM4"

    board = BoardShim(board_id, params)
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
    sfreq = BoardShim.get_sampling_rate(board_id)
    print('sfreq = ', sfreq)

    time.sleep(30)
  
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    eeg_data = data[eeg_channels, :]
    np.save('openbci_raw_30s.npy', eeg_data)

   
if __name__ == '__main__':
    main()
import brainflow
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from mne.channels import read_layout
from psychopy import visual, core, event, gui

CytonDaisy_board_id = 2
serial_port = "COM4"
full_screen = True
use_EEG = True
time_between_imgs = 0.25
time_img_onscreen = 1.00
n_images = 250


def main():

    if use_EEG:
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        board = BoardShim(CytonDaisy_board_id, params)
        board.prepare_session()
        board.start_stream()

    if full_screen:
        win = visual.Window(size=(1920,1080), fullscr=True)
    else:  
        win = visual.Window(size=(1000,700))

    msg = visual.TextStim(win, text="Press spacebar to begin...").draw()
    win.flip()
    event.waitKeys(keyList=["space"])  # wait for a spacebar press before continuing
    event.clearEvents()

    cat_folder = 'C:\\Users\\murph\\Documents\\Blog\\Faces_vs_Cats_BCI\\data\\cats\\'
    faces_folder = 'C:\\Users\\murph\\Documents\\Blog\\Faces_vs_Cats_BCI\\data\\faces\\'
    img_folder = 'C:\\Users\\murph\\Documents\\Blog\\Faces_vs_Cats_BCI\\data\\mixed\\'

    # Extract list of all .jpg files in each folder
    cat_imgs = [img for img in os.listdir(cat_folder) if str(img).lower().endswith('.jpg')]
    face_imgs = [img for img in os.listdir(faces_folder) if str(img).lower().endswith('.jpg')]

    cat_imgs = np.random.choice(cat_imgs, size=n_images)
    face_imgs = np.random.choice(face_imgs, size=n_images)

    cat_imgs = list(cat_imgs)
    cat_imgs.extend(list(face_imgs))
    imgs = cat_imgs
    assert len(imgs) == 2 * n_images, f"Total number of images not equal to 2*n_images (2*{n_images})"

    # shuffle labels and use this to shuffle images, so we know the correct
    # label of each of the shuffled images to send the correct EEG trigger
    # cat = 1, face = 2
    labels = np.array([1] * n_images + [2] * n_images)
    labels_idx = np.array(range(len(imgs)))
    np.random.shuffle(labels_idx)
    imgs = np.array(imgs)[labels_idx]
    labels = labels[labels_idx]

    for img, label in zip(imgs[:100], labels[:100]):

        # Display the image on screen and send EEG trigger
        stim = visual.ImageStim(win, image=img_folder+img, size=0.75).draw()
        if use_EEG:
            board.insert_marker(label)
        win.flip()
        core.wait(time_img_onscreen)

        # Display the fixation cross
        msg = visual.TextStim(win, text="+").draw()
        win.flip()
        core.wait(time_between_imgs)


    # End of presentation stage
    # Begin data writing stage
  
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(CytonDaisy_board_id)
    marker_channel = BoardShim.get_marker_channel(CytonDaisy_board_id)

    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(CytonDaisy_board_id)
    sfreq = BoardShim.get_sampling_rate(CytonDaisy_board_id)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    raw.save('cats_vs_faces_raw.fif', overwrite=True)
    events = data[marker_channel, :]
    np.save('cats_vs_faces.npy', events)

if __name__ == '__main__':
    main()
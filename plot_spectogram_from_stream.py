#!/usr/bin/env python3
from typing import List

import sounddevice as sd
from time import sleep, process_time

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# sampling rate of the device
samplingrate = sd.query_devices("default", "input")["default_samplerate"]
# max total size of the whole recording
recording_size = int(samplingrate * 0.20)
# size of buffer to save before reaching vol_threshold
buffer_size = int(samplingrate * 0.05)

# recording data, array of floats
v = []
# recording triggered on volume over this threshold
vol_threshold = 0.03
# triggered when sound reaches vol_threshold
recording = False
# we want to gather some data before the volume threshold was reached to get a more clear view of the shape of the sound
data_buffer = []


# returns true if any of the channel data values reach vol_threshold
# TODO would be useful to have some algorithm to check if the sound is not a conversation
# to prevent random iot ghosts turning devices on and off while talking
def should_start_recording(channel_data: List[float]) -> bool:
    for f in channel_data:
        val = f if f > 0 else -f
        if val > vol_threshold:
            return True
    return False


def audio_callback(data, frames, time, status):
    global v, recording, data_buffer
    if status:
        print(status, flush=True)
    channel_data = [channels[0] for channels in data]
    if not recording:
        recording = should_start_recording(channel_data)
    if recording:
        v.extend(data_buffer)
        data_buffer = []
        v.extend(channel_data)
    else:
        data_buffer.extend(channel_data)
        # limits max size
        if len(data_buffer) > buffer_size:
            # gets the last $buffer_size samples
            data_buffer = data_buffer[-buffer_size:]


# open default input device
stream = sd.InputStream(
    device="default", channels=1, samplerate=samplingrate, callback=audio_callback
)

with stream:
    print("listening...")
    while True:
        if len(v) >= recording_size:
            # trim v size to be exactly recording_size
            v = v[-recording_size:]
            y = np.asarray(v, dtype=np.float32)

            mel_spec = librosa.feature.melspectrogram(y=y, sr=samplingrate)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

            # normalize data to make each value on a scale between -1 and 1
            normalized = librosa.util.normalize(mel_spec_db)
            # show preview
            plt.imshow(normalized)
            plt.show()

            recording = False
            v = []

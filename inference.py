
# Imports
import time
import argparse as ap
import sounddevice as sd
import time
import psutil
import uuid
import redis
from scipy.io.wavfile import write
import tensorflow as tf
import zipfile
import os
import numpy as np
import paho.mqtt.client as mqtt
import uuid
import platform
import psutil
import time
import json

# Set the MQTT topic
MQTT_TOPIC = "s295573"

# Create the MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect('mqtt.eclipseprojects.io', 1883) #The MQTT broker can be changed


LABELS = ['ANG', 'HAP','NEU']

PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_length_in_s': 0.016,
    'frame_step_in_s': 0.016,
    'lower_frequency': 20,
    'upper_frequency': 4000,
    'num_mel_bins': 40,
    'num_coefficients': 10
}

IS_SILENCE_ARGS = {
    'downsampling_rate' : 16000,
    'frame_length_in_s' : 0.016,
    'dbFSthres' : -120,
    'duration_thres' : 1
}


def send_message(label, confidence):
    
    # Get the current time in milliseconds
    timestamp_ms = int(time.time() * 1000) #from the source we want the timestamp in milliseconds
    # Get the MAC address of the PC
    mac_address =  hex(uuid.getnode())
    print(type(confidence.item()))
    confidence = confidence.item() 
    # Create the message payload as a JSON object
    message = {
        "mac_address": mac_address,
        "timestamp": timestamp_ms,
        "prediction_confidence": confidence,
        "label": label
    }

    # Convert the message payload to a JSON string
    message_json = json.dumps(message)
    #print(message_json)
    # Publish the message to the MQTT topic
    client.publish(MQTT_TOPIC, message_json)
    print()
    print('Timestamp is: ',timestamp_ms)
    print('Message sent')
    print('----------------------------------')
    

def callback(indata, frames, call_back, status):
    #print('im in the callback')
    global store_information
    store_audio = is_silence(indata=indata,
                             downsampling_rate=IS_SILENCE_ARGS['downsampling_rate'],
                             frame_length_in_s=IS_SILENCE_ARGS['frame_length_in_s'],
                             dbFSthres=IS_SILENCE_ARGS['dbFSthres'],
                             duration_thres=IS_SILENCE_ARGS['duration_thres'])
    
    #print(f'callback after is silence NOT STORE AUDIO IS {store_audio}')
    if not store_audio:
        input_details = interpreter.get_input_details()
        #print('input details: ')
        #input details: 
        # [  1 187  10   1]
        # output details:
        # [1 3]
        # print(input_details[0]['shape'])
        output_details = interpreter.get_output_details()
        # print('output details: ')
        # print(output_details[0]['shape'])

        mfccs = get_mfccs(indata=indata,
                          downsampling_rate=PREPROCESSING_ARGS['downsampling_rate'],
                          frame_length_in_s=PREPROCESSING_ARGS['frame_length_in_s'],
                          frame_step_in_s=PREPROCESSING_ARGS['frame_step_in_s'],
                          num_mel_bins=PREPROCESSING_ARGS['num_mel_bins'],
                          lower_frequency=PREPROCESSING_ARGS['lower_frequency'],
                          upper_frequency=PREPROCESSING_ARGS['upper_frequency'],
                          num_coefficients=PREPROCESSING_ARGS['num_coefficients'])
        #print('I have mfccs')
        mfccs = tf.expand_dims(mfccs, 0)
        mfccs = tf.expand_dims(mfccs, -1)
        #print('Start the interpretetion by the model')
        #print(mfccs.shape)

        # is the input tensor
        #GIVES THIS ERROR
        # Cannot set tensor: Dimension mismatch. Got 62 but expected 187 for dimension 1 of input 0.
        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        magnitude_of_index = np.max(output[0])

        top_index = np.argmax(output[0])
        predicted_label = LABELS[top_index]

        print(f'Model prediction - {predicted_label} - {magnitude_of_index}')

        if (magnitude_of_index > 0.95 and predicted_label == 'ANG'):
            send_message('ANG', magnitude_of_index)
        elif (magnitude_of_index > 0.95 and predicted_label == 'HAP'):
            send_message('HAP', magnitude_of_index)
        elif (magnitude_of_index > 0.95 and predicted_label == 'NEU'):
            send_message('NEU', magnitude_of_index)



def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    # print(indata.shape)
    print('...')
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1
    indata = tf.squeeze(indata)

    return indata


# Gets an spectrogram that takes time x amplitude to frequency x magnitude
def get_spectrogram(indata, downsampling_rate, frame_length_in_s, frame_step_in_s):
    audio_padded = get_audio_from_numpy(indata)

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    spectrogram = stft = tf.signal.stft(
        audio_padded,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram


def get_log_mel_spectrogram(indata, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency):
    spectrogram = get_spectrogram(
        indata, downsampling_rate, frame_length_in_s, frame_step_in_s)

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    num_spectrogram_bins = frame_length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=downsampling_rate,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

    return log_mel_spectrogram


def get_mfccs(indata, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients):
    # First you obtain the log_mel_spectrogram
    log_mel_spectrogram = get_log_mel_spectrogram(
        indata, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency)

    # Compute the mfccs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    # Reshapping the mfccs in order to have only the desired coefficients
    mfccs = mfccs[..., :num_coefficients]

    return mfccs


# Classify the audio
def is_silence(indata, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres):

    spectrogram = get_spectrogram(
        indata,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s
    )

    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthres
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_thres:
        return 0
    else:
        return 1


# ------------------------PARSER------------------------
parser = ap.ArgumentParser(
    description='You can choose between --device --host --port --user --password')
parser.add_argument('--device', type=int, help='Insert the device number')
parser.add_argument('--host', type=str, help='Insert the host')
parser.add_argument('--port', type=int, help='Insert the host')
parser.add_argument('--user', type=str, help='Insert the host')
parser.add_argument('--password', type=str, help='Insert the host')
args = parser.parse_args()

REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USER = args.user
REDIS_PASSWORD = args.password
mac_address = hex(uuid.getnode())

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    username=REDIS_USER
)


# Script parameter for the INPUT STREAM most of them fixed
SAMPLE_RATE = 16000
AUDIO_FILE_LENGTH_IN_S = 3.5
DEVICE = args.device  # passed in the command line
CHANNELS = 1
DTYPE = 'int16'


# ping command is used to test if the connection works
print('Is connected:', redis_client.ping())


MODEL_NAME = 'emo_model'
# Unzipping the model
print('Unzipping the model')
# ---> this creates ./emotion_model.tflite
zipped_model_path = os.path.join('.', f'{MODEL_NAME}.tflite.zip')
with zipfile.ZipFile(zipped_model_path, 'r') as zip_ref:
    zip_ref.extractall("./")

# Implement the interpreter
print('Loading the model')
model_path = os.path.join('./emo_model.tflite/tflite_models/', f'{MODEL_NAME}.tflite')
print(model_path)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print('...script is starting...')
store_information = False
with sd.InputStream(device=DEVICE,
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    dtype=DTYPE,
                    blocksize=int(SAMPLE_RATE * AUDIO_FILE_LENGTH_IN_S),
                    callback=callback):

    while True:
        key = input()
        if key in ['Q', 'q']:
            print('Stopping the script')
            break

        time.sleep(1)
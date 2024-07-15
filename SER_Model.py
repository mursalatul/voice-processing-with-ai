import os
import glob
import re
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, LSTM, TimeDistributed, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from google.colab import drive
from google.colab import files

drive.mount('/content/drive', force_remount=True)

# Emotion mapping
int2emotion = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "disgust": "disgust",
    "surprised": "surprised"
}

# Allowed emotions
AVAILABLE_EMOTIONS = {
    "angry",
    "disgust",
    "fearful",
    "sad",
    "neutral",
    "happy"
}

file_pattern = '/content/drive/MyDrive/SER/dataverse_files/**/*.wav'

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc", False)
    chroma = kwargs.get("chroma", False)
    mel = kwargs.get("mel", False)
    contrast = kwargs.get("contrast", False)
    tonnetz = kwargs.get("tonnetz", False)
    n_fft = kwargs.get("n_fft", 2048)

    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        if chroma or contrast:
            stft = np.abs(librosa.stft(X, n_fft=n_fft))

        result = np.array([])

        def ensure_1d(feature, name):
            if feature.ndim > 1:
                feature = np.mean(feature, axis=1)
            return feature.flatten()

        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            result = np.hstack((result, ensure_1d(mfccs, "MFCCs")))

        if chroma:
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            result = np.hstack((result, ensure_1d(chroma, "Chroma")))

        if mel:
            mel = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft)
            result = np.hstack((result, ensure_1d(mel, "MEL")))

        if contrast:
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
            result = np.hstack((result, ensure_1d(contrast, "Contrast")))

        if tonnetz:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
            result = np.hstack((result, ensure_1d(tonnetz, "Tonnetz")))

    return result

def load_data(test_size=0.2):
    X, y = [], []
    files = glob.glob(file_pattern, recursive=True)
    print("Files found:", len(files))
    for file in files:
        print(file)  # Print each file path to verify correct matching
        basename = os.path.basename(file)
        match = re.match(r'.+_.+_(.+)\.wav', basename)

        if not match:
            print(f"Skipping file with unexpected name format: {basename}")
            continue

        emotion_code = match.group(1)
        emotion = int2emotion.get(emotion_code.lower())

        if emotion is None:
            print(f"Skipping file with unrecognized emotion code: {basename} (code: {emotion_code})")
            continue

        print(f"Processing file: {file}, Emotion: {emotion}")

        if emotion in AVAILABLE_EMOTIONS:
            features = extract_feature(file, mfcc=True, chroma=True, mel=True)
            if features is not None:
                print(f"Extracted features shape: {features.shape}")  # Print feature shape for each file
                X.append(features)
                y.append(emotion)
            else:
                print(f"Failed to extract features from file: {file}")
        else:
            print(f"Skipping emotion: {emotion}")

    if not X:
        raise ValueError("No data loaded - check file paths and emotion filtering.")

    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

try:
    X_train, X_test, y_train, y_test = load_data(test_size=0.25)
    print("Data loaded successfully")
    print("[+] Number of training samples:", len(X_train))
    print("[+] Number of testing samples:", len(X_test))
except ValueError as e:
    print("Error:", e)

# Encode labels
le = LabelEncoder()

# Check if labels are not empty
if len(y_train) == 0 or len(y_test) == 0:
    print('Labels array is empty. Please check the data loading process.')
    raise ValueError("Labels array is empty. Please check the data loading process.")

# Debugging: Verify unique labels before encoding
unique_labels_before_encoding = np.unique(y_train + y_test)
print(f"Unique labels before encoding: {unique_labels_before_encoding}")

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Debugging: Verify unique labels after encoding
unique_labels_after_encoding = np.unique(np.argmax(y_train, axis=1))
print(f"Unique labels after encoding: {unique_labels_after_encoding}")

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Verify shapes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Flatten the input features for MLPClassifier
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Training MLP Classifier as a preliminary model
model_params = {
    "alpha": 0.01,
    "batch_size": 256,
    "epsilon": 1e-08,
    "hidden_layer_sizes": (300,),
    "learning_rate": "adaptive",
    "max_iter": 500,
}
model = MLPClassifier(**model_params)

print("[*] Training the model...")
model.fit(X_train_flat, y_train)

y_pred = model.predict(X_test_flat)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))

def extract_features(file_path, max_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs

def load_data(file_path_pattern, test_size=0.2):
    features = []
    labels = []
    for file_path in glob.glob(file_path_pattern, recursive=True):
        try:
            print(f"Processing file: {file_path}")  # Debug statement
            file_name = os.path.basename(file_path)
            print(f"File name: {file_name}")  # Debug statement

            match = re.match(r'.+_.+_(.+)\.wav', file_name)
            if not match:
                print(f"Skipping file with unexpected name format: {file_name}")
                continue

            emotion_code = match.group(1)
            label = int2emotion.get(emotion_code.lower())
            if label not in AVAILABLE_EMOTIONS:
                print(f"Skipping unrecognized or unavailable emotion: {label}")
                continue

            print(f"Extracted label: {label}")  # Debug statement
            data = extract_features(file_path)
            features.append(data)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Total files processed: {len(features)}")  # Debug statement
    print(f"Total labels processed: {len(labels)}")  # Debug statement

    return np.array(features), np.array(labels)

file_path_pattern = '/content/drive/MyDrive/SER/dataverse_files/**/*.wav'
features, labels = load_data(file_path_pattern, test_size=0.2)

if len(labels) == 0:
    raise ValueError("No labels found. Check the data loading process and file patterns.")

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.25, random_state=42)

# Verify shapes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(40, 174, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(128, return_sequences=False, dropout=0.5))

model.add(Dense(len(unique_labels_after_encoding), activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[checkpoint, earlystop, reduce_lr])

import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

plot_history(history)

model.load_weights('model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc * 100:.2f}')

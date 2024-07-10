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
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    LSTM,
    TimeDistributed,
    Flatten,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from google.colab import drive
from google.colab import files

drive.mount("/content/drive", force_remount=True)

# Emotion mapping
int2emotion = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "disgust": "disgust",
    "surprised": "surprised",
}

# Allowed emotions
AVAILABLE_EMOTIONS = {"angry", "disgust", "fearful", "sad", "neutral", "happy"}

file_pattern = r"/content/drive/MyDrive/SER/dataverse_files/**/*.wav"


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
            tonnetz = librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X), sr=sample_rate
            )
            result = np.hstack((result, ensure_1d(tonnetz, "Tonnetz")))

    return result


def load_data(test_size=0.2):
    X, y = [], []
    files = glob.glob(file_pattern, recursive=True)
    print("Files found:", len(files))

    for file in files:
        basename = os.path.basename(file)
        match = re.match(r".+_(.+)\.wav", basename)

        if not match:
            print(f"Skipping file with unexpected name format: {basename}")
            continue

        emotion_code = match.group(1)
        emotion = int2emotion.get(emotion_code.lower())

        if emotion is None:
            print(
                f"Skipping file with unrecognized emotion code: {basename} (code: {emotion_code})"
            )
            continue

        print(f"Processing file: {file}, Emotion: {emotion}")

        if emotion in AVAILABLE_EMOTIONS:
            features = extract_feature(file, mfcc=True, chroma=True, mel=True)
            # shape is constant (180,)
            # print(f"Extracted features shape: {features.shape}")
            X.append(features)
            y.append(emotion)
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
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))


def extract_features(file_path, max_len=174):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs


def extract_label_from_filename(filename):
    parts = filename.split("_")
    if len(parts) > 1:
        return parts[1]
    else:
        return None


def load_data(file_pattern):
    labels = []
    features = []
    for root, dirs, files in os.walk(file_pattern):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")  # Debugging: print file path
                feature = extract_features(file_path)
                label = extract_label_from_filename(file)
                if label:
                    features.append(feature)
                    labels.append(label)
                else:
                    print(f"Skipping file due to invalid label extraction: {file_path}")
    return np.array(features), np.array(labels)


features, labels = load_data(file_pattern)

# Debugging: Print shapes and contents
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels: {labels[:10]}")  # Print first 10 labels for debugging

# Encode labels
le = LabelEncoder()

"""
having issues here as labels are 0
"""

# Check if labels are not empty
if labels.size == 0:
    print("empty array")
    # raise ValueError("Labels array is empty. Please check the data loading process.")

# Debugging: Verify unique labels before encoding
unique_labels_before_encoding = np.unique(labels)
print(f"Unique labels before encoding: {unique_labels_before_encoding}")

labels = le.fit_transform(labels)
labels = to_categorical(labels)

# Debugging: Verify unique labels after encoding
unique_labels_after_encoding = np.unique(labels.argmax(axis=1))
print(f"Unique labels after encoding: {unique_labels_after_encoding}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Verify shapes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")


def create_crnn_model(input_shape, num_classes, learning_rate=0.001):
    model = Sequential()

    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(
        Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=l2(0.001))
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
num_classes = y_train.shape[1]

model = create_crnn_model(input_shape, num_classes)

model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=0.00001
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1,
)

# Print training history
print(history.history)

# Load the best model
model.load_weights("best_model.keras")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the trained model
if not os.path.isdir("result"):
    os.mkdir("result")

model.save("result/crnn_model.h5")

uploaded = files.upload()

for file_name in uploaded.keys():
    print(f"Processing file: {file_name}")
    emotion = file_name
    print(emotion)

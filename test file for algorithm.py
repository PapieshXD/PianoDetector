import numpy as np
import librosa
import sounddevice as sd
import customtkinter as ctk
from tkinter import filedialog, scrolledtext
from queue import Queue
from threading import Thread
import logging
from scipy.signal import butter, lfilter

print(sd.query_devices())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("note_recognition.log"),
        logging.StreamHandler()
    ]
)

# Piano frequency dictionary generator
def generate_piano_frequencies():
    keys = []
    frequencies = []
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    for n in range(88):
        octave = (n + 9) // 12
        note = notes[n % 12]
        freq = 27.5 * (2 ** (n / 12))
        keys.append(f"{note}{octave}")
        frequencies.append(freq)
    return dict(zip(keys, frequencies))

piano_frequencies = generate_piano_frequencies()

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Designs a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def is_piano_sound(dominant_frequency, stft, sr, threshold=0.6):
    """
    Determines if the detected sound is likely from a piano.
    Checks for harmonic structure and amplitude consistency.
    """
    # Calculate harmonic frequencies
    harmonics = [dominant_frequency * i for i in range(2, 6)]  # First 5 harmonics
    harmonic_thresholds = []

    # STFT frequencies
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])

    for harmonic in harmonics:
        # Find the closest bin to the harmonic
        idx = np.argmin(np.abs(frequencies - harmonic))
        harmonic_magnitude = stft[idx]
        fundamental_magnitude = stft[np.argmax(stft)]  # Magnitude of the fundamental frequency

        # Compare harmonic magnitude to fundamental magnitude
        if fundamental_magnitude > 0:
            harmonic_strength = harmonic_magnitude / fundamental_magnitude
        else:
            harmonic_strength = 0
        harmonic_thresholds.append(harmonic_strength)

    # Check if harmonics meet the threshold
    is_harmonic = all(ht > threshold for ht in harmonic_thresholds)

    # Log the harmonic strengths for debugging
    log_to_gui(f"Harmonic strengths: {harmonic_thresholds}")

    return is_harmonic

# Integrating the filter into live recognition
def process_audio():
    """Processes live audio to identify notes and classify as piano or not."""
    global stop_flag
    lowcut = 27.5  # Lower bound of the piano range
    highcut = 4186  # Upper bound of the piano range

    while not stop_flag:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get().flatten()

            # Debug: Check if the audio chunk has non-zero values
            if np.max(np.abs(audio_chunk)) < 1e-3:
                log_to_gui("Audio chunk is too quiet. Try increasing microphone sensitivity.", level="WARNING")
                continue

            # Apply bandpass filter
            filtered_chunk = apply_bandpass_filter(audio_chunk, lowcut, highcut, SAMPLE_RATE, order=6)

            # STFT analysis
            n_fft = min(2048, len(filtered_chunk))  # Adjust to chunk size
            hop_length = n_fft // 4
            stft_matrix = np.abs(librosa.stft(filtered_chunk, n_fft=n_fft, hop_length=hop_length))
            avg_magnitude = np.mean(stft_matrix, axis=1)
            frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=n_fft)

            # Check if there's a dominant frequency
            if len(avg_magnitude) == 0 or np.max(avg_magnitude) < 1e-2:
                log_to_gui("No dominant frequency detected. Ensure proper input.", level="WARNING")
                continue

            # Find dominant frequency and closest note
            dominant_idx = np.argmax(avg_magnitude)
            dominant_frequency = frequencies[dominant_idx]
            closest_note, note_freq = find_closest_note(dominant_frequency, piano_frequencies)

            # Classify sound as piano or not
            is_piano = is_piano_sound(dominant_frequency, avg_magnitude, SAMPLE_RATE)

            sound_type = "Piano" if is_piano else "Not Piano"

            # Update GUI and log
            label_live_note.configure(
                text=f"Dominant Frequency: {dominant_frequency:.2f} Hz\n"
                     f"Closest Note: {closest_note} ({note_freq:.2f} Hz)\n"
                     f"Sound Type: {sound_type}"
            )
            log_to_gui(f"Live Recognition: Dominant Frequency: {dominant_frequency:.2f}Hz, "
                       f"Closest Note: {closest_note} ({note_freq:.2f}Hz), Sound Type: {sound_type}")


def find_closest_note(freq, piano_dict):
    closest_note = min(piano_dict.keys(), key=lambda note: abs(piano_dict[note] - freq))
    return closest_note, piano_dict[closest_note]

# Global variables
y, sr = None, None
BUFFER_DURATION = 0.5
SAMPLE_RATE = 44100
CHUNK_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
audio_queue = Queue()
stop_flag = False

# Logging to the GUI
def log_to_gui(message, level="INFO"):
    """Log messages to the GUI log panel."""
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    text_logs.insert("end", message + "\n")
    text_logs.see("end")

# File Analysis Functions
def load_audio_file():
    global y, sr
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=(("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*"))
    )
    if file_path:
        label_file_path.configure(text=f"Selected File: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=None)
            if y is not None and sr is not None:
                label_audio_info.configure(
                    text=f"Duration: {len(y) / sr:.2f} s, Samples: {len(y)}, Sampling Rate: {sr} Hz"
                )
                log_to_gui(f"Loaded file: {file_path}. Duration: {len(y) / sr:.2f}s, SR: {sr}Hz")
            else:
                label_audio_info.configure(text="Error: Failed to load audio data.")
                log_to_gui("Error: Failed to load audio data.", level="ERROR")
        except Exception as e:
            label_audio_info.configure(text=f"Error: {str(e)}")
            log_to_gui(f"Error loading file: {str(e)}", level="ERROR")

def analyze_audio():
    global y, sr
    if y is None or sr is None:
        label_audio_info.configure(text="No audio file loaded!")
        log_to_gui("No audio file loaded for analysis.", level="WARNING")
        return

    try:
        n_fft = 1024
        hop_length = n_fft // 4
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        avg_magnitude = np.mean(stft, axis=1)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        dominant_idx = np.argmax(avg_magnitude)
        dominant_frequency = frequencies[dominant_idx]
        closest_note, note_freq = find_closest_note(dominant_frequency, piano_frequencies)

        label_analysis_result.configure(
            text=f"Dominant Frequency: {dominant_frequency:.2f} Hz\n"
                 f"Closest Note: {closest_note} ({note_freq:.2f} Hz)"
        )
        log_to_gui(f"Analyzed file: Dominant Frequency: {dominant_frequency:.2f}Hz, "
                   f"Closest Note: {closest_note} ({note_freq:.2f}Hz)")
    except Exception as e:
        label_analysis_result.configure(text=f"Error: {str(e)}")
        log_to_gui(f"Error during analysis: {str(e)}", level="ERROR")

# Live Recognition Functions
def audio_callback(indata, frames, time, status):
    if status:
        log_to_gui(f"Audio Input Error: {status}", level="ERROR")
    audio_queue.put(indata.copy())

def process_audio():
    global stop_flag
    while not stop_flag:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get().flatten()
            n_fft = 1024
            hop_length = n_fft // 4
            stft = np.abs(librosa.stft(audio_chunk, n_fft=n_fft, hop_length=hop_length))
            avg_magnitude = np.mean(stft, axis=1)
            frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=n_fft)
            dominant_idx = np.argmax(avg_magnitude)
            dominant_frequency = frequencies[dominant_idx]
            closest_note, note_freq = find_closest_note(dominant_frequency, piano_frequencies)

            label_live_note.configure(
                text=f"Dominant Frequency: {dominant_frequency:.2f} Hz\n"
                     f"Closest Note: {closest_note} ({note_freq:.2f} Hz)"
            )
            log_to_gui(f"Live Recognition: Dominant Frequency: {dominant_frequency:.2f}Hz, "
                       f"Closest Note: {closest_note} ({note_freq:.2f}Hz)")

def start_live_recognition():
    global stop_flag
    stop_flag = False
    def live_recognition_thread():
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
            while not stop_flag:
                sd.sleep(100)
    Thread(target=process_audio, daemon=True).start()
    Thread(target=live_recognition_thread, daemon=True).start()
    log_to_gui("Started live recognition.")

def stop_live_recognition():
    global stop_flag
    stop_flag = True
    log_to_gui("Stopped live recognition.")

# UI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("Audio Note Analyzer with Logs")
root.geometry("600x800")

label_title = ctk.CTkLabel(root, text="Audio Note Analyzer", font=("Arial", 20))
label_title.pack(pady=20)

button_load_file = ctk.CTkButton(root, text="Load Audio File", command=load_audio_file)
button_load_file.pack(pady=10)

button_analyze = ctk.CTkButton(root, text="Analyze Audio", command=analyze_audio)
button_analyze.pack(pady=10)

label_file_path = ctk.CTkLabel(root, text="No file selected.", font=("Arial", 14))
label_file_path.pack(pady=10)

label_audio_info = ctk.CTkLabel(root, text="", font=("Arial", 12))
label_audio_info.pack(pady=10)

label_analysis_result = ctk.CTkLabel(root, text="", font=("Arial", 14))
label_analysis_result.pack(pady=20)

label_live_recognition = ctk.CTkLabel(root, text="Live Note Recognition", font=("Arial", 20))
label_live_recognition.pack(pady=20)

label_live_note = ctk.CTkLabel(root, text="Waiting for live input...", font=("Arial", 14))
label_live_note.pack(pady=10)

button_start = ctk.CTkButton(root, text="Start Live Recognition", command=start_live_recognition)
button_start.pack(pady=10)

button_stop = ctk.CTkButton(root, text="Stop Live Recognition", command=stop_live_recognition)
button_stop.pack(pady=10)

label_logs = ctk.CTkLabel(root, text="Logs:", font=("Arial", 16))
label_logs.pack(pady=10)

text_logs = scrolledtext.ScrolledText(root, wrap="word", height=10, width=70)
text_logs.pack(pady=10)

root.mainloop()

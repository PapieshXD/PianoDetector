import librosa
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Zmienne globalne na dane audio
y, sr = None, None

def load_audio_file():
    """Funkcja do wgrywania pliku audio."""
    global y, sr  # Deklarujemy zmienne globalne
    file_path = filedialog.askopenfilename(
        title="Wybierz plik audio",
        filetypes=(("Pliki WAV", "*.wav"), ("Pliki MP3", "*.mp3"), ("Wszystkie pliki", "*.*"))
    )
    if file_path:
        label_file_path.configure(text=f"Wybrany plik: {file_path}")
        try:
            # Wczytanie pliku audio za pomocą librosa
            y, sr = librosa.load(file_path, sr=None)  # sr=None zachowuje oryginalną częstotliwość próbkowania
            if y is not None and sr is not None:
                label_audio_info.configure(
                    text=f"Długość: {len(y) / sr:.2f} s, Próbek: {len(y)}, Częstotliwość próbkowania: {sr} Hz"
                )
            else:
                label_audio_info.configure(text="Błąd: Nie udało się wczytać danych audio.")
        except Exception as e:
            label_audio_info.configure(text=f"Błąd podczas ładowania pliku: {str(e)}")

def generate_piano_frequencies():
    """Generowanie częstotliwości dźwięków pianina."""
    keys = []
    frequencies = []

    # Nazwy dźwięków
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    # Generowanie częstotliwości dla 88 klawiszy
    for n in range(88):
        octave = (n + 9) // 12  # Oktawa (np. 0, 1, 2...)
        note = notes[n % 12]  # Nazwa dźwięku (np. C, D, E...)
        freq = 27.5 * (2 ** (n / 12))  # Częstotliwość wg wzoru
        keys.append(f"{note}{octave}")  # Dodanie dźwięku do listy
        frequencies.append(freq)

    # Tworzenie słownika: klawisz -> częstotliwość
    piano_dict = dict(zip(keys, frequencies))
    return piano_dict

x = generate_piano_frequencies()
print(x)

def find_closest_note(freq, piano_dict):
    """Znajdowanie najbliższego dźwięku dla danej częstotliwości."""
    global closest_note
    closest_note = min(piano_dict.keys(), key=lambda note: abs(piano_dict[note] - freq))
    return closest_note, piano_dict[closest_note]

def analyze_audio():
    """Funkcja do analizy FFT i rozpoznawania dźwięku."""
    global y, sr, positive_frequencies, positive_magnitudes, note_freq  # Używamy zmiennych globalnych
    if y is None or sr is None:
        label_audio_info.configure(text="Brak wgranego pliku audio!")
        return

    try:
        # Obliczenie FFT
        fft = np.fft.fft(y)
        frequencies = np.fft.fftfreq(len(fft), 1 / sr)
        magnitudes = np.abs(fft)

        # Filtrujemy tylko częstotliwości dodatnie
        positive_frequencies = frequencies[:len(frequencies) // 2]
        positive_magnitudes = magnitudes[:len(magnitudes) // 2]

        # Znajdujemy dominującą częstotliwość
        dominant_frequency = positive_frequencies[np.argmax(positive_magnitudes)]

        # Dopasowanie do najbliższego dźwięku pianina
        piano_frequencies = generate_piano_frequencies()
        closest_note, note_freq = find_closest_note(dominant_frequency, piano_frequencies)

        # Wyświetlenie wyników
        label_analysis_result.configure(
            text=f"Dominująca częstotliwość: {dominant_frequency:.2f} Hz\n"
                 f"Najbliższy dźwięk: {closest_note} ({note_freq:.2f} Hz)"
        )
    except Exception as e:
        label_analysis_result.configure(text=f"Błąd podczas analizy: {str(e)}")

# UI w CTk
ctk.set_appearance_mode("dark")  # Tryb ciemny
ctk.set_default_color_theme("blue")  # Domyślny motyw

root = ctk.CTk()
root.title("Prosta analiza dźwięków pianina")
root.geometry("600x500")

# Nagłówek
label_title = ctk.CTkLabel(root, text="Analiza dźwięków pianina", font=("Arial", 20))
label_title.pack(pady=20)

# Przycisk wgrywania pliku
button_load_file = ctk.CTkButton(root, text="Wgraj plik audio", command=load_audio_file)
button_load_file.pack(pady=10)

# Przycisk do analizy pliku
button_analyze = ctk.CTkButton(root, text="Analizuj dźwięk", command=analyze_audio)
button_analyze.pack(pady=10)

# Etykieta dla ścieżki pliku
label_file_path = ctk.CTkLabel(root, text="Nie wybrano pliku.", font=("Arial", 14))
label_file_path.pack(pady=10)

# Etykieta dla informacji o audio
label_audio_info = ctk.CTkLabel(root, text="", font=("Arial", 12))
label_audio_info.pack(pady=10)

# Etykieta dla wyników analizy
label_analysis_result = ctk.CTkLabel(root, text="", font=("Arial", 14))
label_analysis_result.pack(pady=20)



def open_new_window():
    global positive_frequencies, positive_magnitudes, closest_note
    # Tworzymy nowe okno
    new_window = ctk.CTkToplevel(root)
    new_window.title("Nowe okno")
    new_window.geometry("400x300")

    # Dodajemy treści do nowego okna
    label = ctk.CTkLabel(new_window, text="Wykresy", font=("Arial", 16))
    label.pack(pady=20)

    button_close = ctk.CTkButton(new_window, text="Zamknij okno", command=new_window.destroy)
    button_close.pack(pady=20)

    # Tworzymy wykres Matplotlib
    fig, ax = plt.subplots()
    ax.plot(positive_frequencies, positive_magnitudes, label=f'Najwyższa częstotliwość')
    ax.set_title(f"fft dźwięku {closest_note} ")
    ax.set_xlabel("częstotlowość [hz]")
    ax.set_ylabel("Amplituda")
    ax.legend()

    # Dodanie wykresu do okna Tkinter
    canvas = FigureCanvasTkAgg(fig, master=new_window)  # Tworzymy widget z wykresem
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    # Dodanie przycisku zamykającego okno
    button_close = ctk.CTkButton(new_window, text="Zamknij", command=new_window.destroy)
    button_close.pack(pady=10)

label_main = ctk.CTkLabel(root, text="Porównanie wykresów", font=("Arial", 20))
label_main.pack(pady=20)

button_open_window = ctk.CTkButton(root, text="Wyświetl wykresy", command=open_new_window)
button_open_window.pack(pady=20)

# Uruchomienie aplikacji
root.mainloop()

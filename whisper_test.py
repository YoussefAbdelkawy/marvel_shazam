import threading
import tkinter as tk
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CLIP_SECONDS = 10

print("Loading Whisper model (first run downloads about 140 MB)...")
model = WhisperModel("base.en", device="cpu", compute_type="int8")


class App:
    def __init__(self, root):
        self.root = root
        self.busy = False
        root.title("Whisper Shazam Test")
        root.geometry("420x520")

        self.canvas = tk.Canvas(root, width=420, height=300, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.circle = self.canvas.create_oval(110, 40, 310, 240, fill="#4a90e2", outline="")
        self.label = self.canvas.create_text(210, 140, text="TAP", fill="white",
                                             font=("Arial", 24, "bold"))
        for item in (self.circle, self.label):
            self.canvas.tag_bind(item, "<Button-1>", self.start)

        self.status = tk.Label(root, text="Tap the circle, then speak", font=("Arial", 12))
        self.status.pack(pady=5)
        self.result = tk.Label(root, text="", wraplength=380, font=("Arial", 16), justify="center")
        self.result.pack(pady=20)

    def start(self, event=None):
        if self.busy:
            return
        self.busy = True
        self.canvas.itemconfig(self.circle, fill="#e24a4a")
        self.canvas.itemconfig(self.label, text="LISTENING")
        self.status.config(text=f"Recording {CLIP_SECONDS} seconds...")
        threading.Thread(target=self.record_and_transcribe, daemon=True).start()

    def record_and_transcribe(self):
        audio = sd.rec(int(CLIP_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        self.root.after(0, lambda: self.status.config(text="Transcribing..."))
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        self.root.after(0, lambda: self.finish(text))

    def finish(self, text):
        self.result.config(text=text or "(nothing heard)")
        self.canvas.itemconfig(self.circle, fill="#4a90e2")
        self.canvas.itemconfig(self.label, text="TAP")
        self.status.config(text="Tap again to listen")
        self.busy = False


root = tk.Tk()
App(root)
root.mainloop()
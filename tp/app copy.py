import tkinter as tk
from tkinter import filedialog
import subprocess


class YOLOv5DetectorGUI:
    def __init__(self, master):
        self.master = master
        master.title("YOLOv5 Detector")

        # self.weights_label = tk.Label(master, text="Weights:")
        # self.weights_label.pack()
        # self.weights_entry = tk.Entry(master)
        # self.weights_entry.pack()

        self.source_label = tk.Label(master, text="Source:")
        self.source_label.pack()
        self.source_entry = tk.Entry(master)
        self.source_entry.pack()
        self.browse_button = tk.Button(
            master, text="Browse", command=self.browse_source
        )
        self.browse_button.pack()

        self.run_button = tk.Button(
            master, text="Run Detection", command=self.run_detection
        )
        self.run_button.pack()

    def browse_source(self):
        filename = filedialog.askopenfilename()
        self.source_entry.delete(0, tk.END)
        self.source_entry.insert(tk.END, filename)

    def run_detection(self):
        # weights = self.weights_entry.get()
        source = self.source_entry.get()
        cmd = ["python", "detect_copy.py", "--source", source]
        subprocess.run(cmd)


def main():
    root = tk.Tk()
    yolo_gui = YOLOv5DetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

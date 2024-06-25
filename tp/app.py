import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import subprocess
import cv2
import os


class YOLOv5App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 Object Detection App")

        self.output_text = tk.Text(root, height=10, width=50, state=tk.DISABLED)
        self.output_text.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.start_button = tk.Button(
            root, text="Start Detection", command=self.start_detection
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(
            root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.RIGHT, padx=10)

        self.output_dir_button = tk.Button(
            root, text="Select Output Directory", command=self.select_output_directory
        )
        self.output_dir_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.pack(side=tk.RIGHT, padx=10)

        self.status_bar = tk.Label(
            root, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            root, variable=self.progress_var, mode="indeterminate"
        )
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.process = None
        self.output_dir = ""

    def update_status(self, message):
        self.status_bar.config(text="Status: " + message)

    def update_progress(self, percentage):
        self.progress_var.set(percentage)

    def start_detection(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)  # Clear previous output
        self.output_text.insert(tk.END, "Starting YOLOv5 object detection...\n")
        self.output_text.config(state=tk.DISABLED)
        self.update_status("Starting YOLOv5 object detection...")

        self.progress_var.set(0)
        self.progress_bar.start()

        # Run YOLOv5 detection in a separate thread
        command = ["python", "detectCopy.py", "--source", "0"]
        if self.output_dir:
            command.extend(["--output", self.output_dir])

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Redirect output and image to the text box and label
        threading.Thread(target=self.update_output_and_image).start()

    def stop_detection(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, "\nDetection process terminated by user.\n")
            self.output_text.config(state=tk.DISABLED)

            self.progress_bar.stop()
            self.progress_var.set(0)

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def update_output_and_image(self):
        def update_image(image_path):
            if os.path.exists(image_path):
                frame = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(img)
                self.image_label.config(image=img)
                self.image_label.image = img

        for line in iter(self.process.stdout.readline, ""):
            self.append_output(line.strip())

            # Update image only if the file exists
            image_path = "output.jpg"
            self.root.after(10, update_image, image_path)

        self.process.wait()

        # Stop and reset the progress bar
        self.progress_bar.stop()
        self.progress_var.set(0)

        self.update_status("Detection process completed.")
        self.enable_start_button()
        self.disable_stop_button()
        self.append_output("\nDetection process completed.")

    def append_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.config(state=tk.DISABLED)

    def enable_start_button(self):
        self.start_button.config(state=tk.NORMAL)

    def disable_stop_button(self):
        self.stop_button.config(state=tk.DISABLED)

    def select_output_directory(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(
                tk.END, f"Selected output directory: {self.output_dir}\n"
            )
            self.output_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv5App(root)
    root.mainloop()

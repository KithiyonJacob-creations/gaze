# gaze_project/main.py (Advanced: RANSAC+Poly, Head pose, Pupil refine, L2CS + RetinaFace)

import cv2
import numpy as np
import tkinter as tk
import time
import threading
import pickle
import csv
import torch
from screeninfo import get_monitors
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from PIL import Image
import math
from l2cs.pipeline import Pipeline

# ---------- HELPER FUNCTIONS ----------
def kalman_filter(init_x=0, init_y=0):
    state = np.array([[init_x], [init_y], [0], [0]])
    A = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    P = np.eye(4) * 1000
    Q = np.eye(4) * 0.1
    R = np.eye(2) * 20

    def update(z):
        nonlocal state, P
        state = A @ state
        P = A @ P @ A.T + Q
        y = z.reshape(2, 1) - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y
        P = (np.eye(4) - K @ H) @ P
        return int(state[0, 0]), int(state[1, 0])
    return update

def angles_to_vector(pitch_rad, yaw_rad):
    x = -np.cos(pitch_rad) * np.sin(yaw_rad)
    y = -np.sin(pitch_rad)
    z = -np.cos(pitch_rad) * np.cos(yaw_rad)
    return [x, y, z]

# ---------- OVERLAY CLASS ----------
class DotOverlay:
    def __init__(self, screen_width, screen_height):
        self.ready = False
        self.dot_coords = [screen_width // 2, screen_height // 2]
        self.trail = []
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.dot_radius = 15
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill='green', outline='white', width=2)
        self.trail_limit = 30
        self.canvas.create_text(screen_width // 2, screen_height // 2,
                                text="Press ENTER to start calibration", fill="white",
                                font=("Helvetica", 32), tags="instruction")
        self.root.bind('<Return>', self.start_calibration)
        self.update_dot()

    def start_calibration(self, event):
        self.canvas.delete("instruction")
        self.ready = True

    def show_message(self, text, duration=1.0):
        label = self.canvas.create_text(self.root.winfo_screenwidth() // 2,
                                        self.root.winfo_screenheight() // 2 + 100,
                                        text=text, fill="white",
                                        font=("Helvetica", 24),
                                        tags="message")
        self.root.update()
        self.root.after(int(duration * 1000), lambda: self.canvas.delete(label))

    def update_dot(self):
        if self.ready:
            x, y = self.dot_coords
            r = self.dot_radius
            self.canvas.coords(self.dot, x - r, y - r, x + r, y + r)
            self.trail.append((x, y))
            if len(self.trail) > self.trail_limit:
                self.trail.pop(0)
            self.canvas.delete("trail")
            for i, (tx, ty) in enumerate(self.trail):
                alpha = int(255 * (i + 1) / self.trail_limit)
                color = f"#{alpha:02x}0000"
                self.canvas.create_oval(tx - 3, ty - 3, tx + 3, ty + 3, fill=color, outline="", tags="trail")
        self.root.after(30, self.update_dot)

    def run(self):
        self.root.mainloop()

# ---------- BACKGROUND LOGIC ----------
def calibration_logic(overlay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline(weights='models/L2CSNet_gaze360.pkl', arch='resnet18', device=device)

    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height
    cap = cv2.VideoCapture(0)
    kf = kalman_filter()
    X_data, Y_data = [], []

    calibration_points = [
        (int(screen_width * 0.1), int(screen_height * 0.1)),
        (int(screen_width * 0.5), int(screen_height * 0.1)),
        (int(screen_width * 0.9), int(screen_height * 0.1)),
        (int(screen_width * 0.1), int(screen_height * 0.5)),
        (int(screen_width * 0.5), int(screen_height * 0.5)),
        (int(screen_width * 0.9), int(screen_height * 0.5)),
        (int(screen_width * 0.1), int(screen_height * 0.9)),
        (int(screen_width * 0.5), int(screen_height * 0.9)),
        (int(screen_width * 0.9), int(screen_height * 0.9)),
    ]

    print("[INSTRUCTION] Please follow the red dot on screen.")
    for count in range(3, 0, -1):
        print(f"Starting in {count}...")
        time.sleep(1)

    for idx, point in enumerate(calibration_points):
        print(f"[LOOK] Target at {point}. Hold gaze...")
        overlay.dot_coords = point
        overlay.show_message(f"Target {idx + 1}/{len(calibration_points)}", 1.0)

        for t in range(3, 0, -1):
            overlay.show_message(f"Hold still: {t}", 1.0)
            time.sleep(1)

        samples = []
        for i in range(20):
            ret, frame = cap.read()
            if not ret:
                continue
            result = pipeline.step(frame)
            if result.pitch.size > 0:
                pitch, yaw = result.pitch[0], result.yaw[0]
                head_pose = result.bboxes[0][0] / frame.shape[1] if result.bboxes.shape[0] > 0 else 0.5
                gaze_vec = angles_to_vector(pitch, yaw) + [head_pose]
                samples.append(gaze_vec)
            overlay.show_message(f"Collecting... {i + 1}/20", 0.05)

        if len(samples) >= 5:
            avg_vec = np.mean(samples, axis=0)
            X_data.append(avg_vec)
            Y_data.append([point[0], point[1]])
            overlay.show_message("✔️ Success", 1.0)
            print(f"[DATA] Samples: {len(samples)}, Gaze+Head: {avg_vec}, Target: {point}")
        else:
            overlay.show_message("❌ Skipped (too few samples)", 1.5)
            print(f"[WARNING] Too few samples collected at point {point}. Skipping...")

    if len(X_data) < 5:
        print("[ERROR] Not enough calibration data collected! Please ensure your face is clearly visible and follow the dot.")
        overlay.show_message("❌ Calibration failed. Try again!", duration=3.0)
        cap.release()
        return

    print("[TRAINING] Fitting screen mapping model using Polynomial RANSAC...")
    model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(min_samples=5)).fit(X_data, Y_data)

    with open("gaze_screen_mapper.pkl", "wb") as f:
        pickle.dump(model, f)

    predictions = model.predict(X_data)
    mse = mean_squared_error(Y_data, predictions)
    print(f"[EVALUATION] Calibration RMSE: {np.sqrt(mse):.2f} pixels")

    log_file = open("gaze_log.csv", mode="w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["timestamp", "screen_x", "screen_y"])

    print("[READY] Real-time tracking starts now...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.step(frame)
        if result.pitch.size > 0:
            pitch, yaw = result.pitch[0], result.yaw[0]
            head_pose = result.bboxes[0][0] / frame.shape[1] if result.bboxes.shape[0] > 0 else 0.5
            gaze_vec = angles_to_vector(pitch, yaw) + [head_pose]
            screen_x, screen_y = model.predict([gaze_vec])[0]
            smoothed = kf(np.array([screen_x, screen_y]))
            overlay.dot_coords = smoothed
            writer.writerow([time.time(), smoothed[0], smoothed[1]])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

# ---------- MAIN ----------
if __name__ == '__main__':
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height
    overlay = DotOverlay(screen_width, screen_height)

    def background_thread():
        while not overlay.ready:
            time.sleep(0.1)
        calibration_logic(overlay)

    threading.Thread(target=background_thread, daemon=True).start()
    overlay.run()

import cv2
import mediapipe as mp
import tkinter as tk
import time
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Tkinter setup
root = tk.Tk()
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
root.attributes('-transparentcolor', 'black')
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()

canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg='black', highlightthickness=0)
canvas.pack()

cap = cv2.VideoCapture(0)

# Calibration points: 3x3 grid (9 points)
calibration_points = [
    (screen_w//5, screen_h//5),
    (screen_w//2, screen_h//5),
    (screen_w - screen_w//5, screen_h//5),
    (screen_w//5, screen_h//2),
    (screen_w//2, screen_h//2),
    (screen_w - screen_w//5, screen_h//2),
    (screen_w//5, screen_h - screen_h//5),
    (screen_w//2, screen_h - screen_h//5),
    (screen_w - screen_w//5, screen_h - screen_h//5)
]
np.random.shuffle(calibration_points)

eye_features = []
screen_targets = []
dot_items = []

# Smoothing buffer
history = []
max_history = 8  # increased for smoother tracking

# Helper: compute center of landmarks
def get_center(landmarks):
    return (sum(p.x for p in landmarks) / len(landmarks),
            sum(p.y for p in landmarks) / len(landmarks))

# Calibration step
def calibrate_next_point(idx):
    if idx >= len(calibration_points):
        print("✅ Calibration done. Fitting models...")
        global model_x, model_y

        X = np.array(eye_features)
        Y = np.array(screen_targets)

        # Use polynomial + RANSAC for robustness
        poly_deg = 3
        model_x = make_pipeline(PolynomialFeatures(poly_deg), RANSACRegressor(min_samples=5)).fit(X, Y[:, 0])
        model_y = make_pipeline(PolynomialFeatures(poly_deg), RANSACRegressor(min_samples=5)).fit(X, Y[:, 1])

        print("✅ Models trained. Starting live tracking.")
        root.after(500, update_pointer)
        return

    dot_x, dot_y = calibration_points[idx]
    dot = canvas.create_oval(dot_x - 30, dot_y - 30, dot_x + 30, dot_y + 30, fill='yellow', outline='')
    root.update()

    time.sleep(2)  # let user look

    features_batch = []
    for _ in range(10):  # more frames per point
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_iris = [face_landmarks.landmark[i] for i in range(474, 478)]
            right_iris = [face_landmarks.landmark[i] for i in range(469, 473)]
            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]

            # Compute basic features
            cx_left, cy_left = get_center(left_iris)
            cx_right, cy_right = get_center(right_iris)
            cx = (cx_left + cx_right) / 2
            cy = (cy_left + cy_right) / 2
            rel_x = cx - nose_tip.x
            rel_y = cy - nose_tip.y

            bbox = face_landmarks.landmark
            min_x = min(p.x for p in bbox)
            min_y = min(p.y for p in bbox)
            max_x = max(p.x for p in bbox)
            max_y = max(p.y for p in bbox)
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y

            # Head pose estimate (rough): angles
            yaw = nose_tip.x - (left_eye_outer.x + right_eye_outer.x) / 2
            pitch = nose_tip.y - chin.y
            roll = left_eye_outer.y - right_eye_outer.y

            feature_vector = [rel_x, rel_y, min_x, min_y, bbox_w, bbox_h, yaw, pitch, roll]
            features_batch.append(feature_vector)

        time.sleep(0.1)

    if features_batch:
        avg_feature = np.mean(features_batch, axis=0)
        eye_features.append(avg_feature)
        screen_targets.append((dot_x, dot_y))
        print(f"✅ Captured & averaged {len(features_batch)} frames for point {idx+1}/{len(calibration_points)}")
    else:
        print(f"⚠️ No face detected at point {idx+1}, skipping.")

    canvas.delete(dot)
    root.after(500, lambda: calibrate_next_point(idx+1))

# Live tracking
def update_pointer():
    global dot_items, history
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_pointer)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    for item in dot_items:
        canvas.delete(item)
    dot_items.clear()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        left_iris = [face_landmarks.landmark[i] for i in range(474, 478)]
        right_iris = [face_landmarks.landmark[i] for i in range(469, 473)]
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        left_eye_outer = face_landmarks.landmark[33]
        right_eye_outer = face_landmarks.landmark[263]

        cx_left, cy_left = get_center(left_iris)
        cx_right, cy_right = get_center(right_iris)
        cx = (cx_left + cx_right) / 2
        cy = (cy_left + cy_right) / 2
        rel_x = cx - nose_tip.x
        rel_y = cy - nose_tip.y

        bbox = face_landmarks.landmark
        min_x = min(p.x for p in bbox)
        min_y = min(p.y for p in bbox)
        max_x = max(p.x for p in bbox)
        max_y = max(p.y for p in bbox)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        yaw = nose_tip.x - (left_eye_outer.x + right_eye_outer.x) / 2
        pitch = nose_tip.y - chin.y
        roll = left_eye_outer.y - right_eye_outer.y

        feature_vector = np.array([[rel_x, rel_y, min_x, min_y, bbox_w, bbox_h, yaw, pitch, roll]])

        pred_x = int(model_x.predict(feature_vector)[0])
        pred_y = int(model_y.predict(feature_vector)[0])

        history.append((pred_x, pred_y))
        if len(history) > max_history:
            history.pop(0)

        avg_x = int(np.mean([p[0] for p in history]))
        avg_y = int(np.mean([p[1] for p in history]))

        avg_x = max(0, min(screen_w, avg_x))
        avg_y = max(0, min(screen_h, avg_y))

        dot = canvas.create_oval(avg_x - 15, avg_y - 15, avg_x + 15, avg_y + 15, fill='lime', outline='')
        dot_items.append(dot)

    root.after(10, update_pointer)

# Start calibration
root.after(1000, lambda: calibrate_next_point(0))
root.mainloop()
cap.release()

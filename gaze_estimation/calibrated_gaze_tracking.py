import cv2
import dlib
import numpy as np
import torch
import pathlib
import pyautogui
import pickle
from l2cs.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# === Setup
SCREEN_RESOLUTION = pyautogui.size()
GRID_SIZE = 5
DELAY = 2  # seconds between points

# === L2CS and Dlib ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
pipe = Pipeline(
    weights=pathlib.Path("models/L2CSNet_gaze360.pkl"),
    arch="ResNet50",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    include_detector=True
)

# === Data containers ===
pitch_yaw_data = []
screen_points = []

# === Generate calibration points ===
def generate_grid(size):
    screen_w, screen_h = SCREEN_RESOLUTION
    x_steps = np.linspace(0.2, 0.8, size)
    y_steps = np.linspace(0.2, 0.8, size)
    points = [(int(screen_w*x), int(screen_h*y)) for y in y_steps for x in x_steps]
    return points

# === Main calibration loop ===
def calibrate():
    global pitch_yaw_data, screen_points
    cap = cv2.VideoCapture(0)
    points = generate_grid(GRID_SIZE)

    for pt in points:
        # Show calibration dot
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        pyautogui.moveTo(pt[0], pt[1])
        cv2.imshow("Look at the dot", img)
        pyautogui.sleep(DELAY)

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            continue
        results = pipe.step(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            continue

        pitch = results.pitch[0]
        yaw = results.yaw[0]

        pitch_yaw_data.append([pitch, yaw])
        screen_points.append(pt)

        print(f"Captured: pitch={pitch:.2f}, yaw={yaw:.2f}, screen={pt}")

        if cv2.waitKey(100) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Train models
    model_x = LinearRegression().fit(pitch_yaw_data, [p[0] for p in screen_points])
    model_y = LinearRegression().fit(pitch_yaw_data, [p[1] for p in screen_points])
    with open("calibration_model.pkl", "wb") as f:
        pickle.dump((model_x, model_y), f)
    print("✅ Calibration complete and saved to calibration_model.pkl")

if __name__ == '__main__':
    calibrate()

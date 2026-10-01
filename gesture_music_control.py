import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import urllib.request


MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# Download model
if not os.path.exists(MODEL_PATH):
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)


# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()


last_action_time = 0
cooldown = 1.0


with HandLandmarker.create_from_options(options) as detector:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = detector.detect(mp_image)

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            # Finger tips
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]

            # Convert coordinates
            thumb_x = int(thumb.x * width)
            thumb_y = int(thumb.y * height)

            index_x = int(index.x * width)
            index_y = int(index.y * height)

            middle_x = int(middle.x * width)
            middle_y = int(middle.y * height)

            # Draw points
            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (middle_x, middle_y),
                10,
                (0, 0, 255),
                -1
            )

            # Draw line
            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 255, 255),
                2
            )

            # Distance between thumb and index
            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            current_time = time.time()

            # Volume Up
            if (
                distance > 150
                and current_time - last_action_time > cooldown
            ):

                pyautogui.press("volumeup")

                cv2.putText(
                    frame,
                    "VOLUME UP",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                last_action_time = current_time

            # Volume Down
            elif (
                distance < 50
                and current_time - last_action_time > cooldown
            ):

                pyautogui.press("volumedown")

                cv2.putText(
                    frame,
                    "VOLUME DOWN",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                last_action_time = current_time

            # Play / Pause
            elif (
                abs(index_y - middle_y) < 20
                and current_time - last_action_time > cooldown
            ):

                pyautogui.press("space")

                cv2.putText(
                    frame,
                    "PLAY / PAUSE",
                    (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    3
                )

                last_action_time = current_time

            # Show distance
            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "SHOW YOUR HAND",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Show webcam
        cv2.imshow(
            "Hand Gesture Music Control",
            frame
        )

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()
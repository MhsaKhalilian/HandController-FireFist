import cv2
import mediapipe as mp

overlay_img = cv2.imread('./src/pngwing.com.png', cv2.IMREAD_UNCHANGED)
overlay_img = cv2.resize(overlay_img, (300, 250))


cap = cv2.VideoCapture(0)
mediapipeHands = mp.solutions.hands
hands = mediapipeHands.Hands()
Draw = mp.solutions.drawing_utils

x_pos, y_pos = 100, 100
show_overlay = False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    if result.multi_hand_landmarks:
        for handLandmark in result.multi_hand_landmarks:
            # Detect the positions of specific landmarks
            landmarks = {}
            for id, lm in enumerate(handLandmark.landmark):
                h, w, c = frame.shape
                landmarks[id] = (int(lm.x * w), int(lm.y * h))

            # Check if the hand is in a fist
            if (
                8 in landmarks and 6 in landmarks and landmarks[8][1] > landmarks[6][1] and
                12 in landmarks and 10 in landmarks and landmarks[12][1] > landmarks[10][1] and
                16 in landmarks and 14 in landmarks and landmarks[16][1] > landmarks[14][1] and
                20 in landmarks and 18 in landmarks and landmarks[20][1] > landmarks[18][1]
            ):
                show_overlay = True
                x_pos, y_pos = landmarks[0]  # Base of the hand (landmark 0)
            else:
                show_overlay = False

            Draw.draw_landmarks(frame, handLandmark, mediapipeHands.HAND_CONNECTIONS)

    if show_overlay:
        # Overlay the picture
        overlay_h, overlay_w, _ = overlay_img.shape
        x1, y1 = x_pos - overlay_w // 2, y_pos - overlay_h // 2
        x2, y2 = x1 + overlay_w, y1 + overlay_h

        # Ensure the overlay image stays within the frame
        if x1 < 0:
            x1, x2 = 0, overlay_w
        if y1 < 0:
            y1, y2 = 0, overlay_h
        if x2 > frame.shape[1]:
            x1, x2 = frame.shape[1] - overlay_w, frame.shape[1]
        if y2 > frame.shape[0]:
            y1, y2 = frame.shape[0] - overlay_h, frame.shape[0]

        # Extract the alpha channel for blending
        if overlay_img.shape[2] == 4:
            for i in range(y1, y2):
                for j in range(x1, x2):
                    alpha = overlay_img[i - y1, j - x1, 3] / 255.0
                    for k in range(3):  # Blend each color channel
                        frame[i, j, k] = int(
                            alpha * overlay_img[i - y1, j - x1, k] +
                            (1 - alpha) * frame[i, j, k]
                        )
        else:
            for i in range(y1, y2):
                for j in range(x1, x2):
                    for k in range(3):  # Copy each color channel
                        frame[i, j, k] = overlay_img[i - y1, j - x1, k]

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load sunglasses image (JPG format)
sunglasses = cv2.imread('sunglassespng.png')

def overlay_image(bg, fg, x, y, scale=1.0):
    """Overlay `fg` onto `bg` at (x, y) using a binary mask for JPG images."""
    fg = cv2.resize(fg, (0, 0), fx=scale, fy=scale)
    h, w, _ = fg.shape

    if y + h > bg.shape[0] or x + w > bg.shape[1]:  # Prevent overlaying out of bounds
        return bg

    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(fg_gray, 1, 255, cv2.THRESH_BINARY)

    roi = bg[y:y+h, x:x+w]

    fg_masked = cv2.bitwise_and(fg, fg, mask=mask)
    bg_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    combined = cv2.add(bg_masked, fg_masked)
    bg[y:y+h, x:x+w] = combined

    return bg

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Overlay sunglasses
        frame = overlay_image(frame, sunglasses, x + int(w * 0.15), y + int(h * 0.2), w / sunglasses.shape[1])

    # Show frame
    cv2.imshow('Real-Time Sunglasses Filter', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

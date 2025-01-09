import numpy as np
import cv2
import time  # To generate unique filenames based on time

# Initialize the face and eye cascade classifiers from XML files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Variable to store execution state
first_read = True

# Attempt to initialize the camera with the first working index
cap = None
for i in range(5):  # Try up to 5 different camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():  # Check if the camera is successfully opened
        print(f"Camera found at index {i}")
        break
else:
    print("No camera found. Exiting.")
    exit()  # Exit if no camera is found

# Start capturing video
ret, img = cap.read()

while ret:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Convert the captured image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detect the face for the region of the image to be fed to the eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ROI (Region of Interest) for the face to input to eye classifier
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Examining the length of eyes object for eye detection
            if len(eyes) >= 2:
                if first_read:
                    cv2.putText(img, "Eye detected press s to begin", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Eyes open!", (70, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            else:
                if first_read:
                    cv2.putText(img, "No eyes detected", (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
                else:
                    # Blink detected, save the image
                    print("Blink detected--------------")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
                    filename = f"blink_detected_{timestamp}.jpg"  # Create filename
                    cv2.imwrite(filename, img)  # Save the image to the disk
                    print(f"Image saved as {filename}")
                    cv2.waitKey(3000)
                    first_read = True

    else:
        cv2.putText(img, "No face detected", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Show the processed image
    cv2.imshow('img', img)

    # Control the algorithm with keys
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord('s') and first_read:
        first_read = False

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

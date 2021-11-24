# Made by KodeHak

import cv2

# Face, nose, and mouth classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_detector = cv2.CascadeClassifier('Nariz.xml')
mouth_detector = cv2.CascadeClassifier('Mouth.xml')

# Choose an image to detect smiles in
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
    # Read frames
    successful_frame_read, frame = webcam.read()

    # If frame doesn't read break out of it
    if not successful_frame_read:
        break

    # Must convert to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(grayscale_frame)

    # Find the faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get the faces
        the_face = frame[y:y + h, x:x + w]
        # Convert to grayscale
        grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_RGB2GRAY)

        # Detect nose
        nose = nose_detector.detectMultiScale(grayscale_face, scaleFactor=1.2, minNeighbors=8)

        # Detect mouth
        mouth = mouth_detector.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=30)

        # Label face as either no mask on, 50% mask on, or full mask on
        if len(nose) == 0 and len(mouth) == 0:
            cv2.putText(frame, 'FM on', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

        elif len(nose) > 0 and len(mouth) == 0:
            cv2.putText(frame, 'FM 50%', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

        elif len(nose) == 0 and len(mouth) > 0:
            cv2.putText(frame, 'FM 50%', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

        else:
            cv2.putText(frame, 'No FM on', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))



    # Display frames with smiles
    cv2.imshow('CodeHack Smile Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")
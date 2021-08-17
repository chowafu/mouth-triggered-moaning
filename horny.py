# This is a comedic personal project
# meant to make learning programming fun for me.
# Credits to my Girlfriend for giving me this dumb but fun idea.

# This program detects whether your mouth is open or closed,
# and triggers a fun sounding audio if it detects that your mouth is open.

# Handles video stream input
import cv2 as cv

# Handles Face Recognition and Landmark Prediction
import dlib

# Handles audio (stop and play)
import vlc

# Tell OpenCV to take video stream input
vid_stream = cv.VideoCapture(0)

# Tell Python-VLC to use "leeroy.mp3"
# This audio can be swapped with
# an audio of your choice
moan = vlc.MediaPlayer("leeroy.mp3")

# Use dlib's "get_frontal_face_detector()" method to detect a frontal face
detector = dlib.get_frontal_face_detector()

# Use dlib's "shape_predictor()" to localize facial landmarks
# with a trained model "shape_predictor_68_face_landmarks.dat".
# Trained model acquired from http://dlib.net/files/
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Loop to process each frame in the video stream
while True:

    # Process Frames Individually
    _, frame = vid_stream.read()

    # Flip current frame horizontally
    frame = cv.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Process grayscale images to detect the whole face
    faces = detector(gray)

    for face in faces:

        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()

        # After a face is detected,
        # Predict the 68 Facial Landmarks
        landmarks = predictor(gray, face)

        # Initialize variables to zero for each iteration
        y_dist = top_y = bottom_y = 0

        # Process all 68 Facial Landmarks
        for n in range(0, 68):

            # Locate the position in the screen of the current landmark
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Top Lip Landmark = Point 62 out of 68
            # Bottom Lip Landmark = Point 66 out of 68
            if n == 62:
                top_y = y
            if n == 66:
                bottom_y = y

            # Face Outline
            if n >= 0 and n <= 16:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Eyebrow
            elif n >= 17 and n <= 26:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Nose Bridge
            elif n >= 27 and n <= 30:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Nose Curve
            elif n >= 31 and n <= 35:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Eyes
            elif n >= 36 and n <= 47:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Outer Lip Outline
            elif n >= 48 and n <= 59:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Inner Lip Outline
            else:
                cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate distance between
        # Top Lip and Bottom Lip
        # Distance = |y1 - y2|
        y_dist = abs(top_y - bottom_y)

        # Play audio if mouth is open
        if y_dist > 20:
            moan.play()
        # Stop audio if mouth is closed
        else:
            moan.stop()

    # Show Video Window
    cv.imshow("Test Frame", frame)

    # Wait for a key event
    key = cv.waitKey(1)

    # ESC key
    if key == 27:
        break

cv.destroyAllWindows()

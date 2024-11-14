import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Set the path where the CSV file will be saved. For example, the desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
csv_file_path = csv_file_path = f"{datetime.now().strftime('%Y-%m-%d')}.csv"


print(f"CSV file will be saved at: {csv_file_path}")

video_capture = cv2.VideoCapture(0)
Aaditya_image = face_recognition.load_image_file("C:\images\pic.jpg")
Aaditya_encoding = face_recognition.face_encodings(Aaditya_image)[0]

Ratan_image = face_recognition.load_image_file("C:\images\Ratan.jpeg")
Ratan_encoding = face_recognition.face_encodings(Ratan_image)[0]

Steve_image = face_recognition.load_image_file("C:\images\Steve.jpeg")
Steve_encoding = face_recognition.face_encodings(Steve_image)[0]

tesla_image = face_recognition.load_image_file("C:/images/tesla.jpeg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

known_face_encoding = [ 
Aaditya_encoding,
Ratan_encoding,
Steve_encoding,
tesla_encoding
]

known_faces_names = [ 
"Aaditya Rathod",
"Ratan tata",
"Steve jobs",
"Nikola tesla"
]


# Initialize the list of expected students
students = known_faces_names.copy()

# Open CSV file with 'a' mode to append data each time a student is detected
try:
    with open(csv_file_path, "a", newline="") as f:
        lnwriter = csv.writer(f)
        print("CSV file opened successfully.")

        while True:
            _, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Recognize faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)

                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                    # Add the text if a person is present
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 100)
                    fontScale = 1.5
                    fontColor = (255, 0, 0)
                    thickness = 3
                    lineType = 2
                    cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                    # Write to CSV only if the student is still in the list
                    if name in students:
                        students.remove(name)  # Remove student from list to avoid duplicate entries
                        current_time = datetime.now().strftime("%H:%M:%S")
                        lnwriter.writerow([name, current_time])  # Write name and time to CSV
                        f.flush()  # Flush the buffer to force writing
                        print(f"Logged {name} at {current_time}")

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except Exception as e:
    print(f"Error opening or writing to CSV file: {e}")

video_capture.release()
cv2.destroyAllWindows()

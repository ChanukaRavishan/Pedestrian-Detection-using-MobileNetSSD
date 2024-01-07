import cv2
import numpy as np
import csv


# Loading the pre-trained MobileNetSSD model

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD_deploy.prototxt", "./MobileNetSSD/MobileNetSSD_deploy.caffemodel")


# Function to detect pedestrians in a frame and save count

def detected_pedestrians(frame_number, i, csv_writer):

    csv_writer.writerow({'Frame': frame_number, 'Pedestrian Count': i})


# Input Video
'''Change the name of the video file as required'''

video_capture = cv2.VideoCapture("./input_images_and_videos/pedestrian_survaillance.mp4")

frame_number = 0
class_id = 0

# Creating a CSV file to save the pedestrian count

csv_file = open('pedestrians_count.csv', 'w', newline='')
csv_writer = csv.DictWriter(csv_file, fieldnames=['Frame', 'Pedestrian Count'])
csv_writer.writeheader()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame to have a maximum width of 800 pixels (can adjust as needed)
    frame = cv2.resize(frame, (800, 600))

    # Converting the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Setting the input to the network and perform forward pass
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # by considering the confidence we can filter out weak directions
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # 15 corresponds to the class 'person' in the MobileNetSSD model
                # Extracting bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([800, 600, 800, 600])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detected_pedestrians(frame_number, i, csv_writer)

                cv2.putText(frame, f'Frame No: {frame_number},Pedestrian Count: {i}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # output frame
    cv2.imshow("Pedestrian Counting", frame)

    frame_number +=1

    # press q to release the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

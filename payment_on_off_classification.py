import cv2
import numpy as np

# Loading the pre-trained MobileNetSSD model
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD_deploy.prototxt", "./MobileNetSSD/MobileNetSSD_deploy.caffemodel")

def classify_pedestrians(frame):

    #letter colors
    Red = 0
    Green = 0

    frame = cv2.resize(frame, (800, 600))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_brown = np.array([10, 60, 60])
    upper_brown = np.array([30, 180, 180])

    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    contours, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                detected_people_boxes = []

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.2:
                        class_id = int(detections[0, 0, i, 1])
                        if class_id == 15:
                            box = detections[0, 0, i, 3:7] * np.array([800, 600, 800, 600])
                            (startX, startY, endX, endY) = box.astype("int")

                            detected_people_boxes.append((startX, startY, endX, endY))

                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            label = f"Person: {confidence:.2f}"
                            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                for (x, y, w, h) in detected_people_boxes:
                    human_midx = x + w // 2
                    human_midy = y + h // 2

                    cv2.line(frame, (human_midx, human_midy), (cx, cy), (0, 0, 255), 2)

                    distance = abs(human_midx - cx) if human_midx < frame.shape[1] * 3 // 4 else -1

                    # Classifying pedestrians based on distance to the pavement
                    if distance != -1 or distance > 80:
                        classification = "Outside Pavement"
                        Red = 255
                    else:
                        classification = "On Pavement"
                        Green = 255

                    # Displaying the classification and distance on the frame
                    cv2.putText(frame, f'{classification}, Dist: {distance}', (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, Green, Red), 2)

    return frame

# Input video
cap = cv2.VideoCapture("./newdata/1.MOV")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    processed_frame = classify_pedestrians(frame)
    
    cv2.imshow('Pedestrian Classification', processed_frame)
    
    # Breaking the loop when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing resources
cap.release()
cv2.destroyAllWindows()

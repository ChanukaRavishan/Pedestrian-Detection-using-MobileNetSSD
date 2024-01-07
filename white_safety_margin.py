import cv2
import numpy as np


# Loading the pre-trained MobileNetSSD model
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD_deploy.prototxt", "./MobileNetSSD/MobileNetSSD_deploy.caffemodel")

'''Function to identify pedestrians on and outside of the brown pavement'''



def classify_pedestrians(frame):

    #letter colors
    Red = 0
    Green = 0

    frame = cv2.resize(frame, (800, 600))
    
    # Converting the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 150])  # Adjust these values as needed, need to apply setting specific white values
    upper_white = np.array([180, 30, 180])  

    # Create a mask for white color in the frame
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_contours = [contour for contour in contours if cv2.contourArea(contour) > 200]

    # Draw white contours on the frame
    cv2.drawContours(frame, white_contours, -1, (255, 255, 255), 2)

    for contour in white_contours:
        # Calculate the bounding box for the white contour
        x, y, w, h = cv2.boundingRect(contour)
        linear_mark = x
        white_margin_mid = (x + w // 2, y + h // 2)

    
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Set the input to the network and perform a forward pass
        net.setInput(blob)
        detections = net.forward()

        # List to store bounding boxes of detected people
        detected_people_boxes = []

        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Consider confidence to filter out weak detections
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15:  # 15 corresponds to the class 'person' in the MobileNetSSD model
                            
                    box = detections[0, 0, i, 3:7] * np.array([800, 600, 800, 600])
                    (startX, startY, endX, endY) = box.astype("int")

                    detected_people_boxes.append((startX, startY, endX, endY))


                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        for (px, py, pw, ph) in detected_people_boxes:
            # Calculate the center of the pedestrian
            pedestrian_mid = (px + pw // 2, py + ph // 2)

            # Check if the pedestrian is between the white margins
            if pedestrian_mid[0] < frame.shape[1] * 1 // 4 or x < pedestrian_mid[0] < linear_mark :
                classification = "Not On Road"
                Green = 255
            else:
                classification = "On Road"
                Red = 255

            cv2.putText(frame, f'{classification}', (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, Green, Red), 2)

    return frame




'''

    for contour in contours:
        # Filter contours based on area (greater than 2000 pixels)
        if cv2.contourArea(contour) > 2000:
            # Draw contour on the frame
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)

            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Convert the frame to a blob
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

                # Set the input to the network and perform a forward pass
                net.setInput(blob)
                detections = net.forward()

                # List to store bounding boxes of detected people
                detected_people_boxes = []

                # Loop over the detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    # Consider confidence to filter out weak detections
                    if confidence > 0.2:
                        class_id = int(detections[0, 0, i, 1])
                        if class_id == 15:  # 15 corresponds to the class 'person' in the MobileNetSSD model
                            
                            box = detections[0, 0, i, 3:7] * np.array([800, 600, 800, 600])
                            (startX, startY, endX, endY) = box.astype("int")

                            detected_people_boxes.append((startX, startY, endX, endY))

                            # Drawing the bounding box and label on the frame
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            label = f"Person: {confidence:.2f}"
                            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                
                # Now, the variable 'detected_people_boxes' contains the bounding boxes of detected people.

                for (x, y, w, h) in detected_people_boxes:

                    human_midx = x + w // 2
                    human_midy = y + h // 2

                    cv2.line(frame, (human_midx, human_midy), (cx, cy), (0, 0, 255), 2)

                    
                    # Classifying pedestrians based on white safety margins, whether they are inside or outside the roadway

                    if :
                        classification = "Outside Pavement"
                    else:
                        classification = "On Pavement"

                    # Displaying the classification and distance on the frame
                    cv2.putText(frame, f'{classification}, Dist: {distance}', (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

'''

# Input video

cap = cv2.VideoCapture("./newdata/white.MOV") 

frame_number = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    processed_frame = classify_pedestrians(frame)
    
    cv2.imshow('Pedestrian Classification', processed_frame)
    
    frame_number += 1
    
    # Breaking the loop when q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing resources
cap.release()
cv2.destroyAllWindows()

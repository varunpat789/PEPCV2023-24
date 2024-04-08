from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import math 

#For an Intel Realsense Camera
#GOAL: Locate Buoys and distance with trained model

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

#NOTE: bestBuoys2.pt was trained using images found in RoboFlow
model = YOLO("bestBuoys2.pt")

classNames = ["buoy!"]

def midpoint(ptA, ptB):
    return int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5)

while True:

    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    depth_frame = frame.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


    #feeds camera image into the model, returns each detected object
    results = model(color_image, stream=True, conf=0.6)

    for r in results:
        #yolo returns the coords of a bounding box around the object
        boxes = r.boxes

        for box in boxes:
            # define boxes around each detected object and draw a rectangle
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #get distance using depth frame
            midpoint_val = midpoint((x1,y1),(x2,y2))
            dist = depth_frame.get_distance(midpoint_val[0], midpoint_val[1])
            print("detected object at {0}".format(int(dist)))

            # confidence of object
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence = ",confidence)

            # object type
            cls = int(box.cls[0])
            print("Object: ", classNames[cls])

            # printing details of object on frame
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2


            cv2.putText(color_image, classNames[cls], org, font, fontScale, color, thickness)
            #cv2.putText(color_image, str(round(dist,2)), org, font, fontScale, color, thickness)


    cv2.imshow("color", color_image)

    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()





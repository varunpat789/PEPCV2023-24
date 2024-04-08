import pyrealsense2 as rs
import numpy as np
import cv2
import os 
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

n = 1

directory = 'treesPoolLines'
os.chdir(directory) 

while True:

    frame = pipeline.wait_for_frames()

    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())


    cv2.imshow("depth", depth_image)
    cv2.imshow("color", color_image)
    


    filename = "treesPoolLine" + str(n) + ".jpg"
    cv2.imwrite(filename, color_image, [cv2.IMWRITE_JPEG_QUALITY, 50])

    n = n + 1

    time.sleep(.1)

    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()



import cv2
import numpy as np


#Load Haar-cascade nose classifiers.
nose_cascade = cv2.CascadeClassifier("./nose_cascade.xml")

#Load mustache image.
mustache_mask = cv2.imread("dali.png")

#Define video stream as coming from first webcam connected to system.
video = cv2.VideoCapture(0)
scaling_factor = 1


'''
Video processing loop
'''

while True:
    
    #Capture webcam stream.
    ret, frame = video.read()

    #Extract luminance values from frame.
    #Haar-cascade classifiers require greyscale images.
    frame = cv2.resize(frame, None, fx = scaling_factor, fy = scaling_factor,
                       interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Find noses.
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    
    #Loop across each detected nose.
    for nose in nose_rects:

        #Get nose coordinates.
        (x, y, w, h) = nose
        h, w = int(2 * h), int(2  *w)
        x -= int(0.25 * w)
        y -= int(0.45 * h)
        frame_roi = frame[y:y + h, x:x + w]

        #Scale mustache to nose.
        mustache_mask_scaled = cv2.resize(mustache_mask, (w, h),
                                         interpolation=cv2.INTER_AREA)

        #Make mask, threshold, and invert for compositing.
        gray_mask = cv2.cvtColor(mustache_mask_scaled, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        #Combine both masked foreground (mustache) and background (video)
        fg = cv2.bitwise_and(mustache_mask_scaled, mustache_mask_scaled,
                             mask = mask)
        bg = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        frame[y:y + h, x:x + w] = cv2.add(fg, bg)
    
    #Display frame.
    cv2.imshow("Ceci n'est pas Dali", frame)

    #Break loop on ESC key press.
    c = cv2.waitKey(1)
    if c == 27:
        break

#Clean up.
video.release()
cv2.destroyAllWindows()

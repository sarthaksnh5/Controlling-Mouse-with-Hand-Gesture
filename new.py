import cv2
import numpy as np
import time
import pyautogui
from keras.models import load_model
import copy

path = 'detection/VGG_cross_validated.h5'

prediction = ''
action = ''
score = 0
img_counter = 500

save_images = selected_gesture = False, 'peace'

gesture_names = {0: 'Fist', 1: 'L', 2: 'Okay', 3: 'Palm', 4: 'Peace'}

model = load_model(path)

def predict_rgb_image(image):
    result = gesture_names[model.predict_classes(img)[0]]
    print(result)
    return(result)

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print('pred_array: ', pred_array)
    result = gesture_names[np.argmax(pred_array)]
    print('Result: ', result)
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

def remove_Background(frame):
    fgmask= bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3,3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#Open Camera object
cap = cv2.VideoCapture(0)

def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
 

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')
pyautogui.FAILSAFE = False

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)

while(cap.isOpened()):

    #Measure execution time 
    start_time = time.time()
    
    #Capture frames from the camera
    ret, frame = cap.read()
    ret1, frame1 = cap.read()
    
    #Blur the image
    blur = cv2.blur(frame,(3,3))
 	
 	#Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    
    #Kernel matrices for morphological transformation    
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)
    
    #Find contours of the filtered frame
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    #Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    #cv2.imshow('Dilation',median)
    
	#Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0	
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
            
	#Largest area contour 			  
    cnts = contours[ci]

    #Find convex hull
    hull = cv2.convexHull(cnts)
    
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    
    #Get defect points and draw them in the original image
    FarDefect = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        #cv2.line(frame,start,end,[0,255,0],1)
        #cv2.circle(frame,far,10,[100,255,255],3)
    
	#Find moments of the largest contour
    moments = cv2.moments(cnts)
    
    #Central mass of first order moments
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    centerMass=(cx,cy)    
    
    #Draw center mass
    cv2.circle(frame,centerMass,7,[100,0,255],2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)     
    
    #Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    for i in range(0,len(FarDefect)):
        x =  np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
        distanceBetweenDefectsToCenter.append(distance)
    
    #Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
 
    #Get fingertip points from contour hull
    #If points are in proximity of 80 pixels, consider as a single point in the group
    finger = []
    for i in range(0,len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])
    
    #The fingertip points are 5 hull points with largest y coordinates  
    finger =  sorted(finger,key=lambda x: x[1])   
    fingers = finger[0:5]
    
    #Calculate distance of each finger tip to the center mass
    fingerDistance = []
    for i in range(0,len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
        fingerDistance.append(distance)
    
    #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    #than the distance of average finger webbing to center mass by 130 pixels
    result = 0
    for i in range(0,len(fingers)):
        if fingerDistance[i] > AverageDefectDistance+130:
            result = result +1
    
    #Print number of pointed fingers
    #cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)
    
    #show height raised fingers
    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        
    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    x1 = (x/480) * 1920
    y1 = (y/200) * 1080
    pyautogui.moveTo(x1,y1, duration=0.1)
    print("X: ",x1, " Y: ", y1)

    frame1 = cv2.bilateralFilter(frame1, 5, 50, 100)
    frame1 = cv2.flip(frame1, 1)
    
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    
    img1 = remove_Background(frame1)
    img1 = img1[0: int(0.8*frame1.shape[0]), int(0.5*frame1.shape[1]):frame1.shape[1]]

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1, (blurValue, blurValue), 0)
    ret1, thresh1 = cv2.threshold(blur1, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = copy.deepcopy(thresh1)
    contours1, hierarchy1 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours1)
    maxArea = -1 
    if length > 0:
        for i in range(length):
            temp = contours1[i]
            area1 = cv2.contourArea(temp)
            if area1 > maxArea:
                maxArea = area1
                ci1 = i

        res1 = contours[ci1]
        hull1 = cv2.convexHull(res1)
        drawing = np.zeros(img1.shape, np.uint8)

    target1 = np.stack((thresh1,)*3, axis=-1)
    target1 = cv2.resize(target1, (224,224))
    target1 = target1.reshape(1, 224, 224, 3)
    prediction1, score1 = predict_rgb_image_vgg(target1)

    print(prediction1)
    if prediction1 == "Fist":
        pyautogui.click(x1,y1)

 
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    
    ##### Show final image ########
    cv2.imshow('Dilation',frame)
    ###############################
    
    #Print execution time
    #print time.time()-start_time
    
    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()

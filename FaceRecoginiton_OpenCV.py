
# coding: utf-8

# In[1]:

# importing opencv2 library
import cv2


# In[4]:

# loading cascades
# cascades are the xml files which serve as filters
# for reconizing particular element in the image
# here we are loading harr cascades

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')


# In[17]:

# defining a function to detect elements
# cascades work only on grayscale images

def detect(gray_scale_img, original_img):
    
    # scale decides by how much the size of filter/kernel should be increased
    # minNeighbors specify how many faces should be neighbours to be accepted as face
    # SomeCascadeClassifier.detectMultiScale returns (x,y,w,h) tuple 
    # x,y are co-ordinates of upper left corner
    # w,h are height and width of rectangles
    
    faces = face_cascade.detectMultiScale(gray_scale_img, scaleFactor=1.3,minNeighbors=5 )
    
    for (x,y,w,h) in faces:
        # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]
        # detect on gray print rectangle on color
        cv2.rectangle(original_img, (x,y),(x+w,y+h), (255,0,0), 2)
        
        # after detecting the face we need to search in the face region
        # for eyes to save computation time
        # hence defining a region of interest roi
        roi_gray = gray_scale_img[x:x+w,y:y+h]
        
        # for displaying rectangle over actual color image and not grayscale img
        roi_color = original_img[x:x+w,y:y+h]
        
        # decting eyes in region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        
        # drawing rectangles on the frame for eyes
        for (ex,ey,ew,eh) in eyes:
            # we need to draw rectangles on roi_gray
            # detect on gray print rectangle on color
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (0,255,0), 2)
    
    # now original image has rectangles drawn on it
    return original_img


# In[20]:

# video capture 0 for internal device
video_capture = cv2.VideoCapture(0)

while True:
    
    _, frame = video_capture.read()
    
    # frame is color image captured
    # converting this image to grayscale
    
    # gray_scale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    
    canvas = detect(im_gray,frame)
    
    cv2.imshow('Video',canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()


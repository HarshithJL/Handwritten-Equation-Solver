
import numpy as np
import cv2
import operator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Activation,Flatten
from tensorflow.keras.models import model_from_json
# import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn import preprocessing
# from keras.models import model_from_json

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 28
RESIZED_IMAGE_HEIGHT = 28

class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True

def load_model():
    try:
        #Loading the model
        json_file = open("newmodel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("newweights.hdf5")
        print("Model successfully loaded from disk.")

        #compile the model again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except:
        print("""Model not found""")
        return None

def oneHotEncoding():
    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(classess)
    onehot_encode = preprocessing.OneHotEncoder(sparse = False)
    label_encoded = label_encoded.reshape( len(label_encoded),1)
    onehot_encode = onehot_encode.fit_transform(label_encoded)
    return onehot_encode

def get_key(val):
	for key,value in labels.items():
		if value == val:
			return key

def quadratic(b):
    X = []
    A=b[0]
    B=b[1]
    C=b[2]
    D=(B**2-(4*A*C))
    D1=(4*A*C-(B**2))

    if D>0:
        r1=(-B+np.sqrt(D))/2*A
        r2=(-B-np.sqrt(D))/2*A

    elif D==0:
        r1=-B/2*A
        r2=r1
    elif D<0:
        r1=(-B+1j*np.sqrt(D1))/2*A
        r2=(-B-1j*np.sqrt(D1))/2*A

    print('Root 1 : ',r1)
    print('Root 2 : ',r2)

def linear(new):
  l = new[1]
  if(l>0):
    l=-l
  return (l/new[0])

classess=['-','+','0','1','2','3','4','5','6','7','8','9','x','y','z']
t = oneHotEncoding()
labels = {}
for x,y in zip(classess,t):
    l = []
    for z in y:
        l.append(z)
    labels[x] = l

model = load_model()
img_size = 28

# print("Enter choice")
# ch = input()
allContoursWithData = []
validContoursWithData = []
img = cv2.imread('cubic.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlurred = cv2.GaussianBlur(gray, (1,1), 0)
ret,imgThresh = cv2.threshold(imgBlurred,95,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
npaContours,npaHierarchy = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

imgThresh2= cv2.bitwise_not(imgThresh)
for npaContour in npaContours:
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)

for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list


validContoursWithData.sort(key = operator.attrgetter("intRectX"))
n = len(validContoursWithData)       # sort contours from left to right
del validContoursWithData[n-3:]

test_data = []

for contourWithData in validContoursWithData:            # for each contour
        imgROI = imgThresh2[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight+5,
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth+5]

        im_resize = cv2.resize(imgROI,(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)
        test_data.append(im_resize)

# print(len(test_data))

index = []

var = ['x','y','z']
numbers = ['1','2','3','4','5','6','7','8','9']
symbol = ['+','-']

first = True
for i in test_data:
  i = np.array(i).reshape(-1,28,28,1)
  i = i/255.0
  y = model.predict(i)
  idx = list(y[0]).index(y[0].max())
  lst = [0]*len(classess)
  lst[idx] = 1
  name = get_key(lst)
  print(name)

#   print(classes[list(y_pred[0]).index(y_pred[0].max())])
#   if(first):
#     if(classes[list(y_pred[0]).index(y_pred[0].max())] in var):
#       index.append('1')
#       index.append(classes[list(y_pred[0]).index(y_pred[0].max())])
#     else:
#       index.append(classes[list(y_pred[0]).index(y_pred[0].max())])
#     first =False
#   else:
#       index.append(classes[list(y_pred[0]).index(y_pred[0].max())])



# print(index)
# coefficients = []

# i=0
# while(i<len(index)):
#     l = index[i]
#     if l in numbers or l in symbol:
#         coefficients.append(index[i])
#     elif l in var :
#         if (i+1 < (len(index)) and index[i+1] in numbers):
#             i+=1
#     i+=1

# print(coefficients)

# new = []
# i=0
# while(i<len(coefficients)):
#     l = coefficients[i]
#     if(i==0):
#       new.append(int(l))
#     elif l in symbol:
#         if(i+1 < len(coefficients)):
#             if(coefficients[i+1] in numbers):
#                 if l =='+':
#                     new.append(int(coefficients[i+1]))
#                 elif l == '-':
#                     new.append(-int(coefficients[i+1]))
#             else:
#                 if l =='+':
#                     new.append(1)
#                 elif l == '-':
#                     new.append(-1)
#     i+=1
# print(new)
# # new.append(0)
# if ch == 'quadratic':
#   quadratic(new)
# elif ch =='linear':
#   print(linear(new))
# elif ch == 'cubic':
#   print(np.roots(new))

import pickle
import json
import numpy as np
import base64
import cv2
from wavelet import w2d



__class_name_to_number = {}
__class_number_to_name = {}
__model = None # when you type like __model that means the model is private to this file


# classifies image, it can either do for given base64 code of an img or a file to the image in your folder
def classify_image(image_base64_data, file_path=None): 
    imgs = get_cropped_image(file_path, image_base64_data)

    result = []

    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1),scaled_img_har.reshape(32 * 32, 1)))


        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float) # some of the api we are going to use need float data

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0], #gives the probabilty comparison to other celebs
            'class_dictionary': __class_name_to_number
        }) # since we supply only one image we only choose first one

         
    return result

#loads the ML model
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number 
    global __class_number_to_name 

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./artifacts/good_model.pkl", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

#number to name
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

# this funciton takes a base 64 string and returns you a cv2 image
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# crops image if it has 2 eyes and clear image and saves it in cropped faces. if it has 2 faces it will save both of them un the cropped faces list and return both
def get_cropped_image(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./open_cv/haracascade_frontalface_default.xml.txt') # these are harrcascade xml that allows you to detect different features of the face
    eye_cascade = cv2.CascadeClassifier('./open_cv/haracascade_eye.xml.txt')

    if image_path:
        img = cv2.imread(image_path)
    else: 
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_Rob():
    with open("base64.txt") as f:  # open the base 64 encoded image u saved
        return f.read()

if __name__=='__main__':
    load_saved_artifacts()
    # print(classify_image(get_b64_test_image_for_Rob(), None))
    print(classify_image(None, "./test_images/brad1.jpg"))
    print(classify_image(None, "./test_images/jolie1.jpg"))
    print(classify_image(None, "./test_images/kate1.jpg"))
    print(classify_image(None, "./test_images/will1.jpg"))
# u need sklearn lib installed in the env even tho u dont have to import it
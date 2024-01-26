import cv2
import dlib
import numpy as np
from imutils import face_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        return None

    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    return shape

def extract_features(landmarks):
    features = []
    for (x, y) in landmarks:
        features.extend([x, y])
    return features

def train_model(real_path, fake_path, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    data = []
    labels = []

    for i in range(1, 51):
        real_image_path = f"{real_path}/real_face_{i}.jpg"
        image = cv2.imread(real_image_path)
        landmarks = get_landmarks(image, detector, predictor)
        if landmarks is not None:
            features = extract_features(landmarks)
            data.append(features)
            labels.append(0)

    for i in range(1, 51):
        fake_image_path = f"{fake_path}/fake_face_{i}.jpg"
        image = cv2.imread(fake_image_path)
        landmarks = get_landmarks(image, detector, predictor)
        if landmarks is not None:
            features = extract_features(landmarks)
            data.append(features)
            labels.append(1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return classifier

def test_model(model, test_image_path, detector, predictor):
    test_image = cv2.imread(test_image_path)
    test_landmarks = get_landmarks(test_image, detector, predictor)

    if test_landmarks is not None:
        test_features = extract_features(test_landmarks)
        prediction = model.predict([test_features])

        if prediction[0] == 0:
            print("Real face")
        else:
            print("Fake face")
    else:
        print("No face detected in the test image")

def main():
    real_faces_path = "dataset/real_faces"
    fake_faces_path = "dataset/fake_faces"
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"
    test_image_path = "test_images/test_image.jpg"

    model = train_model(real_faces_path, fake_faces_path, predictor_path)
    test_model(model, test_image_path, dlib.get_frontal_face_detector(), dlib.shape_predictor(predictor_path))

if __name__ == "__main__":
    main()

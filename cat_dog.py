import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import cv2

# Load pre-trained model
model1 = SVC(max_iter=-1, kernel='linear', class_weight='balanced', gamma='scale')

# Assuming you have the matrix and labels files already loaded
imagematrix = np.load("matrix.npy")
imagelabels = np.load("labels.npy")

# Split the data into training and testing sets
(train_img, test_img, train_label, test_label) = train_test_split(imagematrix, imagelabels, test_size=0.2, random_state=50)

# Train the SVM model
model1.fit(train_img, train_label)

# Streamlit app
st.title("Image Classification with SVM")

# Upload image through file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If image is uploaded, make prediction and display the image
if uploaded_file is not None:
    content = uploaded_file.read()
    st.image(content, caption="Uploaded Image", use_column_width=True)

    # Resize and flatten the uploaded image
    uploaded_img = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), 1)
    pixel = cv2.resize(uploaded_img, (128, 128)).flatten()
    rawImage = np.array([pixel])

    # Make prediction using the SVM model
    prediction1 = model1.predict(rawImage.reshape(1, -1))

    st.write(f"Prediction by SVM: {prediction1[0]}")

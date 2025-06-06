import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

model = load_model("C:/Users/PBanerjee/Desktop/DIGI_PROJ/myenv/resnet50v2.keras")


def preprocessing(img):
    processed_image = img.resize((255, 255))
    processed_image = np.array(processed_image) / 255.0  # Normalize
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image


st.title("ID card classifier")
st.write("Upload an image of Aadhaar card or Pan card or Bank Cheque or CML")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify"):
        # Preprocess image
        processed_image = preprocessing(image)

        # Predict
        prediction = model.predict(processed_image)[0]  # softmax output

        # Get index of highest probability
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index] * 100

        # Define class labels (must match your class_indices order)
        labels = ['UIDAI', 'Cheque', 'CML', 'PAN Card']

        # Show result
        st.success(f"This is a **{labels[predicted_index]}** ({confidence:.2f}% confidence)")
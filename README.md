# Hand Gesture Recognition Model

This project implements a Hand Gesture Recognition model using a Convolutional Neural Network (CNN) to accurately identify and classify different hand gestures from image data. The model enables intuitive human-computer interaction and gesture-based control systems.

## Dataset

The dataset used for training the model is the [LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog) from Kaggle. The dataset contains images of 10 different hand gestures, each categorized into one of the following classes:

- 01_palm
- 02_l
- 03_fist
- 04_fist_moved
- 05_thumb
- 06_index
- 07_ok
- 08_palm_moved
- 09_c
- 10_down

The dataset is organized into folders, where each folder corresponds to a different gesture category.

## Model Architecture

The Hand Gesture Recognition model is built using a Convolutional Neural Network (CNN). Below are the key layers used in the model:

- **Conv2D**: Convolutional layers for feature extraction.
- **MaxPooling2D**: Pooling layers for down-sampling the feature maps.
- **Flatten**: Converts the 2D matrix data to a 1D vector.
- **Dense**: Fully connected layers for classification.
- **Dropout**: Regularization to prevent overfitting.

### Model Summary

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

### Training

The model was trained on the dataset with the following results:

- **Training Accuracy**: 99.84%
- **Validation Accuracy**: 99.90%

The model achieved high accuracy, indicating strong performance in classifying the hand gestures.

## Saving and Loading the Model

The trained model is saved as `hand_gesture_recognition_model.h5` and can be loaded for future use:

```python
from tensorflow.keras.models import load_model

model = load_model('hand_gesture_recognition_model.h5')
## Gradio GUI for Testing

A Gradio-based GUI is provided to allow users to upload images of hand gestures and get real-time predictions.
Running the GUI

To launch the Gradio interface, run the following code:

```python
import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = load_model('hand_gesture_recognition_model.h5')

# Define the label encoder
label_encoder = LabelEncoder()
gesture_labels = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
label_encoder.fit(gesture_labels)

# Define the image size
img_size = 128

# Prediction function
def predict_gesture(image):
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Preprocess the image: resize and normalize
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the gesture
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return predicted_label[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_gesture,
    inputs=gr.Image(),  # Image input
    outputs=gr.Textbox(),  # Text output for prediction
    title="Hand Gesture Recognition",
    description="Upload an image of a hand gesture, and the model will predict the gesture."
)

## Launch the interface
interface.launch()
```

## Using the GUI

    Upload an image of a hand gesture.
    The model will predict the gesture and display the result in the text box.

## Conclusion

This project demonstrates the power of Convolutional Neural Networks (CNNs) in accurately classifying hand gestures, enabling intuitive and interactive applications. The high accuracy achieved in both training and validation showcases the effectiveness of the model.

Feel free to explore and modify the code to suit your needs!

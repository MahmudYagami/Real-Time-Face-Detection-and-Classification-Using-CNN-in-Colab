# Real-Time Face Detection and Recognition with Custom CNN Model

This project implements a real-time face detection and recognition system using Haar cascades and a Convolutional Neural Network (CNN). The project includes preprocessing, model training, and real-time face recognition through a webcam interface. It is built in Google Colab.

## Features

- **Face Detection**: Uses Haar cascades to detect faces in images.
- **Face Preprocessing**: Crops and resizes detected faces for uniform input to the model.
- **CNN Training**: A custom CNN model is trained on the preprocessed data for face recognition.
- **Real-Time Recognition**: Recognizes faces in a live video stream using a webcam.

## Project Workflow

1. **Data Preprocessing**:
   - Faces are detected using Haar cascades.
   - Detected faces are cropped, resized to 224x224 pixels, and saved for model training.

2. **Model Training**:
   - A CNN is designed with layers for feature extraction and classification.
   - The model is trained on the preprocessed face dataset with TensorFlow's `ImageDataGenerator` for data augmentation.

3. **Real-Time Recognition**:
   - A webcam video feed is processed using JavaScript for real-time image capture.
   - The trained CNN model predicts the class of detected faces.
   - Predictions are displayed on the video feed overlay.

## Technologies Used

- **Google Colab**: For development and training.
- **OpenCV**: For image processing and face detection.
- **TensorFlow/Keras**: For building and training the CNN model.
- **JavaScript**: For handling live webcam video streams.
- **Python**: For the overall implementation.

## How to Use

1. Clone or download this repository.
2. Open the `.ipynb` file in Google Colab.
3. Prepare the dataset by organizing images into folders for each class.
4. Update paths to your dataset and pre-trained model in the notebook.
5. Run all cells sequentially:
   - Preprocess the data.
   - Train the CNN model.
   - Start the real-time recognition interface.
6. Point your webcam at a face to see live recognition results.

## Directory Structure

```
project
├── haarcascade_frontalface_default.xml  # Haar cascade XML for face detection
├── Final_Model.h5                      # Pre-trained CNN model
├── dataset/                            # Original dataset
├── preprocessed/                       # Preprocessed face images
├── notebook.ipynb                      # Main project notebook
```

## Requirements

Install the following libraries before running the project:

- Python 3.7+
- OpenCV
- TensorFlow
- NumPy
- Matplotlib
- PIL

## Example Outputs

- Preprocessed images saved in the `preprocessed/` folder.
- Accuracy and loss plots from model training.
- Real-time face detection and recognition displayed in the webcam interface.

## Contributions

Contributions are welcome! If you encounter issues or have suggestions, feel free to create an issue or pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


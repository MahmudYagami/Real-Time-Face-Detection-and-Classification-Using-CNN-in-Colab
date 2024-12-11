# Real-Time-Face-Detection-and-Classification-Using-CNN-in-Colab

# Real-Time Face Detection and Classification Using CNN

This project demonstrates how to build a real-time face detection and classification system using OpenCV, Haar Cascades, and a Convolutional Neural Network (CNN). The project includes face preprocessing, model training, and evaluation.

## Features

- **Face Detection:** Uses Haar Cascade Classifier for detecting faces in grayscale images.
- **Data Preprocessing:** Automatically preprocesses detected faces to a uniform size (224x224).
- **Model Training:** A CNN is implemented for face classification with real-time detection capabilities.
- **Callbacks for Optimization:** Includes early stopping and model checkpointing to optimize training.

## How It Works

1. **Face Detection with Haar Cascade:**
   - Detect faces in images from the dataset.
   - Preprocess the detected faces by converting them to grayscale and resizing them to 224x224 pixels.

2. **Dataset Preparation:**
   - Preprocessed images are stored in folders corresponding to their class labels.
   - The dataset is split into training and validation sets using TensorFlow's `ImageDataGenerator`.

3. **CNN Model:**
   - A Sequential model with multiple convolutional layers, pooling layers, and dropout for regularization.
   - Outputs a softmax probability distribution for classification.

4. **Training and Evaluation:**
   - Model is trained on the preprocessed dataset with early stopping and model checkpointing.
   - Validation data is used to monitor performance.

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow/Keras
- Matplotlib
- NumPy

Install dependencies using:
```bash
pip install opencv-python tensorflow matplotlib numpy
```

## File Structure

- `haar_cascade_face_detection.py`: Code for face detection and preprocessing.
- `cnn_training.py`: Code for training the CNN model.
- `data/`: Contains the original dataset.
- `preprocessed_data/`: Stores the preprocessed dataset.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-time-face-detection-cnn.git
   cd real-time-face-detection-cnn
   ```

2. Run face detection and preprocessing:
   ```bash
   python haar_cascade_face_detection.py
   ```

3. Train the model:
   ```bash
   python cnn_training.py
   ```

4. Evaluate the model or use it for real-time detection.

## Results

- Accuracy: Achieved X% accuracy on the validation set (update with your results).
- Loss: Validation loss was minimized to X (update with your results).

## Future Improvements

- Use a pre-trained model like MobileNet or ResNet for better performance.
- Extend the project to include real-time webcam integration.
- Experiment with other datasets and architectures.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Happy Coding! If you have any questions, feel free to open an issue or contact me!


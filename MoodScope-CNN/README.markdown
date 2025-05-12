# Mood Classification using CNN (Happy/Sad)

## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images as "Happy" or "Sad" based on their visual content. The model uses max pooling to reduce spatial dimensions and improve feature extraction. The project is built using **TensorFlow/Keras** and includes data preprocessing, model training, testing, and saving functionalities.

### Objective
- Build a CNN model to classify images into two categories: "Happy" or "Sad".
- Utilize max pooling as part of the network architecture to enhance feature extraction and reduce computational complexity.

---

## Project Structure

### Directory Layout
The dataset is organized into the following structure:
```
data/
  ├── Training/
  │   ├── happy/
  │   └── not_happy/
  ├── Validation/
  │   ├── happy/
  │   └── not_happy/
  └── Testing/
      ├── (test images, e.g., photo-1494790108377-be9c29b29330.jpg, etc.)
```

- **Training**: Contains 12 images (6 happy, 6 not_happy) for training the model.
- **Validation**: Contains 2 images (1 happy, 1 not_happy) for validating the model during training.
- **Testing**: Contains 8 images for testing the trained model.

### Key Components
1. **Data Organization**: Images are stored in `Training`, `Validation`, and `Testing` folders with subfolders `happy` and `not_happy` for binary classification.
2. **Data Preprocessing**: Images are loaded, resized to 200x200 pixels, and normalized using `ImageDataGenerator`.
3. **Model Architecture**: A CNN with convolutional layers, max pooling, and dense layers is built for classification.
4. **Training**: The model is trained on the training dataset and validated using the validation dataset.
5. **Testing**: The trained model predicts the class ("Happy" or "Sad") of test images.
6. **Model Saving**: The trained model is saved as `happy_sad_model.keras` for future use.

---

## Prerequisites

### Libraries Used
- `tensorflow.keras.preprocessing.image`: For image loading and preprocessing.
- `matplotlib.pyplot`: For visualizing images.
- `tensorflow`: For building and training the CNN.
- `numpy`: For numerical operations on image arrays.
- `cv2`: For potential image processing (though minimally used).
- `os`: For interacting with the file system.

### Requirements
Ensure you have Python 3.7+ installed. Install the required libraries using the following command:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

Alternatively, you can use the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

#### requirements.txt
```
tensorflow==2.15.0
numpy==1.26.4
matplotlib==3.9.2
opencv-python==4.10.0.84
```

---

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/mood-classification-cnn.git
cd mood-classification-cnn
```

### 2. Prepare the Dataset
- Download or prepare your dataset with the structure mentioned above.
- Place the `data` folder in the root directory of the project.
- Update the file paths in the code (e.g., `train_dataset`, `validation_dataset`, and `dir_path`) to match your local directory structure. For example:
  ```python
  train_dataset = train.flow_from_directory(
      'data/Training',  # Update this path
      target_size=(200, 200),
      batch_size=3,
      class_mode='binary'
  )
  ```

### 3. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Project
The project is implemented in a Jupyter Notebook (`Mood_Classification_CNN.ipynb`). You can run it as follows:

1. **Open the Notebook**:
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open `Mood_Classification_CNN.ipynb` in your browser.

2. **Execute the Cells**:
   Run each cell in the notebook sequentially to:
   - Load and preprocess the data.
   - Build and train the CNN model.
   - Test the model on the test dataset.
   - Save the trained model.

3. **Key Outputs**:
   - Training and validation accuracy/loss for each epoch.
   - Visualizations of test images with their predicted labels ("Happy" or "Sad").
   - A saved model file (`happy_sad_model.keras`).

### Loading and Using the Saved Model
To use the saved model for predictions on new images:

1. **Load the Model**:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model('happy_sad_model.keras')
   ```

2. **Classify a New Image**:
   ```python
   from tensorflow.keras.preprocessing import image
   import numpy as np
   import matplotlib.pyplot as plt

   # Load and preprocess the image
   img_path = 'path_to_new_image.jpg'
   img = image.load_img(img_path, target_size=(200, 200))
   plt.imshow(img)
   plt.show()

   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = x / 255.0  # Normalize

   # Predict
   prediction = model.predict(x)
   if prediction[0][0] < 0.5:
       print("Happy")
   else:
       print("Not Happy")
   ```

---

## Model Architecture

The CNN model is built using the following architecture:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Architecture Breakdown
- **Input Shape**: 200x200 RGB images (3 channels).
- **Conv2D Layers**: Three convolutional layers with 16, 32, and 64 filters, respectively, to extract features.
- **MaxPool2D Layers**: Three max pooling layers to reduce spatial dimensions and computational load.
- **Flatten**: Converts the 3D feature maps into a 1D vector.
- **Dense Layers**: A 512-neuron dense layer for feature combination, followed by a single neuron with sigmoid activation for binary classification (Happy = 0, Not Happy = 1).

### Compilation
- **Loss Function**: `binary_crossentropy` for binary classification.
- **Optimizer**: `RMSprop` with a learning rate of 0.001.
- **Metrics**: `accuracy` to monitor performance.

---

## Training Details

- **Epochs**: 10
- **Batch Size**: 3
- **Training Data**: 12 images (6 happy, 6 not_happy)
- **Validation Data**: 2 images (1 happy, 1 not_happy)

### Training Output (Example)
```
Epoch 10/10
4/4 [==============================] - 0s 123ms/step - loss: 0.0219 - accuracy: 1.0000 - val_loss: 0.0080 - val_accuracy: 1.0000
```
- Achieved 100% training and validation accuracy, but the small dataset suggests potential overfitting.

---

## Results

- **Training Accuracy**: 100% (Epoch 10)
- **Validation Accuracy**: 100% (Epoch 10)
- **Test Predictions**: The model successfully predicts test images as "Happy" or "Not Happy" with visualizations.

### Limitations
- **Small Dataset**: Only 12 training and 2 validation images, leading to potential overfitting.
- **No Data Augmentation**: The model may not generalize well to diverse images.
- **Prediction Logic**: Uses a strict `val == 0` check instead of a threshold (e.g., `val < 0.5`).

---

## Improvements

1. **Increase Dataset Size**: Collect more images to improve model generalization.
2. **Add Data Augmentation**:
   ```python
   train = ImageDataGenerator(
       rescale=1/255,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True
   )
   ```
3. **Refine Prediction Logic**:
   ```python
   if prediction[0][0] < 0.5:
       print("Happy")
   else:
       print("Not Happy")
   ```
4. **Use Dropout**:
   Add dropout layers to reduce overfitting:
   ```python
   tf.keras.layers.Dropout(0.5)
   ```
5. **Evaluate Test Set**: Compute accuracy on the test set to assess performance.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, feel free to reach out:
- **Email**: your-email@example.com
- **GitHub**: [your-username](https://github.com/your-username)
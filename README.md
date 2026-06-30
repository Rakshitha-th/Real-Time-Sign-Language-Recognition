Real-Time Sign Language Recognition

This project is a real-time sign language recognition system that detects hand gestures and converts them into text using computer vision and deep learning.

The model is built using a Convolutional Neural Network (CNN) with TensorFlow/Keras.

The model was trained in Google Colab using the ASL Alphabet dataset containing images from A to Z.

After training, the saved model was integrated into a UI built using HTML/CSS (and/or Flask) and executed in VS Code for real-time prediction.

The system supports webcam-based real-time recognition as well as image-based prediction.

The model achieved an accuracy of 95% on the ASL dataset.

The system also includes speech output (text-to-speech) to read the predicted letters aloud.

This project uses OpenCV for image processing, TensorFlow for model inference, and speech libraries for audio output.

It helps in improving communication for hearing-impaired individuals by translating gestures into text and speech.

All prediction outputs are stored in the Results folder.
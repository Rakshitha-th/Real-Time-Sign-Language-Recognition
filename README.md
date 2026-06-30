A real-time sign language recognition system that detects hand gestures using a webcam and converts them into text using a deep learning model. It captures hand gestures from live video, processes them using a MobileNetV2-based transfer learning model, and predicts the corresponding ASL alphabet (A–Z). The predicted output is displayed on screen.
Tech Used & Purpose:

TensorFlow / Keras (MobileNetV2 Transfer Learning) → Pretrained deep learning model fine-tuned for ASL gesture classification
OpenCV → Webcam capture, image preprocessing, and real-time frame processing
Google Colab → Model training and experimentation environment
HTML / CSS (UI) → User interface for displaying predictions
Python → Core backend logic for prediction and integration
Speech Library (gTTS / pyttsx3) → Converts predicted text into speech output
ASL Dataset → Hand gesture images from A–Z used for training

The model achieved an accuracy of 95% on the ASL dataset. The system helps bridge communication by translating sign language gestures into text and speech in real time.

All prediction outputs are stored in the Results folder.
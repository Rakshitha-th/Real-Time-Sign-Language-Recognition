A real-time sign language recognition system that detects hand gestures using a webcam and converts them into text using a trained deep learning model. It captures hand gestures from live video, processes them using a CNN model, and predicts the corresponding ASL alphabet (A–Z). The predicted output is displayed on screen and stored in a Results folder for reference. The system also supports image-based prediction for testing static inputs.

Tech Used & Purpose:

TensorFlow / Keras → Build and train CNN model for gesture classification
OpenCV → Webcam capture, image preprocessing, and real-time frame processing
Google Colab → Model training environment for ASL dataset
HTML / CSS (UI) → User interface for displaying predictions
Python → Core backend logic for prediction and integration
Speech Library (gTTS / pyttsx3) → Converts predicted text into speech output (if enabled)
ASL Dataset → Contains hand gesture images from A–Z used for training

Model achieved an accuracy of 95% on the ASL dataset. The system helps in bridging communication by translating sign language gestures into text and speech in real time.

All prediction outputs are stored in the Results folder.
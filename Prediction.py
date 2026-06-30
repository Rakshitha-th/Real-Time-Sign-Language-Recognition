from flask import Flask, render_template, Response, jsonify
import cv2, numpy as np, tensorflow as tf, webbrowser
from threading import Timer

app = Flask(__name__)

# ---------- MODEL ----------
MODEL_PATH = r"C:\Users\Hp\OneDrive\researchR\saved_asl_model (1)"

model = tf.keras.layers.TFSMLayer(
    MODEL_PATH,
    call_endpoint="serving_default"
)

labels = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'space','del','nothing'
]

cap = cv2.VideoCapture(0)

sentence = ""
stable = "-"
history = []
counter = 0
conf_score = 0.0
recording = False

MIN_CONF = 0.85
STABLE_FRAMES = 8

def predict(frame):
    global sentence, stable
    global counter, conf_score, history

    roi = frame[130:380,180:430]

    if roi.size == 0:
        return "-", 0

    img = cv2.resize(roi, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    pred = model(img)
    pred = list(pred.values())[0].numpy()

    idx = np.argmax(pred)
    conf = float(np.max(pred))
    letter = labels[idx]

    conf_score = conf

    if conf < MIN_CONF:
        counter = 0
        return "-", conf

    if letter == stable:
        counter += 1
    else:
        stable = letter
        counter = 1

    if counter >= STABLE_FRAMES and recording:

        if stable == "space":
            sentence += " "
            history.append("SPACE")

        elif stable == "del":
            if sentence:
                sentence = sentence[:-1]
            history.append("DEL")

        elif stable != "nothing":
            sentence += stable
            history.append(stable)

        history = history[-10:]
        counter = 0

    return stable, conf

# video
def generate():

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        overlay = frame.copy()

        cv2.rectangle(
            overlay, (150,100), (450,400),
            (56,189,248), -1
        )

        frame = cv2.addWeighted(overlay, 0.12,frame, 0.88, 0)
        cv2.rectangle(frame,(180,130),(430,380),(255,215,0),3)
        letter, conf = predict(frame)

        cv2.putText(frame,"PLACE HAND HERE",(170,85),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(frame,letter,(255,70),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,255),3)

        status = "RECORDING" if recording else "STOPPED"
        color = (34,197,94) if recording else (239,68,68)

        cv2.putText(frame,status,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
        cv2.putText(frame,f"Conf: {conf:.2f}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/live")
def live():
    return jsonify({
        "letter": stable,
        "sentence": sentence,
        "confidence": conf_score,
        "recording": recording,
        "history": " → ".join(history)
    })

@app.route("/start")
def start():
    global recording
    recording = True
    return jsonify(status="started")

@app.route("/stop")
def stop():
    global recording
    recording = False
    return jsonify(sentence=sentence)

@app.route("/clear")
def clear():
    global sentence, history
    sentence = ""
    history = []
    return jsonify(status="cleared")

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":

    Timer(1, open_browser).start()

    try:
        app.run(host="0.0.0.0",port=5000,debug=False)
    finally:
        cap.release()
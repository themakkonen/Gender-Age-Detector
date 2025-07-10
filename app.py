from flask import Flask, render_template, Response, request, send_file, redirect, url_for, jsonify, session, flash
import cv2
import csv
import os
import numpy as np
from io import BytesIO
from datetime import datetime
import base64

app = Flask(__name__)
app.secret_key = 'secret-key-for-session'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB limit

# Load Models
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = None
frame_store = None
last_capture_filename = None
face_count = 0
video_streaming = False

log_file = "detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gender", "Age"])

os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/captures", exist_ok=True)

@app.before_request
def setup_theme():
    if 'theme' not in session:
        session['theme'] = 'light'

@app.route('/toggle_theme', methods=['POST'])
def toggle_theme():
    session['theme'] = 'dark' if session.get('theme') == 'light' else 'light'
    return jsonify({'theme': session['theme']})

@app.route('/')
def index():
    return render_template('index.html', face_count=face_count,
                           captured_image=last_capture_filename,
                           theme=session['theme'],
                           pending=session.get('pending_capture', False))

@app.route('/start')
def start():
    global video_streaming, cap
    if not video_streaming:
        cap = cv2.VideoCapture(0)
        video_streaming = True
    return ("", 204)

@app.route('/stop')
def stop():
    global video_streaming, cap
    video_streaming = False
    if cap is not None:
        cap.release()
        cap = None
    return ("", 204)

def gen_frames():
    global frame_store, face_count, video_streaming, cap
    while True:
        if not video_streaming or cap is None:
            break
        success, frame = cap.read()
        if not success:
            break
        face_count = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face_count += 1
            x1, y1, x2, y2 = max(0, x-20), max(0, y-20), min(frame.shape[1], x+w+20), min(frame.shape[0], y+h+20)
            face_img = frame[y1:y2, x1:x2].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
            gender_net.setInput(blob)
            gender = GENDER_LIST[gender_net.forward().argmax()]
            age_net.setInput(blob)
            age = AGE_BUCKETS[age_net.forward().argmax()]
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        frame_store = frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    if video_streaming and cap is not None:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return send_file('static/icons/camera_stream.png', mimetype='image/jpeg')

@app.route('/capture')
def capture():
    global frame_store, last_capture_filename
    if not video_streaming or frame_store is None:
        session['pending_capture'] = False
        return redirect(url_for('index'))
    filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    full_path = os.path.join("static/captures", filename)
    cv2.imwrite(full_path, frame_store)
    last_capture_filename = f"captures/{filename}"
    gray = cv2.cvtColor(frame_store, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face_img = frame_store[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward().argmax()]
        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward().argmax()]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, gender, age])
    session['pending_capture'] = True
    frame_store = None
    return redirect(url_for('index'))

@app.route('/delete_image', methods=['POST'])
def delete_image():
    global last_capture_filename
    if last_capture_filename:
        path = os.path.join("static", last_capture_filename)
        if os.path.exists(path):
            os.remove(path)
        last_capture_filename = None
    session['pending_capture'] = False
    return redirect(url_for('index'))

@app.route('/download')
def download():
    if last_capture_filename:
        return send_file(os.path.join("static", last_capture_filename), as_attachment=True)
    return "No image to download yet."

@app.route('/delete_history', methods=['POST'])
def delete_history():
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gender", "Age"])
    return jsonify({"status": "success"})

@app.route('/admin')
def admin():
    with open(log_file, "r") as f:
        rows = list(csv.reader(f))
    return render_template("admin.html", logs=rows[1:], headers=rows[0], theme=session["theme"])

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    predictions = []
    b64_preview = None
    session.pop('upload_preview', None)
    session.pop('upload_predictions', None)

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            flash("Invalid file format. Please upload a .jpg, .jpeg, or .png file.")
            return redirect(url_for('upload'))
        if file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
            flash("File size exceeds 1MB limit.")
            return redirect(url_for('upload'))

        in_memory_file = BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
            gender_net.setInput(blob)
            gender = GENDER_LIST[gender_net.forward().argmax()]
            age_net.setInput(blob)
            age = AGE_BUCKETS[age_net.forward().argmax()]
            predictions.append(f"{gender}, {age}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"{gender}, {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, gender, age])

        # Save the processed image to uploads
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        upload_path = os.path.join("static/uploads", filename)
        cv2.imwrite(upload_path, frame)

        _, buffer = cv2.imencode('.jpg', frame)
        b64_preview = base64.b64encode(buffer).decode('utf-8')
        session['upload_preview'] = buffer.tobytes()
        session['upload_predictions'] = predictions

        flash("Upload successful!")

    return render_template("upload.html", b64_preview=b64_preview, predictions=predictions, theme=session["theme"], timestamp=timestamp if 'timestamp' in locals() else None)

@app.route('/reset_upload', methods=['POST'])
def reset_upload():
    session.pop('upload_preview', None)
    session.pop('upload_predictions', None)
    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(debug=True)

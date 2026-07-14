import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent.parent
SCREENSHOT_DIR = BASE_DIR / "picture_screenshot"
PROFILE_DIR = BASE_DIR / "profile_uploads"
CONFIG_PATH = BASE_DIR / "app_config.json"
HISTORY_PATH = BASE_DIR / "detection_history.json"
VIDEO_PATH = BASE_DIR / "security_record.avi"
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}

# Keep this app folder from shadowing standard modules such as json/logging.
base_dir_key = os.path.normcase(os.path.abspath(BASE_DIR))
sys.path = [
    path for path in sys.path
    if os.path.normcase(os.path.abspath(path or os.getcwd())) != base_dir_key
]

from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

SCREENSHOT_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

DEFAULT_CONFIG = {
    "password_hash": generate_password_hash("1234"),
    "profile_name": "Sulayman Bah",
    "profile_area": "Banjul, The Gambia",
    "screenshot_delay": 5,
    "alarm_delay": 5,
    "confidence": 0.5,
    "alarm_enabled": True,
}

app = Flask(__name__)
app.secret_key = os.environ.get("CAMERA_APP_SECRET", "change-this-camera-secret")

camera_lock = threading.Lock()
state_lock = threading.Lock()
camera_active = False
last_screenshot_time = None
last_alarm_time = None
last_detection_time = None
last_detection_confidence = None
status_message = "Camera stopped"


def read_json(path, fallback):
    if not path.exists():
        return fallback

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return fallback


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_config():
    config = DEFAULT_CONFIG.copy()
    config.update(read_json(CONFIG_PATH, {}))
    return config


def save_config(config):
    write_json(CONFIG_PATH, config)


def get_history():
    return read_json(HISTORY_PATH, [])


def save_history(history):
    write_json(HISTORY_PATH, history[:200])


def add_history_record(filename, confidence, timestamp):
    history = get_history()
    history.insert(0, {
        "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "confidence": round(float(confidence) * 100, 1),
    })
    save_history(history)


def find_model_file(filename):
    candidates = [
        BASE_DIR / filename,
        WORKSPACE_DIR / filename,
        Path.cwd() / filename,
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"{filename} was not found. Put it beside chatbot.py or in the Js-basis folder.")


net = cv2.dnn.readNetFromCaffe(
    find_model_file("MobileNetSSD_deploy.prototxt"),
    find_model_file("MobileNetSSD_deploy.caffemodel"),
)


def login_required():
    return session.get("logged_in") is True


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def get_profile_image():
    images = sorted(PROFILE_DIR.glob("profile.*"), key=lambda item: item.stat().st_mtime, reverse=True)
    return images[0].name if images else None


def get_screenshots():
    images = sorted(SCREENSHOT_DIR.glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
    return [image.name for image in images]


def get_camera_status():
    with state_lock:
        return {
            "active": camera_active,
            "message": status_message,
            "last_detection_time": last_detection_time,
            "last_detection_confidence": last_detection_confidence,
        }


def set_camera_status(active=None, message=None, detection_time=None, confidence=None):
    global camera_active, last_detection_time, last_detection_confidence, status_message

    with state_lock:
        if active is not None:
            camera_active = active
        if message is not None:
            status_message = message
        if detection_time is not None:
            last_detection_time = detection_time
        if confidence is not None:
            last_detection_confidence = confidence


def make_message_frame(message, color=(255, 255, 255)):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


@app.context_processor
def inject_profile():
    config = load_config()
    return {
        "profile_name": config["profile_name"],
        "profile_area": config["profile_area"],
        "profile_image": get_profile_image(),
        "camera_status": get_camera_status(),
    }


def process_frame(frame, writer):
    global last_alarm_time, last_screenshot_time

    config = load_config()
    detected = False
    best_confidence = 0.0
    timestamp = datetime.datetime.now()
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence <= float(config["confidence"]):
            continue

        idx = int(detections[0, 0, i, 1])

        if idx >= len(CLASSES) or CLASSES[idx] != "person":
            continue

        detected = True
        best_confidence = max(best_confidence, confidence)
        box = detections[0, 0, i, 3:7] * np.array(
            [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
        )
        start_x, start_y, end_x, end_y = box.astype("int")
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
        cv2.putText(
            frame,
            f"PERSON {confidence * 100:.1f}%",
            (start_x, max(start_y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if detected:
        set_camera_status(
            message="Person detected",
            detection_time=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            confidence=round(best_confidence * 100, 1),
        )

        if (
            last_screenshot_time is None
            or (timestamp - last_screenshot_time).total_seconds() >= int(config["screenshot_delay"])
        ):
            filename = timestamp.strftime("person_%Y-%m-%d_%H-%M-%S.jpg")
            cv2.imwrite(str(SCREENSHOT_DIR / filename), frame)
            add_history_record(filename, best_confidence, timestamp)
            last_screenshot_time = timestamp

        if (
            config["alarm_enabled"]
            and winsound is not None
            and (last_alarm_time is None or (timestamp - last_alarm_time).total_seconds() >= int(config["alarm_delay"]))
        ):
            winsound.Beep(1200, 250)
            last_alarm_time = timestamp

        writer.write(frame)
    elif get_camera_status()["active"]:
        set_camera_status(message="Camera online")

    return frame


def generate_frames():
    if not get_camera_status()["active"]:
        frame = make_message_frame("Camera stopped", (255, 255, 255))
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        set_camera_status(active=False, message="Camera not detected")
        frame = make_message_frame("Camera not detected", (0, 0, 255))
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        return

    set_camera_status(message="Camera online")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, 20.0, (640, 480))

    try:
        while get_camera_status()["active"]:
            with camera_lock:
                success, frame = camera.read()

            if not success:
                set_camera_status(active=False, message="Frame not received")
                break

            frame = cv2.resize(frame, (640, 480))
            frame = process_frame(frame, writer)
            ok, buffer = cv2.imencode(".jpg", frame)

            if not ok:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        camera.release()
        writer.release()
        if not get_camera_status()["active"]:
            set_camera_status(message="Camera stopped")


@app.route("/", methods=["GET", "POST"])
def login():
    if login_required():
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        config = load_config()
        if check_password_hash(config["password_hash"], request.form.get("password", "")):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))

        error = "Wrong password"

    return render_template("index.html", page="login", error=error)


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    screenshots = get_screenshots()
    return render_template(
        "index.html",
        page="dashboard",
        screenshot_count=len(screenshots),
        latest_screenshots=screenshots[:4],
        history=get_history()[:5],
    )


@app.route("/scan")
def scan():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("index.html", page="scan")


@app.route("/camera/start", methods=["POST"])
def start_camera():
    if not login_required():
        return redirect(url_for("login"))

    set_camera_status(active=True, message="Camera starting")
    return redirect(url_for("scan"))


@app.route("/camera/stop", methods=["POST"])
def stop_camera():
    if not login_required():
        return redirect(url_for("login"))

    set_camera_status(active=False, message="Camera stopped")
    return redirect(url_for("scan"))


@app.route("/status")
def status():
    if not login_required():
        return redirect(url_for("login"))

    return jsonify(get_camera_status())


@app.route("/video_feed")
def video_feed():
    if not login_required():
        return redirect(url_for("login"))

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/screenshots")
def screenshots():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("index.html", page="screenshots", screenshots=get_screenshots(), history=get_history())


@app.route("/screenshots/delete/<path:filename>", methods=["POST"])
def delete_screenshot(filename):
    if not login_required():
        return redirect(url_for("login"))

    safe_name = Path(filename).name
    image_path = SCREENSHOT_DIR / safe_name

    if image_path.exists():
        image_path.unlink()

    history = [record for record in get_history() if record.get("filename") != safe_name]
    save_history(history)
    return redirect(url_for("screenshots"))


@app.route("/picture_screenshot/<path:filename>")
def screenshot_file(filename):
    if not login_required():
        return redirect(url_for("login"))

    return send_from_directory(SCREENSHOT_DIR, filename)


@app.route("/profile_uploads/<path:filename>")
def profile_file(filename):
    if not login_required():
        return redirect(url_for("login"))

    return send_from_directory(PROFILE_DIR, filename)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if not login_required():
        return redirect(url_for("login"))

    config = load_config()
    message = None
    error = None

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "profile":
            config["profile_name"] = request.form.get("profile_name", "").strip() or config["profile_name"]
            config["profile_area"] = request.form.get("profile_area", "").strip() or config["profile_area"]
            save_config(config)
            message = "Profile updated."

        elif form_type == "picture":
            image = request.files.get("profile_picture")

            if image is None or image.filename == "":
                error = "Choose a picture before uploading."
            elif not allowed_image(image.filename):
                error = "Upload a JPG, PNG, GIF, or WEBP image."
            else:
                extension = secure_filename(image.filename).rsplit(".", 1)[1].lower()

                for old_image in PROFILE_DIR.glob("profile.*"):
                    old_image.unlink()

                image.save(PROFILE_DIR / f"profile.{extension}")
                message = "Profile picture uploaded."

        elif form_type == "security":
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")

            if len(new_password) < 4:
                error = "Password must be at least 4 characters."
            elif new_password != confirm_password:
                error = "Passwords do not match."
            else:
                config["password_hash"] = generate_password_hash(new_password)
                save_config(config)
                message = "Password updated."

        elif form_type == "detector":
            try:
                config["screenshot_delay"] = max(1, int(request.form.get("screenshot_delay", 5)))
                config["alarm_delay"] = max(1, int(request.form.get("alarm_delay", 5)))
                config["confidence"] = min(0.95, max(0.1, float(request.form.get("confidence", 50)) / 100))
                config["alarm_enabled"] = request.form.get("alarm_enabled") == "on"
                save_config(config)
                message = "Detector settings updated."
            except ValueError:
                error = "Settings must use valid numbers."

    config = load_config()
    return render_template(
        "index.html",
        page="settings",
        config=config,
        confidence_percent=int(float(config["confidence"]) * 100),
        message=message,
        error=error,
    )


@app.route("/logout")
def logout():
    session.clear()
    set_camera_status(active=False, message="Camera stopped")
    return redirect(url_for("login"))


if __name__ == "__main__":
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
    if not HISTORY_PATH.exists():
        save_history([])

    app.run(host="127.0.0.1", port=5000, debug=False)

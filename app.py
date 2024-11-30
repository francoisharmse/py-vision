import os
import torch
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, url_for, redirect, Response, session
from flask_sock import Sock
from werkzeug.utils import secure_filename
import logging
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.secret_key = 'your_secret_key_here'  # Required for session

# Create upload directories if they don't exist
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)
os.makedirs(os.path.join('static', 'sounds'), exist_ok=True)

# Global WebSocket connections
ws_clients = set()

@sock.route('/ws')
def ws_handler(ws):
    ws_clients.add(ws)
    try:
        while True:
            data = ws.receive()
            # Keep connection alive
    except:
        ws_clients.remove(ws)

def notify_detection():
    message = json.dumps({'type': 'detection'})
    dead_clients = set()
    for client in ws_clients:
        try:
            client.send(message)
        except:
            dead_clients.add(client)
    # Remove dead connections
    ws_clients.difference_update(dead_clients)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image-detection', methods=['GET', 'POST'])
def image_detection():
    image_file = None
    detected_objects = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_image_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_file = url_for('static', filename=f'uploads/{filename}')
            
            # Perform object detection
            detected_objects = detect_objects(filepath)
    
    return render_template('image_detection.html', image_file=image_file, detected_objects=detected_objects)

@app.route('/video-detection', methods=['GET', 'POST'])
def video_detection():
    global video_state
    video_file = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_video_file(file.filename):
            # Reset video state when new video is uploaded
            if video_state['cap'] is not None:
                video_state['cap'].release()
            video_state['cap'] = None
            video_state['frame_pos'] = 0
            video_state['is_paused'] = False
            video_state['total_frames'] = 0
            video_state['fps'] = 0
            video_state['duration'] = 0
            video_state['detected_objects'] = set()
            video_state['hidden_objects'] = set()
            video_state['alert_objects'] = set(['person', 'dog', 'cell phone', 'bicycle', 'car', 'bus', 'gun', 'rifle'])
            video_state['active_alert_objects'] = set()
            video_state['alert_triggered'] = False
            
            filename = 'current_video.mp4'  # Always use same filename to avoid accumulating files
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', filename)
            file.save(filepath)
            video_file = url_for('static', filename=f'uploads/videos/{filename}')
    
    return render_template('video_detection.html', video_file=video_file)

@app.route('/video_feed')
def video_feed():
    return Response(get_video_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause')
def toggle_pause():
    global video_state
    video_state['is_paused'] = not video_state['is_paused']
    return {'status': 'ok', 'is_paused': video_state['is_paused']}

@app.route('/current_frame')
def current_frame():
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', 'current_video.mp4')
    if not os.path.exists(video_path):
        return '', 404

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Perform detection on frame
        results = model(frame)
        
        # Draw detection boxes
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf > 0.5:  # Confidence threshold
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return '', 404

@app.route('/video_info')
def video_info():
    global video_state
    return {
        'total_frames': video_state['total_frames'],
        'current_frame': video_state['frame_pos'],
        'fps': video_state['fps'],
        'duration': video_state['duration'],
        'current_time': video_state['frame_pos'] / video_state['fps'] if video_state['fps'] else 0,
        'detected_objects': list(video_state['detected_objects']),
        'hidden_objects': list(video_state['hidden_objects']),
        'alert_objects': list(video_state['alert_objects']),
        'active_alert_objects': list(video_state['active_alert_objects']),
        'alert_triggered': video_state['alert_triggered'],
        'video_effect': video_state['video_effect'],
        'playback_speed': video_state['playback_speed'],
        'is_reversed': video_state['is_reversed']
    }

@app.route('/toggle_object')
def toggle_object():
    global video_state
    object_name = request.args.get('name')
    if object_name:
        if object_name in video_state['hidden_objects']:
            video_state['hidden_objects'].remove(object_name)
        else:
            video_state['hidden_objects'].add(object_name)
        return {'status': 'ok', 'hidden': object_name in video_state['hidden_objects']}
    return {'status': 'error', 'message': 'No object name provided'}

@app.route('/toggle_alert_object')
def toggle_alert_object():
    global video_state
    object_name = request.args.get('name')
    if object_name:
        if object_name in video_state['active_alert_objects']:
            video_state['active_alert_objects'].remove(object_name)
        else:
            video_state['active_alert_objects'].add(object_name)
        return {'status': 'ok', 'active': object_name in video_state['active_alert_objects']}
    return {'status': 'error', 'message': 'No object name provided'}

@app.route('/add_alert_object')
def add_alert_object():
    global video_state
    object_name = request.args.get('name', '').strip().lower()
    if object_name:
        video_state['alert_objects'].add(object_name)
        return {'status': 'ok', 'alert_objects': list(video_state['alert_objects'])}
    return {'status': 'error', 'message': 'No object name provided'}

@app.route('/set_video_position')
def set_video_position():
    global video_state
    try:
        time_position = float(request.args.get('time', 0))
        if video_state['cap'] is not None:
            frame_position = int(time_position * video_state['fps'])
            frame_position = max(0, min(frame_position, video_state['total_frames'] - 1))
            video_state['frame_pos'] = frame_position
            video_state['cap'].set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            return {'status': 'ok', 'current_time': time_position}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    return {'status': 'error', 'message': 'No video loaded'}

@app.route('/set_video_effect')
def set_video_effect():
    global video_state
    effect = request.args.get('effect', 'normal')
    valid_effects = [
        'normal', 'wireframe', 'sketch', 'cartoon', 'watercolor', 
        'black_and_white', 'film_grain', 'color_grading', 'hue_shift',
        'invert', 'monochrome', 'duotone', 'posterize', 'gaussian_blur',
        'radial_blur', 'motion_blur', 'tilt_shift', 'sharpen'
    ]
    if effect in valid_effects:
        video_state['video_effect'] = effect
        return {'status': 'ok', 'effect': effect}
    return {'status': 'error', 'message': 'Invalid effect'}

@app.route('/set_playback_speed')
def set_playback_speed():
    global video_state
    try:
        speed = float(request.args.get('speed', 1.0))
        if 0.1 <= speed <= 4.0:
            video_state['playback_speed'] = speed
            return {'status': 'ok', 'speed': speed}
    except ValueError:
        pass
    return {'status': 'error', 'message': 'Invalid speed'}

@app.route('/toggle_direction')
def toggle_direction():
    global video_state
    video_state['is_reversed'] = not video_state['is_reversed']
    return {'status': 'ok', 'is_reversed': video_state['is_reversed']}

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def detect_objects(image_path):
    logger.info("Detecting objects in image: %s", image_path)
    img = Image.open(image_path)
    
    # Perform inference
    results = model(img)
    logger.info("Inference completed")
    
    # Get detected objects
    predictions = results.pandas().xyxy[0]
    detected_objects = []
    
    logger.info("Processing predictions")
    
    for _, pred in predictions.iterrows():
        detected_objects.append({
            'name': pred['name'],
            'confidence': f"{pred['confidence']:.2f}",
            'bbox': {
                'x1': float(pred['xmin']),
                'y1': float(pred['ymin']),
                'x2': float(pred['xmax']),
                'y2': float(pred['ymax'])
            }
        })
    
    logger.info("Detected objects: %s", detected_objects)
    return detected_objects

# Global variable to store video capture object and frame position
video_state = {
    'cap': None,
    'frame_pos': 0,
    'is_paused': False,
    'total_frames': 0,
    'fps': 0,
    'duration': 0,
    'detected_objects': set(),  # Set to store unique object types
    'hidden_objects': set(),  # Set to store objects that should not be displayed
    'alert_objects': set(['person', 'dog', 'cell phone', 'bicycle', 'car', 'bus', 'gun', 'rifle']),  # Objects to alert on
    'active_alert_objects': set(),  # Objects currently selected for alerts
    'alert_triggered': False,  # Whether an alert object was detected in current frame
    'video_effect': 'normal',  # Current video effect
    'playback_speed': 1.0,
    'is_reversed': False,
    'resume_time': None
}

def apply_video_effect(frame, effect):
    if effect == 'normal':
        return frame
    elif effect == 'wireframe':
        # Create wireframe effect using edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif effect == 'sketch':
        # Create pencil sketch effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=5)
        edges = 255 - edges
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif effect == 'cartoon':
        # Create cartoon effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(frame, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    elif effect == 'watercolor':
        # Create watercolor painting effect
        blur = cv2.bilateralFilter(frame, 9, 75, 75)
        edges = cv2.stylization(frame, sigma_s=60, sigma_r=0.6)
        return edges
    elif effect == 'black_and_white':
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif effect == 'film_grain':
        # Add film grain effect
        noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
        grain = cv2.add(frame, noise)
        return grain
    # Color Filters
    elif effect == 'color_grading':
        # Apply cinematic color grading (teal and orange look)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)  # Increase saturation
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.1)  # Increase brightness
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif effect == 'hue_shift':
        # Shift hue by 30 degrees
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 30) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif effect == 'invert':
        # Invert all colors
        return cv2.bitwise_not(frame)
    elif effect == 'monochrome':
        # Single color tint (sepia tone)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sepia = np.array(frame)
        sepia[:,:,0] = np.clip(gray * 0.272, 0, 255)  # Blue channel
        sepia[:,:,1] = np.clip(gray * 0.534, 0, 255)  # Green channel
        sepia[:,:,2] = np.clip(gray * 0.131, 0, 255)  # Red channel
        return sepia
    elif effect == 'duotone':
        # Two-color gradient (cyan and pink)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color1 = np.array([255, 111, 0])  # Cyan
        color2 = np.array([255, 0, 255])  # Pink
        ratio = gray/255.0
        duotone = np.zeros_like(frame)
        for i in range(3):
            duotone[:,:,i] = (ratio * color1[i] + (1-ratio) * color2[i]).astype(np.uint8)
        return duotone
    elif effect == 'posterize':
        # Reduce color palette
        n_colors = 4
        frame = frame.astype(np.float32) / 255
        frame = np.floor(frame * n_colors) / (n_colors - 1)
        frame = (frame * 255).astype(np.uint8)
        return frame
    # Blur and Focus Filters
    elif effect == 'gaussian_blur':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif effect == 'radial_blur':
        # Create radial blur effect
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        
        # Create distance matrix
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt((frame.shape[1]//2)**2 + (frame.shape[0]//2)**2)
        
        # Normalize distances to [0, 1] range
        blur_weights = distances / max_distance
        
        # Apply variable blur
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Create alpha mask for blending
        mask = blur_weights.astype(np.float32)
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
        mask = np.dstack([mask] * 3)  # Create 3-channel mask
        
        # Blend original and blurred images
        result = cv2.addWeighted(frame.astype(np.float32), 1 - mask, 
                               blurred.astype(np.float32), mask, 0)
        
        return result.astype(np.uint8)
    elif effect == 'motion_blur':
        # Create horizontal motion blur
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        return cv2.filter2D(frame, -1, kernel)
    elif effect == 'tilt_shift':
        # Create tilt-shift effect
        height = frame.shape[0]
        center_y = height // 2
        
        # Create gradient mask
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        mask[center_y-50:center_y+50] = 1
        mask = cv2.GaussianBlur(mask, (151, 151), 50)
        mask = np.stack([mask] * 3, axis=2)
        
        # Blur the image
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        return cv2.addWeighted(frame, mask, blurred, 1-mask, 0)
    elif effect == 'sharpen':
        # Create sharpening effect
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        return cv2.filter2D(frame, -1, kernel)
    return frame

def get_video_frame():
    global video_state
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', 'current_video.mp4')
    
    if not os.path.exists(video_path):
        return
        
    if video_state['cap'] is None:
        video_state['cap'] = cv2.VideoCapture(video_path)
        video_state['total_frames'] = int(video_state['cap'].get(cv2.CAP_PROP_FRAME_COUNT))
        video_state['fps'] = video_state['cap'].get(cv2.CAP_PROP_FPS)
        video_state['duration'] = video_state['total_frames'] / video_state['fps']
        video_state['frame_pos'] = 0
        video_state['detected_objects'] = set()
        video_state['hidden_objects'] = set()
        video_state['alert_triggered'] = False
        video_state['video_effect'] = 'normal'
    
    cap = video_state['cap']
    
    while True:
        if video_state.get('resume_time') and video_state['is_paused']:
            if time.time() >= video_state['resume_time']:
                video_state['is_paused'] = False
                video_state['alert_triggered'] = False
                video_state['resume_time'] = None
        
        if not video_state['is_paused']:
            # Calculate frame step based on speed and direction
            frame_step = int(video_state['playback_speed'] * (1 if not video_state['is_reversed'] else -1))
            new_pos = video_state['frame_pos'] + frame_step
            
            # Handle bounds
            if new_pos >= video_state['total_frames']:
                new_pos = 0
            elif new_pos < 0:
                new_pos = video_state['total_frames'] - 1
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            
            if not ret:
                video_state['frame_pos'] = 0
                continue
                
            video_state['frame_pos'] = new_pos
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_state['frame_pos'])
            ret, frame = cap.read()
        
        # Apply video effect before object detection
        frame = apply_video_effect(frame, video_state['video_effect'])
            
        # Perform detection on frame
        results = model(frame)
        
        # Update detected objects set
        frame_objects = set()
        alert_detected = False
        
        # Draw detection boxes
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf > 0.5:  # Confidence threshold
                obj_name = model.names[int(cls)]
                frame_objects.add(obj_name)
                
                # Check if this is an alert object
                is_alert = obj_name in video_state['active_alert_objects']
                
                # Calculate box size and use it to determine intensity
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                frame_area = frame.shape[0] * frame.shape[1]
                
                # Normalize box area relative to frame size (0 to 1)
                size_factor = min(box_area / frame_area * 4, 1.0)  # multiply by 4 to make the effect more pronounced
                
                # Base colors (BGR format)
                if is_alert:
                    base_color = (0, 0, 255)  # Red for alerts
                else:
                    base_color = (0, 255, 100)  # Brighter green (reduced blue component)
                
                # Adjust color intensity based on size_factor with higher minimum intensity
                box_color = tuple(int(c * (0.6 + 0.4 * size_factor)) for c in base_color)  # min 60% intensity
                text_color = tuple(int(c * (0.7 + 0.3 * size_factor)) for c in base_color)  # min 70% intensity
                
                # Only draw box if object type is not hidden
                if obj_name not in video_state['hidden_objects']:
                    label = f"{obj_name} {conf:.2f}"
                    # Slightly thicker box (changed from 1 to 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    # Adjust text size based on distance
                    text_size = 0.4 + (0.3 * size_factor)  # Scale between 0.4 and 0.7
                    cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)
                
                if is_alert and not video_state['is_paused']:
                    alert_detected = True
                    notify_detection()  # Only notify for alerts
        
        # Update the global set of detected objects
        video_state['detected_objects'].update(frame_objects)
        
        # If alert object detected and video is playing, pause it
        if alert_detected:
            video_state['is_paused'] = True
            video_state['alert_triggered'] = True
            # Schedule auto-resume after 1 second
            video_state['resume_time'] = time.time() + 0.0

        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
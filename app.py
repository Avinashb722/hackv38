import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime, timedelta
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import os
from collections import defaultdict
import logging
import json
from database import init_db, save_dashboard_data, create_report, create_calendar_event, get_reports, get_calendar_events, get_dashboard_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize database
init_db()
logger.info("Database initialized successfully")

try:
    # Initialize YOLOv8 model for face detection
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        logger.info("Downloading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        model.export(format='onnx')
    else:
        model = YOLO(model_path)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    raise

try:
    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30)
    logger.info("DeepSORT tracker initialized successfully")
except Exception as e:
    logger.error(f"Error initializing DeepSORT tracker: {str(e)}")
    raise

# Camera settings
CAMERA_SETTINGS = {
    'droidcam': 'http://192.168.23.43:4747/video',
    'system': 0
}

# Zone configurations
ZONE_TYPES = {
    'zone1': 'desk',      # Desk area
    'zone2': 'meeting',   # Meeting room
    'zone3': 'desk',      # Desk area
    'zone4': 'restricted' # Restricted area
}

# Constants for anomaly detection
MAX_IDLE_TIME = 300  # 5 minutes in seconds
MAX_MEETING_CAPACITY = 4
RESTRICTED_ZONE_ALERT = True

# Initialize state tracking for working status
working_states = {
    'working': ['laptop', 'chair','face','person'],
    'not_working': ['cell phone', 'bed', 'couch','sleeping']
}

person_states = defaultdict(lambda: {
    'last_activity_time': time.time(),
    'is_working': True,
    'idle_threshold': 60,  # 1 minute in seconds
    'last_face_detected': time.time(),
    'face_detection_threshold': 5  # 5 seconds threshold for face detection
})

def define_zones(frame_shape):
    try:
        height, width = frame_shape[:2]
        zone_width = width // 4
        zones = {
            'zone1': [(0, 0), (zone_width, height)],  # Leftmost column
            'zone2': [(zone_width, 0), (zone_width * 2, height)],  # Second column
            'zone3': [(zone_width * 2, 0), (zone_width * 3, height)],  # Third column
            'zone4': [(zone_width * 3, 0), (width, height)]  # Rightmost column
        }
        logger.info(f"Zones defined successfully. Frame shape: {frame_shape}, Zone width: {zone_width}")
        return zones
    except Exception as e:
        logger.error(f"Error defining zones: {str(e)}")
        raise

class Metrics:
    def __init__(self):
        # Basic metrics
        self.productivity_hours = 0
        self.room_utilization = 0
        self.desk_occupancy = 0
        self.last_activity = time.time()
        self.is_occupied = False
        self.zone_counts = {'zone1': 0, 'zone2': 0, 'zone3': 0, 'zone4': 0}
        self.zone_availability = {
            'zone1': {'available': True, 'capacity': 4},
            'zone2': {'available': True, 'capacity': 4},
            'zone3': {'available': True, 'capacity': 4},
            'zone4': {'available': True, 'capacity': 4}
        }
        self.tracked_objects = {}
        self.current_camera = 'system'

        # Advanced metrics
        self.person_metrics = defaultdict(lambda: {
            'productive_hours': 0,
            'meeting_hours': 0,
            'break_time': 0,
            'last_seen': time.time(),
            'current_zone': None,
            'zone_history': [],
            'idle_start': None,
            'zone_times': {
                'zone1': 0,  # Time spent in zone 1
                'zone2': 0,  # Time spent in zone 2
                'zone3': 0,  # Time spent in zone 3
                'zone4': 0   # Time spent in zone 4
            },
            'last_zone_change': time.time(),
            'daily_productivity': defaultdict(lambda: {
                'zone1': 0,
                'zone2': 0,
                'zone3': 0,
                'zone4': 0
            })
        })
        
        # Heatmap data - match the frame dimensions
        self.heatmap_data = np.zeros((720, 1280), dtype=np.float32)
        self.heatmap_decay = 0.95
        
        # Historical data
        self.historical_data = {
            'productivity': [],
            'utilization': [],
            'occupancy': [],
            'alerts': []
        }
        
        # Anomaly detection
        self.alerts = []
        self.last_alert_time = time.time()
        self.alert_cooldown = 60

    def update_person_metrics(self, track_id, zone_name, current_time):
        person = self.person_metrics[track_id]
        time_diff = current_time - person['last_seen']
        
        # Update zone history
        if person['current_zone'] != zone_name:
            # Calculate time spent in previous zone
            if person['current_zone'] is not None:
                prev_zone = person['current_zone']
                person['zone_times'][prev_zone] += time_diff
                # Update daily productivity for previous zone
                current_date = datetime.now().strftime('%Y-%m-%d')
                person['daily_productivity'][current_date][prev_zone] += time_diff / 3600

            person['zone_history'].append({
                'zone': zone_name,
                'timestamp': current_time,
                'duration': time_diff
            })
            person['current_zone'] = zone_name
            person['last_zone_change'] = current_time
        
        # Update current zone time
        person['zone_times'][zone_name] += time_diff
        current_date = datetime.now().strftime('%Y-%m-%d')
        person['daily_productivity'][current_date][zone_name] += time_diff / 3600
        
        # Update specific metrics based on zone type
        zone_type = ZONE_TYPES[zone_name]
        if zone_type == 'desk':
            person['productive_hours'] += time_diff / 3600
        elif zone_type == 'meeting':
            person['meeting_hours'] += time_diff / 3600
        else:
            person['break_time'] += time_diff / 3600
        
        person['last_seen'] = current_time
        
        # Check for anomalies
        self.check_anomalies(track_id, zone_name, current_time)

    def check_anomalies(self, track_id, zone_name, current_time):
        person = self.person_metrics[track_id]
        zone_type = ZONE_TYPES[zone_name]
        
        # Check for idle time
        if current_time - person['last_seen'] > MAX_IDLE_TIME:
            if person['idle_start'] is None:
                person['idle_start'] = current_time
            elif current_time - person['idle_start'] > MAX_IDLE_TIME:
                self.add_alert(f"Person {track_id} has been idle for too long")
        else:
            person['idle_start'] = None
        
        # Check meeting room capacity
        if zone_type == 'meeting' and self.zone_counts[zone_name] > MAX_MEETING_CAPACITY:
            self.add_alert(f"Meeting room {zone_name} exceeds capacity")
        
        # Check restricted zone access
        if zone_type == 'restricted' and RESTRICTED_ZONE_ALERT:
            self.add_alert(f"Unauthorized access to restricted zone {zone_name}")

    def add_alert(self, message):
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.alerts.append({
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            self.last_alert_time = current_time
            # Keep only last 100 alerts
            self.alerts = self.alerts[-100:]

    def update_heatmap(self, x, y):
        try:
            # Ensure coordinates are within bounds
            x = min(max(0, int(x)), self.heatmap_data.shape[1] - 1)
            y = min(max(0, int(y)), self.heatmap_data.shape[0] - 1)
            # Update heatmap with new detection
            self.heatmap_data[y, x] = 1.0
            # Apply decay to existing heatmap
            self.heatmap_data *= self.heatmap_decay
        except Exception as e:
            logger.error(f"Error updating heatmap: {str(e)}")

    def get_heatmap_image(self):
        try:
            # Convert heatmap to RGB image
            heatmap = cv2.applyColorMap(
                (self.heatmap_data * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            # Resize heatmap to match the display size
            heatmap = cv2.resize(heatmap, (640, 480))
            return heatmap
        except Exception as e:
            logger.error(f"Error generating heatmap image: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def update_historical_data(self):
        # Update historical data every minute
        current_time = datetime.now()
        self.historical_data['productivity'].append({
            'timestamp': current_time.isoformat(),
            'value': self.productivity_hours
        })
        self.historical_data['utilization'].append({
            'timestamp': current_time.isoformat(),
            'value': self.room_utilization
        })
        self.historical_data['occupancy'].append({
            'timestamp': current_time.isoformat(),
            'value': self.desk_occupancy
        })
        
        # Keep only last 24 hours of data
        cutoff_time = current_time - timedelta(hours=24)
        for key in self.historical_data:
            self.historical_data[key] = [
                entry for entry in self.historical_data[key]
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]

    def get_person_productivity(self):
        current_time = time.time()
        productivity_data = {}
        
        for track_id, person in self.person_metrics.items():
            # Calculate current session time for current zone
            current_session_time = 0
            if person['current_zone']:
                current_session_time = (current_time - person['last_zone_change']) / 3600
                person['zone_times'][person['current_zone']] += current_session_time * 3600
            
            # Calculate total productive hours for each zone
            zone_hours = {
                zone: round(time / 3600, 2)  # Convert seconds to hours
                for zone, time in person['zone_times'].items()
            }
            
            productivity_data[track_id] = {
                'total_productive_hours': round(person['productive_hours'] + current_session_time, 2),
                'current_zone': person['current_zone'],
                'zone_type': ZONE_TYPES[person['current_zone']] if person['current_zone'] else None,
                'meeting_hours': round(person['meeting_hours'], 2),
                'break_time': round(person['break_time'], 2),
                'zone_hours': zone_hours,
                'daily_productivity': {
                    date: {
                        zone: round(hours, 2)
                        for zone, hours in zones.items()
                    }
                    for date, zones in person['daily_productivity'].items()
                }
            }
        
        return productivity_data

metrics = Metrics()

def check_working_status(track_id, detected_objects, current_time):
    person = person_states[track_id]
    
    # Check if face is detected
    if 'face' not in detected_objects:
        detected_objects.append('face not detected')
    
    # Check for not working objects
    not_working_objects = [obj for obj in detected_objects if obj in working_states['not_working']]
    if not_working_objects:
        person['is_working'] = False
        person['last_activity_time'] = current_time
        return False
    
    # Check for working objects
    working_objects = [obj for obj in detected_objects if obj in working_states['working']]
    if working_objects:
        person['is_working'] = True
        person['last_activity_time'] = current_time
        return True
    
    # Check for idle time
    if current_time - person['last_activity_time'] > person['idle_threshold']:
        person['is_working'] = False
        return False
    
    return person['is_working']

def process_frame(frame):
    try:
        if frame is None:
            logger.error("Received empty frame")
            return None

        # Resize frame for YOLO processing
        frame_small = cv2.resize(frame, (640, 384))
        
        # Run YOLOv8 detection with all relevant classes
        results = model(frame_small, classes=[0, 63, 64, 67])  # person, laptop, cell phone, chair
        
        # Get detections
        detections = []
        current_time = time.time()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = r.names[cls]
                
                if conf > 0.5:
                    # Scale coordinates back to original frame size
                    scale_x = frame.shape[1] / frame_small.shape[1]
                    scale_y = frame.shape[0] / frame_small.shape[0]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    if class_name == 'person':
                        width = x2 - x1
                        height = y2 - y1
                        y1 = y1 + height * 0.1
                        y2 = y1 + height * 0.4
                        x1 = x1 + width * 0.1
                        x2 = x2 - width * 0.1
                        detections.append(([x1, y1, x2, y2], conf, 'face'))
                    else:
                        detections.append(([x1, y1, x2, y2], conf, class_name))
        
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Define zones
        zones = define_zones(frame.shape)
        
        # Reset zone counts
        metrics.zone_counts = {'zone1': 0, 'zone2': 0, 'zone3': 0, 'zone4': 0}
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Get detected objects for this person
            person_objects = [d[2] for d in detections if d[2] != 'face']
            
            # Check working status
            is_working = check_working_status(track_id, person_objects, current_time)
            
            # Draw face bounding box with working status
            x1, y1, x2, y2 = map(int, ltrb)
            color = (0, 255, 0) if is_working else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display working status with reason
            status = "Working" if is_working else "Not Working"
            reason = ""
            if not is_working:
                if 'cell phone' in person_objects:
                    reason = " (Using phone)"
                elif any(obj in ['bed', 'couch'] for obj in person_objects):
                    reason = " (Sleeping)"
                elif current_time - person_states[track_id]['last_activity_time'] > person_states[track_id]['idle_threshold']:
                    reason = " (Idle)"
            
            cv2.putText(frame, f"ID: {track_id} - {status}{reason}", (x1, y1-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update metrics and continue with existing zone tracking
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            metrics.update_heatmap(center_x, center_y)
            
            # Determine zone
            zone_width = frame.shape[1] // 4
            if center_x < zone_width:
                zone_name = 'zone1'
            elif center_x < zone_width * 2:
                zone_name = 'zone2'
            elif center_x < zone_width * 3:
                zone_name = 'zone3'
            else:
                zone_name = 'zone4'
                
            metrics.zone_counts[zone_name] += 1
            metrics.update_person_metrics(track_id, zone_name, current_time)
            
            # Update tracked objects with working status
            metrics.tracked_objects[track_id] = {
                'position': (center_x, center_y),
                'last_seen': current_time,
                'zone': zone_name,
                'is_working': is_working
            }

        # Draw zone boundaries and labels with availability status
        for zone_name, (start, end) in zones.items():
            # Determine zone color based on availability
            zone_type = ZONE_TYPES[zone_name]
            count = metrics.zone_counts[zone_name]
            capacity = metrics.zone_availability[zone_name]['capacity']
            
            if zone_type == 'restricted':
                color = (0, 0, 255)  # Red for restricted
            elif count >= capacity:
                color = (0, 0, 255)  # Red for full
            elif count > 0:
                color = (0, 255, 0)  # Green for occupied
            else:
                color = (255, 255, 0)  # Yellow for available
            
            # Draw zone boundary
            cv2.rectangle(frame, start, end, color, 2)
            
            # Draw zone label with count and capacity
            label = f"{zone_name}: {count}/{capacity}"
            cv2.putText(frame, label, 
                        (start[0] + 5, start[1] + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
            
            # Draw availability status
            status = "FULL" if count >= capacity else "AVAILABLE" if count == 0 else "OCCUPIED"
            cv2.putText(frame, status, 
                        (start[0] + 5, start[1] + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, color, 1)
        
        # Update metrics
        total_people = sum(metrics.zone_counts.values())
        metrics.room_utilization = min(100, total_people * 25)
        metrics.desk_occupancy = min(100, total_people * 50)
        
        if total_people > 0:
            metrics.is_occupied = True
            time_diff = current_time - metrics.last_activity
            metrics.productivity_hours += time_diff / 3600
            metrics.last_activity = current_time
        else:
            metrics.is_occupied = False
        
        # Update historical data every minute
        if int(current_time) % 60 == 0:
            metrics.update_historical_data()
        
        return frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def generate_frames():
    try:
        # Get camera source based on current selection
        camera_source = CAMERA_SETTINGS[metrics.current_camera]
        logger.info(f"Using camera source: {camera_source}")
        
        # Connect to camera
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera_source}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame from camera")
                break
                
            # Process every other frame to reduce delay
            frame_count += 1
            if frame_count % 2 == 0:
                continue
            
            # Process frame for object detection and tracking
            processed_frame = process_frame(frame)
            if processed_frame is None:
                continue
            
            # Convert frame to jpg format
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    try:
        camera_type = request.json.get('camera_type', 'system')
        if camera_type in CAMERA_SETTINGS:
            metrics.current_camera = camera_type
            logger.info(f"Switched to camera: {camera_type}")
            return jsonify({'status': 'success', 'camera': camera_type})
        return jsonify({'status': 'error', 'message': 'Invalid camera type'})
    except Exception as e:
        logger.error(f"Error switching camera: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/metrics')
def get_metrics():
    return jsonify({
        'productivity_hours': round(metrics.productivity_hours, 2),
        'room_utilization': metrics.room_utilization,
        'desk_occupancy': metrics.desk_occupancy,
        'zone_counts': metrics.zone_counts,
        'zone_availability': metrics.zone_availability,
        'tracked_objects': len(metrics.tracked_objects),
        'current_camera': metrics.current_camera
    })

@app.route('/api/reports', methods=['GET'])
def handle_reports():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    report_type = request.args.get('report_type')
    
    if start_date:
        start_date = datetime.fromisoformat(start_date)
    if end_date:
        end_date = datetime.fromisoformat(end_date)
        
    reports = get_reports(start_date, end_date, report_type)
    return jsonify(reports)

@app.route('/api/calendar', methods=['GET', 'POST'])
def handle_calendar():
    if request.method == 'POST':
        try:
            event_data = request.json
            event = create_calendar_event(event_data)
            return jsonify(event), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        event_type = request.args.get('event_type')
        
        if start_date:
            start_date = datetime.fromisoformat(start_date)
        if end_date:
            end_date = datetime.fromisoformat(end_date)
            
        events = get_calendar_events(start_date, end_date, event_type)
        return jsonify(events)

@app.route('/api/dashboard/history', methods=['GET'])
def get_dashboard_history():
    try:
        date_str = request.args.get('date')
        limit = request.args.get('limit', 100, type=int)
        
        if not date_str:
            return jsonify({'error': 'Date parameter is required'}), 400
            
        try:
            # Convert string date to datetime object
            selected_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Validate date is not in the future
            if selected_date.date() > datetime.now().date():
                return jsonify({'error': 'Cannot view history for future dates'}), 400
                
            # Create start and end of the selected date
            start_date = datetime.combine(selected_date.date(), datetime.min.time())
            end_date = datetime.combine(selected_date.date(), datetime.max.time())
            
            # Get history data for the selected date
            data = get_dashboard_history(start_date, end_date, limit)
            
            if not data:
                logger.info(f"No history data found for date: {date_str}")
                return jsonify([])
                
            # Sort data by timestamp in descending order
            data.sort(key=lambda x: x['timestamp'], reverse=True)
            
            logger.info(f"Retrieved {len(data)} history records for date: {date_str}")
            return jsonify(data)
            
        except ValueError as e:
            logger.error(f"Invalid date format: {date_str}")
            return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD'}), 400
            
    except Exception as e:
        logger.error(f"Error retrieving dashboard history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def get_analytics():
    try:
        analytics_data = {
            'person_metrics': metrics.get_person_productivity(),
            'historical_data': metrics.historical_data,
            'alerts': metrics.alerts,
            'current_metrics': {
                'productivity_hours': round(metrics.productivity_hours, 2),
                'room_utilization': metrics.room_utilization,
                'desk_occupancy': metrics.desk_occupancy,
                'zone_counts': metrics.zone_counts,
                'tracked_objects': len(metrics.tracked_objects),
                'total_people': sum(metrics.zone_counts.values()),
                'active_zones': {
                    zone: {
                        'count': count,
                        'type': ZONE_TYPES[zone],
                        'capacity': metrics.zone_availability[zone]['capacity']
                    }
                    for zone, count in metrics.zone_counts.items()
                }
            }
        }
        
        # Save dashboard data
        try:
            dashboard_data = {
                'productivity_hours': analytics_data['current_metrics']['productivity_hours'],
                'room_utilization': analytics_data['current_metrics']['room_utilization'],
                'desk_occupancy': analytics_data['current_metrics']['desk_occupancy'],
                'zone_counts': analytics_data['current_metrics']['zone_counts'],
                'alerts': analytics_data['alerts'],
                'person_metrics': analytics_data['person_metrics'],
                'historical_data': analytics_data['historical_data'],
                'total_people': analytics_data['current_metrics']['total_people'],
                'active_zones': analytics_data['current_metrics']['active_zones']
            }
            save_dashboard_data(dashboard_data)
            logger.info("Dashboard data saved successfully")
        except Exception as e:
            logger.error(f"Error saving dashboard data: {str(e)}")
        
        return jsonify(analytics_data)
    except Exception as e:
        logger.error(f"Error in analytics route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap')
def get_heatmap():
    heatmap = metrics.get_heatmap_image()
    ret, buffer = cv2.imencode('.jpg', heatmap)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
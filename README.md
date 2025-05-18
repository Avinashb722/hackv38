# Employee Monitor System

A real-time employee monitoring system that uses computer vision and AI to track workplace productivity and space utilization.

## Features

- Real-time person detection and tracking
- Zone-based occupancy monitoring
- Productivity analytics
- Calendar and event management
- Report generation
- Space utilization heatmap
- Alert system for anomalies

## Technologies Used

- Python
- Flask
- OpenCV
- YOLOv8
- DeepSORT
- SQLite
- HTML/CSS/JavaScript
- Chart.js

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/employee-monitor.git
cd employee-monitor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the YOLOv8 model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Features in Detail

### Dashboard
- Real-time metrics display
- Zone availability monitoring
- Productivity tracking
- Alert notifications

### Camera Monitor
- Live video feed
- Person detection and tracking
- Zone-based occupancy tracking
- Working status monitoring

### Reports
- Historical data analysis
- Export functionality (PDF, Excel, CSV)
- Custom date range selection

### Calendar
- Event management
- Zone scheduling
- Meeting room booking
- Break time tracking

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 for object detection
- DeepSORT for object tracking
- Flask for the web framework
- Chart.js for data visualization 
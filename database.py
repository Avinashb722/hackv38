from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    report_type = Column(String(50))  # daily, weekly, monthly
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)  # Store all metrics as JSON
    notes = Column(String(500))
    
    def to_dict(self):
        return {
            'id': self.id,
            'report_type': self.report_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'created_at': self.created_at.isoformat(),
            'metrics': self.metrics,
            'notes': self.notes
        }

class DashboardData(Base):
    __tablename__ = 'dashboard_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    productivity_hours = Column(Float)
    room_utilization = Column(Float)
    desk_occupancy = Column(Float)
    zone_counts = Column(JSON)
    alerts = Column(JSON)
    heatmap_data = Column(JSON)
    person_metrics = Column(JSON)  # Store individual person metrics
    historical_data = Column(JSON)  # Store historical trends
    total_people = Column(Integer)
    active_zones = Column(JSON)    # Store active zone information
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'productivity_hours': self.productivity_hours,
            'room_utilization': self.room_utilization,
            'desk_occupancy': self.desk_occupancy,
            'zone_counts': self.zone_counts,
            'alerts': self.alerts,
            'heatmap_data': self.heatmap_data,
            'person_metrics': self.person_metrics,
            'historical_data': self.historical_data,
            'total_people': self.total_people,
            'active_zones': self.active_zones
        }

class CalendarEvent(Base):
    __tablename__ = 'calendar_events'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(100))
    description = Column(String(500))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    event_type = Column(String(50))  # meeting, break, work_session
    zone_id = Column(String(50))  # zone1, zone2, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'event_type': self.event_type,
            'zone_id': self.zone_id,
            'created_at': self.created_at.isoformat()
        }

# Create database engine and session
engine = create_engine('sqlite:///employee_monitor.db')
Session = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return Session()

# Helper functions for database operations
def save_dashboard_data(metrics_data):
    session = get_session()
    try:
        dashboard_data = DashboardData(
            productivity_hours=metrics_data['productivity_hours'],
            room_utilization=metrics_data['room_utilization'],
            desk_occupancy=metrics_data['desk_occupancy'],
            zone_counts=metrics_data['zone_counts'],
            alerts=metrics_data.get('alerts', []),
            heatmap_data=metrics_data.get('heatmap_data', {}),
            person_metrics=metrics_data.get('person_metrics', {}),
            historical_data=metrics_data.get('historical_data', {}),
            total_people=metrics_data.get('total_people', 0),
            active_zones=metrics_data.get('active_zones', {})
        )
        session.add(dashboard_data)
        session.commit()
        return dashboard_data.to_dict()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def create_report(report_data):
    session = get_session()
    try:
        report = Report(
            report_type=report_data['report_type'],
            start_date=datetime.fromisoformat(report_data['start_date']),
            end_date=datetime.fromisoformat(report_data['end_date']),
            metrics=report_data['metrics'],
            notes=report_data.get('notes', '')
        )
        session.add(report)
        session.commit()
        return report.to_dict()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def create_calendar_event(event_data):
    session = get_session()
    try:
        event = CalendarEvent(
            title=event_data['title'],
            description=event_data.get('description', ''),
            start_time=datetime.fromisoformat(event_data['start_time']),
            end_time=datetime.fromisoformat(event_data['end_time']),
            event_type=event_data['event_type'],
            zone_id=event_data.get('zone_id')
        )
        session.add(event)
        session.commit()
        return event.to_dict()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_reports(start_date=None, end_date=None, report_type=None):
    session = get_session()
    try:
        query = session.query(Report)
        if start_date:
            query = query.filter(Report.start_date >= start_date)
        if end_date:
            query = query.filter(Report.end_date <= end_date)
        if report_type:
            query = query.filter(Report.report_type == report_type)
        return [report.to_dict() for report in query.all()]
    finally:
        session.close()

def get_calendar_events(start_date=None, end_date=None, event_type=None):
    session = get_session()
    try:
        query = session.query(CalendarEvent)
        if start_date:
            query = query.filter(CalendarEvent.start_time >= start_date)
        if end_date:
            query = query.filter(CalendarEvent.end_time <= end_date)
        if event_type:
            query = query.filter(CalendarEvent.event_type == event_type)
        return [event.to_dict() for event in query.all()]
    finally:
        session.close()

def get_dashboard_data(start_date=None, end_date=None):
    session = get_session()
    try:
        query = session.query(DashboardData)
        if start_date:
            query = query.filter(DashboardData.timestamp >= start_date)
        if end_date:
            query = query.filter(DashboardData.timestamp <= end_date)
        return [data.to_dict() for data in query.all()]
    finally:
        session.close()

def get_dashboard_history(start_date=None, end_date=None, limit=100):
    session = get_session()
    try:
        query = session.query(DashboardData)
        if start_date:
            query = query.filter(DashboardData.timestamp >= start_date)
        if end_date:
            query = query.filter(DashboardData.timestamp <= end_date)
        query = query.order_by(DashboardData.timestamp.desc()).limit(limit)
        return [data.to_dict() for data in query.all()]
    finally:
        session.close() 
import sys
import os
import time
import json
import numpy as np
import cv2
from mss import mss # type: ignore
import psutil
from typing import Dict, Tuple, List, Any, Optional, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import sqlite3
import pandas as pd
from pathlib import Path
import logging
import webbrowser
from queue import Queue, Empty

from PyQt5.QtCore import (
    QObject, pyqtSignal, Qt, QTimer, QThread, QPoint, QSize, QRect, pyqtSlot, QDateTime
)
from PyQt5.QtWidgets import (
    QWidget, QSpinBox, QCheckBox, QProgressBar, QGridLayout, QRadioButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QScrollArea, QApplication,
    QPushButton, QGroupBox, QStyle, QMenu, QStatusBar, QMessageBox, QInputDialog,
    QDialog, QListWidget, QSlider, QDoubleSpinBox, QMainWindow, QSizePolicy,
    QListWidgetItem, QDateTimeEdit, QTableWidget, QTableWidgetItem, QHeaderView,QSplitter, QFrame
)
from PyQt5.QtGui import (
    QColor, QPalette, QFont, QPixmap, QImage, QPainter, QPen, QBrush, QIcon
)
import pyqtgraph as pg # type: ignore
from screeninfo import get_monitors

from flask import Flask, jsonify, render_template # type: ignore
from flask_cors import CORS # type: ignore


# --- Configuration & Constants ---
APP_NAME = "VICE (Visual Identification Conveyance Enhancement) V3.2"
DB_NAME = "vice_main_database.db" # Renamed for clarity
INEFFICIENCY_DB_NAME = "vice_inefficiency_events.db" # NEW: Database for non-critical events
LOG_LEVEL = logging.INFO
CAPTURE_INTERVAL_DEFAULT = 1.0
RESOLUTION_SCALE_DEFAULT = 50
FLASK_HOST = '0.0.0.0'; FLASK_PORT = 5965

DB_BATCH_INSERT_INTERVAL = 5.0 # seconds
HEALTH_SNAPSHOT_INTERVAL = 60.0 # seconds

# Debouncing settings for blinking lights
# Grace period should be slightly longer than the OFF period of a blink.
# If a light blinks ON (1s) - OFF (1s), then a grace period of 1.1s to 1.9s is reasonable.
BLINK_GRACE_PERIOD_SECONDS = 2.8 

# Color Definitions (HSV Lower, HSV Upper, BGR for display)
# Weights adjusted for health impact: Red/Orange = large decrease, Blue/Purple = small decrease
# --- (Located at the top of the file) ---
COLOR_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "red":    {"lower": [np.array([0, 70, 70]), np.array([170, 70, 70])], "upper": [np.array([10, 255, 255]), np.array([180, 255, 255])], "bgr": (0, 0, 255), "weight": 0.9, "threshold_percent": 0.5, "message": "E-STOP", "is_blinking": True, "priority": 0},
    "orange": {"lower": np.array([5, 100, 100]), "upper": np.array([25, 255, 255]), "bgr": (0, 165, 255), "weight": 0.7, "threshold_percent": 0.5, "message": "JAM", "is_blinking": True, "priority": 1},
    "purple": {"lower": np.array([130, 50, 50]), "upper": np.array([160, 255, 255]), "bgr": (128, 0, 128), "weight": 0.3, "threshold_percent": 0.5, "message": "ANTI-GRIDLOCK", "is_blinking": False, "priority": 2},
    "blue":   {"lower": np.array([100, 100, 50]), "upper": np.array([125, 255, 255]), "bgr": (255, 0, 0), "weight": 0.2, "threshold_percent": 0.5, "message": "FULL", "is_blinking": False, "priority": 3},
    "grey":   {"lower": np.array([0, 0, 80]),   "upper": np.array([180, 50, 180]), "bgr": (128, 128, 128), "weight": 0.5, "threshold_percent": 30.0, "message": "STAND DOWN", "is_blinking": False, "priority": 4},
    "green":  {"bgr": (0, 255, 0), "message": "NOMINAL", "is_blinking": False, "weight": 0.0, "priority": 5}
}


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DB_NAME
INEFFICIENCY_DB_PATH = BASE_DIR / INEFFICIENCY_DB_NAME # NEW
REPORTS_DIR = BASE_DIR / "12_Hour_Reports"
REPORTS_DIR.mkdir(exist_ok=True) # Ensure reports directory exists

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static" 


logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] 
)
logger = logging.getLogger(__name__)

# --- Data Structures ---
@dataclass
class ROIInfo:
    roi: Tuple[int, int, int, int]
    monitor_info: Dict[str, int]
    title: str

@dataclass
class DetectedColorInfo:
    name: str
    bgr: Tuple[int, int, int]
    message: str

# --- (Located in the Data Structures section) ---
@dataclass
class RegionStatus:
    title: str; is_anomaly: bool = False; detected_colors: List[DetectedColorInfo] = field(default_factory=list)
    anomaly_start_time: Optional[datetime] = None
    # NEW: Specific event counters
    critical_error_count: int = 0 # Jams + E-Stops
    purple_count: int = 0
    blue_count: int = 0
    grey_count: int = 0
    # Internal state for debouncing
    _blink_color_last_physical_detection_time: Dict[str, Optional[datetime]] = field(default_factory=dict)

    def get_primary_color_message(self) -> str:
        if not self.is_anomaly or not self.detected_colors: return COLOR_DEFINITIONS["green"]["message"]
        msg = self.detected_colors[0].message
        if len(self.detected_colors) > 1: msg += f" (+{len(self.detected_colors)-1})"
        return msg
        
    def get_primary_color_name(self) -> str:
        if not self.is_anomaly or not self.detected_colors: return "green"
        return self.detected_colors[0].name

    def get_primary_bgr(self) -> Tuple[int, int, int]:
        if not self.is_anomaly or not self.detected_colors: return COLOR_DEFINITIONS["green"]["bgr"]
        return self.detected_colors[0].bgr

# --- Utility Functions ---
def get_standard_icon(style_enum):
    return QApplication.style().standardIcon(style_enum)

# --- Database Adapters (for datetime) ---
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter('datetime', lambda s: datetime.fromisoformat(s.decode()))

class DataLogger(QThread):
    def __init__(self, main_db_path: Path, inefficiency_db_path: Path, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.main_db_path = main_db_path
        self.inefficiency_db_path = inefficiency_db_path
        self.log_queue = Queue()
        self.running = True
        self._init_db_lock = threading.Lock()
        self._initialize_databases()
        
    def _initialize_databases(self):
        with self._init_db_lock:
            try: # Main DB
                with sqlite3.connect(self.main_db_path) as conn:
                    c = conn.cursor()
                    c.execute("CREATE TABLE IF NOT EXISTS monitoring_sessions (id INTEGER PRIMARY KEY, start_time DATETIME, end_time DATETIME, roi_count INTEGER, settings_json TEXT)")
                    c.execute("CREATE TABLE IF NOT EXISTS system_snapshots (id INTEGER PRIMARY KEY, session_id INTEGER, timestamp DATETIME, health_percentage REAL, active_anomalies INTEGER, FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id))")
                    c.execute("CREATE TABLE IF NOT EXISTS completed_events (id INTEGER PRIMARY KEY, session_id INTEGER, roi_title TEXT, primary_event_type TEXT, start_time DATETIME, end_time DATETIME, duration_seconds REAL, secondary_colors_json TEXT, FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id))")
            except Exception as e: logger.exception(f"Main DB init error: {e}")

            try: # Inefficiency DB
                with sqlite3.connect(self.inefficiency_db_path) as conn:
                    c = conn.cursor()
                    c.execute("CREATE TABLE IF NOT EXISTS completed_inefficiency_events (id INTEGER PRIMARY KEY, session_id INTEGER, roi_title TEXT, primary_event_type TEXT, start_time DATETIME, end_time DATETIME, duration_seconds REAL, secondary_colors_json TEXT, FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id))")
            except Exception as e: logger.exception(f"Inefficiency DB init error: {e}")

    def run(self):
        logger.info("DataLogger thread started.");
        while self.running:
            try:
                log_batch = []; 
                try: log_batch.append(self.log_queue.get(timeout=DB_BATCH_INSERT_INTERVAL))
                except Empty: pass
                while not self.log_queue.empty() and len(log_batch) < 50: log_batch.append(self.log_queue.get_nowait())
                if log_batch: self._process_batch(log_batch)
            except Exception as e: logger.exception(f"Error in DataLogger thread: {e}")
        logger.info("DataLogger thread finished.")
    
    def _process_batch(self, batch):
        # Separate batches by database to ensure single connection per block
        main_db_batch = [item for item in batch if item[0] != 'LOG_INEFFICIENCY_EVENT']
        inefficiency_db_batch = [item for item in batch if item[0] == 'LOG_INEFFICIENCY_EVENT']

        if main_db_batch:
            try:
                with sqlite3.connect(self.main_db_path, timeout=10) as conn:
                    c = conn.cursor()
                    for cmd, data in main_db_batch:
                        if cmd == 'START_SESSION': c.execute("INSERT INTO monitoring_sessions (start_time, roi_count, settings_json) VALUES (?, ?, ?)", (data['start_time'], data['roi_count'], data['settings_json'])); data['callback'](c.lastrowid)
                        elif cmd == 'END_SESSION': c.execute("UPDATE monitoring_sessions SET end_time = ? WHERE id = ?", (data['end_time'], data['session_id']))
                        elif cmd == 'LOG_SNAPSHOT': c.execute("INSERT INTO system_snapshots (session_id, timestamp, health_percentage, active_anomalies) VALUES (?, ?, ?, ?)", (data['session_id'], data['timestamp'], data['health'], data['active_anomalies']))
                        elif cmd == 'LOG_CRITICAL_EVENT': c.execute("INSERT INTO completed_events (session_id, roi_title, primary_event_type, start_time, end_time, duration_seconds, secondary_colors_json) VALUES (?, ?, ?, ?, ?, ?, ?)", (data['session_id'], data['roi_title'], data['event_type'], data['start_time'], data['end_time'], data['duration'], json.dumps(data['secondary_colors'])))
                    conn.commit()
            except Exception as e: logger.error(f"Failed to process main DB log batch: {e}")
        
        if inefficiency_db_batch:
            try:
                with sqlite3.connect(self.inefficiency_db_path, timeout=10) as conn:
                    c = conn.cursor()
                    for cmd, data in inefficiency_db_batch:
                        c.execute("INSERT INTO completed_inefficiency_events (session_id, roi_title, primary_event_type, start_time, end_time, duration_seconds, secondary_colors_json) VALUES (?, ?, ?, ?, ?, ?, ?)", (data['session_id'], data['roi_title'], data['event_type'], data['start_time'], data['end_time'], data['duration'], json.dumps(data['secondary_colors'])))
                    conn.commit()
            except Exception as e: logger.error(f"Failed to process inefficiency DB log batch: {e}")

    def stop(self): self.running = False; self.log_queue.put((None, None))
    def add_log_entry(self, command, data): self.log_queue.put((command, data))
    
    # Data fetching methods remain the same, they only query the main DB.
    def get_kpi_metrics(self, start_dt, end_dt):
        try:
            with sqlite3.connect(f'file:{self.main_db_path}?mode=ro', uri=True) as conn:
                q = "SELECT duration_seconds FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP') AND end_time BETWEEN ? AND ?"
                rows = conn.execute(q, (start_dt, end_dt)).fetchall()
                if not rows: return {"total_downtime": 0, "mtbf": "N/A", "mttr": "N/A", "failure_count": 0}
                downtime = sum(r[0] for r in rows); failures = len(rows); uptime = (end_dt-start_dt).total_seconds() - downtime
                return {"total_downtime": downtime, "mtbf": uptime / failures if failures > 0 else uptime, "mttr": downtime / failures if failures > 0 else 0, "failure_count": failures}
        except Exception as e: logger.error(f"DB KPI error: {e}"); return {}
    def get_pareto_analysis(self, start_dt, end_dt):
        try:
            with sqlite3.connect(f'file:{self.main_db_path}?mode=ro', uri=True) as conn:
                q = "SELECT roi_title, SUM(duration_seconds) as total_duration FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP') AND end_time BETWEEN ? AND ? GROUP BY roi_title ORDER BY total_duration DESC LIMIT 5"
                return [{"roi": r[0], "duration": r[1]} for r in conn.execute(q, (start_dt, end_dt)).fetchall()]
        except Exception as e: logger.error(f"DB Pareto error: {e}"); return []
    def get_health_snapshots(self, start_dt, end_dt):
        try:
            with sqlite3.connect(f'file:{self.main_db_path}?mode=ro', uri=True) as conn:
                q = "SELECT timestamp, health_percentage FROM system_snapshots WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp"
                return [{"timestamp": r[0].isoformat(), "health": r[1]} for r in conn.execute(q, (start_dt, end_dt)).fetchall()]
        except Exception as e: logger.error(f"DB health snapshot error: {e}"); return []
    def generate_12_hour_report_archive(self):
        logger.info("Generating 12-hour report archives..."); now = datetime.now()
        report_path_main = REPORTS_DIR / f"main_report_archive_{now.strftime('%Y-%m-%d_%H-%M-%S')}.db"
        report_path_inefficiency = REPORTS_DIR / f"inefficiency_report_archive_{now.strftime('%Y-%m-%d_%H-%M-%S')}.db"
        try:
            with sqlite3.connect(self.main_db_path) as main_conn, sqlite3.connect(report_path_main) as report_conn:
                main_conn.backup(report_conn); logger.info(f"Successfully created main backup: {report_path_main}")
            with sqlite3.connect(self.inefficiency_db_path) as main_conn, sqlite3.connect(report_path_inefficiency) as report_conn:
                main_conn.backup(report_conn); logger.info(f"Successfully created inefficiency backup: {report_path_inefficiency}")
        except Exception as e: logger.exception(f"Failed to generate 12-hour report: {e}")


class MonitorSelector(QDialog):
    def __init__(self, monitors: List[Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.monitors = monitors
        self.selected_monitor_index: Optional[int] = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Select Monitor")
        self.setModal(True)
        layout = QVBoxLayout(self)

        title_label = QLabel("Select Monitor for ROI")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        for i, monitor in enumerate(self.monitors):
            btn_text = f"Monitor {i+1} ({monitor.width}x{monitor.height} at {monitor.x},{monitor.y})"
            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, idx=i: self.select_monitor(idx))
            layout.addWidget(btn)

        self.setMinimumWidth(300)

    def select_monitor(self, monitor_index: int):
        self.selected_monitor_index = monitor_index
        self.accept()

    def get_selected_monitor(self) -> Optional[Any]:
        if self.selected_monitor_index is not None:
            return self.monitors[self.selected_monitor_index]
        return None


class AlertWindow(QWidget):
    # --- Configuration for the new AlertWindow ---
    MAX_RECENTLY_CLEARED_TO_SHOW = 10 # Keep this many "green" rows before pruning

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Region Status Monitor")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setMinimumSize(800, 800)

        # Internal state
        self.all_region_statuses: Dict[str, RegionStatus] = {}
        self.recently_cleared_queue: Deque[str] = deque(maxlen=self.MAX_RECENTLY_CLEARED_TO_SHOW)
        
        self.conveyor_health = 100.0
        self.session_start_time = datetime.now()
        
        # UI Elements
        self.region_row_widgets: Dict[str, QWidget] = {} # Only stores widgets currently visible

        self.initUI()
        self.position_window()

    def initUI(self):
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(10)
        
        title_label = QLabel("Region Monitoring Status"); title_label.setFont(QFont("Arial", 16, QFont.Bold)); title_label.setAlignment(Qt.AlignCenter); main_layout.addWidget(title_label)
        
        health_group = QGroupBox("Overall Conveyor Health"); health_layout = QHBoxLayout(); self.health_label = QLabel(f"Health: {self.conveyor_health:.1f}%"); self.health_label.setFont(QFont("Arial", 12, QFont.Bold)); self.health_progress_bar = QProgressBar(); self.health_progress_bar.setRange(0, 100); self.health_progress_bar.setValue(int(self.conveyor_health)); self.health_progress_bar.setTextVisible(False); health_layout.addWidget(self.health_label); health_layout.addWidget(self.health_progress_bar, 1); health_group.setLayout(health_layout); main_layout.addWidget(health_group)
        
        session_info_layout = QHBoxLayout(); self.uptime_label = QLabel("Uptime: 0s"); self.active_rois_label = QLabel("Total ROIs: 0"); session_info_layout.addWidget(self.uptime_label); session_info_layout.addStretch(); session_info_layout.addWidget(self.active_rois_label); main_layout.addLayout(session_info_layout)

        header_widget = QWidget(); header_layout = QHBoxLayout(header_widget); header_layout.setContentsMargins(6, 3, 6, 3); header_font = QFont("Arial", 9, QFont.Bold); header_font.setUnderline(True)
        col_labels = ["Region", "Status", "Duration", "Crit. Errors", "Purple", "Blue", "Grey"]
        col_stretches = [4, 5, 2, 1, 1, 1, 1]
        for text, stretch in zip(col_labels, col_stretches):
            lbl = QLabel(text); lbl.setFont(header_font); header_layout.addWidget(lbl, stretch)
        main_layout.addWidget(header_widget)

        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.alert_container_widget = QWidget()
        self.alert_lines_layout = QVBoxLayout(self.alert_container_widget); self.alert_lines_layout.setContentsMargins(0, 0, 0, 0); self.alert_lines_layout.setSpacing(1); self.alert_lines_layout.addStretch()
        scroll_area.setWidget(self.alert_container_widget)
        main_layout.addWidget(scroll_area, 1)

        self.update_timer = QTimer(self); self.update_timer.timeout.connect(self._update_session_info); self.update_timer.start(1000)
        self.flash_timer = QTimer(self); self.flash_timer.timeout.connect(self._handle_flashing); self.flash_timer.start(750); self._flash_state = False

    def _update_session_info(self):
        uptime_delta = datetime.now() - self.session_start_time
        self.uptime_label.setText(f"Uptime: {str(uptime_delta).split('.')[0]}")
        self.active_rois_label.setText(f"Total ROIs: {len(self.all_region_statuses)}")

    def _handle_flashing(self):
        self._flash_state = not self._flash_state
        for title, row_widget in self.region_row_widgets.items():
            status_obj = self.all_region_statuses.get(title)
            if not status_obj: continue
            is_over_time = status_obj.is_anomaly and status_obj.anomaly_start_time and (datetime.now() - status_obj.anomaly_start_time).total_seconds() > 150
            flash_style = "background-color: #581c1c;"
            current_style = row_widget.styleSheet()
            has_flash = flash_style in current_style
            if is_over_time:
                if self._flash_state and not has_flash: row_widget.setStyleSheet(current_style + flash_style)
                elif not self._flash_state and has_flash: row_widget.setStyleSheet(current_style.replace(flash_style, ""))
            elif has_flash: row_widget.setStyleSheet(current_style.replace(flash_style, ""))

    def position_window(self):
        try:
            screen_geometry = QApplication.primaryScreen().geometry()
            self.move(screen_geometry.width() - self.width() - 20, 30)
        except Exception as e:
            logger.warning(f"Could not auto-position alert window: {e}")

    @pyqtSlot(float, int)
    def update_overall_health(self, health, active_anomalies):
        self.health_label.setText(f"Health: {health:.1f}%")
        self.health_progress_bar.setValue(int(health))
        stylesheet = "QProgressBar::chunk{background-color:%s;}"; color = "green"
        if health < 40: color = "red"
        elif health < 70: color = "orange"
        self.health_progress_bar.setStyleSheet(stylesheet % color)

    @pyqtSlot(RegionStatus)
    def update_region_status(self, status: RegionStatus):
        title = status.title
        was_anomaly = self.all_region_statuses.get(title, RegionStatus(title, is_anomaly=False)).is_anomaly
        self.all_region_statuses[title] = status
        
        # Logic to decide if a row should be added or removed
        should_be_visible = status.is_anomaly or title in self.recently_cleared_queue
        is_currently_visible = title in self.region_row_widgets

        # If an event clears, add it to the recently cleared queue
        if was_anomaly and not status.is_anomaly:
            if title in self.recently_cleared_queue:
                self.recently_cleared_queue.remove(title) # Remove to re-add at the end
            self.recently_cleared_queue.append(title)

        # Prune the oldest "cleared" event if the queue is full AND a new anomaly appears
        # This prevents the list from shrinking while everything is nominal
        if len(self.recently_cleared_queue) == self.MAX_RECENTLY_CLEARED_TO_SHOW and status.is_anomaly and not was_anomaly:
             # The deque automatically handles pruning the oldest item when a new one is appended
             pass

        # Redraw the visible rows, which will handle adding/removing/updating
        self._sort_and_redraw_visible_rows()

    def _sort_and_redraw_visible_rows(self):
        # Determine which statuses should be visible
        visible_titles = {title for title, s in self.all_region_statuses.items() if s.is_anomaly}
        visible_titles.update(self.recently_cleared_queue)

        # Prune widgets that are no longer visible
        for title in list(self.region_row_widgets.keys()):
            if title not in visible_titles:
                widget = self.region_row_widgets.pop(title)
                widget.deleteLater()

        # Get the actual status objects for visible ROIs and sort them
        visible_statuses = [self.all_region_statuses[title] for title in visible_titles if title in self.all_region_statuses]
        sorted_statuses = sorted(visible_statuses, key=lambda s: COLOR_DEFINITIONS[s.get_primary_color_name()]["priority"])

        # Update the layout based on the sorted list
        for i, status in enumerate(sorted_statuses):
            self._create_or_update_row(status, i)

    def _create_or_update_row(self, status: RegionStatus, position: int):
        title = status.title
        
        if title not in self.region_row_widgets:
            line_widget = QWidget(); line_widget.setAutoFillBackground(True); line_layout = QHBoxLayout(line_widget); line_layout.setContentsMargins(6, 2, 6, 2); line_layout.setSpacing(10)
            
            font_small = QFont("Arial", 9); font_bold = QFont("Arial", 10, QFont.Bold)
            
            # Create all the labels for the row
            labels = {
                "title": QLabel(title), "status": QLabel(), "duration": QLabel(),
                "critical": QLabel(), "purple": QLabel(), "blue": QLabel(), "grey": QLabel()
            }
            labels["title"].setFont(font_bold); labels["title"].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for key in ["status", "duration", "critical", "purple", "blue", "grey"]: labels[key].setFont(font_small); labels[key].setAlignment(Qt.AlignCenter)
            
            line_layout.addWidget(labels["title"], 4); line_layout.addWidget(labels["status"], 5); line_layout.addWidget(labels["duration"], 2); line_layout.addWidget(labels["critical"], 1); line_layout.addWidget(labels["purple"], 1); line_layout.addWidget(labels["blue"], 1); line_layout.addWidget(labels["grey"], 1)
            
            self.alert_lines_layout.insertWidget(position, line_widget)
            self.region_row_widgets[title] = line_widget
            line_widget.setProperty("labels", labels) # Store labels with the widget
        
        # Update the content of the labels
        widget = self.region_row_widgets[title]
        labels = widget.property("labels")

        labels["status"].setText(status.get_primary_color_message())
        labels["duration"].setText(f"Dur: {str(datetime.now() - status.anomaly_start_time).split('.')[0]}" if status.anomaly_start_time else "Dur: --")
        labels["critical"].setText(str(status.critical_error_count))
        labels["purple"].setText(str(status.purple_count))
        labels["blue"].setText(str(status.blue_count))
        labels["grey"].setText(str(status.grey_count))
        
        # Update colors
        bgr_color = status.get_primary_bgr(); text_color = QColor(bgr_color[2], bgr_color[1], bgr_color[0])
        labels["status"].setStyleSheet(f"color: {text_color.name()};" + ("font-weight: bold;" if status.is_anomaly else ""))
        labels["duration"].setStyleSheet(f"color: {'#f0f0f0' if status.is_anomaly else '#888'};")
        
    def remove_region(self, title: str):
        if title in self.all_region_statuses: del self.all_region_statuses[title]
        if title in self.region_row_widgets: self.region_row_widgets.pop(title).deleteLater()
        if title in self.recently_cleared_queue: self.recently_cleared_queue.remove(title)
        self._sort_and_redraw_visible_rows()

    def clear_all_regions(self):
        self.all_region_statuses.clear(); self.recently_cleared_queue.clear()
        for widget in self.region_row_widgets.values(): widget.deleteLater()
        self.region_row_widgets.clear()
        
    def closeEvent(self, event):
        self.update_timer.stop(); self.flash_timer.stop(); super().closeEvent(event)


class ReportingDashboard(QMainWindow):
    def __init__(self, data_logger: DataLogger, parent: Optional[QWidget] = None):
        super().__init__(parent); self.data_logger = data_logger; self.setWindowTitle("VICE - Reporting & Analysis"); self.setGeometry(150, 150, 1600, 900); self.initUI()
    def initUI(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        control_bar = QHBoxLayout(); self.start_dt_edit = QDateTimeEdit(QDateTime.currentDateTime().addDays(-1)); self.end_dt_edit = QDateTimeEdit(QDateTime.currentDateTime()); refresh_btn = QPushButton("Load Data")
        control_bar.addWidget(QLabel("From:")); control_bar.addWidget(self.start_dt_edit); control_bar.addWidget(QLabel("To:")); control_bar.addWidget(self.end_dt_edit); control_bar.addWidget(refresh_btn); control_bar.addStretch()
        main_layout.addLayout(control_bar)
        splitter = QSplitter(Qt.Horizontal); main_layout.addWidget(splitter)
        left_panel = QFrame(); left_panel.setFrameShape(QFrame.StyledPanel); left_layout = QVBoxLayout(left_panel)
        kpi_group = QGroupBox("Key Performance Indicators (KPIs)"); kpi_layout = QGridLayout(kpi_group)
        self.kpi_labels = {"downtime":QLabel("N/A"), "mtbf":QLabel("N/A"), "mttr":QLabel("N/A"), "failures":QLabel("N/A")}
        kpi_layout.addWidget(QLabel("<b>Total Downtime:</b>"), 0, 0); kpi_layout.addWidget(self.kpi_labels["downtime"], 0, 1); kpi_layout.addWidget(QLabel("<b>MTBF (Hours):</b>"), 1, 0); kpi_layout.addWidget(self.kpi_labels["mtbf"], 1, 1); kpi_layout.addWidget(QLabel("<b>MTTR (Seconds):</b>"), 2, 0); kpi_layout.addWidget(self.kpi_labels["mttr"], 2, 1); kpi_layout.addWidget(QLabel("<b>Failure Count:</b>"), 3, 0); kpi_layout.addWidget(self.kpi_labels["failures"], 3, 1)
        left_layout.addWidget(kpi_group)
        pareto_group = QGroupBox("Top 5 Downtime Contributors"); pareto_layout = QVBoxLayout(pareto_group)
        self.pareto_table = QTableWidget(5, 2); self.pareto_table.setHorizontalHeaderLabels(["ROI", "Total Downtime (s)"]); self.pareto_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pareto_layout.addWidget(self.pareto_table); left_layout.addWidget(pareto_group); splitter.addWidget(left_panel)
        right_panel = QFrame(); right_panel.setFrameShape(QFrame.StyledPanel); right_layout = QVBoxLayout(right_panel)
        health_group = QGroupBox("Conveyor Health Over Time"); health_layout = QVBoxLayout(health_group)
        self.health_plot = pg.PlotWidget(); self.health_plot.setLabel('left', 'Health %', units='%'); self.health_plot.setLabel('bottom', 'Time'); self.health_plot.setAxisItems({'bottom': pg.DateAxisItem()}); self.health_plot.showGrid(x=True, y=True, alpha=0.3)
        health_layout.addWidget(self.health_plot); right_layout.addWidget(health_group); splitter.addWidget(right_panel); splitter.setSizes([600, 1000])
        refresh_btn.clicked.connect(self.refresh_data); self.refresh_data()
    def refresh_data(self): self.load_report_data(self.start_dt_edit.dateTime().toPyDateTime(), self.end_dt_edit.dateTime().toPyDateTime())
    def load_report_data(self, start_dt, end_dt):
        kpis = self.data_logger.get_kpi_metrics(start_dt, end_dt)
        self.kpi_labels["downtime"].setText(f"{timedelta(seconds=int(kpis.get('total_downtime', 0)))}")
        mtbf_val = kpis.get('mtbf'); self.kpi_labels["mtbf"].setText(f"{mtbf_val / 3600:.2f}" if isinstance(mtbf_val, (int, float)) else "N/A")
        mttr_val = kpis.get('mttr'); self.kpi_labels["mttr"].setText(f"{mttr_val:.2f}" if isinstance(mttr_val, (int, float)) else "N/A")
        self.kpi_labels["failures"].setText(f"{kpis.get('failure_count', 'N/A')}")
        pareto_data = self.data_logger.get_pareto_analysis(start_dt, end_dt); self.pareto_table.setRowCount(len(pareto_data));
        for row, item in enumerate(pareto_data): self.pareto_table.setItem(row, 0, QTableWidgetItem(item["roi"])); self.pareto_table.setItem(row, 1, QTableWidgetItem(f"{item['duration']:.2f}"))
        health_data = self.data_logger.get_health_snapshots(start_dt, end_dt); self.health_plot.clear()
        if health_data:
            timestamps = [datetime.fromisoformat(d['timestamp']).timestamp() for d in health_data]; health_values = [d['health'] for d in health_data]
            self.health_plot.plot(timestamps, health_values, pen='b')


class AnomalyDetector(QObject):
    new_region_status = pyqtSignal(RegionStatus)
    metrics_update_for_logging = pyqtSignal(dict)
    anomaly_ended = pyqtSignal(dict) # Emits a dict payload
    
    def __init__(self, parent=None):
        super().__init__(parent); self.rois={}; self.running=False; self.sct=None; self.monitor_thread=None; self.resolution_scale_factor=RESOLUTION_SCALE_DEFAULT/100.0; self.capture_interval_sec=CAPTURE_INTERVAL_DEFAULT; self.detection_mode_all_colors=False; self._current_roi_statuses={}
    
    def update_config(self, rois, res, interval, all_colors):
        self.rois=rois; self.resolution_scale_factor=max(0.1,min(1.0,res/100.0)); self.capture_interval_sec=max(0.05,interval); self.detection_mode_all_colors=all_colors; self._current_roi_statuses={t:self._current_roi_statuses.get(t,RegionStatus(t)) for t in rois}
    
    def _capture(self, roi_info):
        if not self.sct: return None
        try:
            mon = {"top": roi_info.monitor_info["top"] + roi_info.roi[1], "left": roi_info.monitor_info["left"] + roi_info.roi[0], "width": roi_info.roi[2], "height": roi_info.roi[3]}
            if mon["width"] <= 0 or mon["height"] <= 0: return None
            img = np.array(self.sct.grab(mon)); img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return cv2.resize(img_bgr, (0,0), fx=self.resolution_scale_factor, fy=self.resolution_scale_factor, interpolation=cv2.INTER_AREA) if self.resolution_scale_factor<1.0 else img_bgr
        except Exception: return None

    def _detect(self, img):
        if img is None: return []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); detected = []
        for name, p in COLOR_DEFINITIONS.items():
            if name == "green": continue
            if not self.detection_mode_all_colors and name not in ["orange", "red", "grey"]: continue
            mask = cv2.bitwise_or(cv2.inRange(hsv, p["lower"][0], p["upper"][0]), cv2.inRange(hsv, p["lower"][1], p["upper"][1])) if isinstance(p.get("lower"), list) else cv2.inRange(hsv, p["lower"], p["upper"])
            if np.sum(mask > 0) / mask.size * 100 >= p["threshold_percent"]: detected.append(DetectedColorInfo(name,p["bgr"],p["message"]))
        detected.sort(key=lambda d: list(COLOR_DEFINITIONS.keys()).index(d.name)); return detected
    
    def _loop(self):
        try:
            self.sct = mss(); logger.info("Monitoring loop started.")
            while self.running:
                scan_time = datetime.now(); loop_start = time.perf_counter(); statuses_for_metrics = {}
                for title, roi_info in list(self.rois.items()):
                    if not self.running: break
                    status, was_anomaly = self._current_roi_statuses[title], self._current_roi_statuses[title].is_anomaly
                    prev_primary, prev_start = status.get_primary_color_name(), status.anomaly_start_time; prev_secondary = [c.name for c in status.detected_colors[1:]]
                    img = self._capture(roi_info); frame_colors = self._detect(img); del img
                    persistent_colors = []
                    for name, props in COLOR_DEFINITIONS.items():
                        if name == "green": continue
                        color_in_frame = next((d for d in frame_colors if d.name == name), None)
                        if color_in_frame: status._blink_color_last_physical_detection_time[name] = scan_time; persistent_colors.append(color_in_frame)
                        elif props.get("is_blinking", False):
                            last_seen = status._blink_color_last_physical_detection_time.get(name)
                            if last_seen and (scan_time - last_seen).total_seconds() < BLINK_GRACE_PERIOD_SECONDS: persistent_colors.append(DetectedColorInfo(name, props["bgr"], props["message"]))
                            else: status._blink_color_last_physical_detection_time[name] = None
                    persistent_colors.sort(key=lambda d: list(COLOR_DEFINITIONS.keys()).index(d.name))
                    status.detected_colors = persistent_colors; is_now_anomaly = bool(persistent_colors)
                    if is_now_anomaly and not was_anomaly:
                        status.is_anomaly = True; status.anomaly_start_time = scan_time
                        p_color = status.get_primary_color_name()
                        if p_color in ["red", "orange"]: status.critical_error_count += 1
                        elif p_color == "purple": status.purple_count += 1
                        elif p_color == "blue": status.blue_count += 1
                        elif p_color == "grey": status.grey_count += 1
                    elif not is_now_anomaly and was_anomaly:
                        status.is_anomaly = False; status.anomaly_start_time = None
                        if prev_primary and prev_start:
                            payload = {"roi_title": title, "primary_event_type_name": prev_primary, "event_type": COLOR_DEFINITIONS[prev_primary]['message'], "start_time": prev_start, "end_time": scan_time, "duration": (scan_time - prev_start).total_seconds(), "secondary_colors": prev_secondary}
                            self.anomaly_ended.emit(payload)
                    self.new_region_status.emit(status)
                    statuses_for_metrics[title] = {"is_anomaly": status.is_anomaly, "detected_colors": [vars(d) for d in status.detected_colors], "start_time": status.anomaly_start_time.isoformat() if status.anomaly_start_time else None}
                self.metrics_update_for_logging.emit({"timestamp": scan_time, "statuses": statuses_for_metrics})
                time.sleep(max(0, self.capture_interval_sec - (time.perf_counter() - loop_start)))
        except Exception as e: logger.exception("CRITICAL ERROR in monitoring loop:")
        finally:
            if self.sct: self.sct.close()
            logger.info("Monitoring loop stopped.")

    def start(self):
        if self.running: return
        for s in self._current_roi_statuses.values(): s._blink_color_last_physical_detection_time.clear()
        self.running = True; self.monitor_thread = threading.Thread(target=self._loop, daemon=True); self.monitor_thread.start()
    def stop(self):
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive(): self.monitor_thread.join(timeout=3.0)
        for s in self._current_roi_statuses.values():
            if s.is_anomaly: s.is_anomaly=False; s.detected_colors=[]; s.anomaly_start_time=None; self.new_region_status.emit(s)


class ConveyorMetrics(QObject):
    health_updated = pyqtSignal(float, int)
    @pyqtSlot(dict)
    def process_metrics(self, data):
        num_rois = len(data["statuses"])
        if not num_rois: self.health_updated.emit(100.0, 0); return
        impact, active_anomalies = 0.0, 0
        for s in data["statuses"].values():
            if s["is_anomaly"] and s["detected_colors"]:
                active_anomalies += 1
                p_color = s["detected_colors"][0]["name"]
                impact += COLOR_DEFINITIONS[p_color].get("weight", 0.0)
        health = max(0.0, 100.0 - (impact / num_rois * 100.0))
        self.health_updated.emit(health, active_anomalies)


class FlaskAppWrapper(QThread):
    #FIX: Signals must be defined at the class level
    api_started = pyqtSignal(str, int)
    api_error = pyqtSignal(str)

    def __init__(self, data_logger: DataLogger, host: str, port: int, parent: Optional[QObject]=None):
        super().__init__(parent)
        self.data_logger = data_logger; self.host = host; self.port = port
        self.flask_app = Flask(__name__, template_folder=str(TEMPLATES_DIR)); CORS(self.flask_app)
        self._setup_routes()
    def _setup_routes(self):
        @self.flask_app.route('/')
        def status(): return jsonify({'status': 'online'})
        @self.flask_app.route('/dashboard')
        def dashboard(): return render_template("index.html")
        @self.flask_app.route('/api/live_metrics')
        def live(): return jsonify(self.data_logger.get_latest_metrics() or {})
        @self.flask_app.route('/api/completed_events')
        def completed(): return jsonify(self.data_logger.get_completed_events() or [])
        @self.flask_app.route('/api/hourly_summary')
        def hourly(): return jsonify(self.data_logger.get_hourly_summary() or [])
    def run(self):
        try:
            logger.info(f"Starting Flask API on {self.host}:{self.port}")
            self.api_started.emit(self.host, self.port)
            self.flask_app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        except Exception as e: self.api_error.emit(str(e)); logger.exception("Flask API thread error:")
    def shutdown(self):
        if self.isRunning(): self.quit(); self.wait(2000)


class ROISelectionOverlay(QDialog):
    def __init__(self, monitor_geom: Dict[str, int], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.monitor_geom = monitor_geom # {'top', 'left', 'width', 'height'}
        self.start_pos: Optional[QPoint] = None
        self.end_pos: Optional[QPoint] = None
        self.current_pos: Optional[QPoint] = None
        self.selecting = False
        self.selected_rect: Optional[QRect] = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(monitor_geom['left'], monitor_geom['top'], 
                         monitor_geom['width'], monitor_geom['height'])
        
        # Capture screenshot for background
        with mss() as sct:
            sct_img = sct.grab(self.monitor_geom)
            self.bg_pixmap = QPixmap.fromImage(QImage(sct_img.rgb, sct_img.width, sct_img.height, QImage.Format_RGB888).rgbSwapped())
        
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True) # For live update of current_pos

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.bg_pixmap) # Draw screenshot as background
        
        # Semi-transparent overlay over the entire screen
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.selecting and self.start_pos and self.current_pos:
            rect_to_draw = QRect(self.start_pos, self.current_pos).normalized()
            # Clear the overlay within the selection rectangle
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(rect_to_draw, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            
            # Draw border for the selection
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect_to_draw)
            
            # Draw dimensions
            painter.setPen(Qt.white)
            painter.setFont(QFont('Arial', 10))
            dim_text = f"{rect_to_draw.width()}x{rect_to_draw.height()}"
            painter.drawText(rect_to_draw.topLeft() + QPoint(5, -5), dim_text)


    def mousePressEvent(self, event: Any): # QMouseEvent
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.current_pos = event.pos() # Initialize current_pos
            self.selecting = True
            self.update() # Trigger repaint

    def mouseMoveEvent(self, event: Any): # QMouseEvent
        if self.selecting:
            self.current_pos = event.pos()
            self.update() # Trigger repaint

    def mouseReleaseEvent(self, event: Any): # QMouseEvent
        if event.button() == Qt.LeftButton and self.selecting:
            self.end_pos = event.pos()
            self.selecting = False
            if self.start_pos and self.end_pos:
                self.selected_rect = QRect(self.start_pos, self.end_pos).normalized()
                if self.selected_rect.width() < 2 or self.selected_rect.height() < 2: # Min ROI size
                    QMessageBox.warning(self, "ROI Too Small", "Selected ROI is too small. Please try again.")
                    self.selected_rect = None # Discard small ROI
                    self.start_pos = None # Reset for new selection
                    self.end_pos = None
                    self.update()
                    return
            self.accept() # Close dialog and signal accepted

    def keyPressEvent(self, event: Any): # QKeyEvent
        if event.key() == Qt.Key_Escape:
            self.reject() # Close dialog and signal rejected

    def get_selected_rect(self) -> Optional[QRect]:
        return self.selected_rect


class MetricsDashboard(QMainWindow):
    # This is a simplified version of the original MetricsDashboard
    # focusing on showing live and some historical data.
    # The original had more complex calculations; some are now in ConveyorMetrics.
    
    UPDATE_INTERVAL = 1000  # ms
    HISTORY_POINTS = 300 # Number of points for health history graph (e.g., 5 minutes at 1s interval)

    def __init__(self, data_logger: DataLogger, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.data_logger = data_logger # Used to fetch historical data on open

        self.timestamps_history: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        self.health_values_history: Deque[float] = deque(maxlen=self.HISTORY_POINTS)
        
        self.init_ui()
        self.load_initial_history()

        # Timer for live updates if this dashboard stays open
        self.live_update_timer = QTimer(self)
        self.live_update_timer.timeout.connect(self._request_live_metrics_update) # Placeholder if needed
        # self.live_update_timer.start(self.UPDATE_INTERVAL) # Live updates are now pushed via signal

    def init_ui(self):
        self.setWindowTitle("Conveyor Metrics Dashboard (Live)")
        self.setMinimumSize(1000, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Health Bar (similar to AlertWindow for consistency)
        health_group = QGroupBox("Current Conveyor Health")
        health_layout = QHBoxLayout(health_group)
        self.current_health_label = QLabel("Health: --%")
        self.current_health_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.current_health_bar = QProgressBar()
        self.current_health_bar.setRange(0,100); self.current_health_bar.setTextVisible(False)
        health_layout.addWidget(self.current_health_label)
        health_layout.addWidget(self.current_health_bar, 1)
        main_layout.addWidget(health_group)

        # Graphs
        graphs_group = QGroupBox("Performance Graphs")
        graphs_layout = QHBoxLayout(graphs_group)
        
        # Health History Plot
        self.health_plot_widget = pg.PlotWidget()
        self.health_plot_widget.setTitle("Health History (Recent)", size="12pt")
        self.health_plot_widget.setLabel('left', 'Health %')
        self.health_plot_widget.setLabel('bottom', 'Time (Samples)')
        self.health_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.health_plot_widget.setYRange(0, 100)
        self.health_history_curve = self.health_plot_widget.plot(pen='b')
        graphs_layout.addWidget(self.health_plot_widget)

        # Color Event Counts Plot (Bar Graph)
        self.color_counts_plot_widget = pg.PlotWidget()
        self.color_counts_plot_widget.setTitle("Anomaly Color Counts (Session)", size="12pt")
        self.color_counts_plot_widget.setLabel('left', 'Count')
        # self.color_counts_plot_widget.setLabel('bottom', 'Color Type') # X-axis ticks will be color names
        self.color_bar_graph_item = None # Will be pg.BarGraphItem
        graphs_layout.addWidget(self.color_counts_plot_widget)
        
        main_layout.addWidget(graphs_group)

        # Key Metrics Display (Text)
        key_metrics_group = QGroupBox("Key Metrics (Live)")
        key_metrics_layout = QGridLayout(key_metrics_group)
        self.uptime_metric_label = QLabel("Uptime: --")
        self.active_rois_metric_label = QLabel("Active ROIs: --")
        self.active_jams_metric_label = QLabel("Active Jams: --")
        key_metrics_layout.addWidget(QLabel("<b>Uptime:</b>"), 0, 0); key_metrics_layout.addWidget(self.uptime_metric_label, 0, 1)
        key_metrics_layout.addWidget(QLabel("<b>Active ROIs:</b>"), 1, 0); key_metrics_layout.addWidget(self.active_rois_metric_label, 1, 1)
        key_metrics_layout.addWidget(QLabel("<b>Current Jams:</b>"), 2, 0); key_metrics_layout.addWidget(self.active_jams_metric_label, 2, 1)
        main_layout.addWidget(key_metrics_group)


    def load_initial_history(self):
        historical_data = self.data_logger.get_health_history(limit=self.HISTORY_POINTS)
        for record in historical_data:
            # Assuming record["timestamp"] is ISO string, convert to epoch for plotting if needed, or use as is if pyqtgraph handles it.
            # For simplicity, using sequence number for x-axis for now.
            try:
                # If using actual timestamps for X-axis:
                # dt_obj = datetime.fromisoformat(record["timestamp"])
                # self.timestamps_history.append(dt_obj.timestamp()) # epoch float
                self.health_values_history.append(record["health_value"])
            except Exception as e:
                logger.warning(f"Could not parse historical timestamp: {record.get('timestamp')}, error: {e}")
        
        if self.health_values_history:
             self.timestamps_history.extend(range(len(self.health_values_history))) # Use sequence for x if timestamps are tricky
             self.health_history_curve.setData(list(self.timestamps_history), list(self.health_values_history))


    @pyqtSlot(dict) # Slot to receive live metrics from ConveyorMetrics
    def update_live_metrics(self, metrics: dict):
        # Health Bar and Label
        health = metrics.get("conveyor_health", 100.0)
        self.current_health_label.setText(f"Health: {health:.1f}%")
        self.current_health_bar.setValue(int(health))
        stylesheet = "QProgressBar::chunk { background-color: %s; }"
        if health < 40: color = "red"
        elif health < 70: color = "orange"
        else: color = "green"
        self.current_health_bar.setStyleSheet(stylesheet % color)

        # Health History Graph
        self.health_values_history.append(health)
        if len(self.timestamps_history) == 0 or self.timestamps_history[-1] < self.HISTORY_POINTS -1 : # if not full or just started
            self.timestamps_history.append(len(self.timestamps_history))
        else: # Shift x-axis data
            self.timestamps_history.popleft()
            self.timestamps_history.append(self.timestamps_history[-1] + 1)

        self.health_history_curve.setData(list(self.timestamps_history), list(self.health_values_history))
        
        # Color Counts Bar Graph
        color_counts = metrics.get("color_event_counts", {})
        if color_counts:
            colors_present = [c for c in COLOR_DEFINITIONS.keys() if c != "green" and c in color_counts] # Order from config
            x_ticks_labels = []
            bar_heights = []
            bar_brushes = []

            for i, color_name in enumerate(colors_present):
                x_ticks_labels.append((i, color_name.capitalize()))
                bar_heights.append(color_counts.get(color_name,0))
                bgr = COLOR_DEFINITIONS[color_name]["bgr"]
                bar_brushes.append(QColor(bgr[2],bgr[1],bgr[0])) # RGB for QColor

            if self.color_bar_graph_item:
                self.color_counts_plot_widget.removeItem(self.color_bar_graph_item)

            self.color_bar_graph_item = pg.BarGraphItem(x=list(range(len(colors_present))), 
                                                        height=bar_heights, width=0.6, brushes=bar_brushes)
            self.color_counts_plot_widget.addItem(self.color_bar_graph_item)
            
            ax = self.color_counts_plot_widget.getAxis('bottom')
            ax.setTicks([x_ticks_labels])


        # Key Metrics Labels
        uptime_sec = metrics.get("uptime_seconds", 0)
        self.uptime_metric_label.setText(str(timedelta(seconds=int(uptime_sec))))
        self.active_rois_metric_label.setText(str(metrics.get("active_rois", 0)))
        self.active_jams_metric_label.setText(str(metrics.get("active_jams_count",0)))


    def _request_live_metrics_update(self):
        # This would be used if dashboard pulls data.
        # Now, data is pushed via update_live_metrics slot.
        pass

    def closeEvent(self, event):
        self.live_update_timer.stop()
        logger.info("Metrics Dashboard closed.")
        super().closeEvent(event)


class MainApplicationWindow(QMainWindow):
    session_id_received = pyqtSignal(int)
    def __init__(self):
        super().__init__(); self.rois = {}; self.monitoring_active_flag = False; self.detection_mode_all_colors = False; self.current_session_id = None; self.last_health_snapshot = (100.0, 0)
        self.data_logger = DataLogger(DB_PATH, INEFFICIENCY_DB_PATH); self.alert_window = AlertWindow(); self.anomaly_detector = AnomalyDetector(); self.conveyor_metrics = ConveyorMetrics(); self.reporting_dashboard = None
        self.data_logger.start(); self.initUI(); self.connect_signals(); self.load_settings()
        self.resource_timer = QTimer(self); self.resource_timer.timeout.connect(self.update_resource_usage); self.resource_timer.start(2000)
        self.clock_timer = QTimer(self); self.clock_timer.timeout.connect(self.update_clock); self.clock_timer.start(1000)
        self.report_scheduler = QTimer(self); self.report_scheduler.timeout.connect(self.run_scheduled_reports); self.report_scheduler.start(60000)
        self.health_snapshot_timer = QTimer(self); self.health_snapshot_timer.timeout.connect(self.log_health_snapshot)
    def initUI(self):
        self.setWindowTitle(APP_NAME); self.setGeometry(100, 100, 600, 800)
        central_widget = QWidget(); self.setCentralWidget(central_widget); main_layout = QVBoxLayout(central_widget)
        controls_group = QGroupBox("Controls"); controls_layout = QVBoxLayout(controls_group); resource_layout = QHBoxLayout(); self.cpu_label = QLabel("CPU: --%"); self.memory_label = QLabel("Mem: --%"); resource_layout.addWidget(self.cpu_label); resource_layout.addWidget(self.memory_label); controls_layout.addLayout(resource_layout)
        roi_buttons_layout = QGridLayout(); self.add_roi_btn = QPushButton(get_standard_icon(QStyle.SP_FileDialogNewFolder), " Add ROI"); self.delete_roi_btn = QPushButton(get_standard_icon(QStyle.SP_DialogDiscardButton), " Delete ROI"); self.clear_rois_btn = QPushButton(get_standard_icon(QStyle.SP_TrashIcon), " Clear All ROIs"); roi_buttons_layout.addWidget(self.add_roi_btn, 0, 0); roi_buttons_layout.addWidget(self.delete_roi_btn, 0, 1); roi_buttons_layout.addWidget(self.clear_rois_btn, 0, 2); controls_layout.addLayout(roi_buttons_layout)
        saveload_buttons_layout = QHBoxLayout(); self.save_rois_btn = QPushButton(get_standard_icon(QStyle.SP_DialogSaveButton), " Save ROIs"); self.load_rois_btn = QPushButton(get_standard_icon(QStyle.SP_DialogOpenButton), " Load ROIs"); saveload_buttons_layout.addWidget(self.save_rois_btn); saveload_buttons_layout.addWidget(self.load_rois_btn); controls_layout.addLayout(saveload_buttons_layout); main_layout.addWidget(controls_group)
        regions_group = QGroupBox("Monitored Regions"); regions_layout = QVBoxLayout(regions_group); self.roi_list_widget = QListWidget(); self.roi_list_widget.setContextMenuPolicy(Qt.CustomContextMenu); regions_layout.addWidget(self.roi_list_widget); main_layout.addWidget(regions_group)
        settings_group = QGroupBox("Settings"); settings_layout = QGridLayout(settings_group)
        settings_layout.addWidget(QLabel("Resolution Scale:"), 0, 0); self.resolution_slider = QSlider(Qt.Horizontal); self.resolution_slider.setRange(10, 100); self.resolution_slider.setValue(RESOLUTION_SCALE_DEFAULT); self.resolution_value_label = QLabel(f"{RESOLUTION_SCALE_DEFAULT/100.0:.2f}"); resolution_hbox = QHBoxLayout(); resolution_hbox.addWidget(self.resolution_slider); resolution_hbox.addWidget(self.resolution_value_label); settings_layout.addLayout(resolution_hbox, 0, 1, 1, 2)
        settings_layout.addWidget(QLabel("Capture Interval (s):"), 1, 0); self.interval_spinbox = QDoubleSpinBox(); self.interval_spinbox.setRange(0.1, 10.0); self.interval_spinbox.setValue(CAPTURE_INTERVAL_DEFAULT); self.interval_spinbox.setSingleStep(0.1); settings_layout.addWidget(self.interval_spinbox, 1, 1, 1, 2)
        settings_layout.addWidget(QLabel("Detection Mode:"), 2, 0); self.jams_only_radio = QRadioButton("Jams & E-Stops Only"); self.all_colors_radio = QRadioButton("All Colors"); self.jams_only_radio.setChecked(not self.detection_mode_all_colors); self.all_colors_radio.setChecked(self.detection_mode_all_colors); detection_mode_hbox = QHBoxLayout(); detection_mode_hbox.addWidget(self.jams_only_radio); detection_mode_hbox.addWidget(self.all_colors_radio); settings_layout.addLayout(detection_mode_hbox, 2, 1, 1, 2)
        settings_layout.addWidget(QLabel("System Time:"), 3, 0); self.clock_label = QLabel("--:--:--"); self.clock_label.setFont(QFont("Arial", 10, QFont.Bold)); settings_layout.addWidget(self.clock_label, 3, 1)
        schedule_group = QGroupBox("Auto Monitoring"); schedule_layout = QGridLayout(schedule_group); self.schedule_enabled_check = QCheckBox("Enable Auto Start/Stop"); schedule_layout.addWidget(self.schedule_enabled_check, 0, 0, 1, 2); schedule_layout.addWidget(QLabel("Stop Between:"), 1, 0); self.stop_start_hour_spin = QSpinBox(); self.stop_start_hour_spin.setRange(0,23); self.stop_start_hour_spin.setValue(17); schedule_layout.addWidget(self.stop_start_hour_spin, 1, 1); schedule_layout.addWidget(QLabel("and:"), 2, 0); self.stop_end_hour_spin = QSpinBox(); self.stop_end_hour_spin.setRange(0,23); self.stop_end_hour_spin.setValue(9); schedule_layout.addWidget(self.stop_end_hour_spin, 2, 1); settings_layout.addWidget(schedule_group, 4,0,1,3)
        main_layout.addWidget(settings_group); main_layout.addStretch(1)
        monitoring_buttons_layout = QHBoxLayout(); self.start_monitor_btn = QPushButton(get_standard_icon(QStyle.SP_MediaPlay), " Start Monitoring"); self.stop_monitor_btn = QPushButton(get_standard_icon(QStyle.SP_MediaStop), " Stop Monitoring"); self.reporting_btn = QPushButton(get_standard_icon(QStyle.SP_DialogHelpButton), " Reporting Dashboard"); monitoring_buttons_layout.addWidget(self.start_monitor_btn); monitoring_buttons_layout.addWidget(self.stop_monitor_btn); monitoring_buttons_layout.addWidget(self.reporting_btn)
        main_layout.addLayout(monitoring_buttons_layout)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Ready.")
        self.update_control_states()
    def connect_signals(self):
        self.add_roi_btn.clicked.connect(self.add_new_roi); self.delete_roi_btn.clicked.connect(self.delete_selected_roi); self.clear_rois_btn.clicked.connect(self.clear_all_rois_confirmed); self.save_rois_btn.clicked.connect(self.save_rois_to_file); self.load_rois_btn.clicked.connect(self.load_rois_from_file); self.roi_list_widget.customContextMenuRequested.connect(self.show_roi_context_menu); self.roi_list_widget.itemDoubleClicked.connect(self.rename_selected_roi); self.resolution_slider.valueChanged.connect(self.update_resolution_label); self.start_monitor_btn.clicked.connect(self.start_monitoring_session); self.stop_monitor_btn.clicked.connect(self.stop_monitoring_session); self.reporting_btn.clicked.connect(self.show_reporting_dashboard)
        
        self.session_id_received.connect(self._on_session_id_received)
        self.anomaly_detector.new_region_status.connect(self.alert_window.update_region_status)
        self.anomaly_detector.metrics_update_for_logging.connect(self.conveyor_metrics.process_metrics)
        self.anomaly_detector.anomaly_ended.connect(self.route_completed_event_to_logger)
        self.conveyor_metrics.health_updated.connect(self.alert_window.update_overall_health)
        self.conveyor_metrics.health_updated.connect(lambda h, a: setattr(self, 'last_health_snapshot', (h, a)))

    @pyqtSlot(dict)
    def route_completed_event_to_logger(self, data: dict):
        """Inspects event type and routes it to the correct logger queue."""
        if not self.current_session_id: return
        data['session_id'] = self.current_session_id
        
        primary_color_name = data.get('primary_event_type_name')
        if primary_color_name in ['red', 'orange']:
            self.data_logger.add_log_entry('LOG_CRITICAL_EVENT', data)
        elif primary_color_name in ['blue', 'purple', 'grey']:
            self.data_logger.add_log_entry('LOG_INEFFICIENCY_EVENT', data)
    
    def start_monitoring_session(self):
        if not self.rois: return
        self.start_monitor_btn.setEnabled(False); self.status_bar.showMessage("Initializing session...")
        self.detection_mode_all_colors = self.all_colors_radio.isChecked()
        settings = {"resolution":self.resolution_slider.value(), "interval":self.interval_spinbox.value(), "mode":"all_colors" if self.detection_mode_all_colors else "jams_only"}
        log_data = {'start_time': datetime.now(), 'roi_count': len(self.rois), 'settings_json': json.dumps(settings), 'callback': self.session_id_received.emit }
        self.data_logger.add_log_entry('START_SESSION', log_data)
    @pyqtSlot(int)
    def _on_session_id_received(self, new_id: int):
        self.current_session_id = new_id; logger.info(f"New monitoring session started with ID: {self.current_session_id}")
        settings = {"resolution":self.resolution_slider.value(), "interval":self.interval_spinbox.value(), "mode":"all_colors" if self.all_colors_radio.isChecked() else "jams_only"}
        self.anomaly_detector.update_config(self.rois, settings["resolution"], settings["interval"], self.all_colors_radio.isChecked())
        self.anomaly_detector.start(); self.alert_window.show(); self.monitoring_active_flag = True; self.health_snapshot_timer.start(int(HEALTH_SNAPSHOT_INTERVAL * 1000)); self.update_control_states()
        self.status_bar.showMessage(f"Monitoring active. Session ID: {self.current_session_id}")
    def stop_monitoring_session(self):
        if not self.monitoring_active_flag: return
        self.anomaly_detector.stop(); self.monitoring_active_flag = False; self.health_snapshot_timer.stop()
        if self.current_session_id:
            self.data_logger.add_log_entry('END_SESSION', {'end_time': datetime.now(), 'session_id': self.current_session_id})
            logger.info(f"Monitoring session {self.current_session_id} ended.")
        self.current_session_id = None; self.update_control_states(); self.status_bar.showMessage("Monitoring stopped.")
    def log_health_snapshot(self):
        if self.monitoring_active_flag and self.current_session_id and hasattr(self, 'last_health_snapshot'):
            health, active_anomalies = self.last_health_snapshot
            self.data_logger.add_log_entry('LOG_SNAPSHOT', {'session_id': self.current_session_id, 'timestamp': datetime.now(), 'health': health, 'active_anomalies': active_anomalies})
    def run_scheduled_reports(self):
        now = datetime.now()
        if now.hour in [6, 18] and now.minute == 0: self.data_logger.generate_12_hour_report_archive()
    def show_reporting_dashboard(self):
        if not self.reporting_dashboard or not self.reporting_dashboard.isVisible():
            self.reporting_dashboard = ReportingDashboard(self.data_logger, self); self.reporting_dashboard.show()
        else: self.reporting_dashboard.activateWindow()
    def closeEvent(self, event):
        self.stop_monitoring_session(); self.save_settings(); self.data_logger.stop(); self.data_logger.wait()
        if self.alert_window: self.alert_window.close()
        if self.reporting_dashboard: self.reporting_dashboard.close()
        super().closeEvent(event)
    def update_control_states(self):
        has_rois = bool(self.rois); is_monitoring = self.monitoring_active_flag; self.start_monitor_btn.setEnabled(has_rois and not is_monitoring); self.add_roi_btn.setEnabled(not is_monitoring); self.delete_roi_btn.setEnabled(has_rois and not is_monitoring and bool(self.roi_list_widget.currentItem())); self.clear_rois_btn.setEnabled(has_rois and not is_monitoring); self.save_rois_btn.setEnabled(has_rois and not is_monitoring); self.load_rois_btn.setEnabled(not is_monitoring); self.resolution_slider.setEnabled(not is_monitoring); self.interval_spinbox.setEnabled(not is_monitoring); self.jams_only_radio.setEnabled(not is_monitoring); self.all_colors_radio.setEnabled(not is_monitoring); self.stop_monitor_btn.setEnabled(is_monitoring); self.reporting_btn.setEnabled(True)
    def update_resource_usage(self): self.cpu_label.setText(f"CPU: {psutil.cpu_percent():.1f}%"); self.memory_label.setText(f"Mem: {psutil.virtual_memory().percent:.1f}%")
    def update_clock(self): now = datetime.now(); self.clock_label.setText(now.strftime("%H:%M:%S")); self.check_monitoring_schedule(now)
    def check_monitoring_schedule(self, now):
        if not self.schedule_enabled_check.isChecked(): return
        hour, start, end = now.hour, self.stop_start_hour_spin.value(), self.stop_end_hour_spin.value()
        in_stop_period = (hour >= start or hour < end) if start > end else (start <= hour < end)
        if in_stop_period and self.monitoring_active_flag: self.stop_monitoring_session(); logger.info("Auto-stopped monitoring.")
        elif not in_stop_period and not self.monitoring_active_flag and self.rois: self.start_monitoring_session(); logger.info("Auto-started monitoring.")
    def add_new_roi(self):
        monitors = get_monitors(); selector = MonitorSelector(monitors, self)
        if selector.exec_() == QDialog.Accepted:
            monitor = selector.get_selected_monitor(); title, ok = QInputDialog.getText(self, "ROI Name", "Enter ROI name:")
            if ok and title and title not in self.rois:
                monitor_details = {"top": monitor.y, "left": monitor.x, "width": monitor.width, "height": monitor.height, "mon": monitors.index(monitor) + 1}
                self.hide(); time.sleep(0.2)
                try:
                    overlay = ROISelectionOverlay(monitor_details);
                    if overlay.exec_() == QDialog.Accepted:
                        rect = overlay.get_selected_rect()
                        if rect and rect.width() > 5 and rect.height() > 5:
                            roi_tuple = (rect.x(), rect.y(), rect.width(), rect.height()); self.rois[title] = ROIInfo(roi=roi_tuple, monitor_info=monitor_details, title=title); self.roi_list_widget.addItem(f"{title}: ({rect.width()}x{rect.height()})")
                finally: self.show(); self.activateWindow()
        self.update_control_states()
    def delete_selected_roi(self):
        item = self.roi_list_widget.currentItem();
        if item and QMessageBox.question(self, 'Confirm Delete', f'Delete "{item.text().split(":")[0]}"?', QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            title = item.text().split(":")[0]; del self.rois[title]; self.roi_list_widget.takeItem(self.roi_list_widget.row(item)); self.alert_window.remove_region(title)
        self.update_control_states()
    def clear_all_rois_confirmed(self):
        if self.rois and QMessageBox.question(self, 'Confirm Clear', 'Delete all ROIs?', QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.rois.clear(); self.roi_list_widget.clear(); self.alert_window.clear_all_regions()
        self.update_control_states()
    def rename_selected_roi(self, item):
        old_title = item.text().split(":")[0]; new_title, ok = QInputDialog.getText(self, 'Rename ROI', 'New name:', text=old_title)
        if ok and new_title and new_title != old_title and new_title not in self.rois:
            self.rois[new_title] = self.rois.pop(old_title); self.rois[new_title].title = new_title; item.setText(f"{new_title}: ({self.rois[new_title].roi[2]}x{self.rois[new_title].roi[3]})")
    def show_roi_context_menu(self, pos):
        item = self.roi_list_widget.itemAt(pos)
        if not item: return
        menu = QMenu(); rename = menu.addAction("Rename"); delete = menu.addAction("Delete")
        action = menu.exec_(self.roi_list_widget.mapToGlobal(pos))
        if action == rename: self.rename_selected_roi(item)
        elif action == delete: self.delete_selected_roi()
    def update_resolution_label(self, val): self.resolution_value_label.setText(f"{val/100.0:.2f}")
    def save_rois_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", str(BASE_DIR/"roi_config.json"), "JSON (*.json)")
        if path:
            try:
                with open(path, 'w') as f: json.dump({t: r.__dict__ for t, r in self.rois.items()}, f, indent=4)
            except Exception as e: QMessageBox.critical(self, "Save Error", str(e))
    def load_rois_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load ROIs", str(BASE_DIR), "JSON (*.json)")
        if path:
            try:
                with open(path, 'r') as f: loaded = json.load(f)
                self.rois.clear(); self.roi_list_widget.clear(); self.alert_window.clear_all_regions()
                for title, data in loaded.items(): self.rois[title] = ROIInfo(**data); self.roi_list_widget.addItem(f"{title}: ({data['roi'][2]}x{data['roi'][3]})")
            except Exception as e: QMessageBox.critical(self, "Load Error", str(e))
        self.update_control_states()
    def save_settings(self):
        settings = {"rois": {t: r.__dict__ for t, r in self.rois.items()}, "resolution_scale": self.resolution_slider.value(), "capture_interval": self.interval_spinbox.value(), "detect_all_colors": self.all_colors_radio.isChecked(), "auto_schedule_enabled": self.schedule_enabled_check.isChecked(), "auto_schedule_stop_from": self.stop_start_hour_spin.value(), "auto_schedule_stop_to": self.stop_end_hour_spin.value()}
        try:
            with open(BASE_DIR / "app_settings.json", 'w') as f: json.dump(settings, f, indent=4)
        except Exception as e: logger.error(f"Save settings error: {e}")
    def load_settings(self):
        path = BASE_DIR / "app_settings.json"
        if not path.exists(): return
        try:
            with open(path, 'r') as f: settings = json.load(f)
            self.resolution_slider.setValue(settings.get("resolution_scale", 50)); self.interval_spinbox.setValue(settings.get("capture_interval", 1.0)); 
            self.detection_mode_all_colors = settings.get("detect_all_colors", False)
            self.all_colors_radio.setChecked(self.detection_mode_all_colors); self.jams_only_radio.setChecked(not self.detection_mode_all_colors); 
            self.schedule_enabled_check.setChecked(settings.get("auto_schedule_enabled", False)); self.stop_start_hour_spin.setValue(settings.get("auto_schedule_stop_from", 17)); self.stop_end_hour_spin.setValue(settings.get("auto_schedule_stop_to", 9))
            self.rois.clear(); self.roi_list_widget.clear()
            for title, data in settings.get("rois", {}).items():
                self.rois[title] = ROIInfo(**data)
                self.roi_list_widget.addItem(f"{title}: ({data['roi'][2]}x{data['roi'][3]})")
        except Exception as e: logger.error(f"Load settings error: {e}")
        self.update_control_states()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(BASE_DIR / "app_icon.png")) if (BASE_DIR / "app_icon.png").exists() else get_standard_icon(QStyle.SP_ComputerIcon))
    app.setStyle("Fusion")
    main_window = MainApplicationWindow()
    main_window.show()
    sys.exit(app.exec_())

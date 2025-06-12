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
import pandas as pd # For daily report generation
from pathlib import Path
import logging
import webbrowser

from PyQt5.QtCore import (
    QObject, pyqtSignal, Qt, QTimer, QThread, QPoint, QSize, QRect, pyqtSlot
)
from PyQt5.QtWidgets import (
    QWidget, QSpinBox, QCheckBox, QProgressBar, QGridLayout, QRadioButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QScrollArea, QApplication,
    QPushButton, QGroupBox, QStyle, QMenu, QStatusBar, QMessageBox, QInputDialog,
    QDialog, QListWidget, QSlider, QDoubleSpinBox, QMainWindow, QSizePolicy,
    QListWidgetItem
)
from PyQt5.QtGui import (
    QColor, QPalette, QFont, QPixmap, QImage, QPainter, QPen, QBrush, QIcon
)
import pyqtgraph as pg # type: ignore
from screeninfo import get_monitors

from flask import Flask, jsonify, render_template # type: ignore
from flask_cors import CORS # type: ignore

# --- Configuration & Constants ---
# General Settings
APP_NAME = "VICE (Visual Identification Conveyance Enhancement) V3.0"
DB_NAME = "conveyor_metrics_v3.db"
LOG_LEVEL = logging.INFO
CAPTURE_INTERVAL_DEFAULT = 1.0  # seconds (adjust for responsiveness vs. CPU load)
RESOLUTION_SCALE_DEFAULT = 50   # percent
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5965
FLASK_DEBUG = False

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


# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DB_NAME
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

sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter('datetime', lambda s: datetime.fromisoformat(s.decode()))


# --- Utility Functions ---
def get_standard_icon(style_enum):
    return QApplication.style().standardIcon(style_enum)

# --- Database Adapters (for datetime) ---
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter('datetime', lambda s: datetime.fromisoformat(s.decode()))


# --- Monitor Selection Dialog ---
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

# --- (Located in the Core Application Classes section) ---
class AlertWindow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.region_row_widgets: Dict[str, QWidget] = {} 
        self.all_region_statuses: Dict[str, RegionStatus] = {} # Caches the latest status object for each region
        self.conveyor_health = 100.0
        self.session_start_time = datetime.now()
        self.active_rois_count = 0
        self.initUI()
        self.position_window()

    def initUI(self):
        self.setWindowTitle("Region Status Monitor"); self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setMinimumSize(800, 800) # Increased width for new columns

        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(10)
        
        # --- Top Section (Unchanged) ---
        title_label = QLabel("Region Monitoring Status"); title_label.setFont(QFont("Arial", 16, QFont.Bold)); title_label.setAlignment(Qt.AlignCenter); main_layout.addWidget(title_label)
        health_group = QGroupBox("Overall Conveyor Health"); health_layout = QHBoxLayout(); self.health_label = QLabel(f"Health: {self.conveyor_health:.1f}%"); self.health_label.setFont(QFont("Arial", 12, QFont.Bold)); self.health_progress_bar = QProgressBar(); self.health_progress_bar.setRange(0, 100); self.health_progress_bar.setValue(int(self.conveyor_health)); self.health_progress_bar.setTextVisible(False); health_layout.addWidget(self.health_label); health_layout.addWidget(self.health_progress_bar, 1); health_group.setLayout(health_layout); main_layout.addWidget(health_group)
        session_info_layout = QHBoxLayout(); self.uptime_label = QLabel("Uptime: 0s"); self.active_rois_label = QLabel("ROIs: 0"); session_info_layout.addWidget(self.uptime_label); session_info_layout.addStretch(); session_info_layout.addWidget(self.active_rois_label); main_layout.addLayout(session_info_layout)

        # --- Header Row for the new columns ---
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget); header_layout.setContentsMargins(6, 3, 6, 3)
        header_font = QFont("Arial", 9, QFont.Bold); header_font.setUnderline(True)
        col_labels = ["Region", "Status", "Duration", "Crit. Errors", "Purple", "Blue", "Grey"]
        col_stretches = [4, 5, 2, 1, 1, 1, 1] # Stretch factors for columns
        for text, stretch in zip(col_labels, col_stretches):
            lbl = QLabel(text); lbl.setFont(header_font); header_layout.addWidget(lbl, stretch)
        main_layout.addWidget(header_widget)

        # --- Scroll Area for sorted rows ---
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.alert_container_widget = QWidget()
        self.alert_lines_layout = QVBoxLayout(self.alert_container_widget); self.alert_lines_layout.setContentsMargins(0, 0, 0, 0); self.alert_lines_layout.setSpacing(1); self.alert_lines_layout.addStretch()
        scroll_area.setWidget(self.alert_container_widget)
        main_layout.addWidget(scroll_area, 1)

        self.update_timer = QTimer(self); self.update_timer.timeout.connect(self._update_session_info); self.update_timer.start(1000)
        # NEW: Timer for flashing effect
        self.flash_timer = QTimer(self); self.flash_timer.timeout.connect(self._handle_flashing); self.flash_timer.start(750) # Toggles every 750ms
        self._flash_state = False

    def _update_session_info(self): uptime_delta = datetime.now() - self.session_start_time; self.uptime_label.setText(f"Uptime: {str(uptime_delta).split('.')[0]}"); self.active_rois_label.setText(f"ROIs: {self.active_rois_count}")
    
    def _handle_flashing(self):
        self._flash_state = not self._flash_state # Toggle state
        for title, status_obj in self.all_region_statuses.items():
            if title in self.region_row_widgets:
                row_widget = self.region_row_widgets[title]
                is_over_time_limit = False
                if status_obj.is_anomaly and status_obj.anomaly_start_time:
                    duration = (datetime.now() - status_obj.anomaly_start_time).total_seconds()
                    if duration > 150: # 2.5 minutes
                        is_over_time_limit = True
                
                # Apply or remove flashing stylesheet
                current_stylesheet = row_widget.styleSheet()
                flash_style = "background-color: #581c1c;" # Dark red
                
                if is_over_time_limit:
                    if self._flash_state:
                        if flash_style not in current_stylesheet: row_widget.setStyleSheet(current_stylesheet + flash_style)
                    else:
                        if flash_style in current_stylesheet: row_widget.setStyleSheet(current_stylesheet.replace(flash_style, ""))
                else: # Ensure flashing stops if duration is no longer over limit (or anomaly clears)
                    if flash_style in current_stylesheet: row_widget.setStyleSheet(current_stylesheet.replace(flash_style, ""))

    def position_window(self):
        try: screen_geometry = QApplication.primaryScreen().geometry(); self.move(screen_geometry.width() - self.width() - 20, 30)
        except Exception as e: logger.warning(f"Could not auto-position alert window: {e}"); self.move(300, 300)

    @pyqtSlot(float)
    def update_overall_health(self, health_percentage: float):
        self.conveyor_health = health_percentage; self.health_label.setText(f"Health: {self.conveyor_health:.1f}%"); self.health_progress_bar.setValue(int(self.conveyor_health))
        stylesheet = "QProgressBar::chunk { background-color: %s; }"; color = "green"
        if health_percentage < 40: color = "red"
        elif health_percentage < 70: color = "orange"
        self.health_progress_bar.setStyleSheet(stylesheet % color)

    @pyqtSlot(RegionStatus)
    def update_region_status(self, status: RegionStatus):
        self.all_region_statuses[status.title] = status
        self.active_rois_count = len(self.all_region_statuses)
        self._redraw_all_rows()

    def _redraw_all_rows(self):
        # Clear existing layout
        while self.alert_lines_layout.count() > 1: # Keep the stretch item
            item = self.alert_lines_layout.takeAt(0)
            if item and item.widget(): item.widget().deleteLater()
        
        # Sort statuses by priority
        sorted_statuses = sorted(self.all_region_statuses.values(), key=lambda s: COLOR_DEFINITIONS[s.get_primary_color_name()]["priority"])

        # Re-populate layout in sorted order
        for status in sorted_statuses:
            self._create_or_update_row(status)

    def _create_or_update_row(self, status: RegionStatus):
        # This method creates a single row widget and inserts it into the layout
        # Since we redraw all, this logic can be simplified to just "create"
        title = status.title
        
        line_widget = QWidget(); line_widget.setAutoFillBackground(True) # Needed for background color to work
        line_layout = QHBoxLayout(line_widget); line_layout.setContentsMargins(6, 2, 6, 2); line_layout.setSpacing(10)
        
        font_small = QFont("Arial", 9); font_bold = QFont("Arial", 10, QFont.Bold)
        
        # Define columns with their respective data, stretch factor, and font
        columns_data = [
            (title, 4, font_bold),
            (status.get_primary_color_message(), 5, font_small),
            (f"Dur: {str(datetime.now() - status.anomaly_start_time).split('.')[0]}" if status.anomaly_start_time else "Dur: --", 2, font_small),
            (str(status.critical_error_count), 1, font_small),
            (str(status.purple_count), 1, font_small),
            (str(status.blue_count), 1, font_small),
            (str(status.grey_count), 1, font_small)
        ]

        for text, stretch, font in columns_data:
            lbl = QLabel(text); lbl.setFont(font); lbl.setAlignment(Qt.AlignCenter)
            if "Dur:" in text: lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if text == title: lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            line_layout.addWidget(lbl, stretch)
        
        # Color the status message text
        bgr_color = status.get_primary_bgr(); text_color = QColor(bgr_color[2], bgr_color[1], bgr_color[0])
        status_label_widget = line_layout.itemAt(1).widget()
        status_label_widget.setStyleSheet(f"color: {text_color.name()};" + ("font-weight: bold;" if status.is_anomaly else ""))
            
        self.alert_lines_layout.insertWidget(self.alert_lines_layout.count()-1, line_widget) # Insert before the stretch
        self.region_row_widgets[title] = line_widget # Track the row widget for flashing

    def remove_region(self, title: str):
        if title in self.all_region_statuses: del self.all_region_statuses[title]
        self._redraw_all_rows()

    def clear_all_regions(self):
        self.all_region_statuses.clear(); self._redraw_all_rows()
        
    def closeEvent(self, event): self.update_timer.stop(); self.flash_timer.stop(); super().closeEvent(event)



# --- (Located in the Core Application Classes section) ---
class AnomalyDetector(QObject):
    new_region_status = pyqtSignal(RegionStatus)
    metrics_update_for_logging = pyqtSignal(dict)
    anomaly_ended = pyqtSignal(str, str, datetime, datetime)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent); self.rois = {}; self.running = False; self.sct = None; self.monitor_thread = None
        self.resolution_scale_factor = RESOLUTION_SCALE_DEFAULT / 100.0; self.capture_interval_sec = CAPTURE_INTERVAL_DEFAULT
        self.detection_mode_all_colors = False; self._current_roi_statuses = {}

    def update_config(self, rois, res, interval, all_colors):
        self.rois = rois; self.resolution_scale_factor = max(0.1, min(1.0, res / 100.0))
        self.capture_interval_sec = max(0.05, interval); self.detection_mode_all_colors = all_colors
        new_statuses = {title: self._current_roi_statuses.get(title, RegionStatus(title=title)) for title in rois}
        self._current_roi_statuses = new_statuses

    def _capture_single_roi(self, roi_info):
        if not self.sct: return None
        try:
            mon = {"top": roi_info.monitor_info["top"] + roi_info.roi[1], "left": roi_info.monitor_info["left"] + roi_info.roi[0], "width": roi_info.roi[2], "height": roi_info.roi[3]}
            if mon["width"] <= 0 or mon["height"] <= 0: return None
            img = np.array(self.sct.grab(mon)); img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if self.resolution_scale_factor < 1.0:
                return cv2.resize(img_bgr, (0, 0), fx=self.resolution_scale_factor, fy=self.resolution_scale_factor, interpolation=cv2.INTER_AREA)
            return img_bgr
        except Exception as e: logger.error(f"Capture error for '{roi_info.title}': {e}"); return None

    def _detect_colors_in_single_frame(self, image):
        if image is None: return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); detected = []
        for name, props in COLOR_DEFINITIONS.items():
            if name == "green": continue
            if not self.detection_mode_all_colors and name not in ["orange", "red", "grey"]: continue
            mask = None
            if isinstance(props.get("lower"), list):
                mask = cv2.bitwise_or(cv2.inRange(hsv, props["lower"][0], props["upper"][0]), cv2.inRange(hsv, props["lower"][1], props["upper"][1]))
            else: mask = cv2.inRange(hsv, props["lower"], props["upper"])
            if np.sum(mask > 0) / mask.size * 100 >= props["threshold_percent"]:
                detected.append(DetectedColorInfo(name=name, bgr=props["bgr"], message=props["message"]))
        detected.sort(key=lambda dci: list(COLOR_DEFINITIONS.keys()).index(dci.name))
        return detected

    def _monitoring_loop(self):
        try:
            self.sct = mss(); logger.info("Monitoring loop started.")
            while self.running:
                scan_time = datetime.now(); loop_start = time.perf_counter(); statuses_for_metrics = {}
                for title, roi_info in list(self.rois.items()):
                    if not self.running: break
                    status_obj = self._current_roi_statuses[title]; was_anomaly = status_obj.is_anomaly
                    prev_primary_color = status_obj.get_primary_color_name(); prev_start_time = status_obj.anomaly_start_time
                    img = self._capture_single_roi(roi_info); frame_colors = self._detect_colors_in_single_frame(img); del img
                    persistent_colors = []
                    for name, props in COLOR_DEFINITIONS.items():
                        if name == "green": continue
                        color_in_frame = next((dci for dci in frame_colors if dci.name == name), None)
                        if color_in_frame:
                            status_obj._blink_color_last_physical_detection_time[name] = scan_time
                            persistent_colors.append(color_in_frame)
                        elif props.get("is_blinking", False):
                            last_seen = status_obj._blink_color_last_physical_detection_time.get(name)
                            if last_seen and (scan_time - last_seen).total_seconds() < BLINK_GRACE_PERIOD_SECONDS:
                                persistent_colors.append(DetectedColorInfo(name=name, bgr=props["bgr"], message=props["message"]))
                            else: status_obj._blink_color_last_physical_detection_time[name] = None
                    persistent_colors.sort(key=lambda dci: list(COLOR_DEFINITIONS.keys()).index(dci.name))
                    status_obj.detected_colors = persistent_colors; is_now_anomaly = bool(persistent_colors)
                    if is_now_anomaly and not was_anomaly:
                        status_obj.is_anomaly = True; status_obj.anomaly_start_time = scan_time
                        # NEW: Increment correct counter on transition to anomaly
                        primary_color = status_obj.get_primary_color_name()
                        if primary_color in ["red", "orange"]: status_obj.critical_error_count += 1
                        elif primary_color == "purple": status_obj.purple_count += 1
                        elif primary_color == "blue": status_obj.blue_count += 1
                        elif primary_color == "grey": status_obj.grey_count += 1
                    elif not is_now_anomaly and was_anomaly:
                        status_obj.is_anomaly = False; status_obj.anomaly_start_time = None
                        if prev_primary_color and prev_start_time: self.anomaly_ended.emit(title, prev_primary_color, prev_start_time, scan_time)
                    else: status_obj.is_anomaly = is_now_anomaly
                    self.new_region_status.emit(status_obj)
                    statuses_for_metrics[title] = {"title": status_obj.title, "is_anomaly": status_obj.is_anomaly, "detected_colors": [vars(dci) for dci in status_obj.detected_colors], "anomaly_start_time": status_obj.anomaly_start_time.isoformat() if status_obj.anomaly_start_time else None, "jam_count": status_obj.critical_error_count} # Note: jam_count is now critical_error_count
                self.metrics_update_for_logging.emit({"timestamp": scan_time, "roi_statuses": statuses_for_metrics, "num_rois": len(self.rois)})
                time.sleep(max(0, self.capture_interval_sec - (time.perf_counter() - loop_start)))
        except Exception as e: logger.exception("CRITICAL ERROR in monitoring loop:")
        finally:
            if self.sct: self.sct.close()
            logger.info("Monitoring loop stopped.")

    def start(self):
        if self.running: return
        for status in self._current_roi_statuses.values(): status._blink_color_last_physical_detection_time.clear()
        self.running = True; self.monitor_thread = threading.Thread(target=self._monitoring_loop, name="AnomalyDetectorThread", daemon=True); self.monitor_thread.start()

    def stop(self):
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive(): self.monitor_thread.join(timeout=(self.capture_interval_sec * 2) + BLINK_GRACE_PERIOD_SECONDS + 0.5)
        self.monitor_thread = None
        for status_obj in self._current_roi_statuses.values():
            if status_obj.is_anomaly:
                status_obj.is_anomaly = False; status_obj.detected_colors = []; status_obj.anomaly_start_time = None
                self.new_region_status.emit(status_obj)

# --- Conveyor Metrics Calculation ---

class ConveyorMetrics(QObject):
    overall_health_updated = pyqtSignal(float)
    metrics_for_api_and_dashboard = pyqtSignal(dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.start_time = datetime.now()
        self.num_rois = 0
        self.current_conveyor_health = 100.0
        # This will store cumulative counts of primary anomaly colors for the session
        self.session_primary_color_anomaly_counts: Dict[str, int] = defaultdict(int)

    @pyqtSlot(dict)
    def process_metrics_from_detector(self, payload: dict):
        timestamp_dt = payload["timestamp"] 
        roi_statuses_data: Dict[str, Dict] = payload["roi_statuses"] 
        self.num_rois = payload["num_rois"]

        if not self.num_rois:
            self.current_conveyor_health = 100.0
            self.overall_health_updated.emit(self.current_conveyor_health)
            # self.session_primary_color_anomaly_counts.clear() # Optionally reset if no ROIs
            return

        active_jam_titles = []
        
        # Update session_primary_color_anomaly_counts based on current debounced states
        # This needs to be based on *transitions* into an anomaly state with a primary color,
        # or count how many ROIs are *currently* in an anomaly state with a given primary color.
        # For now, let's count ROIs currently showing a primary color anomaly.
        current_cycle_primary_color_counts: Dict[str, int] = defaultdict(int)

        for title, status_dict in roi_statuses_data.items():
            if status_dict["is_anomaly"]: # is_anomaly is the debounced state
                active_jam_titles.append(title)
                if status_dict["detected_colors"]: # This is the debounced list of color dicts
                    primary_color_name = status_dict["detected_colors"][0]["name"]
                    current_cycle_primary_color_counts[primary_color_name] += 1
        
        # If session_primary_color_anomaly_counts is meant to be a sum over time of how many
        # *frames* a color was primary, then this update is different.
        # Based on original intent, it seemed like a cumulative count of detection events.
        # Let's stick to incrementing for now, assuming each processing cycle is an "event".
        for color, count in current_cycle_primary_color_counts.items():
            self.session_primary_color_anomaly_counts[color] += count


        self._calculate_conveyor_health(roi_statuses_data)
        self.overall_health_updated.emit(self.current_conveyor_health)

        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        api_metrics = {
            "timestamp": timestamp_dt.isoformat(),
            "conveyor_health": self.current_conveyor_health,
            "active_rois": self.num_rois,
            "active_jams_count": len(active_jam_titles),
            "active_jam_rois": active_jam_titles,
            "uptime_seconds": uptime_seconds,
            "color_event_counts": dict(self.session_primary_color_anomaly_counts), 
        }
        self.metrics_for_api_and_dashboard.emit(api_metrics)

    def _calculate_conveyor_health(self, roi_statuses_data: Dict[str, Dict]):
        if not self.num_rois:
            self.current_conveyor_health = 100.0
            return

        total_weighted_impact_sum = 0.0
        
        for title, status_dict in roi_statuses_data.items():
            # status_dict["is_anomaly"] is the debounced state
            if status_dict["is_anomaly"] and status_dict["detected_colors"]:
                # Primary color from the debounced list determines the weight
                primary_color_info = status_dict["detected_colors"][0] 
                primary_color_name = primary_color_info["name"]
                
                if primary_color_name in COLOR_DEFINITIONS:
                    color_props = COLOR_DEFINITIONS[primary_color_name]
                    total_weighted_impact_sum += color_props.get("weight", 0.0)
        
        if self.num_rois > 0:
            average_impact_factor = total_weighted_impact_sum / self.num_rois
            self.current_conveyor_health = max(0.0, min(100.0, 100.0 - (average_impact_factor * 100.0)))
        else:
            self.current_conveyor_health = 100.0
        
        # logger.debug(f"Health: {self.current_conveyor_health:.1f}% (Total Impact Sum: {total_weighted_impact_sum:.2f}, Avg Factor: {average_impact_factor:.2f})")


class DataLogger(QObject):
    def __init__(self, db_path: Path, parent: Optional[QObject] = None):
        super().__init__(parent); self.db_path = db_path; self._init_db_lock = threading.Lock(); self._initialize_database()
    def _initialize_database(self):
        with self._init_db_lock:
            try:
                with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
                    cursor=conn.cursor()
                    cursor.execute("CREATE TABLE IF NOT EXISTS conveyor_log (id INTEGER PRIMARY KEY, timestamp DATETIME, conveyor_health REAL, active_rois INTEGER, active_jams_count INTEGER, active_jam_rois TEXT, uptime_seconds REAL, color_event_counts TEXT)")
                    cursor.execute("CREATE TABLE IF NOT EXISTS completed_events (id INTEGER PRIMARY KEY, roi_title TEXT, event_type TEXT, start_time DATETIME, end_time DATETIME, duration_seconds REAL)")
                    conn.commit()
            except sqlite3.Error as e: logger.exception(f"DB init error: {e}")
    @pyqtSlot(str, str, datetime, datetime)
    def log_completed_event(self, title, color, start, end):
        if color not in ["red", "orange"]: return
        duration = (end - start).total_seconds(); event_type = COLOR_DEFINITIONS.get(color, {}).get("message", "UNKNOWN")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO completed_events (roi_title, event_type, start_time, end_time, duration_seconds) VALUES (?, ?, ?, ?, ?)", (title, event_type, start, end, duration))
        except Exception as e: logger.error(f"Log completed event error: {e}")
    @pyqtSlot(dict)
    def log_conveyor_metrics(self, data):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO conveyor_log (timestamp, conveyor_health, active_rois, active_jams_count, active_jam_rois, uptime_seconds, color_event_counts) VALUES (?, ?, ?, ?, ?, ?, ?)", (data["timestamp"], data["conveyor_health"], data["active_rois"], data["active_jams_count"], json.dumps(data["active_jam_rois"]), data["uptime_seconds"], json.dumps(data["color_event_counts"])))
        except Exception as e: logger.error(f"Log conveyor metrics error: {e}")
    def get_latest_metrics(self):
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
                row = conn.execute("SELECT timestamp, conveyor_health, active_rois, active_jams_count, active_jam_rois, uptime_seconds, color_event_counts FROM conveyor_log ORDER BY timestamp DESC LIMIT 1").fetchone()
                if row:
                    return {
                        "timestamp": row[0].isoformat(), # FIX: Convert datetime to string
                        "conveyor_health": row[1],
                        "active_rois": row[2],
                        "active_jams_count": row[3],
                        "active_jam_rois": json.loads(row[4] or '[]'),
                        "uptime_seconds": row[5],
                        "color_event_counts": json.loads(row[6] or '{}')
                    }
        except Exception as e: logger.error(f"DB get_latest_metrics error: {e}"); return None
    def get_completed_events(self, limit=50):
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
                rows = conn.execute("SELECT roi_title, event_type, end_time, duration_seconds FROM completed_events ORDER BY end_time DESC LIMIT ?", (limit,)).fetchall()
                return [{"roi_title": r[0], "event_type": r[1], "end_time": r[2].isoformat(), "duration_seconds": r[3]} for r in rows] # FIX: Convert datetime to string
        except Exception as e: logger.error(f"DB get_completed_events error: {e}"); return []
    def get_hourly_summary(self):
        try:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
                query = "SELECT strftime('%Y-%m-%d %H:00:00', end_time) as hour, event_type, COUNT(*) as count FROM completed_events WHERE end_time >= ? GROUP BY hour, event_type ORDER BY hour"
                df = pd.read_sql_query(query, conn, params=(datetime.now() - timedelta(hours=24),))
                if not df.empty: return df.pivot(index='hour', columns='event_type', values='count').fillna(0).reset_index().to_dict('records')
        except Exception as e: logger.error(f"DB get_hourly_summary error: {e}"); return []

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


class MainApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rois = {}
        self.monitoring_active_flag = False
        # FIX: Initialize attribute before initUI is called
        self.detection_mode_all_colors = False 
        
        self.alert_window = AlertWindow()
        self.anomaly_detector = AnomalyDetector()
        self.conveyor_metrics = ConveyorMetrics()
        self.data_logger = DataLogger(DB_PATH)
        self.metrics_dashboard_window = None
        self.flask_api_thread = None
        
        self.initUI()
        self.connect_signals()
        self.load_settings()
        
        self.resource_timer = QTimer(self)
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(2000)
        
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        
        self.start_flask_api()

    def initUI(self):
        self.setWindowTitle(APP_NAME); self.setGeometry(100, 100, 600, 750); central_widget = QWidget(); self.setCentralWidget(central_widget); main_layout = QVBoxLayout(central_widget)
        controls_group = QGroupBox("Controls"); controls_layout = QVBoxLayout(); resource_layout = QHBoxLayout(); self.cpu_label = QLabel("CPU: --%"); self.memory_label = QLabel("Mem: --%"); resource_layout.addWidget(self.cpu_label); resource_layout.addWidget(self.memory_label); controls_layout.addLayout(resource_layout); roi_buttons_layout = QGridLayout(); self.add_roi_btn = QPushButton(get_standard_icon(QStyle.SP_FileDialogNewFolder), " Add ROI"); self.delete_roi_btn = QPushButton(get_standard_icon(QStyle.SP_DialogDiscardButton), " Delete ROI"); self.clear_rois_btn = QPushButton(get_standard_icon(QStyle.SP_TrashIcon), " Clear All ROIs"); roi_buttons_layout.addWidget(self.add_roi_btn, 0, 0); roi_buttons_layout.addWidget(self.delete_roi_btn, 0, 1); roi_buttons_layout.addWidget(self.clear_rois_btn, 0, 2); controls_layout.addLayout(roi_buttons_layout); saveload_buttons_layout = QHBoxLayout(); self.save_rois_btn = QPushButton(get_standard_icon(QStyle.SP_DialogSaveButton), " Save ROIs"); self.load_rois_btn = QPushButton(get_standard_icon(QStyle.SP_DialogOpenButton), " Load ROIs"); saveload_buttons_layout.addWidget(self.save_rois_btn); saveload_buttons_layout.addWidget(self.load_rois_btn); controls_layout.addLayout(saveload_buttons_layout); controls_group.setLayout(controls_layout); main_layout.addWidget(controls_group)
        regions_group = QGroupBox("Monitored Regions"); regions_layout = QVBoxLayout(); self.roi_list_widget = QListWidget(); self.roi_list_widget.setContextMenuPolicy(Qt.CustomContextMenu); regions_layout.addWidget(self.roi_list_widget); regions_group.setLayout(regions_layout); main_layout.addWidget(regions_group)
        settings_group = QGroupBox("Settings"); settings_layout = QGridLayout(); settings_layout.addWidget(QLabel("Resolution Scale:"), 0, 0); self.resolution_slider = QSlider(Qt.Horizontal); self.resolution_slider.setRange(10, 100); self.resolution_slider.setValue(RESOLUTION_SCALE_DEFAULT); self.resolution_value_label = QLabel(f"{RESOLUTION_SCALE_DEFAULT/100.0:.2f}"); resolution_hbox = QHBoxLayout(); resolution_hbox.addWidget(self.resolution_slider); resolution_hbox.addWidget(self.resolution_value_label); settings_layout.addLayout(resolution_hbox, 0, 1, 1, 2); settings_layout.addWidget(QLabel("Capture Interval (s):"), 1, 0); self.interval_spinbox = QDoubleSpinBox(); self.interval_spinbox.setRange(0.1, 10.0); self.interval_spinbox.setValue(CAPTURE_INTERVAL_DEFAULT); self.interval_spinbox.setSingleStep(0.1); settings_layout.addWidget(self.interval_spinbox, 1, 1, 1, 2); settings_layout.addWidget(QLabel("Detection Mode:"), 2, 0); self.jams_only_radio = QRadioButton("Jams Only"); self.all_colors_radio = QRadioButton("All Colors"); self.jams_only_radio.setChecked(not self.detection_mode_all_colors); self.all_colors_radio.setChecked(self.detection_mode_all_colors); detection_mode_hbox = QHBoxLayout(); detection_mode_hbox.addWidget(self.jams_only_radio); detection_mode_hbox.addWidget(self.all_colors_radio); settings_layout.addLayout(detection_mode_hbox, 2, 1, 1, 2); settings_layout.addWidget(QLabel("System Time:"), 3, 0); self.clock_label = QLabel("--:--:--"); self.clock_label.setFont(QFont("Arial", 10, QFont.Bold)); settings_layout.addWidget(self.clock_label, 3, 1); schedule_group = QGroupBox("Auto Monitoring"); schedule_layout = QGridLayout(schedule_group); self.schedule_enabled_check = QCheckBox("Enable Auto Start/Stop"); schedule_layout.addWidget(self.schedule_enabled_check, 0, 0, 1, 2); schedule_layout.addWidget(QLabel("Stop Between:"), 1, 0); self.stop_start_hour_spin = QSpinBox(); self.stop_start_hour_spin.setRange(0,23); self.stop_start_hour_spin.setValue(17); schedule_layout.addWidget(self.stop_start_hour_spin, 1, 1); schedule_layout.addWidget(QLabel("and:"), 2, 0); self.stop_end_hour_spin = QSpinBox(); self.stop_end_hour_spin.setRange(0,23); self.stop_end_hour_spin.setValue(9); schedule_layout.addWidget(self.stop_end_hour_spin, 2, 1); settings_layout.addWidget(schedule_group, 4,0,1,3); settings_group.setLayout(settings_layout); main_layout.addWidget(settings_group); main_layout.addStretch(1)
        monitoring_buttons_layout = QHBoxLayout(); self.start_monitor_btn = QPushButton(get_standard_icon(QStyle.SP_MediaPlay), " Start"); self.stop_monitor_btn = QPushButton(get_standard_icon(QStyle.SP_MediaStop), " Stop"); self.metrics_dashboard_btn = QPushButton(get_standard_icon(QStyle.SP_DialogHelpButton), " Metrics"); self.web_dashboard_btn = QPushButton(get_standard_icon(QStyle.SP_ComputerIcon), " Web Report"); monitoring_buttons_layout.addWidget(self.start_monitor_btn); monitoring_buttons_layout.addWidget(self.stop_monitor_btn); monitoring_buttons_layout.addWidget(self.metrics_dashboard_btn); monitoring_buttons_layout.addWidget(self.web_dashboard_btn); main_layout.addLayout(monitoring_buttons_layout)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Ready."); self.update_control_states()
    
    def connect_signals(self):
        self.add_roi_btn.clicked.connect(self.add_new_roi); self.delete_roi_btn.clicked.connect(self.delete_selected_roi); self.clear_rois_btn.clicked.connect(self.clear_all_rois_confirmed); self.save_rois_btn.clicked.connect(self.save_rois_to_file); self.load_rois_btn.clicked.connect(self.load_rois_from_file); self.roi_list_widget.customContextMenuRequested.connect(self.show_roi_context_menu); self.roi_list_widget.itemDoubleClicked.connect(self.rename_selected_roi); self.resolution_slider.valueChanged.connect(self.update_resolution_label); self.start_monitor_btn.clicked.connect(self.start_monitoring_session); self.stop_monitor_btn.clicked.connect(self.stop_monitoring_session); self.metrics_dashboard_btn.clicked.connect(self.show_metrics_dashboard); self.web_dashboard_btn.clicked.connect(self.open_web_dashboard)
        self.anomaly_detector.anomaly_ended.connect(self.data_logger.log_completed_event); self.anomaly_detector.new_region_status.connect(self.alert_window.update_region_status); self.anomaly_detector.metrics_update_for_logging.connect(self.conveyor_metrics.process_metrics_from_detector)
        self.conveyor_metrics.overall_health_updated.connect(self.alert_window.update_overall_health)
        if self.metrics_dashboard_window:
            self.conveyor_metrics.metrics_for_api_and_dashboard.connect(self.metrics_dashboard_window.update_live_metrics)

    def update_control_states(self):
        has_rois = bool(self.rois); is_monitoring = self.monitoring_active_flag; self.add_roi_btn.setEnabled(not is_monitoring); self.delete_roi_btn.setEnabled(has_rois and not is_monitoring and bool(self.roi_list_widget.currentItem())); self.clear_rois_btn.setEnabled(has_rois and not is_monitoring); self.save_rois_btn.setEnabled(has_rois and not is_monitoring); self.load_rois_btn.setEnabled(not is_monitoring); self.resolution_slider.setEnabled(not is_monitoring); self.interval_spinbox.setEnabled(not is_monitoring); self.jams_only_radio.setEnabled(not is_monitoring); self.all_colors_radio.setEnabled(not is_monitoring); self.start_monitor_btn.setEnabled(has_rois and not is_monitoring); self.stop_monitor_btn.setEnabled(is_monitoring); self.metrics_dashboard_btn.setEnabled(True); self.web_dashboard_btn.setEnabled(self.flask_api_thread is not None and self.flask_api_thread.isRunning())
    
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
                            roi_tuple = (rect.x(), rect.y(), rect.width(), rect.height())
                            self.rois[title] = ROIInfo(roi=roi_tuple, monitor_info=monitor_details, title=title)
                            self.roi_list_widget.addItem(f"{title}: ({rect.width()}x{rect.height()})")
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
    
    def start_monitoring_session(self):
        if not self.rois: return
        self.detection_mode_all_colors = self.all_colors_radio.isChecked() # Update mode before passing to detector
        self.anomaly_detector.update_config(self.rois, self.resolution_slider.value(), self.interval_spinbox.value(), self.detection_mode_all_colors)
        self.anomaly_detector.start(); self.alert_window.show(); self.monitoring_active_flag = True; self.update_control_states()
    
    def stop_monitoring_session(self):
        if not self.monitoring_active_flag: return
        self.anomaly_detector.stop(); self.monitoring_active_flag = False; self.update_control_states()
    
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
                for title, data in loaded.items():
                    self.rois[title] = ROIInfo(**data)
                    self.roi_list_widget.addItem(f"{title}: ({data['roi'][2]}x{data['roi'][3]})")
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
    
    def show_metrics_dashboard(self):
        if not self.metrics_dashboard_window or not self.metrics_dashboard_window.isVisible():
            self.metrics_dashboard_window = MetricsDashboard(self.data_logger, self); self.metrics_dashboard_window.show()
        else: self.metrics_dashboard_window.activateWindow()
    
    def start_flask_api(self):
        if self.flask_api_thread and self.flask_api_thread.isRunning(): return
        self.flask_api_thread = FlaskAppWrapper(self.data_logger, FLASK_HOST, FLASK_PORT)
        self.flask_api_thread.api_started.connect(lambda h, p: self.status_bar.showMessage(f"Web API: http://{h}:{p}"))
        self.flask_api_thread.api_error.connect(lambda err: QMessageBox.critical(self, "API Error", err))
        self.flask_api_thread.start(); self.update_control_states()
    
    def open_web_dashboard(self):
        if self.flask_api_thread and self.flask_api_thread.isRunning(): webbrowser.open(f"http://{self.flask_api_thread.host}:{self.flask_api_thread.port}/dashboard")
    
    def closeEvent(self, event):
        self.stop_monitoring_session(); self.save_settings()
        if self.alert_window: self.alert_window.close()
        if self.metrics_dashboard_window: self.metrics_dashboard_window.close()
        if self.flask_api_thread: self.flask_api_thread.shutdown()
        super().closeEvent(event)


# --- ROI Selection Overlay ---
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


# --- Metrics Dashboard (pyqtgraph) ---
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


# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(BASE_DIR / "app_icon.png")) if (BASE_DIR / "app_icon.png").exists() else get_standard_icon(QStyle.SP_ComputerIcon))
    
    # Apply a basic style
    app.setStyle("Fusion")
    # palette = QPalette() # Dark theme example - can be expanded
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # app.setPalette(palette)

    main_window = MainApplicationWindow()
    main_window.show()
    
    exit_code = app.exec_()
    logger.info(f"Application exited with code {exit_code}.")
    sys.exit(exit_code)


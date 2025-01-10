from PyQt5.QtCore import (QObject,pyqtSignal,Qt,QTimer)
from PyQt5.QtWidgets import (QWidget, QSpinBox, QCheckBox, QProgressBar, QGridLayout, QRadioButton,QVBoxLayout,QHBoxLayout,QFileDialog, QLabel,QScrollArea,QApplication,QPushButton,QGroupBox,QStyle,QMenu,QStatusBar,QMessageBox,QInputDialog,QDialog,QListWidget,QSlider,QDoubleSpinBox,QMainWindow)
from PyQt5.QtGui import (QColor,QPalette,QFont,QPixmap,QImage, QPainterPath)
from PIL import Image
from PyQt5.QtGui import QPainter, QPen, QBrush
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
import numpy as np
import cv2
import time
from mss import mss
import psutil
from typing import Dict, Tuple, List
from dataclasses import dataclass
import sys
import requests
import json
import math
from datetime import datetime, timedelta
import threading
from collections import deque, defaultdict
from screeninfo import get_monitors
import pyqtgraph as pg
import sqlite3
import pandas as pd
from datetime import datetime
import os
import json
from pathlib import Path
import csv
import sqlite3
from flask import Flask, jsonify, render_template
from flask_cors import CORS

@dataclass

class ROIInfo:
    roi: Tuple[int, int, int, int]
    monitor: dict
    title: str

sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter('datetime', lambda s: datetime.fromisoformat(s))

class MonitorSelector(QDialog):
    def __init__(self, monitors):
        super().__init__()
        self.monitors = monitors
        self.selected_monitor = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        title = QLabel("Select Monitor")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)

        for i, monitor in enumerate(self.monitors):
            btn = QPushButton(f"Monitor {i+1} ({monitor.width}x{monitor.height})")
            btn.clicked.connect(lambda checked, m=monitor: self.select_monitor(m))
            layout.addWidget(btn)

        self.setLayout(layout)
        self.setWindowTitle('Monitor Selection')
        self.setModal(True)

    def select_monitor(self, monitor):
        self.selected_monitor = monitor
        self.accept()

    def get_selected_monitor(self):
        return self.selected_monitor

class AlertWindow(QWidget):
    def __init__(self, detector = None, parent=None):
        super().__init__()
        self.alert_widgets = {}
        self.alert_timers = {}
        self.slack_timers = {}
        self.anomaly_counts = {}
        self.start_time = time.time()
        self.metric_timer = QTimer()
        self.metric_timer.timeout.connect(self.update_conveyance_health)
        self.metric_timer.start(1000)  
        self.metric_label = None
        self.detector = detector

        self.initUI()
        self.position_window()
        self.schedule_daily_reset()

    def initUI(self):
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        layout = QVBoxLayout()
        title_label = QLabel("Region Monitoring Status")
        title_label.setStyleSheet("""QLabel {font-size: 16px;font-weight: bold;color: #2C3E50;padding: 10px;}""")
        layout.addWidget(title_label)

        self.metric_label = QLabel("Monitoring Metric: 0%")
        self.metric_label.setStyleSheet("""QLabel {font-size: 12px;color: #34495E;padding: 5px;}""")
        layout.addWidget(self.metric_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.alert_container = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_container)
        scroll.setWidget(self.alert_container)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Region Status Monitor')
        counter_label = QLabel("Jam Time: 0")
        counter_label.setStyleSheet("color: #7F8C8D;")

    def update_detector(self, detector):
        """Method to update detector reference after initialization"""
        self.detector = detector
    def update_conveyance_health(self):
        if not self.metric_label:
            return
            
        # print(ooaih)
        #        elapsed_time = time.time() - self.start

        elapsed_time = time.time() - self.start_time
        total_monitoring_time = elapsed_time * len(self.alert_widgets)
        total_anomaly_time = sum(self.anomaly_counts.values()) * 1.74 #DONT ASK WHERE 1.74  COMES FROM
        
        # Calculate percentage but...te inverse
        if total_monitoring_time > 0:
            percentage = 100 - (total_anomaly_time / total_monitoring_time) * 100
        else:
            percentage = 100
        self.metric_label.setText(f"Conveyance Health: {percentage:.2f}% "f"(Total Time: {elapsed_time:.0f}s, "f"ROIs: {len(self.alert_widgets)}, "f"Jam Time : {sum(self.anomaly_counts.values())})")

    def closeEvent(self, event):
        self.metric_timer.stop()
        event.accept()
        
    def reset_metrics(self):
        # Reset all counters and data
        self.anomaly_counts.clear()
        self.start_time = time.time()
        self.orange_count = 0
        self.grey_count = 0
        self.blue_count = 0
        self.purple_count = 0
        self.red_count = 0
        self.total_detections = 0   
        
        if self.metric_label:
            self.metric_label.setText("Conveyance Health: 100%")
            
        # Reset all alert widgets
        for title in list(self.alert_widgets.keys()):
            _, status_label, counter_label = self.alert_widgets[title]
            counter_label.setText("Jam Time: 0")
            status_label.setText("Monitoring")
            status_label.setStyleSheet("color: #2C3E50;")
            
        # Reset timer
        self.clear_alerts()
        
    def schedule_daily_reset(self):
        def run_scheduler():
            while True:
                now = datetime.now()
                if now.hour in [6, 18] and now.minute == 0:
                    self.reset_metrics()
                time.sleep(60)  
        
        # Start the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
    def position_window(self):
        try:
            screen = QApplication.primaryScreen().geometry()
            x = screen.width() - self.width() - 20
            y = screen.height() - self.height() - 20
            self.move(x, y)
        except Exception as e:
            print(f"Error BOZO: {e}")
            self.move(300, 300)
                
    def update_alert(self, title: str, has_anomaly: bool, detection_info=None):
        if title not in self.anomaly_counts:
            self.anomaly_counts[title] = 0
        if title not in self.alert_widgets:
            alert_widget = QWidget()
            alert_layout = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold;")
            alert_layout.addWidget(title_label)
            status_label = QLabel()
            alert_layout.addWidget(status_label)
            counter_label = QLabel(f"Jams: {self.anomaly_counts[title]}")
            counter_label.setStyleSheet("color: #7F8C8D;")
            alert_layout.addWidget(counter_label)
            alert_widget.setLayout(alert_layout)
            self.alert_layout.addWidget(alert_widget)
            self.alert_widgets[title] = (alert_widget, status_label, counter_label)
            self.slack_timers[title] = QTimer()
            self.slack_timers[title].setSingleShot(True)
    
        _, status_label, counter_label = self.alert_widgets[title]
        def slack_sender1(msg):
            webhook_url = "https://hooks.slack.com/triggers/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"
            requests.post(webhook_url, json={"operator": msg})  
        if has_anomaly:
            self.anomaly_counts[title] += 1
            counter_label.setText(f"Jams: {self.anomaly_counts[title]}")
            
            if detection_info and isinstance(detection_info, dict):
                detected_colors = [color for color, info in detection_info.items() if info.get('detected', False)]
                
                if detected_colors:
                    primary_color = detected_colors[0]
                    rgb_values = detection_info[primary_color]['rgb']
                    hex_color = f'rgb({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})'
                    
                    color_messages = {
                        'blue': "FULL/UNAVAILABLE",
                        'red': "E-STOPPED/GL",
                        'purple': "ANTI-GRIDLOCKED",
                        'grey': "STAND DOWN",
                        'orange': "JAM DETECTED"  }
                    
                    # Get the appropriate message for the color
                    status_text = color_messages.get(primary_color, "DETECTED")
                    
                    # Add multiple color indicator if needed
                    if len(detected_colors) > 1:
                        status_text += f" (+{len(detected_colors)-1})"
                    
                    status_label.setText(f"🔔 {status_text}")
                    status_label.setStyleSheet(f"color: {hex_color}; font-weight: bold;")
                    counter_label.setStyleSheet(f"color: {hex_color};")
                else:
                    status_label.setText("🔔 JAM DETECTED!")
                    status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
                    counter_label.setStyleSheet("color: #E74C3C;")
            else:
                status_label.setText("🔔 JAM DETECTED!")
                status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
                counter_label.setStyleSheet("color: #E74C3C;")
    
            if not self.slack_timers[title].isActive():
                try:
                    if detection_info and isinstance(detection_info, dict):
                        if detection_info.get('orange', {}).get('detected', False):
                            slack_sender1(f"JAM detected     -->{title}<-- ")
                            self.slack_timers[title].start(150000)
                except Exception as e:
                    print(f"Failed to send Slack notification for {title}: {e}")
                    
            if title in self.alert_timers and self.alert_timers[title].isActive():
                self.alert_timers[title].stop()
            
            if title not in self.alert_timers:
                self.alert_timers[title] = QTimer()
                self.alert_timers[title].setSingleShot(True)
                self.alert_timers[title].timeout.connect(
                    lambda t=title: self.clear_alert(t)
                )
            
            self.alert_timers[title].start(3000)
        else:
            if title not in self.alert_timers or not self.alert_timers[title].isActive():
                status_label.setText("Monitoring")
                status_label.setStyleSheet("color: #2C3E50;")
                counter_label.setStyleSheet("color: #7F8C8D;")
    
        # Force update
        self.alert_widgets[title][0].update()
    
    
    
    def clear_alert(self, title):
        """Clear the alert for a specific ROI"""
        if title in self.alert_widgets:
            _, status_label, counter_label = self.alert_widgets[title]
            status_label.setText("Monitoring")
            status_label.setStyleSheet("color: #2C3E50;")
            counter_label.setStyleSheet("color: #7F8C8D;")
            self.alert_widgets[title][0].update()
    

    def clear_alerts(self):
        for timer in self.alert_timers.values():
            timer.stop()
        self.alert_timers.clear()
        
        for timer in self.slack_timers.values():
            timer.stop()
        self.slack_timers.clear()
        
        for widget, _ in self.alert_widgets.values():
            widget.deleteLater()
        self.alert_widgets.clear()

class AnomalyDetector(QObject):
    alert_signal = pyqtSignal(str, bool, object)
    metrics_updated = pyqtSignal(dict)  
    
    def __init__(self, roi_infos: Dict[str, ROIInfo], alert_window: AlertWindow, parent=None):
        super().__init__(parent)
        self.roi_infos = roi_infos
        self.alert_window = alert_window
        self.running = False
        self.process = psutil.Process()
        self.last_status = {}

        self.lower_grey = np.array([0, 0, 95])  
        self.upper_grey = np.array([180, 30, 170])

        self.lower_orange = np.array([5, 50, 50])  #  should probs add some resolution handling for slivers of orange
        self.upper_orange = np.array([30, 255, 255])
        
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

        self.lower_purple = np.array([130, 50, 50])
        self.upper_purple = np.array([170, 255, 255])
        
        self.lower_red1 = np.array([0, 50, 50])     
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.detect_grey = False
        self.detection_mode = "orange_only"  # Default     
            
        self.resolution_scale = 0.
        self.capture_interval = 0.99
        
        self.metrics = ConveyorMetrics()
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics_display)
        self.metrics_timer.start(1000)
        self.metrics.set_num_rois(len(self.roi_infos))

        
        self.alert_signal.connect(self.handle_alert)
        self.alert_signal.connect(alert_window.update_alert)
        

    def set_detection_mode(self, enable_grey: bool):
        self.detect_grey = enable_grey
        self.detection_mode = "All Colors" if enable_grey else "orange_only"

    def handle_alert(self, title: str, has_anomaly: bool):
        self.alert_window.update_alert(title, has_anomaly)

    def capture_roi(self, roi_info: ROIInfo):
        try:
            with mss() as sct:
                x, y, w, h = roi_info.roi
                monitor = {
                    "top": roi_info.monitor["top"] + y,
                    "left": roi_info.monitor["left"] + x,
                    "width": w,
                    "height": h
                }
                
                roi = np.array(sct.grab(monitor))
                if roi is None or roi.size == 0:
                    return None
                
                roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
                
                if self.resolution_scale < 1.0:
                    new_width = max(1, int(w * self.resolution_scale))
                    new_height = max(1, int(h * self.resolution_scale))
                    if new_width > 0 and new_height > 0:  # print(Addihewfohweofhowehfowehfoweihfowehfowehfowehfoewhowehfohwef
                        roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)
                return roi
                    
        except Exception as e:
            print(f"error: {e}")
            return None
    def set_resolution_scale(self, scale):
        self.resolution_scale = max(0.01, min(1.0, scale / 100))  
    def update_metrics_display(self):
        self.metrics_updated.emit(self.metrics.get_metrics_display())    
        
        
    def detect_color(self, image):
        # Define default green color info
        default_color_info = {
            'green': {'detected': True, 'rgb': (0, 255, 0)},
            'orange': {'detected': False, 'rgb': (255, 165, 0)},
            'grey': {'detected': False, 'rgb': (128, 128, 128)},
            'blue': {'detected': False, 'rgb': (0, 0, 255)},
            'purple': {'detected': False, 'rgb': (128, 0, 128)},
            'red': {'detected': False, 'rgb': (255, 0, 0)}
        }
    
        if image is None:
            self.detected_color_info = default_color_info
            self.metrics.update_detection(False, False, self.roi_infos, False, False, False, True)  # Added green parameter
            return True, default_color_info
    
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            self.detected_color_info = {
                'green': {'detected': False, 'rgb': (0, 255, 0)},
                'orange': {'detected': False, 'rgb': (255, 165, 0)},
                'grey': {'detected': False, 'rgb': (128, 128, 128)},
                'blue': {'detected': False, 'rgb': (0, 0, 255)},
                'purple': {'detected': False, 'rgb': (128, 0, 128)},
                'red': {'detected': False, 'rgb': (255, 0, 0)}
            }
            # Create mask for orange
            orange_mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
            has_orange = bool(self.check_mask(orange_mask, 'orange'))  
            self.detected_color_info['orange']['detected'] = has_orange
            has_grey = False
            has_blue = False
            has_purple = False
            has_red = False           
            mask = orange_mask
            
            # Check for grey if enabled
            if self.detect_grey:
                grey_mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)
                blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
                purple_mask = cv2.inRange(hsv, self.lower_purple, self.upper_purple)
                red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
                red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                has_grey = bool(self.check_mask(grey_mask, 'grey'))
                has_blue = bool(self.check_mask(blue_mask, 'blue'))
                has_purple = bool(self.check_mask(purple_mask, 'purple'))
                has_red = bool(self.check_mask(red_mask, 'red'))         
                # Update detection info dictionary with Python booleans
                self.detected_color_info['grey']['detected'] = has_grey
                self.detected_color_info['blue']['detected'] = has_blue
                self.detected_color_info['purple']['detected'] = has_purple
                self.detected_color_info['red']['detected'] = has_red
                mask = cv2.bitwise_or(orange_mask, grey_mask)
                mask = cv2.bitwise_or(mask, blue_mask)
                mask = cv2.bitwise_or(mask, purple_mask)
                mask = cv2.bitwise_or(mask, red_mask)   
    
            # Check if any non-green color is detected
            has_any_non_green = bool(any([
                self.detected_color_info[color]['detected'] 
                for color in self.detected_color_info if color != 'green'
            ]))
    
            # If no other color is detected, set green to True
            if not has_any_non_green:
                self.detected_color_info['green']['detected'] = True
                self.metrics.update_detection(False, False, self.roi_infos, False, False, False, True)  # Added green parameter
            else:
                self.metrics.update_detection(has_orange, has_grey, self.roi_infos, has_blue, has_purple, has_red, False)
    
            color_info = {
                color: {
                    'detected': bool(info['detected']),
                    'rgb': info['rgb']
                }
                for color, info in self.detected_color_info.items()
            }
            
            return True, color_info  # Always return True since we'll always have at least green
    
        except Exception as e:
            print(f"Color detection error: {e}")
            self.detected_color_info = default_color_info
            self.metrics.update_detection(False, False, self.roi_infos, False, False, False, True)  # Added green parameter
            return True, default_color_info
    
    
    
    def check_mask(self, mask, color='default'):
        if not np.any(mask):
            return False
        color_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        color_percentage = (color_pixels / total_pixels) * 100
        
        if color == 'grey':
            return color_percentage > 30.0
        else:
            return color_percentage > 0.5
    
    
    def monitoring_loop(self):
        while self.running:
            try:
                for title, roi_info in self.roi_infos.items():
                    self.metrics.current_title = title  # Add this line
                    roi = self.capture_roi(roi_info)
                    has_anomaly, color_info = self.detect_color(roi)
                    
                    # Update the current title in metrics
                    self.alert_signal.emit(title, has_anomaly, color_info)
                    
                    self.last_status[title] = has_anomaly
                    del roi
                time.sleep(self.capture_interval)
            except Exception as e:
                print(f"Monitoring loop error: {str(e)}")
    


    def start_monitoring(self):
        # Check both thread status and running flag
        if (hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive()) or \
           (hasattr(self, 'running') and self.running):
            return  # Exit if monitoring is already running
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    

    def stop_monitoring(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

    def set_resolution_scale(self, scale):
        self.resolution_scale = scale

    def set_capture_interval(self, interval):
        self.capture_interval = interval

class SelectionTool(QWidget):
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.alert_window = AlertWindow(self.detector)
        self.initUI()

        self.roi_infos = {}
        self.detector = None
        self.metrics_dashboard = None
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(999)  # Update every secondish
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)  
        self.monitoring_active = False
        
    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        detection_mode_group = QGroupBox("Detection Mode")
        detection_mode_layout = QVBoxLayout()
        control_group = QGroupBox("Controls")
        regions_group = QGroupBox("Monitored Regions")
        settings_group = QGroupBox("Settings")
        
        control_layout = QVBoxLayout()
        
        resource_layout = QHBoxLayout()
        self.cpu_label = QLabel('CPU: 0%')
        self.memory_label = QLabel('Memory: 0%')
        resource_layout.addWidget(self.cpu_label)
        resource_layout.addWidget(self.memory_label)
        control_layout.addLayout(resource_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Add Region button
        self.add_roi_btn = QPushButton('Add Region')
        self.add_roi_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self.add_roi_btn.clicked.connect(self.add_roi)
        button_layout.addWidget(self.add_roi_btn)
        
        # Clear Regions button
        self.clear_rois_btn = QPushButton('Clear All')
        self.clear_rois_btn.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_rois_btn.clicked.connect(self.clear_rois)
        self.clear_rois_btn.setEnabled(False)
        button_layout.addWidget(self.clear_rois_btn)
        
        self.delete_selected_btn = QPushButton('Delete Selected')
        self.delete_selected_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.delete_selected_btn.clicked.connect(self.delete_roi)
        self.delete_selected_btn.setEnabled(True)  # Disabled by default
        button_layout.addWidget(self.delete_selected_btn)
        
        self.metrics_btn = QPushButton("Metrics Dashboard")
        self.metrics_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogHelpButton))  # Or choose another icon
        self.metrics_btn.clicked.connect(self.show_metrics_dashboard)
        self.metrics_btn.setEnabled(False)  # Will be enabled when monitoring starts
        button_layout.addWidget(self.metrics_btn)
        
        control_layout.addLayout(button_layout)
        control_group.setLayout(control_layout)
        
        # Regions layout
        regions_layout = QVBoxLayout()
        self.roi_list = QListWidget()
        self.roi_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_list.customContextMenuRequested.connect(self.show_context_menu)
        regions_layout.addWidget(self.roi_list)
        regions_group.setLayout(regions_layout)
        
        # Settings layout
        settings_layout = QVBoxLayout()
        
        # Resolution scale slider
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution Scale:"))
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setMinimum(10)
        self.resolution_slider.setMaximum(100)
        self.resolution_slider.setValue(50)
        self.resolution_slider.valueChanged.connect(self.update_resolution)
        self.resolution_value = QLabel("0.5")
        resolution_layout.addWidget(self.resolution_slider)
        resolution_layout.addWidget(self.resolution_value)
        settings_layout.addLayout(resolution_layout)
        
        # Interval control
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Update Interval (s):"))
        self.interval_spinbox = QDoubleSpinBox()
        self.interval_spinbox.setRange(0.1, 10.0)
        self.interval_spinbox.setValue(1.0)
        self.interval_spinbox.setSingleStep(0.1)
        self.interval_spinbox.valueChanged.connect(self.update_interval)
        interval_layout.addWidget(self.interval_spinbox)
        settings_layout.addLayout(interval_layout)
        
        clock_layout = QHBoxLayout()
        clock_layout.addWidget(QLabel("System Time:"))
        self.clock_label = QLabel()
        self.clock_label.setStyleSheet("font-weight: bold;")
        clock_layout.addWidget(self.clock_label)
        settings_layout.addLayout(clock_layout)
        
        # Add monitoring schedule controls
        schedule_layout = QHBoxLayout()
        schedule_layout.addWidget(QLabel("Auto-Stop Hours:"))
        
        # Spinboxes for start
        self.start_hour = QSpinBox()
        self.start_hour.setRange(0, 23)
        self.start_hour.setValue(17)  
        self.end_hour = QSpinBox()
        self.end_hour.setRange(0, 23)
        self.end_hour.setValue(9)  
        
        schedule_layout.addWidget(QLabel("From:"))
        schedule_layout.addWidget(self.start_hour)
        schedule_layout.addWidget(QLabel("To:"))
        schedule_layout.addWidget(self.end_hour)
        
        # Add enable/disable toggle
        self.schedule_enabled = QCheckBox("Enable Auto-Stop")
        schedule_layout.addWidget(self.schedule_enabled)
        
        settings_layout.addLayout(schedule_layout)
        
        settings_group.setLayout(settings_layout)
        
        # Start/Stop buttons
        control_buttons_layout = QHBoxLayout()
        
        self.orange_only_radio = QRadioButton("Jams Only")
        self.orange_grey_radio = QRadioButton("Conveyance Health W/Jams")
        self.orange_only_radio.setChecked(True)  

        # Add radio buttons to layout
        detection_mode_layout.addWidget(self.orange_only_radio)
        detection_mode_layout.addWidget(self.orange_grey_radio)
        detection_mode_group.setLayout(detection_mode_layout)
        self.orange_only_radio.toggled.connect(self.detection_mode_changed)
        self.orange_grey_radio.toggled.connect(self.detection_mode_changed)        
    
        self.start_btn = QPushButton('Start Monitoring')
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        control_buttons_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton('Stop Monitoring')
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_buttons_layout.addWidget(self.stop_btn)
        
        self.save_rois_btn = QPushButton("Save Regions")
        self.load_rois_btn = QPushButton("Load Regions")
        self.save_rois_btn.clicked.connect(self.save_rois)
        self.load_rois_btn.clicked.connect(self.load_rois)    
        control_layout.addWidget(self.save_rois_btn)
        control_layout.addWidget(self.load_rois_btn)    
        
        control_layout.addWidget(detection_mode_group)
        
        main_layout.addWidget(control_group)
        main_layout.addWidget(regions_group)
        main_layout.addWidget(settings_group)
        main_layout.addLayout(control_buttons_layout)
        
        self.setLayout(main_layout)
        self.setGeometry(300, 300, 500, 600)
        self.setWindowTitle('Multi-Region Monitor')
        
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage('Ready')
        self.metrics_labels = {
            'uptime': QLabel('Uptime: 0h 0m'),
            'total_detections': QLabel('Total Detections: 0'),
            'orange_count': QLabel('Orange Objects: 0'),
            'grey_count': QLabel('Grey Objects: 0'),
            'conveyor_health': QLabel('Conveyor Health: 100%'),
            'detection_rate': QLabel('Detection Rate: 0/s')
        }
        
    def detection_mode_changed(self):
        """Handle detection mode changes"""
        if self.detector:
            is_grey_enabled = self.orange_grey_radio.isChecked()
            self.detector.set_detection_mode(is_grey_enabled)            
    def show_metrics_dashboard(self):
        if not self.metrics_dashboard:
            
            if hasattr(self, 'detector') and self.detector:
                self.metrics_dashboard = MetricsDashboard(self.detector.metrics, self)
            else:
                QMessageBox.warning(self, "Warning", 
                                "Please start monitoring to view metrics dashboard.")
                return
        
        # Show the dashboard
        self.metrics_dashboard.show()
        self.metrics_dashboard.activateWindow()

    def update_clock(self):
        current_time = datetime.now()
        self.clock_label.setText(current_time.strftime("%H:%M:%S"))
        
        if self.schedule_enabled.isChecked():
            self.check_monitoring_hours(current_time.hour)

    def check_monitoring_hours(self, current_hour):
        current_time = datetime.now()
        current_minutes = current_time.minute
        current_time_minutes = current_hour * 60 + current_minutes
        
        break_periods = [
            ((9, 30), (9, 45)),
            ((12, 0), (12, 30)),  
            ((15, 0), (15, 15))  
        ]
        is_break_time = False
        for (start_hour, start_min), (end_hour, end_min) in break_periods:
            break_start = start_hour * 60 + start_min
            break_end = end_hour * 60 + end_min
            
            if break_start <= current_time_minutes < break_end:
                is_break_time = True
                break
                            
        # Check user-defined monitoring hours
        start_hour = self.start_hour.value()
        end_hour = self.end_hour.value()
        
        # Convert monitoring window to minutes
        start_time = start_hour * 60
        end_time = end_hour * 60
        
        # Determine if we should be running
        if start_hour > end_hour:
            should_run = not (start_time <= current_time_minutes or current_time_minutes < end_time)
        else:
            should_run = not (start_time <= current_time_minutes < end_time)
        
        should_be_monitoring = should_run and not is_break_time
        
        if should_be_monitoring and not self.monitoring_active:
            if hasattr(self, 'detector') and self.detector:
                self.detector.start_monitoring()
                self.monitoring_active = True
                print(f"Monitoring started at {current_time.strftime('%H:%M:%S')}")
                self.metrics_btn.setEnabled(True)
                
        elif not should_be_monitoring and self.monitoring_active:
            if hasattr(self, 'detector') and self.detector:
                self.detector.stop_monitoring()
                self.monitoring_active = False
                reason = "break time" if is_break_time else "quiet hours"
                print(f"Monitoring stopped at {current_time.strftime('%H:%M:%S')} - {reason}")
                self.metrics_btn.setEnabled(False)

    def update_metrics_display(self, metrics):
        if hasattr(self, 'metrics_labels'):
            for key, value in metrics.items():
                if key in self.metrics_labels:
                    self.metrics_labels[key].setText(f"{key.replace('_', ' ').title()}: {value}")

        
    def save_rois(self):
        try:
            # Create a dictionary to store ROI data
            roi_data = {}
            for title, roi_info in self.roi_infos.items():
                roi_data[title] = {
                    'roi': roi_info.roi,
                    'monitor': roi_info.monitor,
                    'title': title
                }
    
            # Get file name from user
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save ROI Configuration",
                os.path.expanduser("~/Documents"),
                "JSON Files (*.json)"
            )
    
            if file_name:
                if not file_name.endswith('.json'):
                    file_name += '.json'
                
                with open(file_name, 'w') as f:
                    json.dump(roi_data, f, indent=4)
                
                self.status_bar.showMessage(f"ROIs saved successfully to {file_name}")
    
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save ROI configuration: {str(e)}"
            )
    
    def load_rois(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load ROI Configuration",
                os.path.expanduser("~/Documents"),
                "JSON Files (*.json)"
            )
    
            if file_name:
                with open(file_name, 'r') as f:
                    loaded_data = json.load(f)
                    self.clear_rois()
    
                for title, data in loaded_data.items():
                    self.roi_infos[title] = ROIInfo(
                        roi=tuple(data['roi']),
                        monitor=data['monitor'],
                        title=data['title']
                    )
                    
                    x, y, w, h = data['roi']
                    self.roi_list.addItem(f"{title}: ({x}, {y}, {w}, {h})")
    
                has_rois = len(self.roi_infos) > 0
                self.clear_rois_btn.setEnabled(has_rois)
                self.start_btn.setEnabled(has_rois)
                
                self.status_bar.showMessage(f"ROIs loaded successfully from {file_name}")
    
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load ROI configuration: {str(e)}"
            )
    

    def update_resource_usage(self):
        if hasattr(self, 'detector') and self.detector:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.Process().memory_percent()
            self.cpu_label.setText(f'CPU: {cpu_percent:.1f}%')
            self.memory_label.setText(f'Memory: {memory_percent:.1f}%')

    def show_context_menu(self, position):
        menu = QMenu()
        item = self.roi_list.itemAt(position)
        
        if item:
            title = item.text().split(':')[0]
            
            rename_action = menu.addAction("Rename")
            rename_action.triggered.connect(lambda: self.rename_roi(title))
            
            delete_action = menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.delete_roi(title))
            
            menu.exec_(self.roi_list.viewport().mapToGlobal(position))

    def add_roi(self):
        monitors = get_monitors()
        if self.detector is None:
            self.detector = AnomalyDetector(self.roi_infos, self.alert_window)
            self.alert_window.detector = self.detector  # Now 
        else:
            # Update
            self.detector.roi_infos = self.roi_infos
        selector = MonitorSelector(monitors)
        if selector.exec_() == QDialog.Accepted:
            monitor = selector.get_selected_monitor()
            
            # Get region
            title, ok = QInputDialog.getText(self, 'Region Name', 'Enter a name for this region:')
            if ok and title:
                if title in self.roi_infos:
                    QMessageBox.warning(self, "Warning", "A region with this name already exists!")
                    return
                
                # Create monitor dict for mss
                monitor_dict = {
                    "left": monitor.x,
                    "top": monitor.y,
                    "width": monitor.width,
                    "height": monitor.height
                }
                
                self.hide()  
                time.sleep(0.5)  
                
                try:
                    x, y, w, h = self.select_region(monitor_dict)
                    if w > 0 and h > 0:
                        roi_info = ROIInfo(roi=(x, y, w, h), monitor=monitor_dict,title=title)
                        self.roi_infos[title] = roi_info
                        
                        # Add to list widget
                        self.roi_list.addItem(
                            f"{title}: ({x}, {y}, {w}, {h})"
                        )
                        
                        # Update buttons
                        self.clear_rois_btn.setEnabled(True)
                        self.start_btn.setEnabled(True)
                        
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to select region: {str(e)}")
                
                finally:
                    self.show()  # Show main window again
        if self.detector is None:
            self.detector = AnomalyDetector(self.roi_infos, self.alert_window)
            # Use orange_grey_rad
            self.detector.set_detection_mode(self.orange_grey_radio.isChecked())
            self.detector.metrics_updated.connect(self.update_metrics_display)
        

    def toggle_grey_detection(self):
        is_grey = self.grey_toggle.isChecked()
        if self.detector:
            self.detector.set_detection_mode(is_grey)
            self.grey_toggle.setStyleSheet(
                "background-color: #808080;" if is_grey else "background-color: none;"
            )
            self.grey_toggle.setText('GREY Mode ON' if is_grey else 'Toggle GREY')
    

    def select_region(self, monitor):
        selection_window = QWidget()
        selection_window.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        selection_window.setGeometry(monitor["left"], monitor["top"],
                                monitor["width"], monitor["height"])
        selection_window.setWindowState(Qt.WindowFullScreen)
        
        with mss() as sct:
            screenshot = sct.grab(monitor)
            # Convert to QPixmap
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            qimage = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
        
        start_pos = None
        end_pos = None
        current_pos = None
        is_selecting = False
        
        class SelectionOverlay(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setAttribute(Qt.WA_TransparentForMouseEvents)
                self.setAttribute(Qt.WA_NoSystemBackground)
                self.setAttribute(Qt.WA_TranslucentBackground)
        
        overlay = SelectionOverlay(selection_window)
        overlay.resize(selection_window.size())
        
        def paintEvent(event):
            painter = QPainter(overlay)
            painter.fillRect(overlay.rect(), QColor(0, 0, 0, 128))
            
            if is_selecting and start_pos:
                x = min(start_pos.x(), current_pos.x() if current_pos else start_pos.x())
                y = min(start_pos.y(), current_pos.y() if current_pos else start_pos.y())
                width = abs((current_pos.x() if current_pos else start_pos.x()) - start_pos.x())
                height = abs((current_pos.y() if current_pos else start_pos.y()) - start_pos.y())
                
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.fillRect(x, y, width, height, Qt.transparent)
                
                # Draw selection rectangle border
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
                painter.drawRect(x, y, width, height)
                
                # Draw selection dimensions
                painter.setPen(Qt.white)
                painter.setFont(QFont('Arial', 10))
                dimension_text = f'{width} x {height}'
                painter.drawText(x + 5, y - 5, dimension_text)
        
        def mousePressEvent(event):
            nonlocal start_pos, is_selecting, current_pos
            if event.button() == Qt.LeftButton:
                start_pos = event.pos()
                current_pos = event.pos()
                is_selecting = True
                overlay.update()
        
        def mouseMoveEvent(event):
            nonlocal current_pos
            if is_selecting:
                current_pos = event.pos()
                overlay.update()
        
        def mouseReleaseEvent(event):
            nonlocal end_pos, is_selecting
            if event.button() == Qt.LeftButton:
                end_pos = event.pos()
                is_selecting = False
                selection_window.close()
        
        def keyPressEvent(event):
            if event.key() == Qt.Key_Escape:
                selection_window.close()
        
        # Set background screenshot
        selection_window.setAutoFillBackground(True)
        palette = selection_window.palette()
        palette.setBrush(QPalette.Window, QBrush(pixmap))
        selection_window.setPalette(palette)
        
        # Connect events
        overlay.paintEvent = paintEvent
        selection_window.mousePressEvent = mousePressEvent
        selection_window.mouseMoveEvent = mouseMoveEvent
        selection_window.mouseReleaseEvent = mouseReleaseEvent
        selection_window.keyPressEvent = keyPressEvent
        
        selection_window.show()
        overlay.show()
        QApplication.processEvents()
        
        while selection_window.isVisible():
            QApplication.processEvents()
        
        if start_pos and end_pos:
            x1, y1 = start_pos.x(), start_pos.y()
            x2, y2 = end_pos.x(), end_pos.y()
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            return x, y, w, h
        
        return 0, 0, 0, 0


    def rename_roi(self, old_title):
        new_title, ok = QInputDialog.getText(self, 'Rename Region', 'Enter new name:', text=old_title)
        if ok and new_title and new_title != old_title:
            if new_title in self.roi_infos:
                QMessageBox.warning(self, "Warning", "A region with this name already exists!")
                return
                
            # Update roi_infos dictionary
            self.roi_infos[new_title] = self.roi_infos.pop(old_title)
            
            for i in range(self.roi_list.count()):
                item = self.roi_list.item(i)
                if item.text().startswith(old_title + ':'):
                    x, y, w, h = self.roi_infos[new_title].roi
                    item.setText(f"{new_title}: ({x}, {y}, {w}, {h})")
                    break

    def delete_roi(self):
        # Get the currently selected item
        current_item = self.roi_list.currentItem()
        if current_item:
            try:
                # Extract title from the selected item
                title = current_item.text().split(':')[0].strip()
                
                reply = QMessageBox.question(self, 'Delete Region', 
                                        f'Are you sure you want to delete "{title}"?',
                                        QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # Remove from roi_infos if it exists
                    if title in self.roi_infos:
                        self.roi_infos.pop(title)
                    
                    # Remove from list widget
                    self.roi_list.takeItem(self.roi_list.row(current_item))
                    
                    # Update buttons
                    self.clear_rois_btn.setEnabled(len(self.roi_infos) > 0)
                    self.start_btn.setEnabled(len(self.roi_infos) > 0)
                    
                    self.status_bar.showMessage(f'Deleted region: {title}')
            except Exception as e:
                print(f"Error deleting ROI: {str(e)}")
                QMessageBox.warning(self, "Error", 
                                f"Failed to delete region: {str(e)}")

    def clear_rois(self):
        reply = QMessageBox.question(self, 'Clear All Regions', 'Are you sure you want to clear all regions?',QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.roi_infos.clear()
            self.roi_list.clear()
            self.clear_rois_btn.setEnabled(False)
            self.start_btn.setEnabled(False)

    def update_resolution(self):
        scale = self.resolution_slider.value() / 100
        self.resolution_value.setText(f"{scale:.2f}")
        if self.detector:
            self.detector.set_resolution_scale(scale)

    def update_interval(self):
        if self.detector:
            self.detector.set_capture_interval(self.interval_spinbox.value())

    def start_monitoring(self):
        self.detector = AnomalyDetector(self.roi_infos, self.alert_window)
        self.detector.set_resolution_scale(self.resolution_slider.value() / 100)
        self.detector.set_capture_interval(self.interval_spinbox.value())
        
        # Remove this line as it's redundant
        # self.detector.start_monitoring()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.alert_window.show()
        
        if self.detector:
            self.detector.running = True
            self.monitoring_thread = threading.Thread(target=self.detector.monitoring_loop)
            self.monitoring_thread.start()
            # Start the metrics timer when monitoring starts
            if hasattr(self.detector, 'metrics_timer'):
                self.detector.metrics_timer.start(1000)
        
        self.metrics_btn.setEnabled(True)
        self.status_bar.showMessage('Monitoring active')



    def stop_monitoring(self):
        if self.detector:
            self.detector.stop_monitoring()
            self.detector = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.alert_window.hide()
        
        self.status_bar.showMessage('Monitoring stopped')
        self.metrics_btn.setEnabled(False)
        if hasattr(self, 'metrics_dashboard') and self.metrics_dashboard:
            self.metrics_dashboard.close()
            self.metrics_dashboard = None

    def closeEvent(self, event):
        if self.detector:
            self.stop_monitoring()
        event.accept()

class ConveyorMetrics:
    def __init__(self):
        self.print_metrics_counter = 0
        self.total_detections = 0
        self.orange_count = 0
        self.grey_count = 0
        self.blue_count = 0
        self.purple_count = 0
        self.red_count = 0

        self.orange_count_raw = 0
        self.grey_count_raw = 0
        self.blue_count_raw = 0
        self.purple_count_raw = 0
        self.red_count_raw = 0

        self.conveyor_health = 100.0
        self.num_rois = 0
        self.roi_infos = None
        self.current_title = None
        self.start_time = time.time()
        
        self.actual_title = None
        
        self.health_history = deque(maxlen=180)  
        self.health_history.append(100.0)  
        
        self.anomaly_titles = {}
        
        self.detection_history = deque(maxlen=100)  
        self.last_detection_time = time.time()
        self.data_logger = DataLogger()   
        self.detection_history.append({
            'orange': False,
            'grey': False,
            'blue': False,
            'purple': False,
            'red': False,
            'timestamp': self.start_time
        })

    def set_num_rois(self, num_rois):
        """Set the number of regions of interest being monitored"""
        self.num_rois = num_rois
    def set_roi_infos(self, roi_infos):
        """Method to update ROI information"""
        self.roi_infos = roi_infos
        self.num_rois = len(roi_infos) if isinstance(roi_infos, list) else 1
        
    def update_detection(self, has_orange, has_grey, roi_infos, has_blue=False, has_purple=False, has_red=False, has_green=True):
        """Update metrics with new detection information"""
        current_time = datetime.now()
        
        self.set_roi_infos(roi_infos)
        ROIS = self.num_rois
        
        if any([has_orange, has_grey, has_blue, has_purple, has_red]):
            self.total_detections += 1/ROIS
            
        current_detection = {
            'orange': has_orange,
            'grey': has_grey,
            'blue': has_blue,
            'purple': has_purple,
            'red': has_red,
            'timestamp': current_time
        }
        self.detection_history.append(current_detection)
        
        if has_orange:
            self.orange_count_raw = getattr(self, 'orange_count_raw', 0) + 1/ROIS
            self.orange_count = round(self.orange_count_raw)
        if has_grey:
            self.grey_count_raw = getattr(self, 'grey_count_raw', 0) + 1/ROIS
            self.grey_count = round(self.grey_count_raw)
        if has_blue:
            self.blue_count_raw = getattr(self, 'blue_count_raw', 0) + 1/ROIS
            self.blue_count = round(self.blue_count_raw)
        if has_purple:
            
            self.purple_count_raw = getattr(self, 'purple_count_raw', 0) + 1/ROIS
            self.purple_count = round(self.purple_count_raw)
        if has_red:
            self.red_count_raw = getattr(self, 'red_count_raw', 0) + 1/ROIS
            self.red_count = round(self.red_count_raw)
            
            
        if any([has_orange, has_grey, has_blue, has_purple, has_red, has_green]):  # If any anomaly detected
            # Initialize title in anomaly tracking if not exists
            self.actual_title = self.current_title
            if self.current_title not in self.anomaly_titles:
                self.anomaly_titles[self.current_title] = {
                    'total_count': 0,
                    'colors': {
                        'orange': 0,
                        'grey': 0,
                        'blue': 0,
                        'purple': 0,
                        'red': 0
                    },
                    'timestamps': [],
                    'sequence': []
                }
            
            # Update total count for this title
            if any([has_orange, has_grey, has_blue, has_purple, has_red]):
                self.anomaly_titles[self.current_title]['total_count'] += 1
            
            # Update individual color counts for this title
            if has_orange:
                self.anomaly_titles[self.current_title]['colors']['orange'] += 1
            if has_grey:
                self.anomaly_titles[self.current_title]['colors']['grey'] += 1
            if has_blue:
                self.anomaly_titles[self.current_title]['colors']['blue'] += 1
            if has_purple:
                self.anomaly_titles[self.current_title]['colors']['purple'] += 1
            if has_red:
                self.anomaly_titles[self.current_title]['colors']['red'] += 1
            
            # Record timestamp and sequence
            self.anomaly_titles[self.current_title]['timestamps'].append(current_time)
            
            # Record sequence of colors that occurred
            sequence = []
            if has_orange: sequence.append('orange')
            if has_grey: sequence.append('grey')
            if has_blue: sequence.append('blue')
            if has_purple: sequence.append('purple')
            if has_red: sequence.append('red')
            if has_green: sequence.append('green')
            self.anomaly_titles[self.current_title]['sequence'].append(sequence)
            #print(sequence)
            # Log to data logger with enhanced information
            #self.data_logger.log_metrics(self)
        
        self.last_detection_time = current_time
        self.calculate_conveyor_health()
        self.print_metrics_counter += 1/ROIS
        
        self.data_logger.log_metrics(self)
        current_datetime = datetime.now()
        if (current_datetime.hour == 0) and current_datetime.minute == 0:
            self.data_logger.generate_daily_report()
            print(f"Daily report generated at {current_datetime.strftime('%Y-%m-%d %H:%M')}")
    
    def calculate_conveyor_health(self):
        """Calculate the health of the conveyor system"""
        WEIGHTS = {
            'orange': 0.62,  # Medium-high impact
            'grey': 0.7,    # high impact
            'red': 0.9,     # High impact
            'blue': 0.3,    # Medium-low impact
            'purple': 0.45  # Medium impact
        }
        
        if len(self.detection_history) < 2:
            return
        
        # Calculate detection rates from recent history
        detection_counts = {color: 0 for color in WEIGHTS.keys()}
        total_samples = len(self.detection_history)
        
        for detection in self.detection_history:
            for color in WEIGHTS.keys():
                if detection.get(color, False):
                    detection_counts[color] += 1
        
        # Calculate health impact
        total_health_impact = 0
        for color, count in detection_counts.items():
            rate = count / total_samples if total_samples > 0 else 0
            health_impact = rate * WEIGHTS[color] * 100
            total_health_impact += health_impact
        
        # Calculate final health (inverse of total impact)
        health = max(0, 100 - total_health_impact)
        self.conveyor_health = round(health, 1)
        self.health_history.append(self.conveyor_health)

    def generate_performance_metrics(self):
        """Generate current performance metrics"""
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        
        # Find most frequent color
        color_counts = {
            'orange': self.orange_count,
            'grey': self.grey_count,
            'blue': self.blue_count,
            'purple': self.purple_count,
            'red': self.red_count
        }
        most_frequent = max(color_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'health': self.conveyor_health,
            'uptime': f"{hours}h {minutes}m",
            'detection_rate': self.total_detections / uptime if uptime > 0 else 0,
            'most_frequent_color': most_frequent
        }

    def analyze_health_trend(self):
        """Analyze the trend in conveyor health"""
        if len(self.health_history) < 2:
            return {'direction': 'stable', 'change': 0.0}
        
        # Calculate trend over available history maybe
        recent_health = list(self.health_history)
        change = recent_health[-1] - recent_health[0]
        
        # Determine trend direction
        if abs(change) < 1.0:
            direction = 'stable'
        else:
            direction = 'improving' if change > 0 else 'declining'
        
        return {
            'direction': direction,
            'change': abs(change)
        }

    def predict_maintenance_need(self):
        """Predict maintenance needs based on health trends"""
        HEALTH_THRESHOLD = 60
        CRITICAL_THRESHOLD = 30
        
        if self.conveyor_health < HEALTH_THRESHOLD:
            # Calculate rate of health decline
            if len(self.health_history) >= 2:
                health_values = list(self.health_history)
                health_decline_rate = (health_values[-1] - health_values[0]) / len(health_values)
                
                if health_decline_rate < 0:  # Health is declining
                    hours_until_critical = abs((self.conveyor_health - CRITICAL_THRESHOLD) / 
                                             (health_decline_rate * 3600))
                    return {
                        'maintenance_needed': True,
                        'estimated_hours_until_critical': round(hours_until_critical, 1),
                        'recommendation': f'Schedule maintenance within {round(hours_until_critical, 1)} hours'
                    }
        
        return {
            'maintenance_needed': False,
            'estimated_hours_until_critical': 0,
            'recommendation': 'No maintenance needed'
        }

    def detect_color_patterns(self):
        """Detect patterns in color detection sequence"""
        if len(self.detection_history) < 2:
            return "Insufficient detection history"

        PATTERNS = {
            ('blue', 'blue'): "Multiple blue detections - Check alignment",
            ('orange', 'orange'): "Multiple orange detections - Possible backup",
            ('red', 'red'): "Multiple red detections - Urgent attention needed",
            ('blue', 'orange'): "Blue-Orange pattern - Check spacing",
            ('orange', 'red'): "Orange-Red pattern - Check conveyor tension"
        }
        
        recent_detections = list(self.detection_history)[-2:]
        
        # Extract detected colors
        recent_colors = []
        for detection in recent_detections:
            colors_in_detection = [
                color for color in ['orange', 'grey', 'blue', 'purple', 'red']
                if detection.get(color, False)
            ]
            if colors_in_detection:
                recent_colors.extend(colors_in_detection)
        
        if len(recent_colors) >= 2:
            for i in range(len(recent_colors) - 1):
                pattern = (recent_colors[i], recent_colors[i+1])
                if pattern in PATTERNS:
                    return PATTERNS[pattern]
        
        return "No significant patterns detected"

    def get_metrics_display(self):
        """Get formatted metrics for display"""
        return {
            'uptime': self.generate_performance_metrics()['uptime'],
            'total_detections': self.total_detections,
            'orange_count': self.orange_count,
            'grey_count': self.grey_count,
            'blue_count': self.blue_count,
            'purple_count': self.purple_count,
            'red_count': self.red_count,
            'conveyor_health': f"{self.conveyor_health:.1f}%",
            'detection_rate': f"{self.generate_performance_metrics()['detection_rate']:.2f}/s"
        }

    def print_metrics(self):
        """Print current metrics to console with detailed region analysis"""
        metrics = self.get_metrics_display()
        print("\nConveyor Metrics:")
        print(f"Number of ROIs: {self.num_rois}")
        print(f"Metrics Update Count: {self.print_metrics_counter}")
        print("\nGlobal Counts:")
        print(f"Total Detections: {metrics['total_detections']}")
        print(f"Orange Count: {metrics['orange_count']}")
        print(f"Grey Count: {metrics['grey_count']}")
        print(f"Blue Count: {metrics['blue_count']}")
        print(f"Purple Count: {metrics['purple_count']}")
        print(f"Red Count: {metrics['red_count']}")
        print(f"Conveyor Health: {metrics['conveyor_health']}")
        
        print("\nRegion-Specific Anomaly Counts:")
        for title, data in self.anomaly_titles.items():
            print(f"\n{title} Region:")
            print(f"  Total Anomalies: {data['total_count']}")
            print("  Color Breakdown:")
            print(f"    Orange: {data['colors']['orange']}")
            print(f"    Grey: {data['colors']['grey']}")
            print(f"    Blue: {data['colors']['blue']}")
            print(f"    Purple: {data['colors']['purple']}")
            print(f"    Red: {data['colors']['red']}")
            
            # Print last occurrence if available
            if data['timestamps']:
                last_time = data['timestamps'][-1]
                print(f"  Last Anomaly: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Print last sequence if available
            if data['sequence']:
                print(f"  Last Sequence: {' + '.join(data['sequence'][-1])}")

class MetricsDashboard(QMainWindow):
    # Class constants
    UPDATE_INTERVAL = 1000  # milliseconds
    WINDOW_MIN_WIDTH = 1200
    WINDOW_MIN_HEIGHT = 1000
    MAX_HISTORY_LENGTH = 900  # For timestamps and health history
    GRAPH_HEIGHT = 300
    GRAPH_WIDTH = 400
    
    COLORS = {
        'background': 'w',
        'grid': '#cccccc',
        'plot_line': '#2196F3',
        'bar_colors': ['#FF9800', '#9C27B0', '#2196F3', '#4CAF50', '#F44336']
    }

    def __init__(self, metrics, parent=None):
        super().__init__(parent)
        self.metrics = metrics
        
        # Initialize with fixed-size deques to prevent unlimited memory growth
        self.timestamps = deque(maxlen=self.MAX_HISTORY_LENGTH)
        self.health_history = deque(maxlen=self.MAX_HISTORY_LENGTH)
        
        self.init_ui()
        self.setup_timer()
    
    def setup_timer(self):
        """Setup update timer for dashboard"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(self.UPDATE_INTERVAL)

    def init_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle('Conveyor System Metrics Dashboard')
        self.setMinimumSize(self.WINDOW_MIN_WIDTH, self.WINDOW_MIN_HEIGHT)
        self.setup_central_widget()
        self.update_dashboard()

    def setup_central_widget(self):
        """Setup the central widget and main layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        main_layout.addSpacing(10)
        self.create_health_section(main_layout)
        main_layout.addSpacing(20)
        self.create_graphs_section(main_layout)
        main_layout.addSpacing(20)
        self.create_metrics_section(main_layout)
        main_layout.addSpacing(10)

    def create_health_section(self, parent_layout):
        """Create the health monitoring section"""
        health_group = QGroupBox("System Health")
        health_layout = QHBoxLayout()
        
        self.health_bar = self.create_health_bar()
        self.health_status = self.create_health_status_label()
        
        health_layout.addWidget(self.health_bar, stretch=7)
        health_layout.addSpacing(10)
        health_layout.addWidget(self.health_status, stretch=3)
        
        health_group.setLayout(health_layout)
        parent_layout.addWidget(health_group)

    def create_health_bar(self):
        """Create and configure health progress bar"""
        health_bar = QProgressBar()
        health_bar.setMinimum(0)
        health_bar.setMaximum(100)
        health_bar.setStyleSheet(self._get_health_bar_style())
        health_bar.setTextVisible(True)
        health_bar.setFormat("%p%")
        return health_bar

    def _get_health_bar_style(self):
        """Return the stylesheet for health progress bar"""
        return """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 30px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0.5, x2: 1, y2: 0.5,
                    stop: 0 #f44336,
                    stop: 0.5 #ffeb3b,
                    stop: 1 #4caf50
                );
            }
        """

    def create_health_status_label(self):
        """Create and configure health status label"""
        status_label = QLabel()
        status_label.setFont(QFont('Arial', 12, QFont.Bold))
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)
        return status_label

    def create_graphs_section(self, parent_layout):
        """Create the graphs section"""
        graphs_group = QGroupBox("Performance Graphs")
        graphs_layout = QHBoxLayout()
        
        self.health_plot = self.create_health_history_graph()
        self.detection_plot = self.create_detection_rates_graph()
        
        graphs_layout.addWidget(self.health_plot)
        graphs_layout.addWidget(self.detection_plot)
        
        graphs_group.setLayout(graphs_layout)
        parent_layout.addWidget(graphs_group)

    def create_health_history_graph(self):
        """Create and configure health history graph"""
        plot = pg.PlotWidget()
        plot.setBackground(self.COLORS['background'])
        plot.setTitle("Health History (Last 15 Minutes)", size="12pt")
        plot.setLabel('left', 'Health %', size="10pt")
        plot.setLabel('bottom', 'Time (5-min intervals)', size="10pt")
        plot.showGrid(x=True, y=True, alpha=0.3)
        
        plot.setMinimumSize(self.GRAPH_WIDTH, self.GRAPH_HEIGHT)
        plot.getAxis('left').setPen(self.COLORS['grid'])
        plot.getAxis('bottom').setPen(self.COLORS['grid'])
        
        # Configure x-axis
        axis = plot.getAxis('bottom')
        axis.setStyle(showValues=True)
        
        # Set fixed range for y-axis (0-100%)
        plot.setYRange(0, 100)
        
        return plot
    

    def create_detection_rates_graph(self):
        """Create and configure detection rates graph"""
        plot = pg.PlotWidget()
        plot.setBackground(self.COLORS['background'])
        plot.setTitle("Color Detection Rates", size="12pt")
        plot.setLabel('left', 'Count', size="10pt")
        plot.setLabel('bottom', 'Color', size="10pt")
        plot.showGrid(y=True, alpha=0.3)
        plot.getAxis('bottom').setHeight(40)
        
        plot.setMinimumSize(self.GRAPH_WIDTH, self.GRAPH_HEIGHT)
        plot.getAxis('left').setPen(self.COLORS['grid'])
        plot.getAxis('bottom').setPen(self.COLORS['grid'])
        
        return plot

    def create_metrics_section(self, parent_layout):
        """Create the detailed metrics section"""
        metrics_group = QGroupBox("Detailed Metrics")
        metrics_layout = QGridLayout()
        
        self.metric_labels = self.initialize_metric_labels()
        self.populate_metrics_grid(metrics_layout)
        
        metrics_layout.setSpacing(15)
        metrics_group.setLayout(metrics_layout)
        parent_layout.addWidget(metrics_group)

    def initialize_metric_labels(self):
        labels = {
            'uptime': ('Uptime:', QLabel()),
            'most_frequent': ('Most Frequent:', QLabel()),
            'trend': ('Health Trend:', QLabel()),
            'maintenance': ('Maintenance Status:', QLabel()),
            'pattern': ('Pattern Alert:', QLabel()),
            'avg_health': ('15-min Average Health:', QLabel()),
            'uph_loss': ('CAP UPH Loss:', QLabel())
        }
        
        for title, (_, label) in labels.items():
            label.setFont(QFont('Arial', 10))
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setMinimumWidth(150)
            label.setStyleSheet("""
                QLabel {
                    padding: 5px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                }
            """)
            
        return labels
    

    def populate_metrics_grid(self, layout):
        """Populate the metrics grid with labels"""
        # Modified layout with average health and UPH loss
        metrics_pairs = [
            ('uptime', 'most_frequent'),
            ('trend', 'maintenance'),
            ('pattern', None),
            ('avg_health', 'uph_loss')  # Add new row for health metrics
        ]
        
        for row, (metric1, metric2) in enumerate(metrics_pairs):
            layout.addWidget(QLabel(self.metric_labels[metric1][0]), row, 0)
            layout.addWidget(self.metric_labels[metric1][1], row, 1)
            if metric2:
                layout.addWidget(QLabel(self.metric_labels[metric2][0]), row, 2)
                layout.addWidget(self.metric_labels[metric2][1], row, 3)
    
    def calculate_recent_health_metrics(self):
        """Calculate 15-minute average health and UPH loss"""
        if len(self.health_history) == 0:
            return 100.0, 0.0
            
        recent_values = list(self.health_history)[-180:]  # Last 15 minutes (assuming 5-second intervals)
        if recent_values:
            avg_health = sum(recent_values) / len(recent_values)
            health_loss = 100 - avg_health
            uph_loss = health_loss * 1.62 if health_loss > 0 else 0
            return round(avg_health, 1), round(uph_loss, 1)
        return 100.0, 0.0
    
    
    def update_dashboard(self):
        """Update all dashboard components"""
        try:
            health = self.metrics.conveyor_health
            self.update_health_indicators(health)
            self.update_history_graphs()
            self.update_detection_rates()
            self.update_detailed_metrics()
        except Exception as e:
            print(f"Error updating dashboard: {e}")

    def update_health_indicators(self, health):
        """Update health bar and status"""
        self.health_bar.setValue(int(health))
        
        if health >= 80:
            status = "Excellent"
            color = "#4caf50"
        elif health >= 60:
            status = "Good"
            color = "#ffeb3b"
        elif health >= 40:
            status = "Fair"
            color = "#ff9800"
        else:
            status = "Poor"
            color = "#f44336"
            
        self.health_status.setText(status)
        self.health_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def update_history_graphs(self):
        """Update health history graph with 15-minute window and 5-minute increments"""
        current_time = time.time()
        
        # Add new data point
        self.timestamps.append(current_time)
        self.health_history.append(self.metrics.conveyor_health)
        
        # Remove data points older than 15 minutes
        while self.timestamps and (current_time - self.timestamps[0]) > 900:  # 900 seconds = 15 minutes
            self.timestamps.popleft()
            self.health_history.popleft()
        
        if self.timestamps:  # Only plot if we have data
            self.health_plot.clear()
            
            # Calculate 5-minute increment timestamps
            start_time = current_time - 900  # 15 minutes ago
            increment = 300  # 5 minutes in seconds
            tick_timestamps = []
            tick_labels = []
            
            # Generate ticks for each 5-minute increment
            for i in range(4):  # 0, 5, 10, 15 minutes (4 ticks)
                tick_time = start_time + (i * increment)
                tick_timestamps.append(tick_time)
                tick_labels.append(datetime.fromtimestamp(tick_time).strftime('%H:%M'))
            
            # Plot the data
            self.health_plot.plot(
                list(self.timestamps),
                list(self.health_history),
                pen=self.COLORS['plot_line']
            )
            
            # Set the x-axis range to show last 15 minutes
            self.health_plot.setXRange(start_time, current_time)
            
            # Update x-axis tick labels with 5-minute increments
            axis = self.health_plot.getAxis('bottom')
            axis.setTicks([[(ts, label) for ts, label in zip(tick_timestamps, tick_labels)]])

    def update_detection_rates(self):
        """Update detection rates bar graph"""
        self.detection_plot.clear()
        
        counts = [
            self.metrics.orange_count,
            self.metrics.grey_count,
            self.metrics.blue_count,
            self.metrics.purple_count,
            self.metrics.red_count
        ]
        
        x = range(len(counts))
        colors = self.COLORS['bar_colors'][:len(counts)]
        
        for i, (count, color) in enumerate(zip(counts, colors)):
            bar = pg.BarGraphItem(
                x=[i], height=[count],
                width=0.8, brush=color
            )
            self.detection_plot.addItem(bar)

    def update_detailed_metrics(self):
        """Update detailed metrics labels"""
        try:
            # Update uptime
            uptime = time.time() - self.metrics.start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            self.metric_labels['uptime'][1].setText(uptime_str)
            
            # Update most frequent
            if self.metrics.total_detections > 0:
                counts = {
                    'Orange': self.metrics.orange_count,
                    'Grey': self.metrics.grey_count,
                    'Blue': self.metrics.blue_count,
                    'Purple': self.metrics.purple_count,
                    'Red': self.metrics.red_count
                }
                most_frequent = max(counts.items(), key=lambda x: x[1])
                self.metric_labels['most_frequent'][1].setText(
                    f"{most_frequent[0]} ({most_frequent[1]})"
                )
            
            # Update health trend
            if len(self.health_history) >= 2:
                trend = "↑" if self.health_history[-1] > self.health_history[-2] else "↓"
                self.metric_labels['trend'][1].setText(trend)
                
            # Update maintenance status
            if hasattr(self.metrics, 'predict_maintenance_need'):
                maintenance_info = self.metrics.predict_maintenance_need()
                if maintenance_info['maintenance_needed']:
                    self.metric_labels['maintenance'][1].setText(
                        f"Needed in {maintenance_info['estimated_hours_until_critical']}h"
                    )
                else:
                    self.metric_labels['maintenance'][1].setText("No maintenance needed")
            
            # Update pattern detection
            if hasattr(self.metrics, 'detect_color_patterns'):
                pattern = self.metrics.detect_color_patterns()
                self.metric_labels['pattern'][1].setText(str(pattern))
            
            # Calculate and update 15-minute average health
            if len(self.health_history) > 0:
                recent_values = list(self.health_history)[-180:]  # Last 15 minutes (assuming 5-second intervals)
                avg_health = sum(recent_values) / len(recent_values)
                self.metric_labels['avg_health'][1].setText(f"{avg_health:.1f}%")
                
                # Color coding for average health
                if avg_health < 90:
                    self.metric_labels['avg_health'][1].setStyleSheet("""
                        QLabel {
                            padding: 5px;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            color: #f44336;
                        }
                    """)
                else:
                    self.metric_labels['avg_health'][1].setStyleSheet("""
                        QLabel {
                            padding: 5px;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            color: #4caf50;
                        }
                    """)
                
                # Calculate and update UPH loss
                health_loss = 100 - avg_health
                uph_loss = health_loss * 2.12 if health_loss > 0 else 0
                self.metric_labels['uph_loss'][1].setText(f"{uph_loss:.1f}")
                
                # Color coding for UPH loss
                if uph_loss > 0:
                    self.metric_labels['uph_loss'][1].setStyleSheet("""
                        QLabel {
                            padding: 5px;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            color: #f44336;
                        }
                    """)
                else:
                    self.metric_labels['uph_loss'][1].setStyleSheet("""
                        QLabel {
                            padding: 5px;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            color: #4caf50;
                        }
                    """)
            
        except Exception as e:
            print(f"Error updating metrics: {e}")

class DataLogger:
    def __init__(self):
        self.db_path = os.getenv('DB_PATH', 'conveyor_metrics.db')
        self.model_dir = Path('ml_models')
        self.model_dir.mkdir(exist_ok=True)
        self.current_region_states = {}  # Track current color for each region
        
        self.anomaly_weights = {
            'red': 0.9,    
            'orange': 0.5, 
            'grey': 0.7,   
            'blue': 0.3,   
            'purple': 0.35 
        }
        
        # Add initialization delay
        time.sleep(1)  # 1 second delay
        self.initialize_database()

        self.api = ConveyorHealthAPI(self)
        self.api.start()
    
    def initialize_database(self):
        """Create database schema with proper order of operations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create and commit metrics table first
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conveyor_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    current_title TEXT,
                    orange_count INTEGER,
                    grey_count INTEGER,
                    blue_count INTEGER,
                    purple_count INTEGER,
                    red_count INTEGER,
                    conveyor_health REAL,
                    total_detections INTEGER,
                    detection_pattern TEXT,
                    maintenance_flag BOOLEAN
                )
            ''')
            conn.commit()  # Commit the table creation
            
            # Now create indices for metrics
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON conveyor_metrics(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_title 
                ON conveyor_metrics(current_title)
            ''')
            conn.commit()  # Commit the indices
            
            # Create and commit patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metrics_id INTEGER,
                    current_title TEXT,
                    pattern_type TEXT,
                    severity_score REAL,
                    timestamp DATETIME,
                    duration INTEGER,
                    FOREIGN KEY(metrics_id) REFERENCES conveyor_metrics(id)
                )
            ''')
            conn.commit()  # Commit the table creation
            
            # Create index for patterns
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_patterns_title 
                ON anomaly_patterns(current_title)
            ''')
            conn.commit()  # Commit the index
            
            # Create and commit health history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_title TEXT,
                    timestamp DATETIME,
                    health_value REAL,
                    moving_average REAL,
                    trend_direction TEXT,
                    anomaly_count INTEGER
                )
            ''')
            conn.commit()  # Commit the table creation
            
            # Create index for health history
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_health_timestamp 
                ON health_history(timestamp)
            ''')
    
            # Create region metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS region_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    region_title TEXT,
                    total_count INTEGER,
                    orange_count INTEGER,
                    grey_count INTEGER,
                    blue_count INTEGER,
                    purple_count INTEGER,
                    red_count INTEGER,
                    FOREIGN KEY(id) REFERENCES conveyor_metrics(id)
                )
            ''')
            conn.commit()
    
            # Create sequence tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    region_title TEXT,
                    sequence TEXT,
                    FOREIGN KEY(id) REFERENCES conveyor_metrics(id)
                )
            ''')
            conn.commit()  # Commit all remaining changes

    def update_region_state(self, region_title, sequence):
        """Update the current state of a region"""
        self.current_region_states[region_title] = {
            'timestamp': datetime.now(),
            'current_color': sequence
        }

    def log_metrics(self, metrics):
        """Log metrics with enhanced ML features and region tracking"""
        try:
            if metrics.current_title is None or metrics.current_title == "None" or metrics.current_title == "":
                return  # Exit the method without logging
            current_time = datetime.now()
            
            # Calculate derived features
            health_trend = self._calculate_health_trend(metrics.health_history)
            pattern = self._detect_pattern(metrics)
            maintenance_needed = self._evaluate_maintenance_need(metrics)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert main metrics
                cursor.execute('''
                    INSERT INTO conveyor_metrics (
                        timestamp,
                        current_title,
                        orange_count,
                        grey_count,
                        blue_count,
                        purple_count,
                        red_count,
                        conveyor_health,
                        total_detections,
                        detection_pattern,
                        maintenance_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_time,
                    metrics.current_title,
                    metrics.orange_count,
                    metrics.grey_count,
                    metrics.blue_count,
                    metrics.purple_count,
                    metrics.red_count,
                    metrics.conveyor_health,
                    metrics.total_detections,
                    pattern,
                    maintenance_needed
                ))
                
                metrics_id = cursor.lastrowid
                
                # Log pattern if significant
                if pattern != "NORMAL":
                    severity_score = self._calculate_severity_score(metrics)
                    cursor.execute('''
                        INSERT INTO anomaly_patterns (
                            metrics_id,
                            current_title,
                            pattern_type,
                            severity_score,
                            timestamp,
                            duration
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics_id,
                        metrics.current_title,
                        pattern,
                        severity_score,
                        current_time,
                        5  # Duration in seconds
                    ))
                
                # Update health history
                moving_avg = self._calculate_moving_average(metrics.health_history)
                trend_direction = "STABLE" if abs(health_trend) < 0.1 else "DECLINING" if health_trend < 0 else "IMPROVING"
                
                cursor.execute('''
                    INSERT INTO health_history (
                        current_title,
                        timestamp,
                        health_value,
                        moving_average,
                        trend_direction,
                        anomaly_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.current_title,
                    current_time,
                    metrics.conveyor_health,
                    moving_avg,
                    trend_direction,
                    metrics.total_detections
                ))
    
                # Log region-specific metrics
                if hasattr(metrics, 'anomaly_titles'):
                    for region_title, region_data in metrics.anomaly_titles.items():
                        cursor.execute('''
                            INSERT INTO region_metrics (
                                timestamp,
                                region_title,
                                total_count,
                                orange_count,
                                grey_count,
                                blue_count,
                                purple_count,
                                red_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            current_time,
                            region_title,
                            region_data['total_count'],
                            region_data['colors']['orange'],
                            region_data['colors']['grey'],
                            region_data['colors']['blue'],
                            region_data['colors']['purple'],
                            region_data['colors']['red']
                        ))
    
                        # Log sequence if available
                        if region_data['sequence']:
                            cursor.execute('''
                                INSERT INTO detection_sequences (
                                    timestamp,
                                    region_title,
                                    sequence
                                ) VALUES (?, ?, ?)
                            ''', (
                                current_time,
                                region_title,
                                str(region_data['sequence'][-1])  # Latest sequence
                            ))
    
                conn.commit()
        except Exception as e:
            print(f"Error logging metrics: {e}")
    

    def _calculate_severity_score(self, metrics):
        return sum([
            metrics.red_count * self.anomaly_weights['red'],
            metrics.orange_count * self.anomaly_weights['orange'],
            metrics.grey_count * self.anomaly_weights['grey'],
            metrics.blue_count * self.anomaly_weights['blue'],
            metrics.purple_count * self.anomaly_weights['purple']
        ])

    def _detect_pattern(self, metrics):
        patterns = []
        
        if metrics.red_count > 2:
            patterns.append(" ")
        if metrics.orange_count > 4:
            patterns.append(" ")
        if metrics.red_count > 0 and metrics.orange_count > 2:
            patterns.append(" ")
        if metrics.conveyor_health < 75:
            if metrics.blue_count > 3:
                patterns.append(" ")
            if metrics.purple_count > 3:
                patterns.append(" ")
                
        return "_".join(patterns) if patterns else "NORMAL"

    def _calculate_health_trend(self, health_history):
        if len(health_history) < 2:
            return 0.0
        health_list = list(health_history)
        return np.polyfit(range(len(health_list)), health_list, 1)[0]

    def _calculate_moving_average(self, health_history, window=30):
        health_list = list(health_history)
        if len(health_list) < window:
            return sum(health_list) / len(health_list)
        return sum(health_list[-window:]) / window

    def _evaluate_maintenance_need(self, metrics):
        return (metrics.conveyor_health < 75 or 
                metrics.red_count > 2 or 
                metrics.orange_count > 4)

    def generate_daily_report(self):
        """Generate a daily summary report"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                today = datetime.now().date()
                
                # Query for daily statistics
                query = '''
                    SELECT 
                        current_title,
                        COUNT(*) as total_records,
                        AVG(conveyor_health) as avg_health,
                        SUM(orange_count) as total_orange,
                        SUM(grey_count) as total_grey,
                        SUM(blue_count) as total_blue,
                        SUM(purple_count) as total_purple,
                        SUM(red_count) as total_red,
                        SUM(CASE WHEN maintenance_flag = 1 THEN 1 ELSE 0 END) as maintenance_flags
                    FROM conveyor_metrics
                    WHERE date(timestamp) = date('now')
                    GROUP BY current_title
                '''
                
                df = pd.read_sql_query(query, conn)
                
                # Save report
                report_path = f'daily_report_{today}.csv'
                df.to_csv(report_path, index=False)
                print(f"Daily report generated: {report_path}")
                
        except Exception as e:
            print(f"Error generating daily report: {e}")

class ConveyorHealthAPI:
    def __init__(self, data_logger):
        self.data_logger = data_logger
        self.flask_app = Flask(__name__, 
                                template_folder=os.path.abspath(os.path.dirname(__file__)),
                                static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'static')))        
        CORS(self.flask_app)
        self.setup_routes()

    def setup_routes(self):
        @self.flask_app.route('/')
        def root():
            return jsonify({
                'status': 'online',
                'endpoints': ['/health', '/metrics', '/patterns', '/history'],
                'timestamp': datetime.now().isoformat()
            })
        @self.flask_app.route('/dashboard')
        def dashboard():
            return render_template('index.html')
        
        @self.flask_app.route('/health')
        def get_health():
            try:
                with sqlite3.connect(self.data_logger.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT conveyor_health, total_detections, maintenance_flag, timestamp
                        FROM conveyor_metrics 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    ''')
                    result = cursor.fetchone()
                    
                    if result:
                        return jsonify({
                            'conveyor_health': result[0],
                            'total_detections': result[1],
                            'maintenance_flag': bool(result[2]),
                            'timestamp': result[3]
                        })
                    return jsonify({'error': 'No data available'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/metrics')
        def get_metrics():
            try:
                with sqlite3.connect(self.data_logger.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT orange_count, grey_count, blue_count, 
                               purple_count, red_count, current_title,
                               total_detections, timestamp
                        FROM conveyor_metrics 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    ''')
                    result = cursor.fetchone()
                    
                    if result:
                        return jsonify({
                            'orange_count': result[0],
                            'grey_count': result[1],
                            'blue_count': result[2],
                            'purple_count': result[3],
                            'red_count': result[4],
                            'current_title': result[5],
                            'total_detections': result[6],
                            'timestamp': result[7]
                        })
                    return jsonify({'error': 'No data available'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/patterns')
        def get_patterns():
            try:
                with sqlite3.connect(self.data_logger.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT current_title, pattern_type, severity_score, 
                               timestamp, duration
                        FROM anomaly_patterns
                        ORDER BY timestamp DESC
                        LIMIT 10
                    ''')
                    results = cursor.fetchall()
                    
                    patterns = [{
                        'title': row[0],
                        'pattern_type': row[1],
                        'severity_score': row[2],
                        'timestamp': row[3],
                        'duration': row[4]
                    } for row in results]
                    
                    return jsonify(patterns)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/history')
        def get_history():
            try:
                with sqlite3.connect(self.data_logger.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT timestamp, health_value, moving_average, 
                               trend_direction, anomaly_count
                        FROM health_history
                        ORDER BY timestamp DESC
                        LIMIT 100
                    ''')
                    results = cursor.fetchall()
                    
                    history = [{
                        'timestamp': row[0],
                        'health_value': row[1],
                        'moving_average': row[2],
                        'trend_direction': row[3],
                        'anomaly_count': row[4]
                    } for row in results]
                    
                    return jsonify(history)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def start(self, host='0.0.0.0', port=5965):
        def run_flask():
            self.flask_app.run(host=host, port=port, debug=False, use_reloader=False)

        self.api_thread = threading.Thread(target=run_flask, daemon=True)
        self.api_thread.start()
        print(f"\nAPI server started successfully!")
        print(f"Available endpoints:")
        print(f"  - http://{host}:{port}/")
        print(f"  - http://{host}:{port}/health")
        print(f"  - http://{host}:{port}/metrics")
        print(f"  - http://{host}:{port}/patterns")
        print(f"  - http://{host}:{port}/history")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #BDC3C7;
            border-radius: 5px;
            margin-top: 1ex;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
    """)
    
    ex = SelectionTool()
    ex.show()
    sys.exit(app.exec_())

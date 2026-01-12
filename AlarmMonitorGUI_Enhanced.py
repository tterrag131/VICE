#!/usr/bin/env python3
"""
OCR-Based Alarm Monitor with Enhanced Reliability (24/7 Operations)

Enhanced Features:
- System health monitoring and watchdog
- Comprehensive error recovery
- Performance monitoring and diagnostics
- Configuration validation
- Automatic backup and state persistence
- Enhanced logging and alerting
"""

import time
import numpy as np
import cv2
from mss import mss
import pytesseract
from PIL import Image
from datetime import datetime, timedelta
from typing import Set, List, Optional, Dict, Any, Tuple
import logging
import sys
import os
import shutil
import json
import threading
import queue
import traceback
import psutil
from pathlib import Path
from difflib import SequenceMatcher
import sqlite3
import re
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinter import font as tkFont
import gc
import weakref
from logging.handlers import RotatingFileHandler
import signal
import webbrowser


# =============================================================================
# RELIABILITY & HEALTH MONITORING SYSTEM
# =============================================================================

class MemoryManager:
    """Manages memory usage and prevents memory leaks for 24/7 operation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_gui_widgets = config.get("max_gui_widgets", 100)
        self.cleanup_interval = config.get("memory_cleanup_interval", 300)  # 5 minutes
        self.last_cleanup = time.time()
        self.widget_refs = weakref.WeakSet()
        
    def register_widget(self, widget):
        """Register a widget for memory tracking."""
        self.widget_refs.add(widget)
    
    def cleanup_if_needed(self, gui_instance):
        """Perform memory cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._perform_cleanup(gui_instance)
            self.last_cleanup = current_time
    
    def _perform_cleanup(self, gui_instance):
        """Perform comprehensive memory cleanup."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Cleanup old GUI widgets if too many exist
            total_widgets = len(gui_instance.tracked_alarm_widgets) + len(gui_instance.untracked_alarm_widgets)
            if total_widgets > self.max_gui_widgets:
                self._cleanup_old_widgets(gui_instance)
            
            # Log cleanup results
            logging.info(f"Memory cleanup: {collected} objects collected, {total_widgets} widgets active")
            
        except Exception as e:
            logging.error(f"Memory cleanup failed: {e}")
    
    def _cleanup_old_widgets(self, gui_instance):
        """Remove oldest widgets to prevent memory accumulation."""
        try:
            # Sort widgets by timestamp and remove oldest ones
            all_widgets = []
            
            # Collect all widgets with timestamps
            for key, widget_data in gui_instance.tracked_alarm_widgets.items():
                all_widgets.append((widget_data['timestamp'], key, 'tracked'))
            
            for key, widget_data in gui_instance.untracked_alarm_widgets.items():
                all_widgets.append((widget_data['timestamp'], key, 'untracked'))
            
            # Sort by timestamp (oldest first)
            all_widgets.sort(key=lambda x: x[0])
            
            # Remove oldest widgets if we exceed the limit
            widgets_to_remove = len(all_widgets) - self.max_gui_widgets
            if widgets_to_remove > 0:
                for i in range(widgets_to_remove):
                    timestamp, key, widget_type = all_widgets[i]
                    
                    if widget_type == 'tracked' and key in gui_instance.tracked_alarm_widgets:
                        gui_instance.tracked_alarm_widgets[key]['frame'].destroy()
                        del gui_instance.tracked_alarm_widgets[key]
                    elif widget_type == 'untracked' and key in gui_instance.untracked_alarm_widgets:
                        gui_instance.untracked_alarm_widgets[key]['frame'].destroy()
                        del gui_instance.untracked_alarm_widgets[key]
                
                logging.info(f"Cleaned up {widgets_to_remove} old widgets")
                
        except Exception as e:
            logging.error(f"Widget cleanup failed: {e}")

class DebugManager:
    """Enhanced debug capabilities for admin users."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug_enabled = config.get("debug_mode", False)
        self.admin_debug = config.get("admin_debug", False)
        self.debug_log_level = config.get("debug_log_level", "INFO")
        self.debug_output_file = config.get("debug_output_file", "debug_enhanced.log")
        
        # Setup enhanced debug logging
        self.debug_logger = self._setup_debug_logger()
        
        # Debug metrics
        self.debug_metrics = {
            "ocr_operations": 0,
            "alarm_events": 0,
            "gui_updates": 0,
            "database_operations": 0,
            "errors_logged": 0,
            "memory_cleanups": 0
        }
        
        # Performance tracking for debug
        self.performance_history = deque(maxlen=1000)
        
    def _setup_debug_logger(self) -> logging.Logger:
        """Setup enhanced debug logger with rotation."""
        debug_logger = logging.getLogger("enhanced_debug")
        debug_logger.setLevel(getattr(logging, self.debug_log_level.upper()))
        
        # Use rotating file handler to prevent log files from growing too large
        debug_handler = RotatingFileHandler(
            self.debug_output_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        debug_formatter = logging.Formatter(
            '%(asctime)s - DEBUG - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        debug_handler.setFormatter(debug_formatter)
        debug_logger.addHandler(debug_handler)
        
        return debug_logger
    
    def log_debug(self, category: str, message: str, data: Dict = None):
        """Enhanced debug logging with categorization."""
        if not self.debug_enabled:
            return
            
        try:
            # Increment metric counter
            if category in self.debug_metrics:
                self.debug_metrics[category] += 1
            
            # Format debug message
            debug_msg = f"[{category.upper()}] {message}"
            if data:
                debug_msg += f" | Data: {json.dumps(data, default=str, indent=2)}"
            
            self.debug_logger.info(debug_msg)
            
            # Admin debug - also print to console
            if self.admin_debug:
                print(f"🔧 DEBUG: {debug_msg}")
                
        except Exception as e:
            logging.error(f"Debug logging failed: {e}")
    
    def log_performance(self, operation: str, duration: float, success: bool, details: Dict = None):
        """Log performance metrics for debugging."""
        if not self.debug_enabled:
            return
            
        try:
            perf_data = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "duration": duration,
                "success": success,
                "details": details or {}
            }
            
            self.performance_history.append(perf_data)
            
            # Log slow operations
            if duration > 1.0:  # Operations taking more than 1 second
                self.log_debug("performance", f"Slow operation: {operation} took {duration:.3f}s", perf_data)
                
        except Exception as e:
            logging.error(f"Performance logging failed: {e}")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary for admin users."""
        try:
            recent_performance = list(self.performance_history)[-50:]  # Last 50 operations
            
            return {
                "debug_enabled": self.debug_enabled,
                "admin_debug": self.admin_debug,
                "metrics": self.debug_metrics.copy(),
                "recent_performance": recent_performance,
                "avg_operation_time": np.mean([p["duration"] for p in recent_performance]) if recent_performance else 0,
                "error_rate": self.debug_metrics["errors_logged"] / max(1, sum(self.debug_metrics.values())),
                "log_file": self.debug_output_file
            }
            
        except Exception as e:
            logging.error(f"Debug summary generation failed: {e}")
            return {"error": str(e)}

class SystemHealthMonitor:
    """Monitors system health, performance, and handles recovery operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_data = {
            "start_time": datetime.now(),
            "last_heartbeat": datetime.now(),
            "ocr_performance": deque(maxlen=100),  # Last 100 OCR operations
            "error_count": 0,
            "restart_count": 0,
            "memory_usage": deque(maxlen=60),  # Last 60 minutes
            "cpu_usage": deque(maxlen=60),
            "alarm_count_24h": 0,
            "last_backup": None,
            "system_status": "STARTING"
        }
        self.performance_thresholds = {
            "max_ocr_time": 2.0,  # seconds
            "max_memory_mb": 500,
            "max_cpu_percent": 80,
            "max_error_rate": 0.1  # 10% error rate
        }
        self.logger = self._setup_health_logger()
        
        # Initialize memory and debug managers
        self.memory_manager = MemoryManager(config)
        self.debug_manager = DebugManager(config)
        
    def _setup_health_logger(self) -> logging.Logger:
        """Setup dedicated health monitoring logger."""
        health_logger = logging.getLogger("health_monitor")
        health_logger.setLevel(logging.INFO)
        
        # Create health log file handler
        health_handler = logging.FileHandler('system_health.log', encoding='utf-8')
        health_formatter = logging.Formatter(
            '%(asctime)s - HEALTH - %(levelname)s - %(message)s'
        )
        health_handler.setFormatter(health_formatter)
        health_logger.addHandler(health_handler)
        
        return health_logger
    
    def heartbeat(self):
        """Update system heartbeat and collect metrics."""
        self.health_data["last_heartbeat"] = datetime.now()
        
        # Collect system metrics
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            self.health_data["memory_usage"].append(memory_mb)
            self.health_data["cpu_usage"].append(cpu_percent)
            
            # Check thresholds
            self._check_performance_thresholds(memory_mb, cpu_percent)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _check_performance_thresholds(self, memory_mb: float, cpu_percent: float):
        """Check if system metrics exceed thresholds."""
        if memory_mb > self.performance_thresholds["max_memory_mb"]:
            self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            
        if cpu_percent > self.performance_thresholds["max_cpu_percent"]:
            self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
    
    def record_ocr_performance(self, duration: float, success: bool):
        """Record OCR operation performance."""
        self.health_data["ocr_performance"].append({
            "timestamp": datetime.now(),
            "duration": duration,
            "success": success
        })
        
        if duration > self.performance_thresholds["max_ocr_time"]:
            self.logger.warning(f"Slow OCR operation: {duration:.3f}s")
    
    def record_error(self, error_type: str, error_msg: str):
        """Record system error for monitoring."""
        self.health_data["error_count"] += 1
        self.logger.error(f"{error_type}: {error_msg}")
        
        # Check error rate
        recent_errors = sum(1 for perf in self.health_data["ocr_performance"] 
                          if not perf["success"])
        total_operations = len(self.health_data["ocr_performance"])
        
        if total_operations > 10:  # Only check after some operations
            error_rate = recent_errors / total_operations
            if error_rate > self.performance_thresholds["max_error_rate"]:
                self.logger.critical(f"High error rate: {error_rate:.2%}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        uptime = datetime.now() - self.health_data["start_time"]
        
        # Calculate averages
        avg_memory = np.mean(self.health_data["memory_usage"]) if self.health_data["memory_usage"] else 0
        avg_cpu = np.mean(self.health_data["cpu_usage"]) if self.health_data["cpu_usage"] else 0
        
        # OCR performance stats
        ocr_times = [p["duration"] for p in self.health_data["ocr_performance"]]
        avg_ocr_time = np.mean(ocr_times) if ocr_times else 0
        
        return {
            "uptime_hours": uptime.total_seconds() / 3600,
            "system_status": self.health_data["system_status"],
            "error_count": self.health_data["error_count"],
            "restart_count": self.health_data["restart_count"],
            "avg_memory_mb": avg_memory,
            "avg_cpu_percent": avg_cpu,
            "avg_ocr_time": avg_ocr_time,
            "total_operations": len(self.health_data["ocr_performance"]),
            "last_heartbeat": self.health_data["last_heartbeat"]
        }

class ConfigurationValidator:
    """Validates system configuration and provides recommendations."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration and return (is_valid, issues_list)."""
        issues = []
        
        # Validate ROI settings
        roi = config.get("roi", {})
        if not all(key in roi for key in ["x", "y", "width", "height"]):
            issues.append("ROI configuration incomplete - missing coordinates")
        
        if roi.get("width", 0) < 100 or roi.get("height", 0) < 100:
            issues.append("ROI dimensions too small - may affect OCR accuracy")
        
        # Validate file paths
        translation_file = config.get("translation_file", "")
        if translation_file and not Path(translation_file).exists():
            issues.append(f"Translation file not found: {translation_file}")
        
        # Validate OCR settings
        ocr_settings = config.get("ocr_settings", {})
        if ocr_settings.get("psm_mode", 0) not in range(0, 14):
            issues.append("Invalid OCR PSM mode - should be 0-13")
        
        # Validate thresholds
        similarity = config.get("similarity_threshold", 0.85)
        if not 0.5 <= similarity <= 1.0:
            issues.append("Similarity threshold should be between 0.5 and 1.0")
        
        # Check system requirements
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception:
            issues.append("Tesseract OCR not properly installed or configured")
        
        return len(issues) == 0, issues

class BackupManager:
    """Handles automatic backup of database and configuration files."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.max_backups = 30  # Keep 30 days of backups
        
    def create_backup(self) -> bool:
        """Create backup of database and configuration."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup database
            db_file = Path(self.config.get("database_file", "jams.db"))
            if db_file.exists():
                backup_db = self.backup_dir / f"jams_backup_{timestamp}.db"
                shutil.copy2(db_file, backup_db)
            
            # Backup translations
            trans_file = Path(self.config.get("translation_file", "TRANSLATIONS.json"))
            if trans_file.exists():
                backup_trans = self.backup_dir / f"translations_backup_{timestamp}.json"
                shutil.copy2(trans_file, backup_trans)
            
            # Backup configuration
            config_backup = self.backup_dir / f"config_backup_{timestamp}.json"
            with open(config_backup, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            self._cleanup_old_backups()
            return True
            
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backup files."""
        try:
            backup_files = list(self.backup_dir.glob("*_backup_*.db"))
            backup_files.extend(self.backup_dir.glob("*_backup_*.json"))
            
            # Sort by modification time and keep only recent ones
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_backup in backup_files[self.max_backups:]:
                old_backup.unlink()
                
        except Exception as e:
            logging.warning(f"Backup cleanup failed: {e}")

# =============================================================================
# ENHANCED EVENT LOGGER WITH RELIABILITY FEATURES
# =============================================================================

class EnhancedEventLogger:
    """Enhanced database logger with backup, recovery, and health monitoring."""
    
    def __init__(self, db_path: str, health_monitor: SystemHealthMonitor):
        self.db_path = db_path
        self.health_monitor = health_monitor
        self.conn = None
        self.backup_manager = None
        self.last_backup = datetime.now()
        self.backup_interval = timedelta(hours=6)  # Backup every 6 hours
        
        try:
            self._establish_connection()
            self.setup()
            logging.info(f"Enhanced database connection established at {db_path}")
        except Exception as e:
            self.health_monitor.record_error("DATABASE_INIT", str(e))
            raise
    
    def _establish_connection(self):
        """Establish database connection with retry logic and performance optimizations."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
                
                # Performance and reliability optimizations
                self.conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
                self.conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety and performance
                self.conn.execute("PRAGMA cache_size=10000")  # Increase cache size
                self.conn.execute("PRAGMA temp_store=memory")  # Store temp tables in memory
                self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
                self.conn.execute("PRAGMA optimize")  # Optimize database
                
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retry
    
    def setup(self):
        """Creates tables and handles schema migration with enhanced error handling."""
        try:
            # Create main events table
            sql_create_table = """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds INTEGER,
                raw_text TEXT NOT NULL,
                friendly_text TEXT,
                severity TEXT DEFAULT 'NORMAL',
                acknowledged BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Create system health table
            sql_create_health = """
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                status TEXT DEFAULT 'OK'
            );
            """
            
            c = self.conn.cursor()
            c.execute(sql_create_table)
            c.execute(sql_create_health)
            
            # Check and add missing columns
            self._migrate_schema()
            
            self.conn.commit()
            logging.info("Enhanced database tables are ready.")
            
        except Exception as e:
            self.health_monitor.record_error("DATABASE_SETUP", str(e))
            raise
    
    def _migrate_schema(self):
        """Handle database schema migrations."""
        c = self.conn.cursor()
        
        # Get existing columns
        c.execute("PRAGMA table_info(events)")
        existing_columns = {row[1] for row in c.fetchall()}
        
        # Add missing columns including macro_area and alarm_key
        required_columns = {
            'friendly_text': 'TEXT',
            'severity': 'TEXT DEFAULT "NORMAL"',
            'acknowledged': 'BOOLEAN DEFAULT FALSE',
            'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'macro_area': 'TEXT',
            'alarm_key': 'TEXT'
        }
        
        for column, definition in required_columns.items():
            if column not in existing_columns:
                try:
                    c.execute(f"ALTER TABLE events ADD COLUMN {column} {definition}")
                    logging.info(f"Added column {column} to events table")
                except Exception as e:
                    logging.warning(f"Failed to add column {column}: {e}")
    
    def log_new_event(self, key: str, raw_text: str, friendly_text: str, severity: str = "NORMAL", macro_area: str = None):
        """Enhanced event logging with error handling and backup management using 6-digit key."""
        try:
            sql_insert = """
            INSERT INTO events (start_time, raw_text, friendly_text, severity, macro_area, alarm_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            c = self.conn.cursor()
            c.execute(sql_insert, (datetime.now(), raw_text, friendly_text, severity, macro_area, key))
            self.conn.commit()
            
            # Check if backup is needed
            self._check_backup_schedule()
            
        except Exception as e:
            self.health_monitor.record_error("DATABASE_INSERT", str(e))
            self._attempt_recovery()
    
    def log_cleared_event(self, key: str, clear_time: datetime) -> Optional[tuple]:
        """
        Finds the active event using 6-digit key, updates its end time and duration,
        and returns the friendly_text and duration.
        """
        sql_find = """
        SELECT id, start_time, friendly_text FROM events
        WHERE alarm_key = ? AND end_time IS NULL
        ORDER BY start_time DESC
        LIMIT 1
        """
        
        sql_update = """
        UPDATE events
        SET end_time = ?, duration_seconds = ?
        WHERE id = ?
        """
        
        try:
            c = self.conn.cursor()
            c.execute(sql_find, (key,))
            result = c.fetchone()
            
            if result:
                event_id, start_time_str, friendly_text = result
                start_time = datetime.fromisoformat(start_time_str)
                duration = int((clear_time - start_time).total_seconds())
                
                c.execute(sql_update, (clear_time, duration, event_id))
                self.conn.commit()
                return friendly_text, duration
            else:
                logging.warning(f"Could not find matching NEW event to clear for key: {key}")
                return None, None
                
        except Exception as e:
            self.health_monitor.record_error("DATABASE_CLEAR", str(e))
            return None, None
    
    def _check_backup_schedule(self):
        """Check if it's time for a scheduled backup."""
        if datetime.now() - self.last_backup > self.backup_interval:
            if self.backup_manager:
                if self.backup_manager.create_backup():
                    self.last_backup = datetime.now()
                    logging.info("Scheduled database backup completed")
    
    def _attempt_recovery(self):
        """Attempt to recover from database errors."""
        try:
            if self.conn:
                self.conn.close()
            
            # Wait a moment and try to reconnect
            time.sleep(1)
            self._establish_connection()
            logging.info("Database connection recovered")
            
        except Exception as e:
            self.health_monitor.record_error("DATABASE_RECOVERY", str(e))
    
    def get_alarm_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alarm statistics for reporting."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            c = self.conn.cursor()
            
            # Total alarms
            c.execute("SELECT COUNT(*) FROM events WHERE start_time > ?", (cutoff_time,))
            total_alarms = c.fetchone()[0]
            
            # Average duration for cleared alarms
            c.execute("""
                SELECT AVG(duration_seconds) FROM events 
                WHERE start_time > ? AND end_time IS NOT NULL
            """, (cutoff_time,))
            avg_duration = c.fetchone()[0] or 0
            
            # Most frequent alarms
            c.execute("""
                SELECT friendly_text, COUNT(*) as count 
                FROM events 
                WHERE start_time > ? 
                GROUP BY friendly_text 
                ORDER BY count DESC 
                LIMIT 10
            """, (cutoff_time,))
            frequent_alarms = c.fetchall()
            
            return {
                "total_alarms": total_alarms,
                "avg_duration_seconds": avg_duration,
                "frequent_alarms": frequent_alarms,
                "period_hours": hours
            }
            
        except Exception as e:
            self.health_monitor.record_error("DATABASE_STATS", str(e))
            return {}
    
    def get_active_jams_by_macro(self) -> List[Tuple[str, int]]:
        """Get total jams grouped by macro area for dashboard display."""
        try:
            c = self.conn.cursor()
            
            # Get total jams by macro area (all events, not just active)
            c.execute("""
                SELECT macro_area, COUNT(*) as count 
                FROM events 
                WHERE macro_area IS NOT NULL 
                GROUP BY macro_area 
                ORDER BY count DESC
            """)
            
            macro_stats = c.fetchall()
            return macro_stats
            
        except Exception as e:
            self.health_monitor.record_error("DATABASE_MACRO_STATS", str(e))
            return []
    
    def reset_jam_counts(self) -> bool:
        """Reset all jam counts by clearing the events table - Admin function."""
        try:
            c = self.conn.cursor()
            
            # Create backup before clearing
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_table_name = f"events_backup_{backup_timestamp}"
            
            # Create backup table with current data
            c.execute(f"""
                CREATE TABLE {backup_table_name} AS 
                SELECT * FROM events
            """)
            
            # Clear the events table
            c.execute("DELETE FROM events")
            
            # Reset the auto-increment counter
            c.execute("DELETE FROM sqlite_sequence WHERE name='events'")
            
            self.conn.commit()
            
            logging.info(f"Jam counts reset successfully. Backup created as {backup_table_name}")
            return True
            
        except Exception as e:
            self.health_monitor.record_error("DATABASE_RESET", str(e))
            logging.error(f"Failed to reset jam counts: {e}")
            return False
    
    def close(self):
        """Enhanced cleanup with final backup."""
        try:
            if self.backup_manager:
                self.backup_manager.create_backup()
            
            if self.conn:
                self.conn.commit()
                self.conn.close()
                
            logging.info("Enhanced database connection closed with backup")
            
        except Exception as e:
            logging.error(f"Error during database cleanup: {e}")

# =============================================================================
# ENHANCED ALARM MONITOR THREAD WITH RELIABILITY
# =============================================================================

class EnhancedAlarmMonitorThread(threading.Thread):
    """Enhanced monitoring thread with comprehensive error handling and recovery."""
    
    def __init__(self, config: Dict[str, Any], gui_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        
        # Use the provided GUI queue directly
        self.gui_queue = gui_queue
        
        # Thread safety locks
        self._state_lock = threading.RLock()  # For alarm state modifications
        self._db_lock = threading.Lock()      # For database operations
        
        # Initialize reliability components
        self.health_monitor = SystemHealthMonitor(config)
        self.backup_manager = BackupManager(config)
        
        # Validate configuration
        is_valid, issues = ConfigurationValidator.validate_config(config)
        if not is_valid:
            for issue in issues:
                logging.warning(f"Configuration issue: {issue}")
        
        # Initialize core components with error handling
        try:
            # Import Translator from the same directory
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from AlarmMonitorGUI import Translator
            
            self.translator = Translator(
                translation_path=config.get("translation_file", "TRANSLATIONS.json"),
                global_similarity_threshold=config.get("global_keyword_similarity", 0.85)
            )
            self.logger = EnhancedEventLogger(
                config.get("database_file", "jams.db"), 
                self.health_monitor
            )
            self.logger.backup_manager = self.backup_manager
            
        except Exception as e:
            self.health_monitor.record_error("INIT_COMPONENTS", str(e))
            raise
        
        # Monitoring state - REFACTORED to use 6-digit keys
        self.current_active_alarms: Set[str] = set()  # Now stores 6-digit keys
        self.active_alarm_raw_map: Dict[str, str] = {}  # Maps 6-digit key to full raw text
        self.pending_clear_alarms: Dict[str, float] = {}
        self.sct = None
        self.running = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Performance monitoring
        self.loop_times = deque(maxlen=100)
        self.ocr_times = deque(maxlen=100)
        
        self.health_monitor.health_data["system_status"] = "INITIALIZED"
        logging.info("Enhanced AlarmMonitorThread initialized with reliability features")
    
    def run(self):
        """Enhanced monitoring loop with comprehensive error handling."""
        logging.info("Starting enhanced alarm monitoring thread...")
        self.running = True
        self.health_monitor.health_data["system_status"] = "RUNNING"
        
        try:
            self.sct = mss()
            
            while self.running:
                loop_start = time.perf_counter()
                
                try:
                    # Update system health
                    self.health_monitor.heartbeat()
                    
                    # Main monitoring logic
                    self._monitoring_cycle()
                    
                    # Reset error counter on successful cycle
                    self.consecutive_errors = 0
                    
                except Exception as e:
                    self._handle_monitoring_error(e)
                
                # Performance tracking
                loop_duration = time.perf_counter() - loop_start
                self.loop_times.append(loop_duration)
                
                # Adaptive sleep based on performance
                target_interval = self.config['scan_interval']
                sleep_time = max(0.1, target_interval - loop_duration)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.health_monitor.record_error("CRITICAL_MONITORING", str(e))
            self.health_monitor.health_data["system_status"] = "CRITICAL_ERROR"
            logging.critical(f"Critical monitoring error: {e}")
            
        finally:
            self._cleanup()
    
    def _monitoring_cycle(self):
        """Single monitoring cycle with error handling."""
        # 1. Capture screen
        captured_image = self._safe_screen_capture()
        if captured_image is None:
            return
        
        # 2. Process OCR
        ocr_start = time.perf_counter()
        try:
            raw_text = self._safe_ocr_processing(captured_image)
            ocr_duration = time.perf_counter() - ocr_start
            
            self.health_monitor.record_ocr_performance(ocr_duration, True)
            self.ocr_times.append(ocr_duration)
            
        except Exception as e:
            ocr_duration = time.perf_counter() - ocr_start
            self.health_monitor.record_ocr_performance(ocr_duration, False)
            raise
        
        # 3. Process alarms - REFACTORED for 6-digit key system
        parsed_alarms = self._parse_alarm_lines(raw_text)
        self._process_alarm_changes(parsed_alarms)
        
        # 4. Send performance data to GUI
        self._send_performance_update()
    
    def _safe_screen_capture(self) -> Optional[np.ndarray]:
        """Screen capture with error handling and recovery."""
        try:
            if not self.sct:
                self.sct = mss()
            
            monitor_coords = {
                "top": self.config['roi']['y'],
                "left": self.config['roi']['x'],
                "width": self.config['roi']['width'],
                "height": self.config['roi']['height']
            }
            
            screenshot = self.sct.grab(monitor_coords)
            img_array = np.array(screenshot)
            return cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            
        except Exception as e:
            self.health_monitor.record_error("SCREEN_CAPTURE", str(e))
            
            # Attempt to reinitialize screen capture
            try:
                if self.sct:
                    self.sct.close()
                self.sct = mss()
                logging.info("Screen capture reinitialized")
            except Exception:
                pass
            
            return None
    
    def _safe_ocr_processing(self, image: np.ndarray) -> str:
        """OCR processing with error handling."""
        try:
            # Preprocess image (existing logic)
            processed_image = self._preprocess_image(image)
            
            # Show debug windows if debug mode is enabled
            if self.config.get("debug_mode", False):
                self._show_debug_windows(image, processed_image)
            
            # OCR extraction
            pil_image = Image.fromarray(processed_image)
            ocr_config = self._build_ocr_config()
            
            extracted_text = pytesseract.image_to_string(pil_image, config=ocr_config)
            return extracted_text.strip()
            
        except Exception as e:
            self.health_monitor.record_error("OCR_PROCESSING", str(e))
            return ""
    
    def _show_debug_windows(self, original_image: np.ndarray, processed_image: np.ndarray):
        """Show debug windows with original and processed images."""
        try:
            # Show original captured image
            cv2.imshow("OCR Debug - Original Capture", original_image)
            
            # Show processed image (what OCR actually sees)
            cv2.imshow("OCR Debug - Processed Image", processed_image)
            
            # Non-blocking window update
            cv2.waitKey(1)
            
        except Exception as e:
            logging.warning(f"Failed to show debug windows: {e}")
    
    def _close_debug_windows(self):
        """Close debug windows when debug mode is disabled."""
        try:
            cv2.destroyWindow("OCR Debug - Original Capture")
            cv2.destroyWindow("OCR Debug - Processed Image")
        except Exception as e:
            logging.warning(f"Failed to close debug windows: {e}")
    
    def _handle_monitoring_error(self, error: Exception):
        """Handle monitoring errors with escalating recovery strategies."""
        self.consecutive_errors += 1
        error_msg = f"Monitoring error #{self.consecutive_errors}: {error}"
        
        if self.consecutive_errors <= 3:
            logging.warning(error_msg)
        elif self.consecutive_errors <= self.max_consecutive_errors:
            logging.error(error_msg)
            # Attempt component reinitialization
            self._attempt_component_recovery()
        else:
            logging.critical(f"Too many consecutive errors ({self.consecutive_errors}). System may need restart.")
            self.health_monitor.health_data["system_status"] = "DEGRADED"
            
            # Send critical error to GUI
            self.gui_queue.put(("CRITICAL_ERROR", {
                "error_count": self.consecutive_errors,
                "last_error": str(error),
                "recommendation": "System restart recommended"
            }))
    
    def _attempt_component_recovery(self):
        """Attempt to recover system components."""
        try:
            # Reinitialize screen capture
            if self.sct:
                self.sct.close()
            self.sct = mss()
            
            # Test database connection
            self.logger._establish_connection()
            
            logging.info("Component recovery attempted")
            
        except Exception as e:
            logging.error(f"Component recovery failed: {e}")
    
    def _send_performance_update(self):
        """Send comprehensive performance data to GUI dashboard."""
        health_status = self.health_monitor.get_health_status()
        
        # Get 24-hour alarm statistics
        stats_24h: Dict = self.logger.get_alarm_statistics(hours=24)
        
        # Get macro area statistics for dashboard
        macro_stats: List = self.logger.get_active_jams_by_macro()
        
        # Calculate recent performance metrics
        recent_loop_times = list(self.loop_times)[-10:]  # Last 10 loops
        recent_ocr_times = list(self.ocr_times)[-10:]
        
        performance_data = {
            "health_status": health_status,
            "avg_loop_time": np.mean(recent_loop_times) if recent_loop_times else 0,
            "avg_ocr_time": np.mean(recent_ocr_times) if recent_ocr_times else 0,
            "consecutive_errors": self.consecutive_errors,
            "active_alarms": len(self.current_active_alarms),
            "pending_alarms": len(self.pending_clear_alarms),
            # Dashboard-specific data
            "avg_clear_time_24h": stats_24h.get("avg_duration_seconds", 0),
            "macro_area_counts": macro_stats,
            "last_scan_time": datetime.now()
        }
        
        self.gui_queue.put(("ENHANCED_STATS", performance_data))
    
    def _cleanup(self):
        """Enhanced cleanup with state preservation and resource leak prevention."""
        try:
            self.health_monitor.health_data["system_status"] = "SHUTTING_DOWN"
            
            # Close OpenCV debug windows to prevent resource leaks
            self._close_debug_windows()
            
            # Close screen capture resources
            if self.sct:
                try:
                    self.sct.close()
                except Exception as e:
                    logging.warning(f"Error closing screen capture: {e}")
                finally:
                    self.sct = None
            
            # Close database connections
            if self.logger:
                try:
                    self.logger.close()
                except Exception as e:
                    logging.warning(f"Error closing database: {e}")
            
            # Save final state
            self._save_system_state()
            
            # Force garbage collection
            gc.collect()
            
            logging.info("Enhanced alarm monitor thread stopped cleanly")
            
        except Exception as e:
            logging.error(f"Error during enhanced cleanup: {e}")
    
    def _save_system_state(self):
        """Save current system state for recovery."""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "active_alarms": list(self.current_active_alarms),
                "pending_alarms": {k: v for k, v in self.pending_clear_alarms.items()},
                "health_data": self.health_monitor.health_data,
                "consecutive_errors": self.consecutive_errors
            }
            
            with open("system_state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Failed to save system state: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal OCR performance."""
        preprocessing = self.config['preprocessing']
        processed_image = image
        
        if preprocessing['grayscale']:
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        scale_factor = preprocessing.get("scaling_factor", 1.0)
        if scale_factor > 1.0:
            width = int(processed_image.shape[1] * scale_factor)
            height = int(processed_image.shape[0] * scale_factor)
            processed_image = cv2.resize(processed_image, (width, height), interpolation=cv2.INTER_CUBIC)

        manual_threshold = preprocessing.get("binary_threshold", 127)
        _, processed_image = cv2.threshold(
            processed_image, manual_threshold, 255, cv2.THRESH_BINARY
        )
        
        # Static "white-out" masking to remove dotted lines
        if processed_image is not None:
            try:
                mask_coords_y = self.config.get("line_mask_y_coords", [])
                if mask_coords_y:
                    h, w = processed_image.shape
                    for y_center in mask_coords_y:
                        y1 = max(0, y_center - 2)
                        y2 = min(h, y_center + 3)
                        cv2.rectangle(processed_image, (0, y1), (w, y2), 255, -1)
            except Exception as e:
                logging.warning(f"Failed to apply static mask: {e}")
                
        return processed_image
    
    def _build_ocr_config(self) -> str:
        """Build Tesseract configuration string."""
        ocr_settings = self.config['ocr_settings']
        whitelist = ocr_settings.get("whitelist")
        config_parts = [
            f"--oem {ocr_settings['oem_mode']}",
            f"--psm {ocr_settings['psm_mode']}",
            f"-l {ocr_settings['language']}",
        ]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        return ' '.join(config_parts)
    
    def _parse_alarm_lines(self, raw_text: str) -> Dict[str, str]:
        """Parse raw OCR text and extract 6-digit keys mapped to full raw text."""
        if not raw_text:
            return {}
        
        lines = raw_text.split('\n')
        parsed_alarms: Dict[str, str] = {}
        
        for line in lines:
            cleaned_line = line.strip()
            
            # Filter out empty lines
            if len(cleaned_line) < 8:
                continue
            
            # Filter out "noise" lines
            is_noise = False
            for noise_phrase in self.config.get("noise_blocklist", []):
                if noise_phrase in cleaned_line:
                    is_noise = True
                    break
            if is_noise:
                continue
            
            # Extract 6-digit key using regex
            match = re.search(r'(\d{6})', cleaned_line)
            if match:
                key = match.group(1)
                parsed_alarms[key] = cleaned_line
            
        return parsed_alarms
    
    def _process_alarm_changes(self, parsed_alarms: Dict[str, str]):
        """Process alarm changes using 6-digit keys with fuzzy matching and grace period logic."""
        loop_time = time.perf_counter()
        grace_period = self.config.get("clear_grace_period", 3.0)
        
        # Extract current alarm keys from parsed alarms
        current_alarm_keys: Set[str] = set(parsed_alarms.keys())
        
        final_new_alarms: Set[str] = set()
        final_cleared_alarms: Set[str] = set()
        
        unmatched_on_screen = current_alarm_keys.copy()
        unmatched_previously_active = self.current_active_alarms.copy()

        # Direct key matching (no fuzzy matching needed for 6-digit keys)
        for key in list(unmatched_on_screen):
            if key in unmatched_previously_active:
                unmatched_on_screen.remove(key)
                unmatched_previously_active.remove(key)
                # Update raw text mapping
                self.active_alarm_raw_map[key] = parsed_alarms[key]
                # Remove from pending clear if it was there
                if key in self.pending_clear_alarms:
                    del self.pending_clear_alarms[key]

        # New alarms (keys not previously active)
        for key in unmatched_on_screen:
            final_new_alarms.add(key)
            self.current_active_alarms.add(key)
            self.active_alarm_raw_map[key] = parsed_alarms[key]

        # Potentially cleared alarms (keys that were active but not seen)
        for key in unmatched_previously_active:
            if key not in self.pending_clear_alarms:
                self.pending_clear_alarms[key] = loop_time
                # Don't remove from current_active_alarms yet - wait for grace period

        # Process grace period for cleared alarms
        for key in list(self.pending_clear_alarms.keys()):
            time_disappeared = self.pending_clear_alarms[key]
            if loop_time - time_disappeared > grace_period:
                final_cleared_alarms.add(key)
                # Now remove from current_active_alarms after grace period
                if key in self.current_active_alarms:
                    self.current_active_alarms.remove(key)
                del self.pending_clear_alarms[key]
        
        # Process new and cleared alarms
        self._process_new_alarms(final_new_alarms, parsed_alarms)
        self._process_cleared_alarms(final_cleared_alarms)
    
    def _process_new_alarms(self, new_alarms: Set[str], parsed_alarms: Dict[str, str]):
        """Process new alarms using 6-digit keys with translation and logging."""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        debug_mode = self.config.get("debug_mode", False)

        for key in new_alarms:
            try:
                # Get the full raw text for this key
                raw_text: str = parsed_alarms[key]
                
                # Translate using the 6-digit key
                translation_result = self.translator.translate(key)
                
                # For tracked alarms, we need to get the macro area from TRANSLATIONS.json directly
                macro_area = 'Unknown'
                is_critical = False
                if key in self.translator.translations:
                    translation_data = self.translator.translations[key]
                    if isinstance(translation_data, dict):
                        macro_area = translation_data.get('macro_area', 'Unknown')
                        is_critical = translation_data.get('critical', False)
                
                # Handle both string and dictionary returns from translator
                # Handle translation result
                # Priority: If key is in translations, use raw data to enforce our format
                if key in self.translator.translations and isinstance(self.translator.translations[key], dict):
                    data = self.translator.translations[key]
                    macro_area = data.get('macro_area', 'Unknown')
                    micro_area = data.get('micro_area', '')
                    descriptor = data.get('descriptor', 'Unknown')
                    is_critical = data.get('critical', False)
                    
                    # Format display as: macro_area + micro_area + descriptor (complete information)
                    parts = []
                    if macro_area and macro_area != 'Unknown':
                        parts.append(str(macro_area))
                    if micro_area and str(micro_area).strip():
                        parts.append(str(micro_area).strip())
                    if descriptor and descriptor != 'Unknown':
                        parts.append(str(descriptor))
                    
                    # Join all parts with spaces for complete information display
                    # User requested format: Macro Micro Descriptor     BED: Code
                    base_text = ' '.join(parts) if parts else 'Unknown'
                    friendly_text = f"{base_text}     BED: {key}"
                    
                    is_tracked = True
                    
                elif isinstance(translation_result, dict):
                    # Fallback for dict result if not in translations (unlikely given logic above)
                    macro_area: str = translation_result.get('macro_area', 'Unknown')
                    micro_area: str = translation_result.get('micro_area', '')
                    descriptor: str = translation_result.get('descriptor', 'Unknown')
                    
                    parts = []
                    if macro_area and macro_area != 'Unknown': parts.append(str(macro_area))
                    if micro_area and str(micro_area).strip(): parts.append(str(micro_area).strip())
                    if descriptor and descriptor != 'Unknown': parts.append(str(descriptor))
                    
                    base_text = ' '.join(parts) if parts else 'Unknown'
                    friendly_text = f"{base_text}     BED: {key}"
                    is_tracked = True

                elif isinstance(translation_result, str):
                    # Check if it's an untracked string (starts with ⚠️ UNTRACKED:)
                    if translation_result.startswith("⚠️ UNTRACKED:"):
                        friendly_text = translation_result
                        macro_area = 'Unknown'
                        is_tracked = False
                    else:
                        # This is a tracked alarm that returned a formatted string
                        friendly_text = translation_result
                        # macro_area already extracted above from translations
                        is_tracked = True
                else:
                    friendly_text: str = f"⚠️ UNTRACKED: {key}"
                    macro_area: str = 'Unknown'
                    is_tracked = False
                
                if friendly_text:
                    # Determine severity based on tracking status
                    severity = "HIGH" if is_tracked else "LOW"
                    
                    # Only log to database with valid macro_area (not 'Unknown')
                    db_macro_area = macro_area if macro_area != 'Unknown' else None
                    self.logger.log_new_event(key, raw_text, friendly_text, severity, db_macro_area)
                    
                    # Send to GUI with timestamp for timer
                    self.gui_queue.put(("NEW_ALARM", {
                        "text": friendly_text,
                        "is_tracked": is_tracked,
                        "raw_text": raw_text,
                        "severity": severity,
                        "timestamp": timestamp,
                        "key": key,
                        "macro_area": macro_area,
                        "is_critical": is_critical
                    }))
                    
                    # Terminal output with debug info
                    message = f"NEW: {friendly_text} [TRACKED: {is_tracked}]"
                    print(f"[{timestamp_str}] {message}")
                    logging.info(message)
                    
            except Exception as e:
                self.health_monitor.record_error("ALARM_PROCESSING", str(e))
    
    def _process_cleared_alarms(self, cleared_alarms: Set[str]):
        """Process cleared alarms using 6-digit keys with logging."""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        for key in cleared_alarms:
            try:
                # Get the original raw text from the state map and remove it
                raw_text: str = self.active_alarm_raw_map.pop(key, f"Unknown Cleared Key: {key}")
                
                # Log cleared event using 6-digit key to find and update the correct database entry
                friendly_text, duration = self.logger.log_cleared_event(key, timestamp)
                
                if friendly_text:
                    # Send to GUI
                    self.gui_queue.put(("CLEAR_ALARM", {
                        "text": friendly_text,
                        "duration": duration,
                        "timestamp": timestamp_str,
                        "key": key
                    }))
                    
                    # Terminal output
                    message = f"CLEARED: {friendly_text} (Duration: {duration}s)"
                    print(f"[{timestamp_str}] {message}")
                    logging.info(message)
                    
            except Exception as e:
                self.health_monitor.record_error("CLEAR_PROCESSING", str(e))

# =============================================================================
# ENHANCED GUI WITH RELIABILITY DASHBOARD
# =============================================================================

class EnhancedAlarmMonitorGUI:
    """Enhanced GUI with system health monitoring and reliability features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gui_queue = queue.Queue()
        self.active_alarms = []
        self.health_data = {}
        
        # Create enhanced main window
        self.root = tk.Tk()
        self.root.title("Enhanced Alarm Monitor - 24/7 Operations")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Setup enhanced UI
        self.setup_enhanced_styles()
        self.create_enhanced_widgets()
        
        # Start enhanced monitoring thread
        self.monitor_thread = EnhancedAlarmMonitorThread(config, self.gui_queue)
        self.monitor_thread.start()
        
        # Start GUI update loop
        self.update_gui()
        
        logging.info("Enhanced AlarmMonitorGUI initialized")
    
    def setup_enhanced_styles(self):
        """Setup enhanced UI styles."""
        self.fonts = {
            'title': tkFont.Font(family="Arial", size=16, weight="bold"),
            'subtitle': tkFont.Font(family="Arial", size=12, weight="bold"),
            'normal': tkFont.Font(family="Arial", size=10),
            'mono': tkFont.Font(family="Consolas", size=10)
        }
        
        # Enhanced color scheme
        self.colors = {
            'bg_primary': '#2b2b2b',
            'bg_secondary': '#3b3b3b',
            'bg_tertiary': '#4b4b4b',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'accent_green': '#4CAF50',
            'accent_red': '#f44336',
            'accent_orange': '#ff9500',
            'accent_blue': '#2196F3'
        }
    
    def create_enhanced_widgets(self):
        """Create enhanced GUI with dashboard layout."""
        # Main horizontal container
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: Active Alarms (60% width)
        self.alarms_frame = tk.Frame(main_container, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=2)
        self.alarms_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right side: Dashboard (40% width)
        self.dashboard_frame = tk.Frame(main_container, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=2)
        self.dashboard_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        self.dashboard_frame.configure(width=400)
        
        # Create the content for both sides
        self._create_alarms_section()
        self._create_dashboard_section()
        
        # Status bar at bottom
        self._create_status_bar()
    
    def _create_alarms_section(self):
        """Create the active alarms section (left side) with separate tracked and untracked sections."""
        # Main title
        main_title_label = tk.Label(
            self.alarms_frame, 
            text="🚨 Active Alarms", 
            font=self.fonts['title'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary']
        )
        main_title_label.pack(pady=(10, 5))
        
        # Tracked Alarms Section (75% of space)
        tracked_frame = tk.Frame(self.alarms_frame, bg=self.colors['bg_secondary'])
        tracked_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        
        tracked_title = tk.Label(
            tracked_frame, 
            text="📍 Tracked Jams", 
            font=self.fonts['subtitle'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['accent_green']
        )
        tracked_title.pack(pady=(0, 5))
        
        # Scrollable frame for tracked alarms
        self.tracked_canvas = tk.Canvas(
            tracked_frame, 
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )
        self.tracked_scrollbar = ttk.Scrollbar(
            tracked_frame, 
            orient="vertical", 
            command=self.tracked_canvas.yview
        )
        self.tracked_scrollable_frame = tk.Frame(self.tracked_canvas, bg=self.colors['bg_secondary'])
        
        self.tracked_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.tracked_canvas.configure(scrollregion=self.tracked_canvas.bbox("all"))
        )
        
        self.tracked_canvas.create_window((0, 0), window=self.tracked_scrollable_frame, anchor="nw")
        self.tracked_canvas.configure(yscrollcommand=self.tracked_scrollbar.set)
        
        self.tracked_canvas.pack(side="left", fill="both", expand=True)
        self.tracked_scrollbar.pack(side="right", fill="y")
        
        # Untracked Alarms Section (25% of space)
        untracked_frame = tk.Frame(self.alarms_frame, bg=self.colors['bg_secondary'])
        untracked_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        untracked_frame.configure(height=150)  # Fixed smaller height
        
        untracked_title = tk.Label(
            untracked_frame, 
            text="❓ Untracked", 
            font=self.fonts['normal'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_secondary']
        )
        untracked_title.pack(pady=(0, 3))
        
        # Scrollable frame for untracked alarms
        self.untracked_canvas = tk.Canvas(
            untracked_frame, 
            bg=self.colors['bg_secondary'],
            highlightthickness=0,
            height=120
        )
        self.untracked_scrollbar = ttk.Scrollbar(
            untracked_frame, 
            orient="vertical", 
            command=self.untracked_canvas.yview
        )
        self.untracked_scrollable_frame = tk.Frame(self.untracked_canvas, bg=self.colors['bg_secondary'])
        
        self.untracked_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.untracked_canvas.configure(scrollregion=self.untracked_canvas.bbox("all"))
        )
        
        self.untracked_canvas.create_window((0, 0), window=self.untracked_scrollable_frame, anchor="nw")
        self.untracked_canvas.configure(yscrollcommand=self.untracked_scrollbar.set)
        
        self.untracked_canvas.pack(side="left", fill="both", expand=True)
        self.untracked_scrollbar.pack(side="right", fill="y")
        
        # Store alarm widgets for updates
        self.tracked_alarm_widgets = {}
        self.untracked_alarm_widgets = {}
    
    def _create_dashboard_section(self):
        """Create the dashboard section (right side)."""
        # Title
        title_label = tk.Label(
            self.dashboard_frame, 
            text="📊 Dashboard", 
            font=self.fonts['title'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary']
        )
        title_label.pack(pady=(10, 20))
        
        # 24h Average Clear Time
        avg_frame = tk.Frame(self.dashboard_frame, bg=self.colors['bg_tertiary'], relief=tk.RAISED, bd=1)
        avg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            avg_frame, 
            text="Avg. Clear Time (24h):", 
            font=self.fonts['subtitle'],
            bg=self.colors['bg_tertiary'], 
            fg=self.colors['text_primary']
        ).pack(pady=5)
        
        self.avg_clear_time_label = tk.Label(
            avg_frame, 
            text="0s", 
            font=self.fonts['title'],
            bg=self.colors['bg_tertiary'], 
            fg=self.colors['accent_blue']
        )
        self.avg_clear_time_label.pack(pady=(0, 10))
        
        # Jams by Macro Area with Reset Button
        macro_header_frame = tk.Frame(self.dashboard_frame, bg=self.colors['bg_secondary'])
        macro_header_frame.pack(fill=tk.X, padx=10, pady=(20, 10))
        
        macro_title = tk.Label(
            macro_header_frame, 
            text="Total Jams by Area:", 
            font=self.fonts['subtitle'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_primary']
        )
        macro_title.pack(side=tk.LEFT)
        
        # Reset Button for Admin Users
        self.reset_button = tk.Button(
            macro_header_frame,
            text="🔄 Reset",
            font=self.fonts['normal'],
            bg=self.colors['accent_red'],
            fg='#ffffff',
            activebackground='#d32f2f',
            activeforeground='#ffffff',
            relief=tk.RAISED,
            bd=2,
            padx=8,
            pady=2,
            command=self.reset_jam_counts
        )
        self.reset_button.pack(side=tk.RIGHT)

        # Settings Button
        self.settings_button = tk.Button(
            macro_header_frame,
            text="⚙️",
            font=self.fonts['normal'],
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            relief=tk.FLAT,
            command=self.create_config_window
        )
        self.settings_button.pack(side=tk.RIGHT, padx=5)

        # TT Button
        self.tt_button = tk.Button(
            macro_header_frame,
            text="TT",
            font=self.fonts['normal'],
            bg=self.colors['accent_blue'],
            fg='#ffffff',
            relief=tk.RAISED,
            bd=2,
            padx=8,
            pady=2,
            command=lambda: webbrowser.open("https://t.corp.amazon.com/create/copy/V2013005644")
        )
        self.tt_button.pack(side=tk.RIGHT, padx=5)
        
        # Scrollable frame for macro area stats
        self.macro_stats_frame = tk.Frame(self.dashboard_frame, bg=self.colors['bg_tertiary'], relief=tk.SUNKEN, bd=1)
        self.macro_stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Last Scan heartbeat
        heartbeat_frame = tk.Frame(self.dashboard_frame, bg=self.colors['bg_secondary'])
        heartbeat_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        tk.Label(
            heartbeat_frame, 
            text="Last Scan:", 
            font=self.fonts['normal'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['text_secondary']
        ).pack(side=tk.LEFT)
        
        self.last_scan_label = tk.Label(
            heartbeat_frame, 
            text="--:--:--", 
            font=self.fonts['mono'],
            bg=self.colors['bg_secondary'], 
            fg=self.colors['accent_green']
        )
        self.last_scan_label.pack(side=tk.RIGHT)
    
    def _create_status_bar(self):
        """Create enhanced status bar at bottom."""
        self.status_frame = tk.Frame(self.root, bg=self.colors['bg_tertiary'], relief=tk.SUNKEN, bd=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="System Status: Initializing...", 
            font=self.fonts['normal'],
            bg=self.colors['bg_tertiary'], 
            fg=self.colors['text_secondary']
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=2)
        
        self.error_count_label = tk.Label(
            self.status_frame, 
            text="Errors: 0", 
            font=self.fonts['normal'],
            bg=self.colors['bg_tertiary'], 
            fg=self.colors['text_secondary']
        )
        self.error_count_label.pack(side=tk.RIGHT, padx=10, pady=2)
    
    def update_gui(self):
        """Enhanced GUI update loop with comprehensive data handling."""
        try:
            # Process all queued messages
            while True:
                try:
                    message_type, data = self.gui_queue.get_nowait()
                    
                    if message_type == "NEW_ALARM":
                        self.add_alarm(data)
                    elif message_type == "CLEAR_ALARM":
                        self.remove_alarm(data)
                    elif message_type == "ENHANCED_STATS":
                        self.update_stats(data)
                    elif message_type == "CRITICAL_ERROR":
                        self.handle_critical_error(data)
                    elif message_type == "ERROR":
                        self.show_error(data)
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            logging.error(f"GUI update error: {e}")
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def add_alarm(self, alarm_data):
        """Add alarm to appropriate section (tracked or untracked) with timestamp for running timer."""
        try:
            key = alarm_data.get('key', 'unknown')
            friendly_text = alarm_data.get('text', 'Unknown Alarm')
            timestamp = alarm_data.get('timestamp', datetime.now())
            is_tracked = alarm_data.get('is_tracked', False)
            severity = alarm_data.get('severity', 'NORMAL')
            is_critical = alarm_data.get('is_critical', False)
            
            
            # Determine which section to add to
            if is_tracked:
                parent_frame = self.tracked_scrollable_frame
                canvas = self.tracked_canvas
                widgets_dict = self.tracked_alarm_widgets
            else:
                parent_frame = self.untracked_scrollable_frame
                canvas = self.untracked_canvas
                widgets_dict = self.untracked_alarm_widgets
            
            # Create alarm frame
            alarm_frame = tk.Frame(
                parent_frame, 
                bg=self.colors['bg_tertiary'], 
                relief=tk.RAISED, 
                bd=1
            )
            alarm_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Create alarm label with initial timer
            duration_str = self._format_duration(datetime.now() - timestamp)
            display_text = f"({duration_str}) {friendly_text}"
            
            # Use smaller font for untracked alarms
            font_to_use = self.fonts['normal'] if is_tracked else tkFont.Font(family="Arial", size=8)
            
            alarm_label = tk.Label(
                alarm_frame,
                text=display_text,
                font=font_to_use,
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_primary'],
                anchor='w',
                justify='left'
            )
            alarm_label.pack(fill=tk.X, padx=3, pady=2)
            
            # Store alarm widget with metadata for updates
            widgets_dict[key] = {
                'frame': alarm_frame,
                'label': alarm_label,
                'timestamp': timestamp,
                'friendly_text': friendly_text,
                'is_tracked': is_tracked,
                'is_tracked': is_tracked,
                'severity': severity,
                'is_critical': is_critical
            }

            # Bind click to copy
            def copy_to_clipboard(event):
                self.root.clipboard_clear()
                self.root.clipboard_append(friendly_text)
                self.root.update() # Required for clipboard to update
                # Visual feedback
                original_bg = alarm_frame.cget("bg")
                alarm_frame.config(bg="#ffffff")
                self.root.after(100, lambda: alarm_frame.config(bg=original_bg))
                
            alarm_frame.bind("<Button-1>", copy_to_clipboard)
            alarm_label.bind("<Button-1>", copy_to_clipboard)
            
            # Update scroll region
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        except Exception as e:
            logging.error(f"Error adding alarm to GUI: {e}")
    
    def remove_alarm(self, alarm_data):
        """Remove alarm from appropriate section (tracked or untracked)."""
        try:
            key = alarm_data.get('key', 'unknown')
            
            # Check tracked alarms first
            if key in self.tracked_alarm_widgets:
                self.tracked_alarm_widgets[key]['frame'].destroy()
                del self.tracked_alarm_widgets[key]
                self.tracked_canvas.configure(scrollregion=self.tracked_canvas.bbox("all"))
            # Check untracked alarms
            elif key in self.untracked_alarm_widgets:
                self.untracked_alarm_widgets[key]['frame'].destroy()
                del self.untracked_alarm_widgets[key]
                self.untracked_canvas.configure(scrollregion=self.untracked_canvas.bbox("all"))
                
        except Exception as e:
            logging.error(f"Error removing alarm from GUI: {e}")
    
    def update_stats(self, stats_data):
        """Update dashboard statistics display."""
        try:
            # Update 24h average clear time
            avg_clear_time = stats_data.get('avg_clear_time_24h', 0)
            self.avg_clear_time_label.config(text=f"{avg_clear_time:.0f}s")
            
            # Update last scan time
            last_scan_time = stats_data.get('last_scan_time')
            if last_scan_time:
                time_str = last_scan_time.strftime("%H:%M:%S")
                self.last_scan_label.config(text=time_str)
            
            # Update macro area statistics
            self._update_macro_stats(stats_data.get('macro_area_counts', []))
            
            # Update status bar
            health_status = stats_data.get('health_status', {})
            system_status = health_status.get('system_status', 'Unknown')
            error_count = stats_data.get('consecutive_errors', 0)
            
            self.status_label.config(text=f"System Status: {system_status}")
            self.error_count_label.config(text=f"Errors: {error_count}")
            
            # Update running timers and color-coding for all active alarms
            self._update_alarm_timers()
            
        except Exception as e:
            logging.error(f"Error updating stats: {e}")
    
    def _update_macro_stats(self, macro_stats):
        """Update the macro area statistics display."""
        try:
            # Clear existing widgets
            for widget in self.macro_stats_frame.winfo_children():
                widget.destroy()
            
            if not macro_stats:
                # Show "No active jams" message
                no_jams_label = tk.Label(
                    self.macro_stats_frame,
                    text="No active jams",
                    font=self.fonts['normal'],
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['text_secondary']
                )
                no_jams_label.pack(pady=10)
            else:
                # Display each macro area with count
                for area, count in macro_stats:
                    area_frame = tk.Frame(self.macro_stats_frame, bg=self.colors['bg_tertiary'])
                    area_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    area_label = tk.Label(
                        area_frame,
                        text=f"{area}:",
                        font=self.fonts['normal'],
                        bg=self.colors['bg_tertiary'],
                        fg=self.colors['text_primary']
                    )
                    area_label.pack(side=tk.LEFT)
                    
                    count_label = tk.Label(
                        area_frame,
                        text=str(count),
                        font=self.fonts['subtitle'],
                        bg=self.colors['bg_tertiary'],
                        fg=self.colors['accent_red'] if count > 3 else self.colors['accent_orange'] if count > 1 else self.colors['accent_green']
                    )
                    count_label.pack(side=tk.RIGHT)
                    
        except Exception as e:
            logging.error(f"Error updating macro stats: {e}")
    
    def _update_alarm_timers(self):
        """Update running timers and color-coding for all active alarms in both sections."""
        try:
            current_time = datetime.now()
            
            # Update tracked alarms
            for key, alarm_widget in self.tracked_alarm_widgets.items():
                self._update_single_alarm_timer(current_time, alarm_widget, is_tracked=True)
            
            # Update untracked alarms (with less prominent color coding)
            for key, alarm_widget in self.untracked_alarm_widgets.items():
                self._update_single_alarm_timer(current_time, alarm_widget, is_tracked=False)
                
        except Exception as e:
            logging.error(f"Error updating alarm timers: {e}")
    
    def _update_single_alarm_timer(self, current_time, alarm_widget, is_tracked=True):
        """Update a single alarm's timer and color coding."""
        try:
            timestamp = alarm_widget['timestamp']
            friendly_text = alarm_widget['friendly_text']
            label = alarm_widget['label']
            label = alarm_widget['label']
            frame = alarm_widget['frame']
            is_critical = alarm_widget.get('is_critical', False)
            
            # Calculate duration
            duration = current_time - timestamp
            duration_str = self._format_duration(duration)
            
            # Update label text with running timer
            display_text = f"({duration_str}) {friendly_text}"
            label.config(text=display_text)
            
            # Apply color-coding based on duration (more prominent for tracked alarms)
            duration_seconds = duration.total_seconds()
            
            if is_tracked:
                # Full color coding for tracked alarms
                if duration_seconds > 300:  # > 30 minutes
                    bg_color = '#800080'  # Purple for stale alarms
                    fg_color = '#ffffff'
                elif duration_seconds > 150:  # > 5 minutes
                    bg_color = self.colors['accent_red']  # Red for critical
                    fg_color = '#ffffff'
                elif duration_seconds > 90:  # > 1 minute
                    bg_color = self.colors['accent_orange']  # Orange for warning
                    fg_color = '#000000'
                else:
                    bg_color = self.colors['bg_tertiary']  # Normal background
                    fg_color = self.colors['text_primary']
                
                # Critical Alarm Flashing Override
                if is_critical:
                    # Flash between Red and Dark Red every 500ms
                    if int(current_time.timestamp() * 2) % 2 == 0:
                        bg_color = self.colors['accent_red']
                        fg_color = '#ffffff'
                    else:
                        bg_color = '#8b0000' # Dark Red
                        fg_color = '#ffffff'
                    
                    # Make font larger/bold if critical
                    label.configure(font=tkFont.Font(family="Arial", size=14, weight="bold"))
            else:
                # Muted color coding for untracked alarms
                if duration_seconds > 1800:  # > 30 minutes
                    bg_color = '#4a4a4a'  # Muted purple
                    fg_color = '#cccccc'
                elif duration_seconds > 300:  # > 5 minutes
                    bg_color = '#5a4a4a'  # Muted red
                    fg_color = '#cccccc'
                else:
                    bg_color = self.colors['bg_tertiary']  # Normal background
                    fg_color = self.colors['text_secondary']
            
            # Update colors
            frame.config(bg=bg_color)
            label.config(bg=bg_color, fg=fg_color)
            
        except Exception as e:
            logging.error(f"Error updating single alarm timer: {e}")
    
    def _format_duration(self, duration):
        """Format duration as HH:MM:SS or MM:SS."""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def handle_critical_error(self, error_data):
        """Handle critical system errors."""
        try:
            error_count = error_data.get('error_count', 0)
            last_error = error_data.get('last_error', 'Unknown error')
            recommendation = error_data.get('recommendation', 'Check system logs')
            
            # Show critical error dialog
            messagebox.showerror(
                "Critical System Error",
                f"Error Count: {error_count}\n"
                f"Last Error: {last_error}\n"
                f"Recommendation: {recommendation}"
            )
            
            # Update status bar with critical status
            self.status_label.config(
                text="System Status: CRITICAL ERROR",
                fg=self.colors['accent_red']
            )
            
        except Exception as e:
            logging.error(f"Error handling critical error: {e}")
    
    def show_error(self, error_message):
        """Show error message."""
        try:
            messagebox.showwarning("System Warning", str(error_message))
        except Exception as e:
            logging.error(f"Error showing error message: {e}")
    
    def reset_jam_counts(self):
        """Reset all jam counts with admin confirmation - Connected to reset button."""
        try:
            # Show confirmation dialog
            confirm = messagebox.askyesno(
                "Reset Jam Counts",
                "⚠️ ADMIN FUNCTION ⚠️\n\n"
                "This will reset ALL jam count statistics to zero.\n"
                "A backup will be created before clearing the data.\n\n"
                "Are you sure you want to continue?",
                icon='warning'
            )
            
            if confirm:
                # Show progress message
                progress_msg = messagebox.showinfo(
                    "Resetting Data",
                    "Resetting jam counts...\nPlease wait while backup is created.",
                    icon='info'
                )
                
                # Call the database reset method through the monitor thread
                if hasattr(self, 'monitor_thread') and self.monitor_thread.logger:
                    success = self.monitor_thread.logger.reset_jam_counts()
                    
                    if success:
                        # Show success message
                        messagebox.showinfo(
                            "Reset Complete",
                            "✅ Jam counts have been successfully reset!\n\n"
                            "• All statistics cleared to zero\n"
                            "• Backup created automatically\n"
                            "• Dashboard will update momentarily",
                            icon='info'
                        )
                        
                        # Log the admin action
                        logging.info("ADMIN ACTION: Jam counts reset by user")
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ADMIN: Jam counts reset")
                        
                        # Force immediate dashboard update
                        self.root.after(1000, lambda: self._update_macro_stats([]))
                        
                    else:
                        # Show error message
                        messagebox.showerror(
                            "Reset Failed",
                            "❌ Failed to reset jam counts.\n\n"
                            "Please check the system logs for details.\n"
                            "Contact system administrator if the issue persists.",
                            icon='error'
                        )
                else:
                    messagebox.showerror(
                        "System Error",
                        "❌ Cannot access database system.\n\n"
                        "System may still be initializing.\n"
                        "Please try again in a few moments.",
                        icon='error'
                    )
            
        except Exception as e:
            logging.error(f"Error in reset_jam_counts: {e}")
            messagebox.showerror(
                "Unexpected Error",
                f"❌ An unexpected error occurred:\n{str(e)}\n\n"
                "Please contact system administrator.",
                icon='error'
            )
    
    def on_closing(self):
        """Handle window closing event."""
        print("Shutting down enhanced alarm monitor...")
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.running = False
        self.root.quit()
        self.root.destroy()
    
    def start(self):
        """Start the enhanced GUI application."""
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the GUI
        print("Starting Enhanced Alarm Monitor GUI...")
        print("Enhanced features: Health monitoring, Performance analytics, Reliability")
        self.root.mainloop()

    def create_config_window(self):
        """Create a configuration window to edit settings."""
        config_window = tk.Toplevel(self.root)
        config_window.title("Settings")
        config_window.geometry("400x500")
        config_window.configure(bg=self.colors['bg_primary'])
        
        # Helper to create labeled entries
        entries = {}
        
        def create_entry(parent, label_text, key, default_val):
            frame = tk.Frame(parent, bg=self.colors['bg_primary'])
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            lbl = tk.Label(frame, text=label_text, bg=self.colors['bg_primary'], fg=self.colors['text_primary'])
            lbl.pack(side=tk.LEFT)
            
            entry = tk.Entry(frame)
            entry.insert(0, str(default_val))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(10, 0))
            
            entries[key] = entry
            
        # ROI Settings
        roi_frame = tk.LabelFrame(config_window, text="ROI Settings", bg=self.colors['bg_primary'], fg=self.colors['text_primary'])
        roi_frame.pack(fill=tk.X, padx=10, pady=10)
        
        create_entry(roi_frame, "X:", "roi_x", self.config['roi']['x'])
        create_entry(roi_frame, "Y:", "roi_y", self.config['roi']['y'])
        create_entry(roi_frame, "Width:", "roi_w", self.config['roi']['width'])
        create_entry(roi_frame, "Height:", "roi_h", self.config['roi']['height'])
        
        # General Settings
        gen_frame = tk.LabelFrame(config_window, text="General", bg=self.colors['bg_primary'], fg=self.colors['text_primary'])
        gen_frame.pack(fill=tk.X, padx=10, pady=10)
        
        create_entry(gen_frame, "Scan Interval (s):", "scan_interval", self.config['scan_interval'])
        create_entry(gen_frame, "Similarity (0.0-1.0):", "similarity", self.config['similarity_threshold'])
        
        # Preprocessing Settings
        prep_frame = tk.LabelFrame(config_window, text="Preprocessing & OCR", bg=self.colors['bg_primary'], fg=self.colors['text_primary'])
        prep_frame.pack(fill=tk.X, padx=10, pady=10)
        
        create_entry(prep_frame, "Binary Threshold (0-255):", "binary_threshold", self.config['preprocessing']['binary_threshold'])
        create_entry(prep_frame, "Scaling Factor:", "scaling_factor", self.config['preprocessing']['scaling_factor'])
        create_entry(prep_frame, "PSM Mode:", "psm_mode", self.config['ocr_settings']['psm_mode'])

        def save_config():
            try:
                # Update config object
                self.config['roi']['x'] = int(entries['roi_x'].get())
                self.config['roi']['y'] = int(entries['roi_y'].get())
                self.config['roi']['width'] = int(entries['roi_w'].get())
                self.config['roi']['height'] = int(entries['roi_h'].get())
                self.config['scan_interval'] = float(entries['scan_interval'].get())
                self.config['similarity_threshold'] = float(entries['similarity'].get())
                
                # Update Preprocessing
                self.config['preprocessing']['binary_threshold'] = int(entries['binary_threshold'].get())
                self.config['preprocessing']['scaling_factor'] = float(entries['scaling_factor'].get())
                self.config['ocr_settings']['psm_mode'] = int(entries['psm_mode'].get())
                
                # Save to file
                with open("config.json", "w") as f:
                    json.dump(self.config, f, indent=2)
                
                messagebox.showinfo("Success", "Configuration saved! Restart application to apply changes.")
                config_window.destroy()
                
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
        
        save_btn = tk.Button(config_window, text="Save", command=save_config, bg=self.colors['accent_green'], fg='white')
        save_btn.pack(pady=20)


def main():
    """Main entry point for the enhanced alarm monitor."""
    
    # Enhanced configuration
    CONFIG = {
        "roi": {
            "x": 2000,  # X-coord of Description column
            "y": 400,  # Y-coord (top of list)
            "width": 436, # Width of Description column
            "height": 400  # Height of visible list
        },
        "scan_interval": 1.0,
        "clear_grace_period": 3.0,
        "similarity_threshold": 0.85,
        "global_keyword_similarity": 0.85,
        "database_file": "jams.db",
        "translation_file": "TRANSLATIONS.json",
        "line_mask_y_coords": [2, 28, 53, 79, 104, 130, 156, 181, 207, 233, 258, 284, 309, 335, 361, 386, 412, 437, 463, 488, 514, 539, 565, 590, 616, 641, 667, 692, 718, 743, 769, 794, 820, 845, 871, 896],
        "noise_blocklist": ["Alarm subscriotion", "OPC Event Serer"],
        "ocr_settings": {
            "language": "eng",
            "whitelist": "",
            "psm_mode": 6,
            "oem_mode": 1
        },
        "preprocessing": {
            "scaling_factor": 1.6,
            "grayscale": True,
            "binary_threshold": 175,
            "adaptive_threshold": False,
            "noise_removal": False
        },
        "debug_mode": True,
        # Enhanced reliability settings
        "health_monitoring": {
            "enabled": True,
            "check_interval": 30.0,
            "memory_threshold_mb": 500,
            "cpu_threshold_percent": 80,
            "error_threshold": 10
        },
        "backup_settings": {
            "enabled": True,
            "interval_hours": 6,
            "max_backups": 10,
            "backup_directory": "backups"
        }
    }
    
    # Load saved configuration if exists
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                saved_config = json.load(f)
                # Deep update for nested dictionaries (roi, etc)
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = update_dict(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d
                
                update_dict(CONFIG, saved_config)
                
                # Force scaling factor to 1.6 if it's 1.0 (default fallback) or missing
                # This ensures the user's request for 1.6 default is respected even if old config exists
                if CONFIG['preprocessing'].get('scaling_factor') == 1.0:
                     CONFIG['preprocessing']['scaling_factor'] = 1.6
                     
                print("Loaded configuration from config.json")
        except json.JSONDecodeError:
            logging.error("config.json is empty or invalid. Using default configuration.")
            print("Warning: config.json is invalid. Using defaults.")
        except Exception as e:
            logging.error(f"Failed to load config.json: {e}")
            print(f"Warning: Failed to load config.json: {e}")

    try:
        # Create and run the enhanced GUI
        app = EnhancedAlarmMonitorGUI(config=CONFIG)
        app.start()
    except Exception as e:
        logging.error(f"Failed to start enhanced alarm monitor: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


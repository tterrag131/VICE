# Enhanced Alarm Monitor - 24/7 Operations Ready

A comprehensive, enterprise-grade OCR-based alarm monitoring system with real-time dashboard, health monitoring, and reliability features designed for 24/7 continuous operation in industrial environments.

## 🚀 **Key Features**

### **Core Monitoring Capabilities**
- **6-Digit Key-Based Identification**: Advanced alarm matching using unique numerical identifiers (e.g., `140015`)
- **Enhanced OCR Accuracy**: Robust text extraction with error recovery and fuzzy matching
- **Real-Time Dashboard**: Live KPI display with running timers and color-coded severity
- **Grace Period Logic**: Prevents false alarm clearing due to temporary OCR misreadings

### **24/7 Operation Optimizations**
- **Memory Management**: Automatic cleanup to prevent memory leaks during continuous operation
- **Database Performance**: SQLite WAL mode with performance optimizations for high throughput
- **Thread Safety**: Comprehensive locking mechanisms for concurrent operations
- **Resource Leak Prevention**: Proper cleanup of OpenCV windows, screen capture, and database connections

### **Enhanced Reliability Features**
- **Health Monitoring**: Real-time system metrics (memory, CPU, OCR performance)
- **Error Recovery**: Escalating recovery strategies with component reinitialization
- **Automatic Backups**: Scheduled database backups every 6 hours with retention management
- **Debug Capabilities**: Rotating log files and admin-friendly debugging tools

### **User Interface**
- **Dual-Section Layout**: Tracked alarms (75%) and untracked alarms (25%) with separate displays
- **Running Timers**: Real-time duration display for all active alarms
- **Color-Coded Severity**: Purple (>30min), Red (>5min), Orange (>1min), Normal (<1min)
- **Admin Controls**: Reset button for granular jam count tracking

## 📁 **Project Structure**

```
projectSanDWRM/
├── AlarmMonitorGUI_Enhanced.py    # Main enhanced application
├── AlarmMonitorGUI.py             # Legacy GUI version
├── alarm_monitor_ocr.py           # Core OCR engine
├── TRANSLATIONS.json              # Alarm code translation database
├── requirements.txt               # Python dependencies
├── config.json                    # System configuration
├── roi_config.json               # ROI configuration
├── roi_selector.py               # Interactive ROI setup tool
├── test_ocr_setup.py             # OCR validation script
├── jams.db                       # SQLite alarm database
├── system_health.log             # Health monitoring logs
├── debug_enhanced.log            # Debug output logs
├── backups/                      # Automatic backup storage
└── README_Enhanced.md            # This documentation
```

## 🛠 **Installation Guide**

### **Prerequisites**

#### **System Requirements**
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB, Recommended 8GB for 24/7 operation
- **Disk Space**: 1GB free space (including backup storage)
- **Display**: Support for screen capture (not compatible with some virtualized environments)

#### **1. Install Tesseract OCR Engine**

**Windows (Recommended Method):**
```cmd
# Download the official installer
# Go to: https://github.com/UB-Mannheim/tesseract/wiki
# Download: tesseract-ocr-w64-setup-5.3.3.20231005.exe (or latest version)
# Install to: C:\Program Files\Tesseract-OCR\
# Add to PATH: C:\Program Files\Tesseract-OCR\
```

**Alternative Windows Locations:**
```cmd
# If installing to custom location, update PATH or modify config:
# Common paths:
# C:\Users\[username]\AppData\Local\Tesseract-OCR\
# C:\tesseract\
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev  # Development headers
```

**macOS:**
```bash
# Using Homebrew
brew install tesseract
```

**Verify Installation:**
```cmd
tesseract --version
# Should output: tesseract 5.x.x
```

#### **2. Install Python Dependencies**

**Method 1: Using pip (Recommended)**
```cmd
# Navigate to project directory
cd path\to\projectSanDWRM

# Install all required packages
pip install -r requirements.txt

# Verify critical packages
pip list | findstr "pytesseract opencv-python pillow numpy mss psutil"
```

**Method 2: Individual Package Installation**
```cmd
pip install pytesseract>=0.3.10
pip install Pillow>=10.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install mss>=9.0.1
pip install psutil>=5.9.0
```

**Troubleshooting Dependencies:**
```cmd
# If installation fails, try upgrading pip first:
python -m pip install --upgrade pip

# For Windows compilation issues:
pip install --only-binary=all numpy opencv-python

# For Linux missing development headers:
sudo apt install python3-dev python3-pip
```

## ⚙️ **Configuration Setup**

### **1. ROI (Region of Interest) Configuration**

The system monitors a specific screen region containing the alarm display. You must configure this area before first use.

#### **Interactive Setup (Recommended)**
```cmd
# Run the ROI selector tool
python roi_selector.py

# OR use the integrated setup
python AlarmMonitorGUI_Enhanced.py --setup-roi
```

**ROI Setup Steps:**
1. **Full screen capture** appears
2. **Click and drag** to select the alarm log area
3. **Preview window** shows selected region
4. **Press SPACE** to confirm selection
5. **Configuration saved** automatically to `roi_config.json`

#### **Manual Configuration**
Edit `roi_config.json`:
```json
{
    "roi": {
        "x": 2000,        // Left edge of alarm area (pixels from left)
        "y": 305,         // Top edge of alarm area (pixels from top)  
        "width": 436,     // Width of alarm area
        "height": 290     // Height of alarm area
    }
}
```

**Finding ROI Coordinates:**
- Use Windows Snipping Tool or similar to measure pixel coordinates
- Ensure the ROI captures only the alarm text area (not headers/borders)
- ROI should be large enough to capture all visible alarms

### **2. TRANSLATIONS.json Configuration**

This file maps 6-digit alarm codes to human-readable descriptions with location information.

#### **File Structure**
```json
{
    "140015": {
        "macro_area": "SORTER",
        "micro_area": "MRS", 
        "descriptor": "Incline"
    },
    "140016": {
        "macro_area": "SORTER",
        "micro_area": "MRS",
        "descriptor": "Horseshoe"
    },
    "291822": {
        "macro_area": "MRS",
        "micro_area": "AFE1",
        "descriptor": "Divert 1"
    }
}
```

#### **Adding New Alarm Codes**
1. **Identify the 6-digit code** from alarm text (e.g., `140015` from `"AREA CC4 BTt140015 MOTOR FAULTED"`)
2. **Determine location information**:
   - `macro_area`: High-level area (SORTER, MRS, OUTBOUND, etc.)
   - `micro_area`: Specific zone within macro area
   - `descriptor`: Equipment/location description
3. **Add entry** to TRANSLATIONS.json
4. **Restart application** to load new translations

#### **Translation Display Format**
The GUI displays: `{macro_area} {micro_area} {descriptor}`
- Example: `"SORTER MRS Incline"` (complete information in one line)
- Untracked alarms show as: `"⚠️ UNTRACKED: 291822"`

### **3. System Configuration**

Edit `config.json` for advanced settings:

```json
{
    "scan_interval": 1.0,                    // Seconds between screen scans
    "clear_grace_period": 3.0,               // Seconds before marking alarm as cleared
    "similarity_threshold": 0.85,            // Fuzzy matching threshold (0.0-1.0)
    "database_file": "jams.db",              // SQLite database filename
    "translation_file": "TRANSLATIONS.json", // Translation database filename
    "debug_mode": false,                     // Enable debug OCR windows
    "admin_debug": false,                    // Enable admin console output
    
    "ocr_settings": {
        "language": "eng",                   // OCR language
        "psm_mode": 6,                      // Page segmentation mode
        "oem_mode": 1,                      // OCR engine mode
        "whitelist": ""                     // Character whitelist (empty = all)
    },
    
    "preprocessing": {
        "scaling_factor": 1.6,              // Image scaling for OCR
        "grayscale": true,                  // Convert to grayscale
        "binary_threshold": 175,            // Threshold for text extraction
        "adaptive_threshold": false,         // Use adaptive thresholding
        "noise_removal": false              // Apply noise removal filters
    },
    
    "health_monitoring": {
        "enabled": true,                    // Enable health monitoring
        "check_interval": 30.0,             // Health check frequency
        "memory_threshold_mb": 500,         // Memory usage alert threshold
        "cpu_threshold_percent": 80,        // CPU usage alert threshold
        "error_threshold": 10               // Error count threshold
    },
    
    "backup_settings": {
        "enabled": true,                    // Enable automatic backups
        "interval_hours": 6,                // Backup frequency
        "max_backups": 10,                  // Maximum backup files to keep
        "backup_directory": "backups"       // Backup storage directory
    }
}
```

## 🚀 **Running the Application**

### **Standard Operation**
```cmd
# Navigate to project directory
cd path\to\projectSanDWRM

# Start the enhanced alarm monitor
python AlarmMonitorGUI_Enhanced.py
```

### **Debug Mode**
```cmd
# Enable debug OCR windows and verbose output
# Edit config.json: set "debug_mode": true, "admin_debug": true
python AlarmMonitorGUI_Enhanced.py
```

### **Expected Startup Sequence**
```
2025-11-11 03:52:30,123 - INFO - Loaded 448 translation keys from TRANSLATIONS.json
2025-11-11 03:52:30,127 - INFO - Enhanced database tables are ready.
2025-11-11 03:52:30,128 - INFO - Enhanced database connection established at jams.db
2025-11-11 03:52:30,129 - INFO - Enhanced AlarmMonitorThread initialized with reliability features
2025-11-11 03:52:30,133 - INFO - Starting enhanced alarm monitoring thread...
2025-11-11 03:52:30,134 - INFO - Enhanced AlarmMonitorGUI initialized
Starting Enhanced Alarm Monitor GUI...
Enhanced features: Health monitoring, Performance analytics, Reliability
```

## 🖥️ **User Interface Guide**

### **Main Window Layout**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Enhanced Alarm Monitor - 24/7 Operations                                       │
├────────────────────────────────┬────────────────────────────────────────────────┤
│ 🚨 Active Alarms (75%)         │ 📊 Dashboard (25%)                             │
│                                │                                                │
│ 📍 Tracked Jams               │ Avg. Clear Time (24h): 245s                   │
│ ┌────────────────────────────┐ │                                                │
│ │ (05:23) SORTER MRS Incline │ │ Total Jams by Area:     🔄 Reset             │
│ │ (03:14) MRS AFE1 Divert 1  │ │ ┌────────────────────────────────────────────┐ │
│ │ (01:45) OUTBOUND Line 3    │ │ │ SORTER:                               15   │ │
│ └────────────────────────────┘ │ │ MRS:                                   8   │ │
│                                │ │ OUTBOUND:                              3   │ │
│ ❓ Untracked                   │ └────────────────────────────────────────────┘ │
│ ┌────────────────────────────┐ │                                                │
│ │ ⚠️ UNTRACKED: 291822       │ │ Last Scan: 15:32:45                           │
│ └────────────────────────────┘ │                                                │
├────────────────────────────────┴────────────────────────────────────────────────┤
│ System Status: RUNNING                                          Errors: 0      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **Color Coding System**
- **Normal** (< 1 minute): Default colors
- **Orange** (1-5 minutes): Warning level
- **Red** (5-30 minutes): Critical level  
- **Purple** (> 30 minutes): Stale/urgent attention needed

### **Dashboard Features**
- **Avg. Clear Time**: 24-hour average alarm resolution time
- **Total Jams by Area**: Count of jams per macro area with color coding
- **Reset Button**: Admin function to zero out statistics for shift tracking
- **Last Scan**: Heartbeat showing system is actively monitoring

### **Admin Functions**

#### **Reset Jam Counts**
1. Click the **🔄 Reset** button in the dashboard
2. Confirm the warning dialog
3. System creates automatic backup before clearing
4. All statistics reset to zero
5. Dashboard updates immediately

**Reset Process:**
```
⚠️ ADMIN FUNCTION ⚠️

This will reset ALL jam count statistics to zero.
A backup will be created before clearing the data.

Are you sure you want to continue?
[Yes] [No]
```

## 📊 **Database Management**

### **Database Files**
- **jams.db**: Main SQLite database storing alarm events
- **jams.db-shm**: Shared memory file (WAL mode)
- **jams.db-wal**: Write-ahead log file (performance optimization)

### **Database Schema**
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    raw_text TEXT NOT NULL,
    friendly_text TEXT,
    severity TEXT DEFAULT 'NORMAL',
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    macro_area TEXT,
    alarm_key TEXT
);
```

### **Backup Management**
- **Automatic Backups**: Created every 6 hours during operation
- **Manual Backups**: Created before admin reset operations
- **Backup Location**: `backups/` directory with timestamped files
- **Backup Format**: `events_backup_YYYYMMDD_HHMMSS.db`
- **Retention**: Configurable maximum backup count (default: 10)

### **Database Queries**
```sql
-- View active alarms
SELECT * FROM events WHERE end_time IS NULL;

-- Get alarm statistics for last 24 hours
SELECT 
    macro_area, 
    COUNT(*) as total_alarms,
    AVG(duration_seconds) as avg_duration
FROM events 
WHERE start_time > datetime('now', '-24 hours')
GROUP BY macro_area;

-- Find longest running alarms
SELECT 
    friendly_text,
    start_time,
    (julianday('now') - julianday(start_time)) * 86400 as current_duration_seconds
FROM events 
WHERE end_time IS NULL
ORDER BY current_duration_seconds DESC;
```

## 🔧 **Troubleshooting Guide**

### **Common Issues and Solutions**

#### **1. Application Won't Start**

**Symptom**: Application crashes immediately or shows import errors
```python
ModuleNotFoundError: No module named 'pytesseract'
```

**Solution**:
```cmd
# Verify Python environment
python --version
pip --version

# Reinstall dependencies
pip install -r requirements.txt

# Check Tesseract installation
tesseract --version
```

#### **2. OCR Not Detecting Text**

**Symptom**: GUI shows "System initializing..." indefinitely, no alarms detected

**Solutions**:
1. **Check ROI Configuration**:
   ```cmd
   python test_ocr_setup.py
   # Review generated images: captured_roi_original.png, captured_roi_preprocessed.png
   ```

2. **Verify Screen Capture**:
   - Ensure ROI coordinates are correct for your display resolution
   - Check if using multiple monitors (update monitor_index in config)
   - Verify application has permission to capture screen

3. **OCR Settings Adjustment**:
   ```json
   // Try different OCR modes in config.json
   "ocr_settings": {
       "psm_mode": 7,  // Try 6, 7, or 8
       "oem_mode": 3,  // Try 1, 2, or 3
   }
   ```

#### **3. Poor Alarm Recognition**

**Symptom**: System detects alarms but translations are wrong or missing

**Solutions**:
1. **Check 6-Digit Key Extraction**:
   - Enable debug mode to see OCR output
   - Verify alarm text contains 6-digit codes
   - Update regex pattern if needed

2. **Update TRANSLATIONS.json**:
   - Add missing alarm codes
   - Verify JSON syntax is correct
   - Restart application to reload translations

#### **4. High Memory Usage**

**Symptom**: System memory continuously increases during operation

**Solutions**:
1. **Adjust Memory Settings**:
   ```json
   "max_gui_widgets": 50,           // Reduce widget limit
   "memory_cleanup_interval": 180,   // More frequent cleanup
   ```

2. **Monitor Health Status**:
   - Check system_health.log for memory warnings
   - Use Task Manager to monitor actual usage

#### **5. Database Errors**

**Symptom**: Database connection failures or corruption warnings

**Solutions**:
1. **Check Database Files**:
   ```cmd
   # Verify database integrity
   sqlite3 jams.db "PRAGMA integrity_check;"
   
   # Check file permissions
   dir jams.* /Q
   ```

2. **Recovery Options**:
   ```cmd
   # Restore from backup
   copy "backups\events_backup_YYYYMMDD_HHMMSS.db" jams.db
   
   # Or delete corrupted database (will recreate)
   del jams.db jams.db-shm jams.db-wal
   ```

#### **6. Performance Issues**

**Symptom**: Slow response, high CPU usage, delayed alarm detection

**Solutions**:
1. **Optimize Configuration**:
   ```json
   "scan_interval": 1.5,        // Increase scan interval
   "preprocessing": {
       "scaling_factor": 1.0,   // Reduce image scaling
       "noise_removal": false   // Disable if not needed
   }
   ```

2. **ROI Optimization**:
   - Reduce ROI size to minimum required area
   - Ensure ROI doesn't include unnecessary UI elements

### **Debug Mode Usage**

**Enable Debug Mode**:
```json
// In config.json
"debug_mode": true,
"admin_debug": true
```

**Debug Features**:
- **OCR Windows**: Shows original and processed images
- **Console Output**: Real-time debug messages with 🔧 prefix
- **Debug Logs**: Detailed logging to `debug_enhanced.log`
- **Performance Metrics**: OCR timing and system performance data

**Debug Output Example**:
```
🔧 DEBUG: [OCR_OPERATIONS] OCR processing completed | Duration: 0.234s
🔧 DEBUG: [ALARM_EVENTS] New alarm detected: 140015 -> SORTER MRS Incline
🔧 DEBUG: [DATABASE_OPERATIONS] Event logged: ID 1234, Key: 140015
```

## 📦 **Deployment to Another Computer**

### **Complete Transfer Package**

To deploy this system to another computer, create a transfer package with these files:

#### **Essential Files**
```
Enhanced_Alarm_Monitor_Package/
├── AlarmMonitorGUI_Enhanced.py    ✅ Main application
├── AlarmMonitorGUI.py             ✅ Dependency (imported by enhanced version)
├── alarm_monitor_ocr.py           ✅ Core OCR engine
├── TRANSLATIONS.json              ✅ Alarm translation database
├── requirements.txt               ✅ Python dependencies
├── config.json                    ✅ System configuration
├── roi_config.json               ⚠️ ROI settings (may need reconfiguration)
├── roi_selector.py               ✅ ROI setup tool
├── test_ocr_setup.py             ✅ Validation script
└── README_Enhanced.md            ✅ This documentation
```

#### **Optional Files** (if transferring existing data)
```
├── jams.db                       📊 Existing alarm database
├── system_health.log            📋 Historical health data
├── backups/                     💾 Database backup files
└── debug_enhanced.log           🔧 Debug history
```

### **Deployment Steps**

#### **1. Prepare Source System**
```cmd
# Create deployment package
mkdir Enhanced_Alarm_Monitor_Package
copy AlarmMonitorGUI_Enhanced.py Enhanced_Alarm_Monitor_Package\
copy AlarmMonitorGUI.py Enhanced_Alarm_Monitor_Package\
copy alarm_monitor_ocr.py Enhanced_Alarm_Monitor_Package\
copy TRANSLATIONS.json Enhanced_Alarm_Monitor_Package\
copy requirements.txt Enhanced_Alarm_Monitor_Package\
copy config.json Enhanced_Alarm_Monitor_Package\
copy roi_config.json Enhanced_Alarm_Monitor_Package\
copy roi_selector.py Enhanced_Alarm_Monitor_Package\
copy test_ocr_setup.py Enhanced_Alarm_Monitor_Package\
copy README_Enhanced.md Enhanced_Alarm_Monitor_Package\

# Optional: Include database
copy jams.db Enhanced_Alarm_Monitor_Package\
xcopy backups Enhanced_Alarm_Monitor_Package\backups\ /E /I
```

#### **2. Target System Setup**

**Install Prerequisites:**
```cmd
# 1. Install Python 3.8+
# Download from: https://www.python.org/downloads/

# 2. Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install to: C:\Program Files\Tesseract-OCR\

# 3. Verify PATH includes Tesseract
tesseract --version
```

**Deploy Application:**
```cmd
# 1. Extract package to target directory
# Example: C:\AlarmMonitor\

# 2. Install Python dependencies
cd C:\AlarmMonitor
pip install -r requirements.txt

# 3. Test basic functionality
python test_ocr_setup.py
```

#### **3. Target System Configuration**

**Configure ROI (Required):**
```cmd
# Run ROI selector for new display configuration
python roi_selector.py

# OR manually adjust roi_config.json for target screen resolution
```

**Verify Configuration:**
```cmd
# Test OCR setup
python test_ocr_setup.py

# Check generated files:
# - captured_roi_original.png (should show correct alarm area)
# - captured_roi_preprocessed.png (should show clear text)
# - ocr_test_results.txt (should contain readable alarm text)
```

**Start Application:**
```cmd
# Launch enhanced alarm monitor
python AlarmMonitorGUI_Enhanced.py
```

#### **4. Validation Checklist**

✅ **Pre-Deployment**
- [ ] Tesseract OCR installed and in PATH
- [ ] Python 3.8+ installed
- [ ] All package files copied
- [ ] Dependencies installed successfully

✅ **Post-Deployment**
- [ ] ROI configured for target display
- [ ] OCR test shows readable text extraction
- [ ] TRANSLATIONS.json loads correctly (check startup logs)
- [ ] GUI launches without errors
- [ ] System detects test alarms (if available)
- [ ] Database creates and connects successfully

✅ **Production Readiness**
- [ ] 24-hour test run completed successfully
- [ ] Memory usage remains stable
- [ ] Automatic backups working
- [ ] Admin reset function tested
- [ ] Health monitoring active

### **Network Deployment Considerations**

#### **Shared TRANSLATIONS.json**
For multiple systems using the same alarm codes:
```cmd
# Option 1: Network share
# Place TRANSLATIONS.json on shared drive
# Update config.json: "translation_file": "\\\\server\\share\\TRANSLATIONS.json"

# Option 2: Centralized updates
# Use version control or file sync to distribute updates
# Restart applications after translation updates
```

#### **Centralized Database**
For enterprise environments requiring centralized data:
```json
// Consider upgrading to PostgreSQL/MySQL for multi-system deployment
// Current SQLite implementation is single-system only
```

## 📈 **Performance Optimization**

### **System Tuning**

#### **For High-Performance Systems**
```json
{
    "scan_interval": 0.5,           // Faster scanning
    "preprocessing": {
        "scaling_factor": 2.0,      // Higher quality OCR
        "noise_removal": true       // Better accuracy
    },
    "health_monitoring": {
        "check_interval": 15.0      // More frequent health checks
    }
}
```

#### **For Resource-Constrained Systems**
```json
{
    "scan_interval": 2.0,           // Slower scanning
    "preprocessing": {
        "scaling_factor": 1.0,      // Lower CPU usage
        "noise_removal": false      // Faster processing
    },
    "backup_settings": {
        "interval_hours": 12        // Less frequent backups
    }
}
```

#### **Memory Optimization**
```json
{
    "max_gui_widgets": 25,          // Fewer widgets in memory
    "memory_cleanup_interval": 120, // More frequent cleanup
    "max_gui_queue_size": 500      // Smaller queue buffer
}
```

## 🛡️ **Security Considerations**

### **File Permissions**
- Ensure database files are writable by application user
- Protect TRANSLATIONS.json from unauthorized modification
- Backup directory should have appropriate access controls

### **Network Security**
- System captures screen content - ensure compliance with security policies
- Database contains operational data - consider encryption for sensitive environments
- Log files may contain operational information - implement appropriate retention policies

### **Access Control**
- Admin functions (reset button) have confirmation dialogs but no authentication
- Consider implementing user authentication for production environments
- Debug mode may display sensitive operational data

## 📞 **Support and Maintenance**

### **Log File Locations**
```
project_directory/
├── alarm_monitor.log           # General application logs
├── system_health.log          # Health monitoring data
├── debug_enhanced.log         # Debug output (if enabled)
└── backups/                   # Database backup files
```

### **Regular Maintenance**
1. **Weekly**: Check system_health.log for warnings
2. **Monthly**: Verify backup integrity and cleanup old files
3. **Quarterly**: Review TRANSLATIONS.json for new alarm codes
4. **Annually**: Update Python dependencies and Tesseract OCR

### **Version Information**
- **Application Version**: Enhanced v2.0
- **Python Requirements**: 3.8+
- **Tesseract Requirements**: 5.0+
- **Database Schema Version**: 2.1

### **Change Log**
- **v2.0**: Enhanced reliability features, 24/7 operation optimization
- **v1.5**: Dashboard implementation, health monitoring
- **v1.0**: 6-digit key system, basic GUI implementation

---

## 📄 **License and Credits**

This Enhanced Alarm Monitor system is designed for industrial alarm monitoring applications. 

**Dependencies:**
- Tesseract OCR (Apache License 2.0)
- OpenCV (BSD License)
- PIL/Pillow (HPND License)
- NumPy (BSD License)
- MSS (MIT License)
- psutil (BSD License)

**System Requirements**: Designed and tested for Windows 10/11 environments with industrial display systems.

---

*For technical support or feature requests, please refer to the system administrator or development team.*

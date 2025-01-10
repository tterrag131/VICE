
# VICE
Visual Identification and Conveyance Enhancement
=======
VICE (BETA 2.0)
Real-Time Anomaly Detection and Analytics Platform
Overview

This application is a sophisticated monitoring system designed for industrial conveyor belts, providing real-time detection of operational anomalies, performance metrics, and automated reporting capabilities. It uses computer vision and color detection to identify various operational states and potential issues.
The previously unmonitored conveyance has been a thorn in the side of SMF1 for a very long time and diagnosis was based on human insight and prone to error. VICE introduces necessary data collection and a modern approach to an archaic system. The prior workflow consisted of 3 AAs watching a screen that runs on internet explorer only and creating jam callouts. VICE absolves those positions as well as collecting data and sending notifications via slack when it detects jams.

Currently with over 800 HRS up and running VICE has produced some incredible results and could be ready for a larger rollout soon. With over 2.2 million data points collected over one shift the insight VICE has offered has been pivotal to SMF1 success. The White paper and Mathematical proof demonstrate the capabilities of VICE and the derivation of the necessary calculations behind the site. 

UPDATE - DEC 29th VICE was converted to have a website shell that displayed the hallmark conveyance health metric and vital graphs. This site was hosted locally and accrued 32 daily active users for the first 2 weeks of testing. The exposed API and simple integration for other sites also offers a unique opportunity for VICE to fill a very gaping hole in operations.
- the data has also been cleaned and is poised to allow for machine learning algorithms to begin to create reccomendations and predictions absed on daily reports, this is the final step before VICE is at the edge of my capabilities and will only be refined instead of enhanced after this final update.

The primary use of this is to detect and monitor SMF1 Conveyance and diligence
The secondary use is to provide vital data to senior operations to make decisions and determine functionality.

Key Features
1. Real-Time Monitoring
Multi-Region Support :
Monitor multiple regions of interest (ROIs) simultaneously
Orange: Jam detection
Grey: Standdown conditions
Blue: Full capacity indicators
Purple: Gridlock situations
Red: Emergency stops/critical issues

2. Alert System
Real-Time Notifications :
Immediate visual alerts for detected anomalies

Slack Integration : 
Automated Slack notifications for jam conditions

Configurable Thresholds :
30% coverage threshold for grey conditions
0.1% threshold for other anomalies

Alert Window : 
Dedicated window for real-time status display of ROIs

3. Performance Metrics
Health Scoring :
Real-time conveyor health calculation
Event Counting
Jam occurrences
Standdowns
Emergency stops
Full capacity events

Performance Analytics :
Mean time between failures
Operational efficiency metrics
System uptime tracking

4. Data Management
Automated Reporting :
Daily reports at 12:00 and 23:59
CSV export capabilities
Historical data tracking

Database Integration :
QLite database for persistent storage

Data Logging : 
Comprehensive event and metric logging

Main Interface Controls
ROI Management :

Add Region: Define new monitoring areas
Delete Region: Remove existing regions
Clear All: Reset all monitoring regions
Start: Begin monitoring process
Stop: Halt monitoring
Resolution Slider: Adjust capture resolution
Interval Control: Set capture frequency
Metrics Dashboard :
Real-time health display
Event counters
Performance graphs

Data Analysis Features
Health Metrics
Real-time health score calculation

Historical trend analysis

Performance degradation tracking

Reporting
Automated daily reports

Custom date range exports

Multiple export formats (CSV, JSON)
>>>>>>> c1c6791 (commit #1)

# VICE (Visual Identification and Conveyance Enhancement)

## Real-Time Anomaly Detection and Analytics Platform for Industrial Conveyor Systems

because this is an external source all AMZN Specific data and information has been stripped. Please reach out to TTERRAG for more information on VICE Specifics.

VICE is an advanced monitoring system designed for industrial conveyor belts, providing real-time detection of operational anomalies, performance metrics, and automated reporting capabilities. It leverages computer vision and color detection to identify various operational states and potential issues.

VICE was created to fill a gap in operational data collection. The largest organ of the FC cuurently collects no logs and the only visualization of the conveyance is an internet explorer page that only 13 CPUs have access to in the wareouse. The increase accountability on RME for conveyance faults and to highlight errors and collect data VICE has been created to absolve these issues.

Since the initial VICE application was launched at SMF1 there has been 3.2 versions the latest of which revised the database structure to its current form, the current state of VICE is highly optimized for database creation with the added benefit of user visuals. the highly optimized data collection of VICE is housed in seperate databases that are relational by nature. These relational datastores are secured on AWS servers and local housings on SMF1 property. 

### Needed features

1. ** ROI Families**
   - VICE is desperately in need of ROI families. because the user is responsible for defining ROIs they need the ability to house these ROIs in nesting families, This will ensure that each ROI is tracked in its microclimate as well as the macro. providing these families will also create valid data keeping that will enhance data output and insight creation.
   - Additionally VICE would benefit greatly from a UI and ROI simplification to make the user experience of adding ROIs simpler.

### Key Features

1. **Real-Time Monitoring**
   - Multi-region support for simultaneous monitoring
   - Detection of jams, standdowns, full capacity, gridlock, and emergency stops

2. **Alert System**
   - Real-time visual alerts for detected anomalies
   - Slack integration for automated notifications
   - Configurable thresholds for different conditions

3. **Performance Metrics**
   - Real-time conveyor health calculation
   - Event counting (jams, standdowns, emergency stops, full capacity events)
   - Mean time between failures and operational efficiency metrics

4. **Data Management**
   - Automated daily reporting
   - CSV export capabilities
   - SQLite database integration for persistent storage

5. **User Interface**
   - Intuitive ROI management
   - Resolution and interval controls
   - Real-time metrics dashboard

### Technical Highlights

- Developed using Python with OpenCV for computer vision tasks
- Utilizes machine learning algorithms for predictive maintenance
- RESTful API for seamless integration with other systems
- Web-based interface for easy access and monitoring

### Impact and Results

- Over 800 hours of successful operation
- 2.2 million+ data points collected per shift
- 32 daily active users during initial testing phase
- Significant reduction in manual monitoring requirements
- Improved response time to conveyor issues by an average of 32 seconds

### Future Development

- Implementation of advanced machine learning models for predictive analytics
- Expansion of the system to cover additional facilities and departments
- Integration with existing warehouse management and maintenance systems

### Skills Demonstrated

- Computer Vision
- Machine Learning
- Data Analysis and Visualization
- Real-time Systems Development
- Database Management
- API Development
- User Interface Design
- Industrial IoT Applications
- End to end development of software
- Data pipeline creation

This project showcases a comprehensive approach to industrial process optimization, combining cutting-edge technologies with practical applications to solve real-world challenges in logistics and operations.

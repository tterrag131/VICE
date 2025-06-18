import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

# --- Configuration ---
# The script will look for the databases in the same directory it is run from.
CRITICAL_DB_NAME = "vice_main_database.db"
INEFFICIENCY_DB_NAME = "vice_inefficiency_events.db"
ANALYSIS_TIMEFRAME_HOURS = 24  # Set the lookback period for "recent" stats.

# --- Helper Functions ---
def format_duration(seconds: float, long_format: bool = False) -> str:
    """Converts seconds into a human-readable string like '1h 15m 30s'."""
    if seconds is None or seconds < 0:
        return "N/A"
    
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if long_format:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
        
    return " ".join(parts)

def print_header(title: str):
    """Prints a formatted header."""
    print("\n" + "=" * 50)
    print(f" {title.upper()}")
    print("=" * 50)

def print_stat(label: str, value: Any, unit: str = ""):
    """Prints a formatted key-value stat line."""
    print(f"  {label:<30} | {value} {unit}")

# --- Main Pulse Check Function ---
def generate_pulse_check():
    """Connects to the DBs and prints a summary report."""
    base_dir = Path(__file__).resolve().parent
    critical_db_path = base_dir / CRITICAL_DB_NAME
    inefficiency_db_path = base_dir / INEFFICIENCY_DB_NAME

    # --- Critical Events Analysis ---
    if not critical_db_path.exists():
        print(f"ERROR: Critical events database not found at '{critical_db_path}'")
    else:
        try:
            with sqlite3.connect(f'file:{critical_db_path}?mode=ro', uri=True) as conn:
                cursor = conn.cursor()
                print_header("VICE Pulse Check - Critical Events")
                
                # Total event count
                cursor.execute("SELECT COUNT(*) FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP')")
                total_events = cursor.fetchone()[0]
                print_stat("Total Critical Events Logged", total_events)

                # Total downtime
                cursor.execute("SELECT SUM(duration_seconds) FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP')")
                total_downtime = cursor.fetchone()[0] or 0.0
                print_stat("Total Downtime", format_duration(total_downtime, long_format=True))

                # Average Time To Recover (MTTR)
                avg_duration = total_downtime / total_events if total_events > 0 else 0
                print_stat("Average Event Duration (MTTR)", f"{avg_duration:.2f}", "seconds")
                
                # Most problematic ROI
                cursor.execute("""
                    SELECT roi_title, SUM(duration_seconds) as total_duration 
                    FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP')
                    GROUP BY roi_title ORDER BY total_duration DESC LIMIT 1
                """)
                top_offender = cursor.fetchone()
                if top_offender:
                    print_stat("Top Downtime Contributor", f"{top_offender[0]} ({format_duration(top_offender[1])})")
                else:
                    print_stat("Top Downtime Contributor", "N/A")

                # --- Recent Critical Stats (Last 24 Hours) ---
                print_header(f"Recent Critical Performance - Last {ANALYSIS_TIMEFRAME_HOURS} Hours")
                time_threshold = datetime.now() - timedelta(hours=ANALYSIS_TIMEFRAME_HOURS)
                
                cursor.execute("SELECT COUNT(*) FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP') AND end_time >= ?", (time_threshold,))
                recent_events = cursor.fetchone()[0]
                print_stat("Critical Events (Recent)", recent_events)
                
                cursor.execute("SELECT SUM(duration_seconds) FROM completed_events WHERE primary_event_type IN ('JAM', 'E-STOP') AND end_time >= ?", (time_threshold,))
                recent_downtime = cursor.fetchone()[0] or 0.0
                print_stat("Downtime (Recent)", format_duration(recent_downtime, long_format=True))

                cursor.execute("SELECT primary_event_type, COUNT(*) FROM completed_events WHERE end_time >= ? AND primary_event_type IN ('JAM', 'E-STOP') GROUP BY primary_event_type", (time_threshold,))
                event_breakdown = cursor.fetchall()
                if event_breakdown:
                    print("  Recent Critical Breakdown:")
                    for event_type, count in event_breakdown:
                        print(f"    - {event_type:<26} | {count}")
                else:
                    print_stat("Recent Critical Breakdown", "None")

        except sqlite3.Error as e:
            print(f"\n--- DATABASE ERROR (Critical Events) ---")
            print(f"An error occurred: {e}")

    # --- Inefficiency Events Analysis ---
    if not inefficiency_db_path.exists():
        print(f"\nINFO: Inefficiency events database not found at '{inefficiency_db_path}'. Skipping.")
    else:
        try:
            with sqlite3.connect(f'file:{inefficiency_db_path}?mode=ro', uri=True) as conn:
                cursor = conn.cursor()
                print_header(f"Inefficiency Events - Last {ANALYSIS_TIMEFRAME_HOURS} Hours")
                
                time_threshold = datetime.now() - timedelta(hours=ANALYSIS_TIMEFRAME_HOURS)

                cursor.execute("SELECT COUNT(*) FROM completed_inefficiency_events WHERE end_time >= ?", (time_threshold,))
                recent_inefficiency_events = cursor.fetchone()[0]
                print_stat("Total Inefficiency Events", recent_inefficiency_events)

                cursor.execute("SELECT SUM(duration_seconds) FROM completed_inefficiency_events WHERE end_time >= ?", (time_threshold,))
                total_inefficiency_time = cursor.fetchone()[0] or 0.0
                print_stat("Total Inefficiency Time", format_duration(total_inefficiency_time, long_format=True))

                cursor.execute("SELECT primary_event_type, COUNT(*), SUM(duration_seconds) FROM completed_inefficiency_events WHERE end_time >= ? GROUP BY primary_event_type ORDER BY COUNT(*) DESC", (time_threshold,))
                inefficiency_breakdown = cursor.fetchall()
                if inefficiency_breakdown:
                    print("  Inefficiency Breakdown (Recent):")
                    print(f"    {'Event Type':<18} | {'Count':<8} | {'Total Time'}")
                    print(f"    {'-'*18} | {'-'*8} | {'-'*12}")
                    for event_type, count, total_time in inefficiency_breakdown:
                        print(f"    {event_type:<18} | {count:<8} | {format_duration(total_time)}")
                else:
                    print_stat("Recent Inefficiency Events", "None")

        except sqlite3.Error as e:
            print(f"\n--- DATABASE ERROR (Inefficiency Events) ---")
            print(f"An error occurred: {e}")

    print("=" * 50)

if __name__ == "__main__":
    generate_pulse_check()

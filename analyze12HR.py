import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse, sys
from datetime import time as dt_time, datetime, timedelta
from typing import List, Any

# --- Configuration & Style ---
# Matplotlib style for a professional, dark-themed look
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    "figure.facecolor": "#2c2c2c", "axes.facecolor": "#2c2c2c",
    "axes.edgecolor": "#f0f0f0", "axes.labelcolor": "#f0f0f0",
    "xtick.color": "#f0f0f0", "ytick.color": "#f0f0f0",
    "text.color": "#f0f0f0", "figure.figsize": (18, 10),
    "legend.facecolor": "#3c3c3c", "legend.edgecolor": "#5c5c5c"
})

# Define the scheduled shutdown period to exclude from analysis
SHUTDOWN_START = dt_time(5, 0)  # 05:00
SHUTDOWN_END = dt_time(7, 0)    # 07:00

# --- Helper Functions ---
def format_duration(seconds: float, long_format: bool = False) -> str:
    """Converts seconds into a human-readable string like '1h 15m 30s'."""
    if pd.isna(seconds) or seconds < 0: return "N/A"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if long_format:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    if hours > 0: return f"{hours:.1f} hr"
    if minutes > 0: return f"{minutes:.1f} min"
    return f"{seconds:.1f} sec"

def print_header(title: str):
    print("\n" + "█" * 80)
    print(f"██ {title.upper():^74} ██")
    print("█" * 80)

def print_kpi(label: str, value: Any, unit: str = ""):
    print(f"  {label:<40} : {value} {unit}")

# --- Analysis Functions ---

def analyze_kpis(df: pd.DataFrame, time_period_hours: float) -> dict:
    """Calculates Key Performance Indicators on the filtered dataframe."""
    if df.empty or time_period_hours <= 0:
        return {"downtime": 0, "availability": 100.0, "mtbf": "N/A", "mttr": "N/A", "count": 0}
    
    total_downtime = df['duration_seconds'].sum()
    failure_count = len(df)
    total_time_seconds = time_period_hours * 3600
    total_uptime = total_time_seconds - total_downtime
    
    availability = (total_uptime / total_time_seconds) * 100 if total_time_seconds > 0 else 100.0
    mttr = total_downtime / failure_count if failure_count > 0 else 0
    mtbf = total_uptime / failure_count if failure_count > 0 else total_uptime

    return {"downtime": total_downtime, "availability": availability, "mtbf": mtbf, "mttr": mttr, "count": failure_count}

def create_detailed_roi_report(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a comprehensive breakdown for every ROI with a critical event."""
    if df.empty: return pd.DataFrame()

    summary = df.groupby('roi_title').agg(
        total_downtime=('duration_seconds', 'sum'),
        event_count=('duration_seconds', 'count'),
        avg_duration=('duration_seconds', 'mean'),
        max_duration=('duration_seconds', 'max')
    ).reset_index()
    type_counts = df.groupby(['roi_title', 'primary_event_type']).size().unstack(fill_value=0)
    report = pd.merge(summary, type_counts, on='roi_title', how='left').fillna(0)
    return report.sort_values('total_downtime', ascending=False)

def perform_pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies top offenders and their primary failure mode by duration."""
    if df.empty: return pd.DataFrame()

    # Determine dominant event type by duration for each ROI
    dominant_event_type = df.groupby(['roi_title', 'primary_event_type'])['duration_seconds'].sum().unstack(fill_value=0)
    dominant_event_type['dominant_event'] = dominant_event_type.idxmax(axis=1)

    # Calculate total duration and merge
    pareto_duration = df.groupby('roi_title')['duration_seconds'].sum().reset_index()
    pareto_combined = pd.merge(pareto_duration, dominant_event_type[['dominant_event']], on='roi_title')
    
    return pareto_combined.sort_values('duration_seconds', ascending=False).head(10)

# --- Visualization Functions ---

def generate_events_by_hour_chart(df: pd.DataFrame, output_path: Path):
    """Creates a bar chart of critical events per hour."""
    if df.empty or 'start_time' not in df.columns: return
    df['hour'] = df['start_time'].dt.hour
    hourly_events = df.groupby('hour')['primary_event_type'].value_counts().unstack(fill_value=0)
    
    if not hourly_events.empty:
        if 'JAM' not in hourly_events: hourly_events['JAM'] = 0
        if 'E-STOP' not in hourly_events: hourly_events['E-STOP'] = 0
        
        ax = hourly_events[['JAM', 'E-STOP']].plot(kind='bar', stacked=True, color={'JAM': '#f97316', 'E-STOP': '#ef4444'})
        plt.title('Critical Events by Hour of Day', fontsize=18, weight='bold')
        plt.xlabel('Hour of Day (24h format)', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Event Type')
        plt.tight_layout()
        plt.savefig(output_path, facecolor=plt.rcParams['figure.facecolor'])
        plt.close()
        print(f"\n[+] Chart saved: {output_path}")

def generate_pareto_chart(pareto_df: pd.DataFrame, output_path: Path):
    """Creates a color-coded bar chart for the top 5 downtime contributors."""
    if pareto_df.empty: return
    
    pareto_top5 = pareto_df.head(5).sort_values('duration_seconds', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors based on dominant event type
    colors = pareto_top5['dominant_event'].map({'JAM': '#f97316', 'E-STOP': '#ef4444'}).fillna('grey')
    
    ax.barh(pareto_top5['roi_title'], pareto_top5['duration_seconds'], color=colors)
    ax.set_title('Top 5 Downtime Contributors by Duration', fontsize=18, weight='bold')
    ax.set_xlabel('Total Downtime (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=plt.rcParams['figure.facecolor'])
    plt.close()
    print(f"[+] Pareto Chart saved: {output_path}")

def generate_event_timeline_chart(df: pd.DataFrame, output_path: Path):
    """Creates a Gantt-style chart of all critical events."""
    if df.empty: return
    
    df_sorted = df.sort_values('start_time')
    rois = df_sorted['roi_title'].unique()
    
    fig, ax = plt.subplots(figsize=(20, max(10, len(rois) * 0.5)))
    
    # Map event types to their colors
    colors = {'JAM': '#f97316', 'E-STOP': '#ef4444'}
    y_ticks = []
    y_labels = []

    for i, roi in enumerate(rois):
        y_ticks.append(i)
        y_labels.append(roi)
        for _, event in df_sorted[df_sorted['roi_title'] == roi].iterrows():
            start = mdates.date2num(event['start_time'])
            end = mdates.date2num(event['end_time'])
            ax.barh(i, end - start, left=start, height=0.6, align='center',
                    color=colors.get(event['primary_event_type'], 'grey'),
                    edgecolor='none') # No border for better color visibility
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel("Time")
    ax.set_title("Timeline of Critical Events", fontsize=20, weight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=plt.rcParams['figure.facecolor'])
    plt.close()
    print(f"\n[+] Event Timeline Chart saved: {output_path}")

# --- Main Script ---

def main(db_files: List[str]):
    shift_data = []

    for db_path_str in db_files:
        db_path = Path(db_path_str)
        if not db_path.exists():
            print(f"WARNING: Database file not found, skipping: {db_path}"); continue
        try:
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
                df = pd.read_sql_query("SELECT * FROM completed_events", conn, parse_dates=['start_time', 'end_time'])
                if not df.empty:
                    shift_data.append({'name': db_path.stem, 'data': df})
        except Exception as e:
            print(f"ERROR: Could not read from database {db_path}: {e}"); return
    
    if not shift_data:
        print("No valid event data found in the specified database(s)."); return

    # Combine all data for overall analysis
    all_events_df = pd.concat([s['data'] for s in shift_data], ignore_index=True)
    
    # --- Filter out scheduled downtime ---
    original_count = len(all_events_df)
    filtered_df = all_events_df[~all_events_df['start_time'].dt.time.between(SHUTDOWN_START, SHUTDOWN_END)].copy()
    shutdown_events = original_count - len(filtered_df)
    
    if filtered_df.empty:
        print("No event data remains after excluding shutdown period."); return

    start_date = filtered_df['start_time'].min(); end_date = filtered_df['end_time'].max()
    total_hours = (end_date - start_date).total_seconds() / 3600
    
    print_header("VICE Overall Analysis Report")
    print(f"  Analysis Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  (Excluded {shutdown_events} events during {SHUTDOWN_START}-{SHUTDOWN_END} shutdown window)")

    # --- Overall KPIs ---
    kpis = analyze_kpis(filtered_df[filtered_df['primary_event_type'].isin(['JAM', 'E-STOP'])], total_hours)
    print_header("Overall Key Performance Indicators")
    print_kpi("System Availability", f"{kpis['availability']:.2f}%")
    print_kpi("Total Critical Events", kpis['count'])
    print_kpi("Total Downtime", format_duration(kpis['downtime'], long_format=True))
    print_kpi("Mean Time Between Failures (MTBF)", f"{kpis['mtbf']/3600:.2f} hours" if isinstance(kpis['mtbf'], (int, float)) else "N/A")
    print_kpi("Mean Time To Recover (MTTR)", f"{kpis['mttr']:.2f} seconds")

    # --- Shift Comparison ---
    if len(shift_data) > 1:
        print_header("Shift-over-Shift Performance Comparison")
        shift_kpis = []
        for shift in shift_data:
            shift_df_filtered = shift['data'][~shift['data']['start_time'].dt.time.between(SHUTDOWN_START, SHUTDOWN_END)]
            shift_hours = 12.0 # Assume each file is a 12 hour report
            kpis_shift = analyze_kpis(shift_df_filtered[shift_df_filtered['primary_event_type'].isin(['JAM', 'E-STOP'])], shift_hours)
            shift_kpis.append({'Shift': shift['name'], **kpis_shift})
        
        shift_df = pd.DataFrame(shift_kpis)
        shift_df['downtime'] = shift_df['downtime'].apply(format_duration)
        shift_df['availability'] = shift_df['availability'].apply(lambda x: f"{x:.2f}%")
        shift_df['mtbf'] = shift_df['mtbf'].apply(lambda x: f"{x/3600:.2f}h" if isinstance(x, (int, float)) else x)
        shift_df['mttr'] = shift_df['mttr'].apply(lambda x: f"{x:.2f}s" if isinstance(x, (int, float)) else x)
        print(shift_df.rename(columns={'count': 'events'}).to_string(index=False))

    # --- Detailed ROI Breakdown ---
    roi_report_df = create_detailed_roi_report(filtered_df[filtered_df['primary_event_type'].isin(['JAM', 'E-STOP'])])
    print_header("Comprehensive ROI Breakdown (All ROIs)")
    if not roi_report_df.empty:
        roi_report_df['total_downtime'] = roi_report_df['total_downtime'].apply(format_duration)
        roi_report_df['avg_duration'] = roi_report_df['avg_duration'].apply(lambda x: f"{x:.1f}s")
        roi_report_df['max_duration'] = roi_report_df['max_duration'].apply(lambda x: f"{x:.1f}s")
        print(roi_report_df.to_string(index=False))
    else:
        print("  No critical events to analyze.")
        
    # --- Generate Visualizations ---
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    generate_events_by_hour_chart(filtered_df, output_dir / f"events_by_hour_{datetime.now():%Y%m%d_%H%M}.png")
    
    pareto_df = perform_pareto_analysis(filtered_df)
    generate_pareto_chart(pareto_df, output_dir / f"top_5_downtime_rois_{datetime.now():%Y%m%d_%H%M}.png")
    
    generate_event_timeline_chart(filtered_df, output_dir / f"event_timeline_{datetime.now():%Y%m%d_%H%M}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VICE 12-hour report databases.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('db_files', nargs='*', help="One or more specific paths to .db report files.")
    parser.add_argument('--dir', '-d', dest='report_dir', type=str, help="Path to a directory containing .db report files to analyze.")
    args = parser.parse_args()
    files_to_process = args.db_files
    if args.report_dir:
        report_directory = Path(args.report_dir)
        if report_directory.is_dir():
            print(f"Searching for .db files in: {report_directory}")
            db_in_dir = list(report_directory.glob("*.db"))
            if not db_in_dir: print(f"No .db files found in the specified directory.")
            files_to_process.extend(db_in_dir)
        else:
            print(f"ERROR: Provided directory path does not exist: {args.report_dir}"); sys.exit(1)
    if not files_to_process:
        parser.error("No input files provided. Please specify one or more .db files or use the --dir option.")
    
    main(sorted(list(set(files_to_process))))


# ============= DATA EXPORT & VISUALIZATION =============

import re
from datetime import datetime, time
import uuid
import openpyxl
from openpyxl import load_workbook
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, HourLocator
import pandas as pd
from profile import UserProfile, ProfileCollector

import re
import pandas as pd
from datetime import datetime, timedelta

def parse_plan_to_timeline(plan_text: str, date_str: str = None) -> pd.DataFrame:
    """Parse LLM plan text into structured timeline"""

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Unified regex:
    # Supports:
    # 09:00-10:00: Activity
    # 09:00 - Activity (60min)
    # * 09:00: Activity (30min)
    # - 09:00 - Activity
    pattern = re.compile(
        r'[*\-•]?\s*'
        r'(?P<start>\d{1,2}:\d{2})'
        r'(?:\s*[-–]\s*(?P<end>\d{1,2}:\d{2}))?'
        r'\s*[:\-]?\s*'
        r'(?P<activity>.+?)'
        r'(?:\((?P<duration>\d+)\s*min\))?'
        r'(?=\n|$)'
    )

    timeline_data = []

    for match in pattern.finditer(plan_text):
        start_time = match.group("start")
        end_time = match.group("end")
        activity = match.group("activity").strip()
        duration = match.group("duration")

        # Clean activity text
        activity = re.sub(r'\(\d+\s*min\)', '', activity).strip()

        # If no explicit end time but duration exists → calculate
        if not end_time:
            if duration:
                duration_min = int(duration)
            else:
                duration_min = 60  # default

            start_dt = datetime.strptime(start_time, "%H:%M")
            end_dt = start_dt + timedelta(minutes=duration_min)
            end_time = end_dt.strftime("%H:%M")

        # Skip 0-minute tasks if you want
        if duration == "0":
            continue

        category = categorize_activity(activity)

        timeline_data.append({
            "date": date_str,
            "start_time": start_time,
            "end_time": end_time,
            "activity": activity[:100],
            "category": category
        })

    df = pd.DataFrame(timeline_data)
    print(f"Parsed {len(df)} activities")
    return df


def categorize_activity(activity_text: str) -> str:
    """Categorize activity based on keywords"""
    activity_lower = activity_text.lower()
    
    categories = {
        'Exercise': ['exercise', 'workout', 'gym', 'run', 'walk', 'yoga'],
        'Work': ['work', 'project', 'meeting', 'study', 'coding', 'email'],
        'Learning': ['learn', 'read', 'course', 'practice', 'study'],
        'Health': ['breakfast', 'lunch', 'dinner', 'meal', 'eat', 'sleep'],
        'Leisure': ['relax', 'hobby', 'game', 'watch', 'music', 'friends'],
        'Routine': ['shower', 'commute', 'prepare', 'clean', 'organize']
    }
    
    for category, keywords in categories.items():
        if any(keyword in activity_lower for keyword in keywords):
            return category
    
    return 'Other'


def save_to_excel(profile: UserProfile, plan_text: str, timeline_df: pd.DataFrame, 
                  excel_path: str = "planning_data.xlsx"):
    """Save user profile and plan to Excel with multiple sheets"""
    
    # Generate IDs
    if os.path.exists(excel_path):
        existing_log_ids = pd.read_excel(excel_path, sheet_name='user_logs')['log_id']
        log_id = int(max(existing_log_ids.astype(int))) + 1
    else:
        log_id = 0
    name_id = f"user_{profile.name.lower().replace(' ', '_')}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data
    log_data = {
        'log_id': [log_id],
        'name_id': [name_id],
        'name': [profile.name],
        'timestamp': [timestamp],
        'free_time_hours': [profile.free_time_hours],
        'goals': [profile.goals],
        'struggles': [profile.struggles],
        'routine': [profile.daily_routine.to_json() if not profile.daily_routine.empty else '']
    }
    
    plan_summary_data = {
        'log_id': [log_id],
        'name_id': [name_id],
        'date': [datetime.now().strftime("%Y-%m-%d")],
        'plan_text': [plan_text],
        'tasks_count': [len(timeline_df)]
    }
    
    # Add log_id to timeline
    timeline_df['log_id'] = log_id
    timeline_df['name_id'] = name_id
    
    # Create or append to Excel
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Read existing data
            existing_logs = pd.read_excel(excel_path, sheet_name='user_logs')
            existing_plans = pd.read_excel(excel_path, sheet_name='plan_summary')
            existing_timeline = pd.read_excel(excel_path, sheet_name='plan_timeline')
            
            # Append new data
            pd.concat([existing_logs, pd.DataFrame(log_data)]).to_excel(
                writer, sheet_name='user_logs', index=False
            )
            pd.concat([existing_plans, pd.DataFrame(plan_summary_data)]).to_excel(
                writer, sheet_name='plan_summary', index=False
            )
            pd.concat([existing_timeline, timeline_df]).to_excel(
                writer, sheet_name='plan_timeline', index=False
            )
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(log_data).to_excel(writer, sheet_name='user_logs', index=False)
            pd.DataFrame(plan_summary_data).to_excel(writer, sheet_name='plan_summary', index=False)
            timeline_df.to_excel(writer, sheet_name='plan_timeline', index=False)
    
    print(f"\n💾 Data saved to {excel_path}")
    print(f"   Log ID: {log_id}")
    return log_id


def visualize_daily_plan(timeline_df: pd.DataFrame, save_path: str = "daily_plan_viz.png"):
    """Create a timeline visualization of the daily plan"""
    
    if timeline_df.empty:
        print("⚠️  No timeline data to visualize")
        return
    
    # Convert times to datetime for plotting
    date_str = timeline_df['date'].iloc[0]
    timeline_df['start_dt'] = pd.to_datetime(date_str + ' ' + timeline_df['start_time'])
    timeline_df['end_dt'] = pd.to_datetime(date_str + ' ' + timeline_df['end_time'])
    timeline_df['duration'] = (timeline_df['end_dt'] - timeline_df['start_dt']).dt.total_seconds() / 3600
    
    # Category colors
    category_colors = {
        'Exercise': '#FF6B6B',
        'Work': '#4ECDC4',
        'Learning': '#95E1D3',
        'Health': '#FFD93D',
        'Leisure': '#C7CEEA',
        'Routine': '#B8B8D1',
        'Other': '#D3D3D3'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot each activity as a vertical bar
    for idx, row in timeline_df.iterrows():
        color = category_colors.get(row['category'], '#D3D3D3')
        ax.bar(
            x=idx,
            height=row['duration'],
            bottom=row['start_dt'].hour + row['start_dt'].minute/60,
            width=0.6,
            color=color,
            edgecolor='white',
            linewidth=2
        )
        
        # Add activity label
        mid_point = row['start_dt'].hour + row['start_dt'].minute/60 + row['duration']/2
        ax.text(
            idx,
            mid_point,
            row['activity'][:30],
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='black'
        )
    
    # Formatting
    ax.set_ylim(6, 24)  # 6 AM to midnight
    ax.set_ylabel('Time of Day', fontsize=14, fontweight='bold')
    ax.set_title(f'Daily Plan Timeline - {date_str}', fontsize=16, fontweight='bold', pad=20)
    
    # Y-axis: hourly ticks
    ax.set_yticks(range(6, 25, 2))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(6, 25, 2)], fontsize=11)
    ax.set_xticks(range(len(timeline_df)))
    ax.set_xticklabels([])
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    legend_patches = [mpatches.Patch(color=color, label=cat) 
                     for cat, color in category_colors.items() 
                     if cat in timeline_df['category'].values]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Daily plan visualization saved to: {save_path}")
    
    return save_path


# ============= MAIN RUNNER =============

def visualize_graph(agent, save_path: str = "planning_graph.png"):
    """Generate and save graph visualization"""
    try:
        # Get PNG data from mermaid
        png_data = agent.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(save_path, 'wb') as f:
            f.write(png_data)
        
        print(f"\n📊 Graph visualization saved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"\n⚠️  Could not generate graph visualization: {e}")
        print(f"   Attempting fallback: saving Mermaid diagram as ASCII...")
        try:
            # Fallback: save the mermaid diagram as text
            diagram_text = agent.get_graph().draw_mermaid()
            mermaid_path = save_path.replace('.png', '.mmd')
            with open(mermaid_path, 'w') as f:
                f.write(diagram_text)
            print(f"   Mermaid diagram saved to: {mermaid_path}")
            return mermaid_path
        except Exception as fallback_error:
            print(f"   Fallback also failed: {fallback_error}")
            return None

def save_log(log_id: str, plan_text: str):
    # Save plan to text file debug
    txt_path = f"logs\daily_plan_{log_id}.txt"
    with open(txt_path, 'w') as f:
        f.write(plan_text)
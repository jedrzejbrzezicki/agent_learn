"""
Streamlit App - Human-in-the-Loop Planning Agent

Simple UI for:
1. Create profile
2. Generate plan
3. Approve/Reject/Modify plan
4. View history
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from user_profile import UserProfile
from planning_agent_with_checkpoints import create_planning_agent, view_checkpoint_history
from visualization_utils import parse_plan_to_timeline, visualize_daily_plan
from loguru import logger

# Page config
st.set_page_config(
    page_title="AI Planning Assistant",
    page_icon="📅",
    layout="wide"
)

# Initialize session state
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None
if 'plan_state' not in st.session_state:
    st.session_state.plan_state = None
if 'history' not in st.session_state:
    st.session_state.history = []


def create_profile_form():
    """Form to create user profile"""
    st.header("📝 Step 1: Your Profile")
    
    with st.form("profile_form"):
        name = st.text_input("Name", value="Jedrzej")
        free_time = st.number_input("Free time (hours/day)", min_value=0.5, max_value=24.0, value=3.0, step=0.5)
        goals = st.text_area("What do you want to accomplish?", value="learn langchain and langgraph")
        struggles = st.text_area("What do you struggle with?", value="not working during working hours, waiting till evening to start")
        
        submitted = st.form_submit_button("Create Profile")
        
        if submitted:
            profile = UserProfile()
            profile.name = name
            profile.free_time_hours = free_time
            profile.goals = goals
            profile.struggles = struggles
            
            # Simple routine
            profile.daily_routine = pd.DataFrame([
                {'time': '09:00', 'activity': 'Start work', 'duration_mins': 480},
                {'time': '17:00', 'activity': 'Finish work', 'duration_mins': 60},
                {'time': '18:00', 'activity': 'Free time starts', 'duration_mins': 180},
            ])
            
            st.session_state.profile = profile
            st.success("✅ Profile created!")
            st.rerun()


def generate_plan():
    """Generate plan using the agent"""
    st.header("🤖 Step 2: Generate Plan")
    
    profile = st.session_state.profile
    
    # Show profile summary
    with st.expander("👤 Your Profile", expanded=False):
        st.write(f"**Name:** {profile.name}")
        st.write(f"**Free time:** {profile.free_time_hours} hours/day")
        st.write(f"**Goals:** {profile.goals}")
        st.write(f"**Struggles:** {profile.struggles}")
    
    if st.button("🚀 Generate Daily Plan", type="primary"):
        with st.spinner("Creating your plan..."):
            # Create agent if not exists
            if st.session_state.agent is None:
                st.session_state.agent = create_planning_agent("streamlit_checkpoints.db")
            
            # Create thread ID
            thread_id = f"user_{profile.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.thread_id = thread_id
            
            config = {"configurable": {"thread_id": thread_id}}
            
            # Initial state
            initial_state = {
                "user_profile": profile.to_context_string(),
                "initial_plan": "",
                "planning_attempt_count": 0,
                "is_valid": False,
                "validation_issues": [],
                "feedback_for_planner": "",
                "final_plan": "",
                "optimization_notes": "",
                "messages": []
            }
            
            # Run agent
            result = st.session_state.agent.invoke(initial_state, config)
            
            # Store results
            st.session_state.current_plan = result.get('final_plan') or result.get('initial_plan', '')
            st.session_state.plan_state = result
            
            st.success("✅ Plan generated!")
            st.rerun()


def review_plan():
    """Review and approve/reject plan"""
    st.header("📅 Step 3: Review Your Plan")
    
    plan = st.session_state.current_plan
    state = st.session_state.plan_state
    
    # Show plan
    st.subheader("Your Daily Plan")
    st.text_area("", value=plan, height=300, disabled=True, label_visibility="collapsed")
    
    # Show metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Planning Attempts", state.get('planning_attempt_count', 0))
    with col2:
        st.metric("Validated", "✅ Yes" if state.get('is_valid') else "❌ No")
    with col3:
        if state.get('optimization_notes'):
            st.info(f"💡 {state['optimization_notes']}")
    
    # Action buttons
    st.subheader("What do you want to do?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve Plan", type="primary", use_container_width=True):
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'plan': plan,
                'status': 'Approved',
                'thread_id': st.session_state.thread_id
            })
            st.success("✅ Plan approved and saved!")
            
            # Visualize and save
            try:
                timeline_df = parse_plan_to_timeline(plan)
                if not timeline_df.empty:
                    visualize_daily_plan(timeline_df, save_path=f"streamlit_plan_plots/streamlit_plan_{st.session_state.thread_id}.png")
                    st.balloons()
            except Exception as e:
                logger.error(f"Visualization error: {e}")
    
    with col2:
        if st.button("❌ Reject & Regenerate", use_container_width=True):
            st.session_state.current_plan = None
            st.session_state.plan_state = None
            st.warning("Plan rejected. Generate a new one!")
            st.rerun()
    
    with col3:
        if st.button("✏️ Modify Plan", use_container_width=True):
            st.session_state.show_modify = True
            st.rerun()
    
    # Modify section (if requested)
    if st.session_state.get('show_modify', False):
        st.subheader("✏️ Modify Your Plan")
        
        modified_plan = st.text_area(
            "Edit your plan:",
            value=plan,
            height=200,
            key="modified_plan_text"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Modified Plan", type="primary"):
                st.session_state.current_plan = modified_plan
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'plan': modified_plan,
                    'status': 'Modified & Approved',
                    'thread_id': st.session_state.thread_id
                })
                st.session_state.show_modify = False
                st.success("✅ Modified plan saved!")
                st.rerun()
        
        with col2:
            if st.button("Cancel"):
                st.session_state.show_modify = False
                st.rerun()


def show_history():
    """Show plan history"""
    st.header("📜 Plan History")
    
    if not st.session_state.history:
        st.info("No plans yet. Generate your first plan!")
        return
    
    for i, entry in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Plan #{len(st.session_state.history) - i + 1} - {entry['timestamp']} ({entry['status']})"):
            st.text_area("", value=entry['plan'], height=150, disabled=True, key=f"history_{i}", label_visibility="collapsed")
            st.caption(f"Thread ID: {entry['thread_id']}")


def main():
    """Main app"""
    
    # Title
    st.title("🤖 AI Daily Planning Assistant")
    st.markdown("*Plan your day with AI, approve or modify as needed*")
    st.divider()
    
    # Sidebar - Navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to:",
            ["Create Profile", "Generate Plan", "Plan History"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Show current profile if exists
        if st.session_state.profile:
            st.success(f"✅ Profile: {st.session_state.profile.name}")
            if st.button("Reset Profile"):
                st.session_state.profile = None
                st.session_state.current_plan = None
                st.session_state.plan_state = None
                st.rerun()
        
        st.divider()
        
        # Quick stats
        if st.session_state.history:
            st.metric("Total Plans Created", len(st.session_state.history))
            approved = sum(1 for h in st.session_state.history if 'Approved' in h['status'])
            st.metric("Approved Plans", approved)
    
    # Main content based on page
    if page == "Create Profile":
        if st.session_state.profile is None:
            create_profile_form()
        else:
            st.success(f"✅ Profile exists for: {st.session_state.profile.name}")
            st.info("Navigate to 'Generate Plan' to create your daily plan")
            
            with st.expander("View Current Profile"):
                profile = st.session_state.profile
                st.write(f"**Name:** {profile.name}")
                st.write(f"**Free time:** {profile.free_time_hours} hours/day")
                st.write(f"**Goals:** {profile.goals}")
                st.write(f"**Struggles:** {profile.struggles}")
    
    elif page == "Generate Plan":
        if st.session_state.profile is None:
            st.warning("⚠️ Please create a profile first!")
        else:
            if st.session_state.current_plan is None:
                generate_plan()
            else:
                review_plan()
    
    elif page == "Plan History":
        show_history()


if __name__ == "__main__":
    main()

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
from typing import TypedDict, Annotated
import operator
from voice import wait_for_wake_word, record_and_transcribe

@dataclass
class UserProfile:
    """Store user's context for planning"""
    name: str = ""
    free_time_hours: float = 0.0
    goals: str = ""  # What they want to do
    struggles: str = ""  # What they struggle with
    daily_routine: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=['time', 'activity', 'duration_mins']
    ))
    
    def is_complete(self) -> bool:
        """Check if we have minimum info to plan"""
        return bool(self.name and self.free_time_hours > 0 and self.goals)
    
    def to_context_string(self) -> str:
        """Convert profile to LLM-friendly context"""
        routine_str = ""
        if not self.daily_routine.empty:
            routine_str = "\n".join([
                f"- {row['time']}: {row['activity']} ({row['duration_mins']}min)"
                for _, row in self.daily_routine.iterrows()
            ])
        
        return f"""
User Profile:
- Name: {self.name}
- Available free time: {self.free_time_hours} hours/day
- Goals: {self.goals}
- Struggles with: {self.struggles}
- Current routine:
{routine_str if routine_str else "  No routine provided yet"}
"""


class ProfileCollector:
    """Interactive voice-based profile collector"""
    
    QUESTIONS = {
        'name': "What's your name?",
        'free_time': "How many hours of free time do you have per day?",
        'goals': "What do you want to accomplish or do more of?",
        'struggles': "What do you struggle to do or procrastinate on?",
        'routine': "Describe your typical daily routine (or say 'skip')"
    }
    
    def __init__(self):
        self.profile = UserProfile()
    
    def collect_profile(self, use_voice: bool = True) -> UserProfile:
        """Collect user profile interactively with voice by default"""
        
        # Wait for wake word first
        if use_voice:
            wait_for_wake_word()
        else:
            print("\n🎯 Let's build your profile!\n")
        
        # Name
        self.profile.name = self._ask_with_confirm('name', use_voice)
        
        # Free time
        time_str = self._ask_with_confirm('free_time', use_voice)
        try:
            # Extract first number from response
            import re
            numbers = re.findall(r'\d+', time_str)
            self.profile.free_time_hours = float(numbers[0]) if numbers else 2.0
        except:
            self.profile.free_time_hours = 2.0  # Default
        
        # Goals
        self.profile.goals = self._ask_with_confirm('goals', use_voice)
        
        # Struggles
        self.profile.struggles = self._ask_with_confirm('struggles', use_voice)
        
        # Routine (optional)
        routine_response = self._ask_with_confirm('routine', use_voice)
        if 'skip' not in routine_response.lower():
            self.profile.daily_routine = self._parse_routine(routine_response)
        
        print("\n✅ Profile complete!\n")
        return self.profile
    
    def _ask_with_confirm(self, key: str, use_voice: bool) -> str:
        """Ask question, show what was captured, allow override"""
        response = self._ask(key, use_voice)
        
        # Display what was captured
        print(f"\n📝 Captured: '{response}'")
        
        # Ask for confirmation
        override = input("   Press Enter to confirm, or type new answer: ").strip()
        
        return override if override else response
    
    def _ask(self, key: str, use_voice: bool) -> str:
        """Ask a question via text or voice"""
        question = self.QUESTIONS[key]
        print(f"\n❓ {question}")
        
        if use_voice:
            return record_and_transcribe()
        else:
            return input("> ")
    
    def _parse_routine(self, routine_text: str) -> pd.DataFrame:
        """Basic routine parser - placeholder for now"""
        # For MVP, just store as a single activity
        return pd.DataFrame([{
            'time': '09:00',
            'activity': routine_text[:50],  # Truncate
            'duration_mins': 60
        }])


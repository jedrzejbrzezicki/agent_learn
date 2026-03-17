# LangGraph Checkpointing - Complete Guide

## 🎯 What Was Added

Your planning agent now has **full checkpointing** using `SqliteSaver`. Every step of execution is saved and can be:
- Replayed
- Resumed
- Time-traveled
- Branched
- Debugged

---

## 📁 Files

1. **`planning_agent_with_checkpoints.py`** - Main agent with checkpointing
2. **`checkpoint_explorer.py`** - Utility to explore saved checkpoints
3. **`planning_checkpoints.db`** - SQLite database (auto-created)

---

## 🚀 Quick Start

### Run the Agent
```bash
python planning_agent_with_checkpoints.py
```

This will:
1. Create a plan with validation & optimization
2. Save checkpoints at EVERY node
3. Show you the checkpoint history at the end
4. Save to `planning_checkpoints.db`

### Explore Checkpoints
```bash
# List all sessions
python checkpoint_explorer.py list

# View specific session details
python checkpoint_explorer.py view user_jedrzej_20260308_011030

# Compare state evolution
python checkpoint_explorer.py compare user_jedrzej_20260308_011030

# Export checkpoint to JSON
python checkpoint_explorer.py export user_jedrzej_20260308_011030

# Time travel demo
python checkpoint_explorer.py timetravel user_jedrzej_20260308_011030
```

---

## 🔍 What Gets Checkpointed

Every time a node completes, LangGraph saves:

```
Checkpoint #1: After PLANNER node
  - State: {user_profile, initial_plan, attempt=1, ...}
  - Timestamp: 2024-03-07 14:30:52
  
Checkpoint #2: After VALIDATOR node
  - State: {..., is_valid=False, issues=[...]}
  - Timestamp: 2024-03-07 14:30:55
  
Checkpoint #3: After FEEDBACK node
  - State: {..., feedback_for_planner="Fix these..."}
  - Timestamp: 2024-03-07 14:30:56

Checkpoint #4: After PLANNER node (retry)
  - State: {..., attempt=2, initial_plan="revised..."}
  - Timestamp: 2024-03-07 14:30:58

... and so on
```

---

## 💡 Key Concepts

### Thread ID
Each user session gets a unique thread ID:
```python
thread_id = f"user_{profile.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

Example: `user_jedrzej_20240307_143052`

All checkpoints for that session are grouped under this ID.

### Config Object
```python
config = {
    "configurable": {
        "thread_id": thread_id
    }
}
```

This tells LangGraph which "conversation" you're working with.

---

## 🎓 Advanced Features

### 1. Resume from Checkpoint

If your agent crashes or you stop it mid-execution:

```python
from planning_agent_with_checkpoints import create_planning_agent, resume_from_checkpoint

agent = create_planning_agent()
config = {"configurable": {"thread_id": "user_jedrzej_20240307_143052"}}

# Resume from last checkpoint
result = resume_from_checkpoint(agent, config)
```

### 2. View History

```python
from planning_agent_with_checkpoints import create_planning_agent, view_checkpoint_history

agent = create_planning_agent()
config = {"configurable": {"thread_id": "user_jedrzej_20240307_143052"}}

view_checkpoint_history(agent, config)
```

Output:
```
📜 CHECKPOINT HISTORY
==================================================

Checkpoint #1:
  Node: ['planner']
  Attempt: 1
  Valid: N/A
  Created: 2024-03-07 14:30:52.123456

Checkpoint #2:
  Node: ['validator']
  Attempt: 1
  Valid: False
  Issues: 2
  Created: 2024-03-07 14:30:55.789012

...
```

### 3. Time Travel

Get state from any point in history:

```python
agent = create_planning_agent()
config = {"configurable": {"thread_id": "user_jedrzej_20240307_143052"}}

# Get all history
history = list(agent.get_state_history(config))

# Get state from 2 checkpoints ago
past_state = list(reversed(history))[2]

print(f"State at {past_state.created_at}:")
print(f"  Attempt: {past_state.values['planning_attempt_count']}")
print(f"  Plan: {past_state.values['initial_plan'][:100]}...")
```

### 4. Branch from Checkpoint

Modify a past state and continue from there:

```python
# Get state from checkpoint #3
past_state = list(reversed(history))[3]

# Modify it
modified_values = past_state.values.copy()
modified_values['feedback_for_planner'] = "Different feedback here"

# Update state
agent.update_state(config, modified_values)

# Continue execution from modified state
result = agent.invoke(None, config)
```

This creates a **branch** - a different execution path from the same starting point!

---

## 🗄️ Database Schema

The SQLite database has this structure:

```sql
CREATE TABLE checkpoints (
    thread_id TEXT,              -- Session identifier
    checkpoint_id TEXT,          -- Unique checkpoint ID
    parent_checkpoint_id TEXT,   -- Previous checkpoint (for branching)
    checkpoint BLOB,             -- Serialized state
    metadata JSON,               -- Additional info
    created_at TIMESTAMP,        -- When it was saved
    PRIMARY KEY (thread_id, checkpoint_id)
)
```

You can query it directly:

```python
import sqlite3

conn = sqlite3.connect("planning_checkpoints.db")
cursor = conn.cursor()

# Count checkpoints per thread
cursor.execute("""
    SELECT thread_id, COUNT(*) 
    FROM checkpoints 
    GROUP BY thread_id
""")

for thread_id, count in cursor.fetchall():
    print(f"{thread_id}: {count} checkpoints")
```

---

## 🔧 Practical Use Cases

### 1. Debugging
"Why did the planner loop 3 times?"

```bash
python checkpoint_explorer.py compare user_jedrzej_20240307_143052
```

See exactly what changed at each step.

### 2. A/B Testing
Want to try different validation rules?

```python
# Run with strict validation
result1 = agent.invoke(initial_state, config1)

# Branch from checkpoint before validation
past_state = get_checkpoint_before_validator()
modified_state = relax_validation_rules(past_state)
agent.update_state(config2, modified_state)
result2 = agent.invoke(None, config2)

# Compare results!
```

### 3. User Undo
User doesn't like the optimized plan?

```python
# Go back to the validated (but not optimized) plan
checkpoint_before_optimizer = get_state_before_node("optimizer")
agent.update_state(config, checkpoint_before_optimizer.values)

# Show user the un-optimized version instead
```

### 4. Crash Recovery
Agent crashed during optimization?

```python
# Next run - automatically resumes
result = agent.invoke(None, config)  # Continues from last checkpoint!
```

---

## 📊 Visualizing Checkpoints

After running, you'll see:

```
🤖 Creating your daily plan with checkpointing...

📂 Checkpointer initialized: planning_checkpoints.db
🧵 Thread ID: user_jedrzej_20240307_143052

Planning attempt #1
...
✓ Plan validation passed
Plan optimization complete

📜 CHECKPOINT HISTORY
==================================================

Checkpoint #1:
  Node: ['planner']
  Attempt: 1
  ...

Checkpoint #5:
  Node: []
  Attempt: 2
  Valid: True
  Created: 2024-03-07 14:31:05.123456

==================================================

✅ Session complete! Thread ID: user_jedrzej_20240307_143052
💾 Checkpoints saved to: planning_checkpoints.db
```

---

## 🎯 Key Takeaways

1. **Every node execution = 1 checkpoint** automatically saved
2. **Thread ID** groups all checkpoints for a session
3. **State is fully serialized** - can resume from anywhere
4. **Time travel** lets you go back to any point
5. **Branching** lets you try different paths
6. **SQLite = queryable** - you can analyze execution patterns

---

## 🚀 Next Steps

Now that you have checkpointing, you can:

1. **Add human-in-the-loop** - Pause for user approval, resume after
2. **Build multi-day sessions** - Same thread_id across days
3. **A/B test prompts** - Branch and compare different approaches
4. **Add reflection** - Load yesterday's checkpoints to inform today's plan
5. **Analytics** - Query database to find patterns (avg attempts, validation failures, etc)

---

## 🐛 Troubleshooting

**"No checkpoints found"**
- Make sure you ran `planning_agent_with_checkpoints.py` first
- Check that `planning_checkpoints.db` exists

**"Thread ID not found"**
- Use `python checkpoint_explorer.py list` to see all thread IDs
- Copy the exact thread_id (case-sensitive!)

**"Database locked"**
- Close other connections to the database
- SQLite only allows one writer at a time

---

## 📚 Learn More

- [LangGraph Checkpointing Docs](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [Human-in-the-loop Guide](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)
- [Time Travel Tutorial](https://langchain-ai.github.io/langgraph/how-tos/time-travel/)

Happy checkpointing! 🎉

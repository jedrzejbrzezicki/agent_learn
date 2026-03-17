"""
Checkpoint Explorer - Advanced LangGraph Checkpointing Features

This script demonstrates:
1. Viewing checkpoint history
2. Time-traveling to past states
3. Branching from checkpoints
4. Querying checkpoint database directly
"""

import sqlite3
from datetime import datetime
from loguru import logger
import json


def list_all_threads(db_path: str = "planning_checkpoints.db"):
    """List all thread IDs in the checkpoint database"""
    logger.info("\n📋 ALL THREADS IN DATABASE")
    logger.info("=" * 60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # New schema doesn't have created_at, use checkpoint_id ordering instead
    cursor.execute("""
        SELECT DISTINCT thread_id, 
               COUNT(*) as checkpoint_count,
               MIN(checkpoint_id) as first_checkpoint_id,
               MAX(checkpoint_id) as last_checkpoint_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY last_checkpoint_id DESC
    """)
    
    threads = cursor.fetchall()
    
    if not threads:
        logger.warning("No threads found in database")
        conn.close()
        return
    
    for thread_id, count, first_id, last_id in threads:
        logger.info(f"\nThread: {thread_id}")
        logger.info(f"  Checkpoints: {count}")
        logger.info(f"  First ID: {first_id}")
        logger.info(f"  Last ID: {last_id}")
    
    conn.close()
    logger.info("\n" + "=" * 60 + "\n")


def view_thread_details(thread_id: str, db_path: str = "planning_checkpoints.db"):
    """View detailed checkpoint information for a specific thread"""
    logger.info(f"\n🔍 THREAD DETAILS: {thread_id}")
    logger.info("=" * 60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # New schema - no created_at, metadata is BLOB
    cursor.execute("""
        SELECT checkpoint_id, parent_checkpoint_id, checkpoint_ns, type
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id ASC
    """, (thread_id,))
    
    checkpoints = cursor.fetchall()
    
    if not checkpoints:
        logger.warning(f"No checkpoints found for thread: {thread_id}")
        conn.close()
        return
    
    for i, (cp_id, parent_id, cp_ns, cp_type) in enumerate(checkpoints, 1):
        logger.info(f"\n#{i} Checkpoint: {cp_id}")
        logger.info(f"  Namespace: {cp_ns}")
        logger.info(f"  Type: {cp_type}")
        logger.info(f"  Parent: {parent_id if parent_id else 'None'}")
    
    conn.close()
    logger.info("\n" + "=" * 60 + "\n")


def compare_checkpoints(thread_id: str, db_path: str = "planning_checkpoints.db"):
    """Compare state evolution across checkpoints"""
    from planning_agent_with_checkpoints import create_planning_agent
    
    logger.info(f"\n📊 STATE EVOLUTION: {thread_id}")
    logger.info("=" * 60)
    
    agent = create_planning_agent(db_path)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        history = list(agent.get_state_history(config))
    except Exception as e:
        logger.error(f"Error getting state history: {e}")
        return
    
    for i, state_snapshot in enumerate(reversed(history)):
        values = state_snapshot.values
        
        logger.info(f"\n--- Checkpoint #{i+1} ---")
        logger.info(f"Next node: {state_snapshot.next}")
        logger.info(f"Planning attempt: {values.get('planning_attempt_count', 0)}")
        logger.info(f"Is valid: {values.get('is_valid', 'N/A')}")
        
        if values.get('validation_issues'):
            logger.info(f"Issues found: {len(values['validation_issues'])}")
            for issue in values['validation_issues'][:2]:  # Show first 2
                logger.info(f"  - {issue}")
        
        if values.get('initial_plan'):
            plan_preview = values['initial_plan'][:100].replace('\n', ' ')
            logger.info(f"Plan preview: {plan_preview}...")
    
    logger.info("\n" + "=" * 60 + "\n")


def export_checkpoint_to_json(thread_id: str, checkpoint_num: int = -1, 
                               output_file: str = None, 
                               db_path: str = "planning_checkpoints.db"):
    """Export a specific checkpoint state to JSON file"""
    from planning_agent_with_checkpoints import create_planning_agent
    
    agent = create_planning_agent(db_path)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        history = list(agent.get_state_history(config))
    except Exception as e:
        logger.error(f"Error getting state history: {e}")
        return
    
    if not history:
        logger.error(f"No checkpoints found for thread: {thread_id}")
        return
    
    # Get checkpoint (default to last one)
    state_snapshot = list(reversed(history))[checkpoint_num]
    
    if output_file is None:
        output_file = f"checkpoint_{thread_id}_{checkpoint_num}.json"
    
    export_data = {
        "thread_id": thread_id,
        "checkpoint_num": checkpoint_num,
        "next_node": state_snapshot.next,
        "state": state_snapshot.values
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    logger.success(f"✅ Checkpoint exported to: {output_file}")


def time_travel_demo(thread_id: str, db_path: str = "planning_checkpoints.db"):
    """Demonstrate time-traveling to a past checkpoint"""
    from planning_agent_with_checkpoints import create_planning_agent
    
    logger.info(f"\n⏰ TIME TRAVEL DEMO: {thread_id}")
    logger.info("=" * 60)
    
    agent = create_planning_agent(db_path)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        history = list(agent.get_state_history(config))
    except Exception as e:
        logger.error(f"Error getting state history: {e}")
        return
    
    if len(history) < 3:
        logger.warning("Need at least 3 checkpoints for time travel demo")
        return
    
    # Get checkpoint from middle of execution
    mid_checkpoint = list(reversed(history))[len(history)//2]
    
    logger.info(f"\nCheckpoint from middle of execution:")
    logger.info(f"State at that point:")
    logger.info(f"  Attempt: {mid_checkpoint.values.get('planning_attempt_count', 0)}")
    logger.info(f"  Valid: {mid_checkpoint.values.get('is_valid', 'N/A')}")
    
    # You could modify state here and continue from this point
    # agent.update_state(config, {"some_field": "modified_value"})
    # result = agent.invoke(None, config)
    
    logger.info("\n💡 From here you could:")
    logger.info("  - Modify the state")
    logger.info("  - Resume execution from this point")
    logger.info("  - Create a branch with different decisions")
    
    logger.info("\n" + "=" * 60 + "\n")


def main():
    """Interactive checkpoint explorer"""
    import sys
    
    db_path = "planning_checkpoints.db"
    
    if len(sys.argv) < 2:
        print("\n🔧 Checkpoint Explorer - Usage:")
        print("  python checkpoint_explorer.py list                    # List all threads")
        print("  python checkpoint_explorer.py view <thread_id>        # View thread details")
        print("  python checkpoint_explorer.py compare <thread_id>     # Compare checkpoints")
        print("  python checkpoint_explorer.py export <thread_id>      # Export to JSON")
        print("  python checkpoint_explorer.py timetravel <thread_id>  # Time travel demo")
        print()
        
        # Default: list all threads
        list_all_threads(db_path)
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_all_threads(db_path)
    
    elif command == "view" and len(sys.argv) > 2:
        thread_id = sys.argv[2]
        view_thread_details(thread_id, db_path)
    
    elif command == "compare" and len(sys.argv) > 2:
        thread_id = sys.argv[2]
        compare_checkpoints(thread_id, db_path)
    
    elif command == "export" and len(sys.argv) > 2:
        thread_id = sys.argv[2]
        export_checkpoint_to_json(thread_id, db_path=db_path)
    
    elif command == "timetravel" and len(sys.argv) > 2:
        thread_id = sys.argv[2]
        time_travel_demo(thread_id, db_path)
    
    else:
        logger.error("Invalid command or missing thread_id")


if __name__ == "__main__":
    main()
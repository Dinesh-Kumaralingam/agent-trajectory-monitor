"""Ingest real SWE-agent trajectory files (.traj JSON) into telemetry.db.

This script:
1. Connects to data/telemetry.db and clears existing data (agent_actions, agent_sessions)
2. Searches for all .traj files under the project root
3. Parses JSON safely and extracts session outcomes and trajectory steps
4. Inserts real agent sessions and actions into the database
"""

import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Project root: two levels up from this script (data/ -> agent-telemetry/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "telemetry.db"


def clear_existing_data(conn):
    """Clear existing data from agent_actions and agent_sessions tables."""
    c = conn.cursor()
    c.execute("DELETE FROM agent_actions")
    c.execute("DELETE FROM agent_sessions")
    c.execute("DELETE FROM hallucination_events")
    conn.commit()
    print("Cleared existing data from agent_actions, agent_sessions, hallucination_events")


def insert_session(conn, session_id, task_desc, outcome):
    """Insert a session into agent_sessions table."""
    c = conn.cursor()
    start_time = datetime.now().isoformat()
    end_time = (datetime.now() + timedelta(minutes=10)).isoformat()
    c.execute(
        "INSERT INTO agent_sessions (session_id, task_description, start_time, end_time, outcome) VALUES (?, ?, ?, ?, ?)",
        (session_id, task_desc[:200], start_time, end_time, outcome)
    )
    conn.commit()


def count_error_keywords(text):
    """Count error-related keywords in text (case-insensitive)."""
    if not text:
        return 0
    text_lower = text.lower()
    keywords = ['error', 'traceback', 'exception', 'syntaxerror', 'typeerror']
    return sum(text_lower.count(kw) for kw in keywords)


def insert_action(conn, session_id, step_idx, action_type, command, reasoning_len,
                  error_kw, exit_code, is_repeat, time_gap, sem_sim):
    """Insert an action into agent_actions table."""
    c = conn.cursor()
    timestamp = (datetime.now() + timedelta(seconds=step_idx * 5)).isoformat()
    c.execute("""
        INSERT INTO agent_actions
        (session_id, action_type, command, args, exit_code, timestamp,
         time_since_last_action, reasoning_length, semantic_similarity,
         error_keywords, is_repeat_command, step_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, action_type, command, json.dumps([]),
        exit_code, timestamp, time_gap, reasoning_len,
        sem_sim, error_kw, int(is_repeat), step_idx
    ))
    conn.commit()


def process_trajectory_file(filepath):
    """Process a single .traj file and return session data and actions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use filename (without extension) as session_id
    session_id = filepath.stem

    # Get outcome from info.exit_status
    info = data.get('info', {})
    exit_status = info.get('exit_status', '')
    outcome = 'success' if exit_status == 'submitted' else 'hallucination'

    # Get task description
    task_desc = data.get('problem_statement', data.get('instance_id', 'Unknown task'))

    # Process trajectory steps
    trajectory = data.get('trajectory', [])
    actions = []

    prev_action = None

    for step_idx, step in enumerate(trajectory):
        # action_type: first word of action string
        action_str = step.get('action', step.get('command', ''))
        action_type = action_str.split()[0] if action_str else 'unknown'

        # command: full action string
        command = action_str

        # reasoning_length: length of thought or response
        thought = step.get('thought', step.get('response', ''))
        reasoning_len = len(thought) if thought else 0

        # error_keywords: count in observation
        observation = step.get('observation', step.get('output', ''))
        error_kw = count_error_keywords(observation)

        # exit_code: 1 if errors found, else 0
        exit_code = 1 if error_kw > 0 else 0

        # is_repeat_command: compare with previous action
        is_repeat = (action_str == prev_action) if prev_action else False
        prev_action = action_str

        # time_since_last_action: random float between 2.0 and 15.0
        time_gap = random.uniform(2.0, 15.0)

        # semantic_similarity: random float between 0.5 and 1.0
        sem_sim = random.uniform(0.5, 1.0)

        actions.append({
            'step_idx': step_idx,
            'action_type': action_type,
            'command': command,
            'reasoning_len': reasoning_len,
            'error_kw': error_kw,
            'exit_code': exit_code,
            'is_repeat': is_repeat,
            'time_gap': time_gap,
            'sem_sim': sem_sim
        })

    return session_id, outcome, task_desc, actions


def main():
    # Search for .traj files anywhere under the project root
    traj_files = list(PROJECT_ROOT.rglob("*.traj"))
    if not traj_files:
        print(f"No .traj files found under {PROJECT_ROOT}")
        print("Please place your .traj files anywhere in the project directory.")
        sys.exit(1)

    print(f"Found {len(traj_files)} .traj files to process:")
    for f in traj_files:
        print(f"  - {f}")

    conn = sqlite3.connect(DB_PATH)
    clear_existing_data(conn)

    total_sessions = 0
    total_actions = 0

    for filepath in traj_files:
        try:
            session_id, outcome, task_desc, actions = process_trajectory_file(filepath)

            # Insert session
            insert_session(conn, session_id, task_desc, outcome)
            total_sessions += 1

            # Insert actions
            for action in actions:
                insert_action(
                    conn, session_id, action['step_idx'],
                    action['action_type'], action['command'],
                    action['reasoning_len'], action['error_kw'],
                    action['exit_code'], action['is_repeat'],
                    action['time_gap'], action['sem_sim']
                )
                total_actions += 1

            print(f"  Processed {filepath.name}: {len(actions)} actions, outcome={outcome}")

        except json.JSONDecodeError as e:
            print(f"  Error parsing {filepath.name}: {e}")
        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    conn.close()
    print(f"\nDone! Ingested {total_sessions} sessions with {total_actions} total actions into {DB_PATH}")


if __name__ == "__main__":
    main()

"""Mock telemetry generator: creates realistic agent failure data for the ROCKET classifier.

This script generates 100 synthetic agent sessions with mathematically realistic patterns:
- Success sessions: steady, low-error, high-diversity command sequences
- Infinite loops: high repeat rate, short inter-command gaps, low semantic diversity
- Hallucinations: high error rates, erratic timing, semantic drift, command complexity spikes
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)

DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS agent_sessions")
    c.execute("DROP TABLE IF EXISTS agent_actions")
    c.execute("DROP TABLE IF EXISTS hallucination_events")

    c.execute("""
    CREATE TABLE agent_sessions (
        session_id TEXT PRIMARY KEY,
        task_description TEXT,
        start_time DATETIME,
        end_time DATETIME,
        outcome TEXT
    )""")

    c.execute("""
    CREATE TABLE agent_actions (
        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        action_type TEXT,
        command TEXT,
        args TEXT,
        exit_code INTEGER,
        timestamp DATETIME,
        time_since_last_action REAL,
        reasoning_length INTEGER,
        semantic_similarity REAL,
        error_keywords INTEGER,
        is_repeat_command BOOLEAN,
        step_index INTEGER
    )""")

    c.execute("""
    CREATE TABLE hallucination_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        detected_at DATETIME,
        event_type TEXT
    )""")

    conn.commit()
    conn.close()

def generate_success_session(session_num, base_time):
    """Generate a successful agent session with steady progress."""
    session_id = f"success_{session_num:04d}"
    n_steps = np.random.randint(15, 35)
    start = base_time
    actions = []

    cmd_pool = [
        ("git", "clone", "https://github.com/example/repo.git"),
        ("cd", "project"),
        ("python", "-m", "venv", ".venv"),
        ("pip", "install", "-r", "requirements.txt"),
        ("pytest", "tests/"),
        ("python", "train.py"),
        ("git", "add", "."),
        ("git", "commit", "-m", "fix"),
    ]

    prev_reasoning = ""
    for step in range(n_steps):
        dt = start + timedelta(seconds=step * np.random.uniform(5, 15))
        cmd = list(cmd_pool[np.random.randint(0, len(cmd_pool))])
        action_type = np.random.choice(["bash", "think", "write", "read"], p=[0.5, 0.2, 0.2, 0.1])

        reasoning_len = int(np.random.normal(150, 50))
        if prev_reasoning:
            # High similarity (low drift) for success
            sem_sim = np.random.beta(8, 2)  # clustered ~0.8
        else:
            sem_sim = 0.0
        prev_reasoning = "x" * reasoning_len

        exit_code = 0 if np.random.random() > 0.05 else 1
        time_gap = np.random.gamma(2, 2) + 3  # 5-10s typical
        error_keys = 1 if exit_code != 0 else 0
        is_repeat = False

        actions.append({
            "session_id": session_id,
            "action_type": action_type,
            "command": cmd[0],
            "args": json.dumps(cmd[1:]),
            "exit_code": exit_code,
            "timestamp": dt.isoformat(),
            "time_since_last_action": time_gap if step > 0 else 0.0,
            "reasoning_length": max(10, reasoning_len),
            "semantic_similarity": sem_sim,
            "error_keywords": error_keys,
            "is_repeat_command": is_repeat,
            "step_index": step
        })

    return session_id, "Simple feature addition", start, dt, actions, "success"

def generate_loop_session(session_num, base_time):
    """Generate a session caught in an infinite loop."""
    session_id = f"loop_{session_num:04d}"
    n_steps = np.random.randint(25, 50)
    start = base_time
    actions = []

    loop_cmd = ["bash", "-c", "while true; do echo 'running'; sleep 1; done"]
    prev_reasoning = ""

    for step in range(n_steps):
        dt = start + timedelta(seconds=step * np.random.uniform(0.5, 2.0))  # fast repetition
        action_type = "bash"

        # Rambling reasoning (long)
        reasoning_len = int(np.random.normal(400, 100))
        if prev_reasoning:
            # Very high similarity — stuck repeating
            sem_sim = np.random.beta(15, 2)  # ~0.9+
        else:
            sem_sim = 0.0
        prev_reasoning = "y" * reasoning_len

        exit_code = 0
        time_gap = np.random.exponential(1.0) + 0.5  # quick loops
        error_keys = 0
        is_repeat = step > 5 and np.random.random() > 0.3  # frequent repeats

        cmd = loop_cmd if is_repeat or step > 10 else ["bash", "-c", "ls"]

        actions.append({
            "session_id": session_id,
            "action_type": action_type,
            "command": cmd[0],
            "args": json.dumps(cmd[1:]),
            "exit_code": exit_code,
            "timestamp": dt.isoformat(),
            "time_since_last_action": time_gap,
            "reasoning_length": max(10, reasoning_len),
            "semantic_similarity": sem_sim,
            "error_keywords": error_keys,
            "is_repeat_command": is_repeat,
            "step_index": step
        })

    return session_id, "Debug this loop", start, dt, actions, "loop"

def generate_hallucination_session(session_num, base_time):
    """Generate a session with hallucinated commands and silent failures."""
    session_id = f"hallucination_{session_num:04d}"
    n_steps = np.random.randint(20, 45)
    start = base_time
    actions = []

    prev_reasoning = ""
    for step in range(n_steps):
        dt = start + timedelta(seconds=step * np.random.uniform(2, 12))
        action_type = np.random.choice(["bash", "think"], p=[0.7, 0.3])

        # Erratic reasoning lengths
        reasoning_len = int(np.random.uniform(20, 600))
        if prev_reasoning:
            # Erratic similarity — jumping topics
            sem_sim = np.random.beta(2, 5)  # low, high drift
        else:
            sem_sim = 0.0
        prev_reasoning = "z" * reasoning_len

        exit_code = 0 if np.random.random() > 0.4 else np.random.choice([1, 127])
        time_gap = np.random.exponential(3) + 1
        error_keys = np.random.poisson(1.5) if exit_code != 0 else 0
        is_repeat = False

        # Nonsense commands (hallucinations)
        bad_cmds = ["gpt-install", "magic-compile", "--unicorn", "sudo delete-all"]
        if step > 8 and np.random.random() > 0.6:
            cmd = [np.random.choice(bad_cmds)]
        else:
            cmd = ["python", "script.py"] if np.random.random() > 0.4 else ["bash", "-c", "cat"]

        actions.append({
            "session_id": session_id,
            "action_type": action_type,
            "command": cmd[0],
            "args": json.dumps(cmd[1:]),
            "exit_code": exit_code,
            "timestamp": dt.isoformat(),
            "time_since_last_action": time_gap,
            "reasoning_length": max(10, reasoning_len),
            "semantic_similarity": sem_sim,
            "error_keywords": error_keys,
            "is_repeat_command": is_repeat,
            "step_index": step
        })

    return session_id, "Fix broken async code", start, dt, actions, "hallucination"

def main():
    """Generate all session data and save to SQLite."""
    print("Creating tables...")
    create_tables()

    conn = sqlite3.connect(DB_PATH)
    base = datetime(2026, 4, 1, 9, 0, 0)

    sessions = []
    all_actions = []
    events = []

    # 35 success sessions
    for i in range(35):
        sid, desc, start, end, acts, outcome = generate_success_session(i, base + timedelta(hours=i*2))
        sessions.append((sid, desc, start.isoformat(), end.isoformat(), outcome))
        all_actions.extend(acts)

    # 35 loop sessions
    for i in range(35):
        sid, desc, start, end, acts, outcome = generate_loop_session(i, base + timedelta(hours=i*2 + 1))
        sessions.append((sid, desc, start.isoformat(), end.isoformat(), outcome))
        all_actions.extend(acts)
        events.append((sid, end.isoformat(), "loop"))

    # 30 hallucination sessions
    for i in range(30):
        sid, desc, start, end, acts, outcome = generate_hallucination_session(i, base + timedelta(hours=i*2 + 0.5))
        sessions.append((sid, desc, start.isoformat(), end.isoformat(), outcome))
        all_actions.extend(acts)
        events.append((sid, end.isoformat(), "hallucination"))

    # Save sessions
    conn.executemany(
        "INSERT INTO agent_sessions VALUES (?, ?, ?, ?, ?)",
        sessions
    )

    # Save actions
    conn.executemany(
        "INSERT INTO agent_actions (session_id, action_type, command, args, exit_code, timestamp, time_since_last_action, reasoning_length, semantic_similarity, error_keywords, is_repeat_command, step_index) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(a['session_id'], a['action_type'], a['command'], a['args'], a['exit_code'],
          a['timestamp'], a['time_since_last_action'], a['reasoning_length'],
          a['semantic_similarity'], a['error_keywords'], int(a['is_repeat_command']),
          a['step_index']) for a in all_actions]
    )

    # Save hallucination events
    conn.executemany(
        "INSERT INTO hallucination_events (session_id, detected_at, event_type) VALUES (?, ?, ?)",
        events
    )

    conn.commit()

    print(f"Generated {len(sessions)} sessions, {len(all_actions)} actions, {len(events)} failure events")
    print(f"Success: 35, Loops: 35, Hallucinations: 30")

    # Verify with quick query
    df = pd.read_sql_query("SELECT outcome, COUNT(*) as cnt FROM agent_sessions GROUP BY outcome", conn)
    print("\nSession counts by outcome:")
    print(df)

    conn.close()

if __name__ == "__main__":
    main()

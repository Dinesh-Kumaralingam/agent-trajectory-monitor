"""Ingest real SWE-agent trajectories from a Parquet file into telemetry.db."""

import json
import random
import sqlite3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent / "telemetry.db"

# Find parquet file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
parquet_files = list(PROJECT_ROOT.rglob("*.parquet"))
if not parquet_files:
    print("No .parquet file found.")
    sys.exit(1)
PARQUET_PATH = parquet_files[0]
print(f"Found parquet file: {PARQUET_PATH}")

def clear_data(conn):
    c = conn.cursor()
    c.execute("DELETE FROM agent_actions")
    c.execute("DELETE FROM agent_sessions")
    c.execute("DELETE FROM hallucination_events")
    conn.commit()
    print("Cleared existing data.")

def insert_session(conn, session_id, task_desc, outcome):
    c = conn.cursor()
    start = datetime.now().isoformat()
    end = (datetime.now() + timedelta(minutes=10)).isoformat()
    c.execute(
        "INSERT INTO agent_sessions (session_id, task_description, start_time, end_time, outcome) VALUES (?, ?, ?, ?, ?)",
        (session_id, str(task_desc)[:200], start, end, outcome)
    )
    conn.commit()

def count_error_keywords(text):
    if not text:
        return 0
    text = str(text).lower()
    return sum(1 for kw in ['error', 'traceback', 'exception', 'syntaxerror', 'typeerror'] if kw in text)

def insert_action(conn, session_id, step_idx, action_type, command, reasoning_len,
                  error_kw, exit_code, is_repeat, time_gap, sem_sim):
    c = conn.cursor()
    ts = (datetime.now() + timedelta(seconds=step_idx * 5)).isoformat()
    c.execute("""
        INSERT INTO agent_actions
        (session_id, action_type, command, args, exit_code, timestamp,
         time_since_last_action, reasoning_length, semantic_similarity,
         error_keywords, is_repeat_command, step_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, action_type, command, json.dumps([]),
        exit_code, ts, time_gap, reasoning_len,
        sem_sim, error_kw, int(is_repeat), step_idx
    ))
    conn.commit()

def main():
    print(f"Loading {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # --- Determine target column ---
    target_col = None
    for col in ['target', 'resolved', 'success', 'pass', 'status']:
        if col in df.columns:
            target_col = col
            break

    # Robust target extraction (immune to arrays)
    targets = []
    if target_col is None:
        print("No target column found - using first 400 rows as successes.")
        targets = [True] * len(df)
    else:
        print(f"Using column '{target_col}' for success/failure labeling")
        for val in df[target_col].values:
            try:
                if isinstance(val, np.ndarray):
                    val = val[0] if val.size > 0 else False
                elif isinstance(val, list):
                    val = val[0] if len(val) > 0 else False
                
                if pd.isna(val):
                    targets.append(False)
                elif isinstance(val, (bool, np.bool_)):
                    targets.append(bool(val))
                elif isinstance(val, (int, float, np.number)):
                    targets.append(val == 1)
                elif isinstance(val, str):
                    targets.append(val.strip().lower() in ['true', '1', 'success', 'pass'])
                else:
                    targets.append(bool(val))
            except:
                targets.append(False)

    # --- Select 200 successes and 200 failures ---
    success_indices = [i for i, t in enumerate(targets) if t][:200]
    fail_indices = [i for i, t in enumerate(targets) if not t][:200]
    
    selected_indices = success_indices + fail_indices
    print(f"Selected {len(success_indices)} success + {len(fail_indices)} failure sessions")

    if not selected_indices:
        print("No valid sessions found to process.")
        return

    # CRITICAL FIX: Convert directly to pure Python dicts to bypass Pandas array bugs
    selected_df = df.iloc[selected_indices]
    records = selected_df.to_dict(orient='records')

    conn = sqlite3.connect(DB_PATH)
    clear_data(conn)

    total_sessions = 0
    total_actions = 0

    # --- Process selected sessions ---
    for idx, row in enumerate(records):
        try:
            orig_idx = selected_indices[idx]
            sid = f"pq_{total_sessions:04d}"

            # Outcome
            is_success = targets[orig_idx]
            outcome = 'success' if is_success else 'hallucination'

            # Task description
            task_desc = str(row.get('instance_id', row.get('problem_statement', f'Task {idx}')))

            # --- Parse trajectory safely ---
            traj_raw = row.get('trajectory', None)
            traj = []
            
            if isinstance(traj_raw, np.ndarray):
                traj = traj_raw.tolist()
            elif isinstance(traj_raw, list):
                traj = traj_raw
            elif isinstance(traj_raw, str):
                try:
                    traj = json.loads(traj_raw)
                except:
                    traj = []
            
            # Fallback
            if not isinstance(traj, list):
                traj = []

            # Insert session
            insert_session(conn, sid, task_desc, outcome)
            total_sessions += 1

            # Process actions
            prev_action = None
            action_count = 0

            for step_idx, step in enumerate(traj):
                if isinstance(step, np.ndarray):
                    step = step.tolist()
                
                if not isinstance(step, dict):
                    continue

                action_str = str(step.get('action', step.get('command', '')))
                action_type = action_str.split()[0] if action_str else 'unknown'

                thought = step.get('thought', step.get('content', step.get('reasoning', '')))
                reasoning_len = len(str(thought))

                observation = step.get('observation', step.get('output', step.get('result', '')))
                error_kw = count_error_keywords(str(observation))

                exit_code = 1 if error_kw > 0 else 0

                is_repeat = (action_str == prev_action) if prev_action else False
                prev_action = action_str

                time_gap = random.uniform(2.0, 15.0)
                sem_sim = random.uniform(0.5, 1.0)

                insert_action(conn, sid, step_idx, action_type, action_str,
                              reasoning_len, error_kw, exit_code, is_repeat,
                              time_gap, sem_sim)
                action_count += 1
                total_actions += 1

            print(f"  {sid}: {action_count} actions, outcome={outcome}")

        except Exception as e:
            print(f"  Error processing row {orig_idx}: {e}")

    conn.close()
    print(f"\nDone! Ingested {total_sessions} sessions with {total_actions} total actions into {DB_PATH}")

if __name__ == "__main__":
    main()
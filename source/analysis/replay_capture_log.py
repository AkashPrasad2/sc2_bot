"""
replay_capture_log.py

This script processes a single replay using the same logic as the main replay parser,
but instead of saving an npz file, it prints out a human-readable log of the events
captured in each grid window, along with the state of resources at that time.
"""

import sys
import sc2reader
from pathlib import Path

# Add parent directory to path so we can import replay_parser
sys.path.append(str(Path(__file__).resolve().parent.parent))
from replay_parser import GameState, GRID_INTERVAL_SECONDS
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, UnitDoneEvent,
    BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent,
)

def analyze_replay(replay_path: str):
    print(f"Analyzing replay: {replay_path}")
    try:
        replay = sc2reader.load_replay(replay_path, load_level=4)
    except Exception as e:
        print(f"Failed to load replay: {e}")
        return

    protoss_player = None
    for player in replay.players:
        if player.play_race == "Protoss":
            protoss_player = player
            break
            
    if not protoss_player:
        print("No Protoss player found in this replay.")
        return
        
    pid = protoss_player.pid
    
    EVENT_TO_ACTION = {
        "TrainProbe":             1,
        "BuildPylon":             2,
        "BuildGateway":           3,
        "BuildCyberneticsCore":   4,
        "BuildAssimilator":       5,
        "BuildNexus":             6,
        "BuildForge":             7,
        "BuildStargate":          8,
        "BuildRoboticsFacility":  9,
        "BuildTwilightCouncil":  10,
        "BuildPhotonCannon":     11,
        "BuildFleetBeacon":      12,
        "BuildTemplarArchive":   13,
        "TrainZealot":           14,
        "TrainStalker":          15,
        "TrainImmortal":         16,
        "TrainVoidRay":          17,
        "TrainCarrier":          18,
        "TrainHighTemplar":      19,
        "WarpInZealot":          20,
        "WarpInStalker":         21,
        "WarpInHighTemplar":     22,
        "ArchonWarp":            23,
        "ArchonWarpSelection":   23,
        "MorphToArchon":         23,
        "ResearchCharge":        24,
        "ResearchWarpGate":      25,
        "UpgradeGroundWeapons1": 26,
        "UpgradeGroundWeapons2": 26,
        "UpgradeGroundWeapons3": 26,
        "UpgradeAirWeapons1":    27,
        "UpgradeAirWeapons2":    27,
        "UpgradeAirWeapons3":    27,
        "UpgradeShields1":       28,
        "UpgradeShields2":       28,
        "UpgradesShields3":      28, 
        "UpgradeShields3":       28,
        "TrainAdept":            30,
        "TrainPhoenix":          31,
        "TrainColossus":         32,
        "WarpInAdept":           33,
    }

    # First pass: map commands to grid slots (to handle queueing correctly)
    grid_actions = {}
    last_window = 0
    
    for event in replay.events:
        t = event.second
        if isinstance(event, (BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent)):
            if event.player.pid == pid:
                ability = event.ability_name
                action_id = EVENT_TO_ACTION.get(ability)
                if action_id is not None:
                    cmd_window = int(t / GRID_INTERVAL_SECONDS)
                    slot = cmd_window
                    while slot in grid_actions:
                        slot += 1
                        
                    grid_actions[slot] = ability
                    last_window = max(last_window, slot)

    # Second pass: simulate state and print
    print("\n--- Replay Capture Log ---")
    print(f"Grid Interval: {GRID_INTERVAL_SECONDS}s")
    print(f"{'Window':<8} | {'Time (s)':<10} | {'Mins':<6} | {'Gas':<6} | {'Action Captured'}")
    print("-" * 65)
    
    state = GameState()
    current_grid = 0
    event_idx = 0
    num_events = len(replay.events)
    
    # Track the last time PlayerStatsEvent fired
    last_stats_time = 0.0
    
    while current_grid <= last_window:
        window_start_time = current_grid * GRID_INTERVAL_SECONDS
        
        # Advance state to the start of this window
        while event_idx < num_events and replay.events[event_idx].second < window_start_time:
            event = replay.events[event_idx]
            
            if isinstance(event, PlayerStatsEvent):
                if event.player.pid == pid:
                    state.update_from_stats(event)
                    last_stats_time = event.second
            elif isinstance(event, (UnitBornEvent, UnitDoneEvent)):
                unit = getattr(event, 'unit', None)
                if unit:
                    owner = getattr(unit, "owner", None)
                    if owner and owner.pid == pid:
                        state.unit_born_or_done(unit.name)
            elif isinstance(event, UnitDiedEvent):
                unit = getattr(event, 'unit', None)
                if unit:
                    owner = getattr(unit, "owner", None)
                    if owner and owner.pid == pid:
                        state.unit_died(unit.name)
            elif isinstance(event, (BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent)):
                if event.player.pid == pid:
                    state.on_build_command(event.ability_name)
                    state.on_train_command(event.ability_name)
                    state.on_upgrade_command(event.ability_name)
                    
            event_idx += 1
            
        action = grid_actions.get(current_grid, "do_nothing")
        stats_age = window_start_time - last_stats_time
        
        # Flag if resources are particularly stale (e.g., > 5 seconds old)
        stale_flag = " (Stale)" if stats_age > 5 else ""
        
        print(f"{current_grid:<8} | {window_start_time:<10.1f} | {int(state.minerals):<6} | {int(state.vespene):<6} | {action}{stale_flag}")
        
        current_grid += 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_replay(sys.argv[1])
    else:
        import os
        replay_dir = r"C:\dev\BetaStar\replays\raw"
        replays = [f for f in os.listdir(replay_dir) if f.endswith('.SC2Replay')]
        if replays:
            analyze_replay(os.path.join(replay_dir, replays[0]))
        else:
            print(f"No replays found in {replay_dir}")

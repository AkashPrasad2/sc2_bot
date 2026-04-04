"""
verify_dataset_completeness.py - Verify parsed dataset matches replay events
"""
import sc2reader
from sc2reader.events import BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent
import numpy as np
import sys

# Configuration
replay_path = r"C:\dev\BetaStar\replays\raw\Classic v Serral_ Game 1 - Magannatha LE.SC2Replay"
dataset_path = r"C:\dev\BetaStar\replays\parsed\dataset.npz"

ACTIONS = [
    "do_nothing",               # 0
    "train_probe",              # 1
    "build_pylon",              # 2
    "build_gateway",            # 3
    "build_cyberneticscore",    # 4
    "build_assimilator",        # 5
    "build_nexus",              # 6
    "build_forge",              # 7
    "build_stargate",           # 8
    "build_robotics_facility",  # 9
    "build_twilight_council",   # 10
    "build_photon_cannon",      # 11
    "build_fleet_beacon",       # 12
    "build_templar_archive",    # 13
    "train_zealot",             # 14
    "train_stalker",            # 15
    "train_immortal",           # 16
    "train_voidray",            # 17
    "train_carrier",            # 18
    "train_high_templar",       # 19
    "warp_in_zealot",           # 20
    "warp_in_stalker",          # 21
    "warp_in_high_templar",     # 22
    "archon_warp_selection",    # 23
    "research_charge",          # 24
    "research_warp_gate",       # 25
    "upgrade_ground_weapons",   # 26
    "upgrade_air_weapons",      # 27
    "upgrade_shields",          # 28
    "attack_enemy_base",        # 29
    "train_adept",              # 30
    "train_phoenix",            # 31
    "train_colossus",           # 32
    "warp_in_adept",            # 33
]

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
    "TrainAdept":            30,
    "TrainPhoenix":          31,
    "TrainColossus":         32,
    "WarpInAdept":           33,
}

print("=" * 120)
print("REPLAY vs DATASET COMPARISON")
print("=" * 120)

# Step 1: Extract commands from replay
print("\n[1/3] Extracting commands from replay...")
replay = sc2reader.load_replay(replay_path, load_level=4)

protoss_player = None
for p in replay.players:
    if p.play_race == "Protoss":
        protoss_player = p
        break

if not protoss_player:
    print("ERROR: No Protoss player found!")
    sys.exit(1)

# Collect all macro commands with their 4s window assignment
replay_commands = []
for event in replay.events:
    if isinstance(event, (BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent)):
        if event.player.pid == protoss_player.pid:
            ability = event.ability_name
            action_id = EVENT_TO_ACTION.get(ability)
            if action_id is not None:
                window = int(event.second / 4)
                replay_commands.append({
                    'time': event.second,
                    'window': window,
                    'ability': ability,
                    'action_id': action_id,
                    'action_name': ACTIONS[action_id]
                })

print(f"Found {len(replay_commands)} mapped macro commands in replay")

# Step 2: Load parsed dataset
print("\n[2/3] Loading parsed dataset...")
try:
    data = np.load(dataset_path, allow_pickle=True)
    sequences = data['sequences']
    print(f"Dataset contains {len(sequences)} sequences")
    
    # Find the sequence that matches this replay (by checking if it has similar action pattern)
    # For now, just show the first sequence
    if len(sequences) > 0:
        seq = sequences[0]
        print(f"Analyzing first sequence: {len(seq)} timesteps")
        
        # Extract actions from dataset
        dataset_actions = []
        for i, row in enumerate(seq):
            action_id = int(row[-1])  # Last column is action
            if action_id != 0:  # Skip do_nothing
                dataset_actions.append({
                    'window': i,
                    'time': i * 4,
                    'action_id': action_id,
                    'action_name': ACTIONS[action_id]
                })
        
        print(f"Found {len(dataset_actions)} non-zero actions in dataset")
    else:
        print("ERROR: Dataset is empty!")
        sys.exit(1)
        
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Please run replay_parser.py first to generate the dataset.")
    sys.exit(1)

# Step 3: Compare
print("\n[3/3] Comparing replay commands with dataset actions...")
print("=" * 120)
print(f"{'Window':<8} | {'Time':<8} | {'Replay Command':<30} | {'Dataset Action':<30} | {'Match':<10}")
print("-" * 120)

# Build a map of windows to actions for easy comparison
replay_by_window = {}
for cmd in replay_commands:
    window = cmd['window']
    if window not in replay_by_window:
        replay_by_window[window] = []
    replay_by_window[window].append(cmd)

dataset_by_window = {}
for action in dataset_actions:
    window = action['window']
    if window not in dataset_by_window:
        dataset_by_window[window] = []
    dataset_by_window[window].append(action)

# Get all windows that have either replay or dataset actions
all_windows = sorted(set(list(replay_by_window.keys()) + list(dataset_by_window.keys())))

matches = 0
mismatches = 0
replay_only = 0
dataset_only = 0

for window in all_windows[:100]:  # Show first 100 windows
    replay_cmds = replay_by_window.get(window, [])
    dataset_acts = dataset_by_window.get(window, [])
    
    if replay_cmds and dataset_acts:
        # Both have actions - check if they match
        replay_str = ", ".join([cmd['action_name'] for cmd in replay_cmds])
        dataset_str = ", ".join([act['action_name'] for act in dataset_acts])
        
        if replay_str == dataset_str:
            status = "✓ MATCH"
            matches += 1
        else:
            status = "✗ DIFFER"
            mismatches += 1
        
        print(f"{window:<8} | {window*4:<8} | {replay_str:<30} | {dataset_str:<30} | {status}")
    
    elif replay_cmds:
        # Only in replay
        replay_str = ", ".join([cmd['action_name'] for cmd in replay_cmds])
        status = "✗ MISSING"
        replay_only += 1
        print(f"{window:<8} | {window*4:<8} | {replay_str:<30} | {'(not in dataset)':<30} | {status}")
    
    elif dataset_acts:
        # Only in dataset
        dataset_str = ", ".join([act['action_name'] for act in dataset_acts])
        status = "✗ EXTRA"
        dataset_only += 1
        print(f"{window:<8} | {window*4:<8} | {'(not in replay)':<30} | {dataset_str:<30} | {status}")

if len(all_windows) > 100:
    print(f"... and {len(all_windows) - 100} more windows")

print("\n" + "=" * 120)
print("SUMMARY:")
print("=" * 120)
print(f"Total replay commands:     {len(replay_commands)}")
print(f"Total dataset actions:     {len(dataset_actions)}")
print(f"Matching windows:          {matches}")
print(f"Mismatched windows:        {mismatches}")
print(f"Replay-only (missing):     {replay_only}")
print(f"Dataset-only (extra):      {dataset_only}")

if mismatches > 0 or replay_only > 0 or dataset_only > 0:
    print("\n⚠ WARNING: Replay and dataset do not match perfectly!")
    print("This could be due to:")
    print("  - Queuing: Commands pushed to later windows when slots are full")
    print("  - Filtering: Illegal actions demoted to do_nothing")
    print("  - Different replay: Dataset may contain different replays")
else:
    print("\n✓ Perfect match! All replay commands are in the dataset.")

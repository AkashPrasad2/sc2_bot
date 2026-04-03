"""
compare_replay_to_dataset.py - Compare raw replay events with parsed training data
"""
import sc2reader
from sc2reader.events import BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent
import numpy as np

# Load the same replay
replay_path = r"C:\dev\BetaStar\replays\raw\Fjant v LiquidMaNa_ Game 2 - Goldenaura LE.SC2Replay"
print(f"Loading replay: {replay_path}\n")
replay = sc2reader.load_replay(replay_path, load_level=4)

# Find Protoss player
protoss_player = None
for p in replay.players:
    if p.play_race == "Protoss":
        protoss_player = p
        break

if not protoss_player:
    print("No Protoss player found!")
    exit()

print(f"Protoss player: {protoss_player.name} (PID {protoss_player.pid})")
print("=" * 100)

# Collect all macro commands from replay
macro_commands = []
for event in replay.events:
    if isinstance(event, (BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent)):
        if event.player.pid == protoss_player.pid:
            ability = event.ability_name
            # Filter to only macro actions (build, train, research, upgrade)
            if any(keyword in ability for keyword in ['Build', 'Train', 'WarpIn', 'Research', 'Morph', 'Archon']):
                macro_commands.append((event.second, ability))

print(f"\nFound {len(macro_commands)} macro commands in replay:")
print("-" * 100)
print(f"{'Time':>8} | {'Command':<30} | {'4s Window':<10}")
print("-" * 100)

for t, ability in macro_commands[:50]:  # Show first 50
    window = int(t / 4)
    print(f"{t:7.1f}s | {ability:<30} | Window {window:3d} (t={window*4}s)")

if len(macro_commands) > 50:
    print(f"... and {len(macro_commands) - 50} more commands")

# Now show what the parser would map these to
print("\n" + "=" * 100)
print("MAPPING TO TRAINING ACTIONS:")
print("=" * 100)

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
    "MorphToArchon":         23,
    "ResearchCharge":        24,
    "ResearchWarpGate":      25,
}

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
    "archon_warp_selecton",     # 23
    "research_charge",          # 24
    "research_warp_gate",       # 25
    "upgrade_ground_weapons",   # 26
    "upgrade_air_weapons",      # 27
    "upgrade_shields",          # 28
    "attack_enemy_base",        # 29
]

mapped_count = 0
unmapped_count = 0
unmapped_abilities = set()

print(f"{'Time':>8} | {'Replay Event':<30} | {'Action ID':<10} | {'Training Action':<30} | {'Status':<10}")
print("-" * 100)

for t, ability in macro_commands[:50]:
    action_id = EVENT_TO_ACTION.get(ability)
    if action_id is not None:
        action_name = ACTIONS[action_id]
        status = "✓ MAPPED"
        mapped_count += 1
        print(
            f"{t:7.1f}s | {ability:<30} | {action_id:<10} | {action_name:<30} | {status}")
    else:
        status = "✗ UNMAPPED"
        unmapped_count += 1
        unmapped_abilities.add(ability)
        print(
            f"{t:7.1f}s | {ability:<30} | {'N/A':<10} | {'(not in training set)':<30} | {status}")

if len(macro_commands) > 50:
    # Count remaining
    for t, ability in macro_commands[50:]:
        action_id = EVENT_TO_ACTION.get(ability)
        if action_id is not None:
            mapped_count += 1
        else:
            unmapped_count += 1
            unmapped_abilities.add(ability)

print("\n" + "=" * 100)
print("SUMMARY:")
print("=" * 100)
print(f"Total macro commands in replay: {len(macro_commands)}")
print(
    f"Mapped to training actions:     {mapped_count} ({100*mapped_count/len(macro_commands):.1f}%)")
print(
    f"Unmapped (ignored):             {unmapped_count} ({100*unmapped_count/len(macro_commands):.1f}%)")

if unmapped_abilities:
    print(f"\nUnmapped abilities found:")
    for ability in sorted(unmapped_abilities):
        print(f"  - {ability}")
    print("\nThese commands will be ignored during training data generation.")
else:
    print("\n✓ All macro commands are mapped to training actions!")

# Show action distribution
print("\n" + "=" * 100)
print("ACTION DISTRIBUTION IN REPLAY:")
print("=" * 100)

action_counts = {}
for t, ability in macro_commands:
    action_id = EVENT_TO_ACTION.get(ability)
    if action_id is not None:
        action_name = ACTIONS[action_id]
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

for action_name in sorted(action_counts.keys(), key=lambda x: action_counts[x], reverse=True):
    count = action_counts[action_name]
    print(f"  {action_name:<30}: {count:3d} times")

"""
show_replay_events.py - Display raw event log from a single replay
"""
import sc2reader
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent,
    UnitDoneEvent, BasicCommandEvent,
)

replay_path = r"C:\dev\BetaStar\replays\raw\Classic v Serral_ Game 1 - Magannatha LE.SC2Replay"

print(f"Loading replay: {replay_path}\n")
replay = sc2reader.load_replay(replay_path, load_level=4)

print(f"Map: {replay.map_name}")
print(f"Duration: {replay.game_length}")
print(f"Players:")
for p in replay.players:
    print(f"  {p.name} ({p.play_race}) - PID {p.pid}")

# Find Protoss player
protoss_player = None
for p in replay.players:
    if p.play_race == "Protoss":
        protoss_player = p
        break

if not protoss_player:
    print("\nNo Protoss player found!")
    exit()

print(f"\nShowing ALL event types for Protoss player: {protoss_player.name} (PID {protoss_player.pid})")
print("=" * 80)

event_types = set()
for event in replay.events:
    t = event.second
    event_type = type(event).__name__
    event_types.add(event_type)
    
    # Check if event has player attribute
    has_player = hasattr(event, 'player')
    if has_player and event.player.pid == protoss_player.pid:
        print(f"[{t:6.1f}s] {event_type}: {event}")
    # Check if event has unit with owner
    elif hasattr(event, 'unit'):
        unit = event.unit
        owner = getattr(unit, "owner", None)
        if owner and owner.pid == protoss_player.pid:
            print(f"[{t:6.1f}s] {event_type}: {unit.name}")

print("\n" + "=" * 80)
print("All event types found in replay:")
for et in sorted(event_types):
    print(f"  - {et}")

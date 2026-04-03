"""
explore_sc2reader.py - Discover all available events and capabilities
"""
import sc2reader
import sc2reader.events as events_module
import inspect

print("=" * 80)
print("ALL EVENT CLASSES IN sc2reader.events:")
print("=" * 80)

# Get all classes from the events module
event_classes = []
for name, obj in inspect.getmembers(events_module):
    if inspect.isclass(obj) and name.endswith('Event'):
        event_classes.append(name)
        
for ec in sorted(event_classes):
    print(f"  - {ec}")

print("\n" + "=" * 80)
print("LOADING A SAMPLE REPLAY TO SEE ACTUAL EVENTS:")
print("=" * 80)

replay_path = r"C:\dev\BetaStar\replays\raw\Classic v Serral_ Game 1 - Magannatha LE.SC2Replay"
replay = sc2reader.load_replay(replay_path, load_level=4)

# Find Protoss player
protoss_player = None
for p in replay.players:
    if p.play_race == "Protoss":
        protoss_player = p
        break

print(f"\nProtoss player: {protoss_player.name} (PID {protoss_player.pid})")

# Collect all event types and sample events
event_samples = {}
for event in replay.events:
    event_type = type(event).__name__
    
    # Only collect first sample of each type for Protoss player
    if event_type not in event_samples:
        # Check if it's a player event
        if hasattr(event, 'player') and event.player.pid == protoss_player.pid:
            event_samples[event_type] = event
        # Check if it's a unit event
        elif hasattr(event, 'unit'):
            unit = event.unit
            owner = getattr(unit, "owner", None)
            if owner and owner.pid == protoss_player.pid:
                event_samples[event_type] = event

print("\n" + "=" * 80)
print("EVENT TYPES FOUND FOR PROTOSS PLAYER (with attributes):")
print("=" * 80)

for event_type in sorted(event_samples.keys()):
    event = event_samples[event_type]
    print(f"\n{event_type}:")
    print(f"  Time: {event.second}s")
    
    # Show all attributes
    attrs = [attr for attr in dir(event) if not attr.startswith('_')]
    for attr in attrs[:15]:  # Limit to first 15 to avoid clutter
        try:
            value = getattr(event, attr)
            if not callable(value):
                print(f"    {attr}: {value}")
        except:
            pass

print("\n" + "=" * 80)
print("SEARCHING FOR BUILD-RELATED COMMANDS:")
print("=" * 80)

build_commands = []
for event in replay.events:
    if hasattr(event, 'player') and event.player.pid == protoss_player.pid:
        if hasattr(event, 'ability_name'):
            ability = event.ability_name
            if 'Build' in ability or 'Morph' in ability or 'Warp' in ability:
                build_commands.append((event.second, type(event).__name__, ability))

print(f"\nFound {len(build_commands)} build-related commands:")
for t, event_type, ability in build_commands[:20]:  # Show first 20
    print(f"  [{t:6.1f}s] {event_type}: {ability}")

print("\n" + "=" * 80)
print("CHECKING FOR TargetUnitCommandEvent or UnitInitEvent:")
print("=" * 80)

# Look for any events that might capture building starts
special_events = []
for event in replay.events:
    event_type = type(event).__name__
    if 'Target' in event_type or 'Init' in event_type or 'Command' in event_type:
        if hasattr(event, 'player') and event.player.pid == protoss_player.pid:
            special_events.append((event.second, event_type, str(event)[:100]))

print(f"\nFound {len(special_events)} special command events:")
for t, event_type, desc in special_events[:30]:  # Show first 30
    print(f"  [{t:6.1f}s] {event_type}: {desc}")

import sc2reader
from sc2reader.events import BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent

replay_path = r'C:\dev\BetaStar\replays\raw\Railgan v ShaDoWn - Abyssal Reef LE.SC2Replay'
replay = sc2reader.load_replay(replay_path, load_level=4)

count = 0
for event in replay.events:
    if isinstance(event, (BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent)) and event.player.pid == 1:
        has_id = hasattr(event, 'ability_id')
        ab_id = hex(event.ability_id) if has_id else 'None'
        print(f"{event.second}s: {event.ability_name} (ID: {ab_id})")
        count += 1
        if count >= 30:
            break

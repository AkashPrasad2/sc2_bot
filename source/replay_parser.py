"""
SC2 Replay Parser — Sequence Dataset Builder
=============================================
Key changes from previous version:
  - do_nothing (action 0) is NOT included in training sequences.
    The model only sees real macro decisions in order. The bot handles
    waiting implicitly by retrying failed actions each cooldown tick.
  - In-progress structure and unit counts are tracked and included in
    the observation vector, mirroring observation_wrapper.py exactly.
  - OBS_SIZE updated to 53.
"""

from collections import defaultdict
import os
import sc2reader
import numpy as np
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent,
    UnitDoneEvent, BasicCommandEvent,
)

STRUCTURE_NAME_MAP = {
    "Nexus":             "NEXUS",
    "Pylon":             "PYLON",
    "Gateway":           "GATEWAY",
    "WarpGate":          "WARPGATE",
    "Forge":             "FORGE",
    "TwilightCouncil":   "TWILIGHTCOUNCIL",
    "PhotonCannon":      "PHOTONCANNON",
    "ShieldBattery":     "SHIELDBATTERY",
    "TemplarArchive":    "TEMPLARARCHIVE",
    "RoboticsBay":       "ROBOTICSBAY",
    "RoboticsFacility":  "ROBOTICSFACILITY",
    "Assimilator":       "ASSIMILATOR",
    "CyberneticsCore":   "CYBERNETICSCORE",
    "Stargate":          "STARGATE",
    "FleetBeacon":       "FLEETBEACON",
}

UNIT_NAME_MAP = {
    "Probe":        "PROBE",
    "Zealot":       "ZEALOT",
    "Stalker":      "STALKER",
    "HighTemplar":  "HIGHTEMPLAR",
    "Archon":       "ARCHON",
    "Immortal":     "IMMORTAL",
    "Carrier":      "CARRIER",
    "VoidRay":      "VOIDRAY",
}

STRUCTURES = [
    "NEXUS", "PYLON", "GATEWAY", "WARPGATE", "FORGE", "TWILIGHTCOUNCIL",
    "PHOTONCANNON", "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY",
    "ROBOTICSFACILITY", "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON",
]
UNITS = [
    "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON", "IMMORTAL", "CARRIER", "VOIDRAY",
]

OBS_SIZE = 53  # 6 base + 15 structures + 8 units + 15 pending structures + 8 pending units + 1 opp

# Build command ability name -> structure key (for tracking pending structures)
BUILD_COMMAND_TO_STRUCTURE = {
    "BuildNexus":             "NEXUS",
    "BuildPylon":             "PYLON",
    "BuildGateway":           "GATEWAY",
    "BuildForge":             "FORGE",
    "BuildTwilightCouncil":   "TWILIGHTCOUNCIL",
    "BuildPhotonCannon":      "PHOTONCANNON",
    "BuildTemplarArchive":    "TEMPLARARCHIVE",
    "BuildRoboticsFacility":  "ROBOTICSFACILITY",
    "BuildAssimilator":       "ASSIMILATOR",
    "BuildCyberneticsCore":   "CYBERNETICSCORE",
    "BuildStargate":          "STARGATE",
    "BuildFleetBeacon":       "FLEETBEACON",
}

# Train/warp command ability name -> unit key (for tracking pending units)
TRAIN_COMMAND_TO_UNIT = {
    "TrainProbe":         "PROBE",
    "TrainZealot":        "ZEALOT",
    "TrainStalker":       "STALKER",
    "TrainImmortal":      "IMMORTAL",
    "TrainVoidRay":       "VOIDRAY",
    "TrainCarrier":       "CARRIER",
    "TrainHighTemplar":   "HIGHTEMPLAR",
    "WarpInZealot":       "ZEALOT",
    "WarpInStalker":      "STALKER",
    "WarpInHighTemplar":  "HIGHTEMPLAR",
}

MORPH_MAP = {
    "WarpGate": "GATEWAY",
}


class GameState:
    """
    Tracks full Protoss game state including in-progress counts.

    pending_structures: incremented on build command, decremented on UnitDoneEvent.
    pending_units:      incremented on train command, decremented on UnitBornEvent.

    This mirrors what python-sc2's already_pending() returns in the live bot,
    giving the model consistent features between training and inference.
    """

    def __init__(self):
        self.time = 0.0
        self.minerals = 50.0
        self.vespene = 0.0
        self.supply_used = 12.0
        self.supply_cap = 15.0

        self.counts = {k: 0 for k in STRUCTURES + UNITS}
        self.pending_structures = {k: 0 for k in STRUCTURES}
        self.pending_units = {k: 0 for k in UNITS}

        self.counts["NEXUS"] = 1
        self.counts["PROBE"] = 12

        self.opp_supply_used = 0.0

    def update_from_stats(self, event: PlayerStatsEvent):
        self.time = event.second
        self.minerals = getattr(event, "minerals_current",
                                getattr(event, "minerals", 0))
        self.vespene = getattr(event, "vespene_current",
                               getattr(event, "vespene",  0))
        self.supply_used = getattr(
            event, "supply_used",      getattr(event, "food_used", 0))
        self.supply_cap = getattr(
            event, "supply_made",      getattr(event, "food_made", 0))

    def update_opp_from_stats(self, event: PlayerStatsEvent):
        self.opp_supply_used = getattr(
            event, "supply_used", getattr(event, "food_used", 0))

    def on_build_command(self, ability_name: str):
        key = BUILD_COMMAND_TO_STRUCTURE.get(ability_name)
        if key:
            self.pending_structures[key] += 1

    def on_train_command(self, ability_name: str):
        key = TRAIN_COMMAND_TO_UNIT.get(ability_name)
        if key:
            self.pending_units[key] += 1

    def unit_born_or_done(self, unit_type_name: str):
        """Moves a unit/structure from pending to completed."""
        unit_key = UNIT_NAME_MAP.get(unit_type_name)
        structure_key = STRUCTURE_NAME_MAP.get(unit_type_name)

        if unit_key:
            self.counts[unit_key] += 1
            # Clamp to 0 to handle starting probes whose train commands aren't in the replay
            self.pending_units[unit_key] = max(
                0, self.pending_units[unit_key] - 1)

        if structure_key:
            self.counts[structure_key] += 1
            self.pending_structures[structure_key] = max(
                0, self.pending_structures[structure_key] - 1)

        predecessor = MORPH_MAP.get(unit_type_name)
        if predecessor:
            self.counts[predecessor] = max(0, self.counts[predecessor] - 1)

    def unit_died(self, unit_type_name: str):
        key = UNIT_NAME_MAP.get(
            unit_type_name) or STRUCTURE_NAME_MAP.get(unit_type_name)
        if key:
            self.counts[key] = max(0, self.counts[key] - 1)

    def to_obs(self) -> list[float]:
        """
        Serialize to the same flat vector as ObservationWrapper.get_observation().

        Layout (53 floats):
            time, minerals, vespene, supply_used, supply_cap, worker_saturation,
            [15 completed structures], [8 completed units],
            [15 pending structures],   [8 pending units],
            opp_supply_used
        """
        ideal_workers = max(self.counts["NEXUS"], 1) * 22
        worker_saturation = self.counts["PROBE"] / ideal_workers

        obs = [
            self.time / 720.0,
            self.minerals / 1800.0,
            self.vespene / 700.0,
            self.supply_used / 200.0,
            self.supply_cap / 200.0,
            worker_saturation,
        ]
        for s in STRUCTURES:
            obs.append(self.counts[s] / 10.0)
        for u in UNITS:
            obs.append(self.counts[u] / 30.0)
        for s in STRUCTURES:
            obs.append(self.pending_structures[s] / 10.0)
        for u in UNITS:
            obs.append(self.pending_units[u] / 30.0)
        obs.append(self.opp_supply_used / 200.0)

        assert len(
            obs) == OBS_SIZE, f"Obs size mismatch: got {len(obs)}, expected {OBS_SIZE}"
        return obs


class ReplayParser:
    """
    Parses SC2 replays into ordered action sequences for LSTM training.

    Each replay becomes one (T, OBS_SIZE+1) array where every row is a
    real macro decision — do_nothing is never recorded. The LSTM learns
    the sequence of decisions directly; waiting is handled implicitly
    by the bot retrying failed actions on each cooldown tick.
    """

    def __init__(
        self,
        replay_folder=r"C:\dev\BetaStar\replays\raw",
        output_file=r"C:\dev\BetaStar\replays\parsed\dataset.npz",
        debug=True,
    ):
        self.replay_folder = replay_folder
        self.output_file = output_file
        self.debug = debug

        self.unmapped_abilities = defaultdict(int)
        self.mapped_actions = defaultdict(int)

        # Only real macro decisions — no do_nothing
        self.EVENT_TO_ACTION = {
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

    def parse_replay(self, replay, min_length: int = 10) -> np.ndarray | None:
        """
        Walk all events chronologically and record one row per real macro action.
        The obs snapshot is taken AFTER updating pending counts so the observation
        already reflects that the issued command is now in progress.
        Returns float32 array of shape (T, OBS_SIZE+1), or None if too short.
        """
        protoss_player = None
        zerg_player = None
        for player in replay.players:
            if player.play_race == "Protoss":
                protoss_player = player
            elif player.play_race == "Zerg":
                zerg_player = player

        if protoss_player is None or zerg_player is None:
            return None

        pid = protoss_player.pid
        opp_pid = zerg_player.pid
        state = GameState()
        rows = []

        for event in replay.events:

            if isinstance(event, PlayerStatsEvent):
                if event.player.pid == pid:
                    state.update_from_stats(event)
                elif event.player.pid == opp_pid:
                    state.update_opp_from_stats(event)

            elif isinstance(event, (UnitBornEvent, UnitDoneEvent)):
                unit = event.unit
                owner = getattr(unit, "owner", None)
                if owner is None or owner.pid != pid:
                    continue
                state.unit_born_or_done(unit.name)

            elif isinstance(event, UnitDiedEvent):
                unit = event.unit
                owner = getattr(unit, "owner", None)
                if owner is None or owner.pid != pid:
                    continue
                state.unit_died(unit.name)

            elif isinstance(event, BasicCommandEvent):
                if event.player.pid != pid:
                    continue

                ability_name = event.ability_name

                # Update pending counts for ALL commands, not just mapped ones,
                # so the state stays accurate even for actions we don't train on
                state.on_build_command(ability_name)
                state.on_train_command(ability_name)

                action_id = self.EVENT_TO_ACTION.get(ability_name)
                if action_id is not None:
                    # Snapshot AFTER pending update — obs now shows this
                    # building/unit as in-progress, which is what the bot
                    # will see at the moment it issues the same command
                    rows.append(state.to_obs() + [float(action_id)])
                    self.mapped_actions[ability_name] += 1
                else:
                    self.unmapped_abilities[ability_name] += 1
                    if self.debug and self.unmapped_abilities[ability_name] == 1:
                        print(f"    [UNMAPPED] {ability_name}")

        if len(rows) < min_length:
            return None

        return np.array(rows, dtype=np.float32)

    def print_statistics(self):
        print("\n" + "=" * 60)
        print("PARSING STATISTICS")
        print("=" * 60)
        print("\nMapped Actions (included in dataset):")
        for ability, count in sorted(self.mapped_actions.items(), key=lambda x: -x[1]):
            action_id = self.EVENT_TO_ACTION.get(ability, 0)
            print(f"  [{action_id:2d}] {ability:30s}: {count:5d} samples")
        print(f"\nTotal mapped samples: {sum(self.mapped_actions.values())}")
        if self.unmapped_abilities:
            print("\nUnmapped Abilities (omitted):")
            for ability, count in sorted(self.unmapped_abilities.items(), key=lambda x: -x[1]):
                print(f"  {ability:30s}: {count:5d} occurrences")
        else:
            print("\nNo unmapped abilities found!")

    def parse_replay_folder(self):
        sequences = []
        bot_replays = []
        skipped = 0
        failed = 0

        replay_files = [f for f in os.listdir(
            self.replay_folder) if f.endswith(".SC2Replay")]
        print(f"Found {len(replay_files)} replay(s) to process.")

        for fname in replay_files:
            path = os.path.join(self.replay_folder, fname)
            try:
                replay = sc2reader.load_replay(path, load_level=4)

                races = {p.play_race for p in replay.players}
                if races != {"Protoss", "Zerg"}:
                    skipped += 1
                    continue

                if not all(p.is_human for p in replay.players):
                    skipped += 1
                    bot_replays.append(fname)
                    continue

                seq = self.parse_replay(replay)
                if seq is None:
                    skipped += 1
                    print(f"  {fname}: too short, skipped")
                    continue

                sequences.append(seq)
                print(f"  {fname}: {len(seq)} actions")

            except Exception as e:
                print(f"  FAILED {fname}: {e}")
                failed += 1

        if not sequences:
            print("No training data collected.")
            return

        seq_array = np.empty(len(sequences), dtype=object)
        for i, s in enumerate(sequences):
            seq_array[i] = s

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        np.savez(self.output_file, sequences=seq_array)

        total_steps = sum(len(s) for s in sequences)
        lengths = [len(s) for s in sequences]
        print(
            f"\nDone. {len(sequences)} sequences | {total_steps} total actions")
        print(
            f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
        print(f"Skipped: {skipped}  |  Failed: {failed}")
        if bot_replays:
            print(f"Bot replays skipped: {bot_replays}")
        print(f"Saved to: {self.output_file}")
        self.print_statistics()


if __name__ == "__main__":
    parser = ReplayParser()
    parser.parse_replay_folder()

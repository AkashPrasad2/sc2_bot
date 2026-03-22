from sc2.bot_ai import BotAI
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId

# 15 structures
PROTOSS_STRUCTURES = [
    UnitTypeId.NEXUS,
    UnitTypeId.PYLON,
    UnitTypeId.GATEWAY,
    UnitTypeId.WARPGATE,
    UnitTypeId.FORGE,
    UnitTypeId.TWILIGHTCOUNCIL,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.TEMPLARARCHIVE,
    UnitTypeId.ROBOTICSBAY,
    UnitTypeId.ROBOTICSFACILITY,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.STARGATE,
    UnitTypeId.FLEETBEACON,
]

# 8 units
PROTOSS_UNITS = [
    UnitTypeId.PROBE,
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.CARRIER,
    UnitTypeId.VOIDRAY,
]


class ObservationWrapper:
    """
    Converts game state into a flat vector for neural network input.

    Feature layout (53 total):
        [0]     game time (normalized)
        [1]     minerals
        [2]     vespene
        [3]     supply_used
        [4]     supply_cap
        [5]     worker saturation
        [6:21]  completed structure counts   (15)
        [21:29] completed unit counts        (8)
        [29:44] in-progress structure counts (15)
        [44:52] in-progress unit counts      (8)
        [52]    opponent supply_used
    """

    def __init__(self):
        self.observation_size = self.calculate_obs_size()

    def calculate_obs_size(self):
        # 6 base + 15 structures + 8 units + 15 structures_in_progress + 8 units_in_progress + 1 opp
        return 6 + len(PROTOSS_STRUCTURES) + len(PROTOSS_UNITS) + len(PROTOSS_STRUCTURES) + len(PROTOSS_UNITS) + 1

    def get_observation(self, bot, opponent=None):
        obs = []

        # --- Base features ---
        obs.append(bot.time / 720.0)
        obs.append(bot.minerals / 1800.0)
        obs.append(bot.vespene / 700.0)
        obs.append(bot.supply_used / 200.0)
        obs.append(bot.supply_cap / 200.0)

        worker_supply = bot.units(UnitTypeId.PROBE).amount
        ideal_workers = bot.townhalls.amount * 22
        obs.append(worker_supply / max(ideal_workers, 1))

        # --- Completed structures ---
        for structure in PROTOSS_STRUCTURES:
            obs.append(bot.structures(structure).ready.amount / 10.0)

        # --- Completed units ---
        for unit in PROTOSS_UNITS:
            obs.append(bot.units(unit).amount / 30.0)

        # --- In-progress structures (under construction) ---
        for structure in PROTOSS_STRUCTURES:
            obs.append(bot.structures(structure).not_ready.amount / 10.0)

        # --- In-progress units (queued in production buildings) ---
        # already_pending() counts units currently being trained across all
        # production buildings. Normalised identically to completed units.
        for unit in PROTOSS_UNITS:
            obs.append(bot.already_pending(unit) / 30.0)

        # --- Opponent ---
        obs.append(opponent.supply_used / 200.0)

        return obs

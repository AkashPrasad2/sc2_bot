from sc2.bot_ai import BotAI  # for our bot
from sc2.player import Bot, Computer  # for opponent
from sc2.ids.unit_typeid import UnitTypeId

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
    UnitTypeId.DARKSHRINE,
    UnitTypeId.ROBOTICSBAY,
    UnitTypeId.ROBOTICSFACILITY,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.STARGATE,
    UnitTypeId.FLEETBEACON,
]

PROTOSS_UNITS = [
    UnitTypeId.PROBE,
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.SENTRY,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.DARKTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.DISRUPTOR,
    UnitTypeId.OBSERVER,
    UnitTypeId.WARPPRISM,
    UnitTypeId.CARRIER,
    UnitTypeId.VOIDRAY,
    UnitTypeId.MOTHERSHIP,
]


class ObservationWrapper:
    """
    Converts game into a vector for neural network input

    NN will take resources, supply, etc as input
    """

    def __init__(self):
        self.observation_size = self.calculate_obs_size()

    def calculate_obs_size(self):
        return len(PROTOSS_UNITS) + len(PROTOSS_STRUCTURES) + 10

    def get_observation(self, bot, opponent):
        obs = []

        # Player resources (normalized)
        obs.append(bot.minerals / 1000.0)
        obs.append(bot.vespene / 500.0)
        obs.append(bot.supply_used / 200.0)
        obs.append(bot.supply_cap / 200.0)
        obs.append(len(bot.units) / 100.0)
        obs.append(len(bot.structures) / 20.0)

        # Player structures
        for structure in PROTOSS_STRUCTURES:
            obs.append(bot.structures(structure).amount / 10.0)

        # Player units
        for unit in PROTOSS_UNITS:
            obs.append(bot.units(unit).amount / 50.0)

        # Opponent info (normalized)
        obs.append(opponent.supply_used / 200.0)
        obs.append(opponent.supply_cap / 200.0)
        obs.append(len(opponent.units) / 100.0)
        obs.append(len(opponent.structures) / 20.0)

        return obs

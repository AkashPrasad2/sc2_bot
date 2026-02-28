from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

# output layer will be an array of numbers corresponding to the differnet actions the model can take
ACTIONS = [
    "do_nothing",               # 0
    "build_probe",              # 1
    "build_pylon",              # 2
    "build_gateway",            # 3
    "build_cyberneticscore",    # 4
    "build_assimilator",        # 5
    "assign_probe_to_gas",      # 6
    "train_zealot",             # 7
    "train_stalker",            # 8
    "attack_enemy_base",        # 9
]


async def execute_action(action_id: int, bot: BotAI):
    """Execute an action with minimal safety checks for faster training"""
    action_name = ACTIONS[action_id]

    if action_name == "do_nothing":
        pass

    elif action_name == "build_probe":
        if bot.can_afford(UnitTypeId.PROBE) and bot.townhalls.ready:
            bot.townhalls.ready.first.train(UnitTypeId.PROBE)

    elif action_name == "build_pylon":
        if bot.can_afford(UnitTypeId.PYLON) and bot.townhalls:
            await bot.build(UnitTypeId.PYLON, near=bot.townhalls.first)

    elif action_name == "build_gateway":
        if bot.can_afford(UnitTypeId.GATEWAY) and bot.structures(UnitTypeId.PYLON).ready:
            await bot.build(UnitTypeId.GATEWAY, near=bot.townhalls.first)

    elif action_name == "build_cyberneticscore":
        if bot.can_afford(UnitTypeId.CYBERNETICSCORE) and bot.structures(UnitTypeId.GATEWAY).ready:
            await bot.build(UnitTypeId.CYBERNETICSCORE, near=bot.townhalls.first)

    elif action_name == "build_assimilator":
        if bot.can_afford(UnitTypeId.ASSIMILATOR) and bot.townhalls:
            for vespene in bot.vespene_geyser.closer_than(15, bot.townhalls.first):
                # Check if already built here
                if bot.gas_buildings.filter(lambda u: u.distance_to(vespene) < 1):
                    continue
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                break

    elif action_name == "assign_probe_to_gas":
        # assign one probe to gas
        for assimilator in bot.gas_buildings.ready:
            if assimilator.assigned_harvesters < assimilator.ideal_harvesters:
                # find closest idle probe or mineral gatherer
                probe = None
                if bot.workers.idle:
                    probe = bot.workers.idle.closest_to(assimilator)
                elif bot.workers.gathering:
                    probe = bot.workers.gathering.closest_to(assimilator)

                if probe:
                    probe.gather(assimilator)
                    return

    elif action_name == "train_zealot":
        if bot.can_afford(UnitTypeId.ZEALOT) and bot.structures(UnitTypeId.GATEWAY).ready.idle:
            bot.structures(UnitTypeId.GATEWAY).ready.idle.first.train(
                UnitTypeId.ZEALOT)

    elif action_name == "train_stalker":
        if bot.can_afford(UnitTypeId.STALKER) and bot.structures(UnitTypeId.CYBERNETICSCORE).ready:
            if bot.structures(UnitTypeId.GATEWAY).ready.idle:
                bot.structures(UnitTypeId.GATEWAY).ready.idle.first.train(
                    UnitTypeId.STALKER)

    elif action_name == "attack_enemy_base":
        for unit in bot.units.of_type([UnitTypeId.ZEALOT, UnitTypeId.STALKER]).idle:
            unit.attack(bot.enemy_start_locations[0])

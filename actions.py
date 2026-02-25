from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

# output layer will be an array of numbers corresponding to the differnet actions the model can take
ACTIONS = [
    "do_nothing",
    "build_probe",

    "build_pylon",
    "build_gateway",
    "build_cyberneticscore",
    "build_assimilator",

    "train_zealot",
    "train_stalker",

    "attack_enemy_base",
]


async def execute_action(action_id: int, bot: BotAI):
    action_name = ACTIONS[action_id]

    if action_name == "do_nothing":
        pass

    elif action_name == "build_probe":
        if bot.can_afford(UnitTypeId.PROBE):
            for nexus in bot.townhalls.ready.idle:
                nexus.train(UnitTypeId.PROBE)
                break

    elif action_name == "build_pylon":
        if bot.can_afford(UnitTypeId.PYLON) and bot.already_pending(UnitTypeId.PYLON) < 2:
            await bot.build(UnitTypeId.PYLON, near=bot.townhalls.first)

    elif action_name == "build_gateway":
        if bot.can_afford(UnitTypeId.GATEWAY) and bot.structures(UnitTypeId.PYLON).ready:
            await bot.build(UnitTypeId.GATEWAY, near=bot.townhalls.first)

    elif action_name == "build_cyberneticscore":
        if bot.can_afford(UnitTypeId.CYBERNETICSCORE) and bot.structures(UnitTypeId.GATEWAY).ready:
            await bot.build(UnitTypeId.CYBERNETICSCORE, near=bot.townhalls.first)

    elif action_name == "build_assimilator":
        if bot.can_afford(UnitTypeId.ASSIMILATOR):
            for vespene in bot.vespene_geyser.closer_than(15, bot.townhalls.first):
                if bot.gas_buildings.filter(lambda unit: unit.distance_to(vespene) < 1):
                    continue
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                break

    elif action_name == "train_zealot":
        if bot.can_afford(UnitTypeId.ZEALOT):
            for gateway in bot.structures(UnitTypeId.GATEWAY).ready.idle:
                gateway.train(UnitTypeId.ZEALOT)
                break

    elif action_name == "train_stalker":
        if bot.can_afford(UnitTypeId.STALKER) and bot.structures(UnitTypeId.CYBERNETICSCORE).ready:
            for gateway in bot.structures(UnitTypeId.GATEWAY).ready.idle:
                gateway.train(UnitTypeId.STALKER)
                break

    elif action_name == "attack_enemy_base":
        for unit in bot.units.of_type([UnitTypeId.ZEALOT, UnitTypeId.STALKER]):
            # attack everything on the way to enemy base
            unit.attack(bot.enemy_start_locations[0])

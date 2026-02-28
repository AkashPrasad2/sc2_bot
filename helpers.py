from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId


async def build_structure(bot: BotAI, building: UnitTypeId):
    """Helper to systematically builds sturctures depending on the type"""

    if building == UnitTypeId.ASSIMILATOR:
        if bot.can_afford(UnitTypeId.ASSIMILATOR) and bot.townhalls:
            for vespene in bot.vespene_geyser.closer_than(15, bot.townhalls.first):
                # Check if already built here
                if bot.gas_buildings.filter(lambda u: u.distance_to(vespene) < 1):
                    continue
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                return

    elif building == UnitTypeId.PYLON:
        if bot.can_afford(UnitTypeId.PYLON) and bot.townhalls:
            # spread them out to maximize buildable surface area
            placement = await bot.find_placement(
                UnitTypeId.PYLON,
                near=bot.townhalls.first.position,
                placement_step=4
            )
            if placement:
                await bot.build(UnitTypeId.PYLON, placement)
                return

    elif building == UnitTypeId.NEXUS:
        if bot.can_afford(UnitTypeId.NEXUS):
            # Expand to nearest mineral field
            location = await bot.get_next_expansion()
            if location:
                await bot.build(UnitTypeId.NEXUS, location)
                return

    # all other buildings just need to be powered by pylon
    else:
        if bot.can_afford(building) and bot.structures(UnitTypeId.PYLON).ready:
            placement = await bot.find_placement(
                building,
                near=bot.townhalls.first.position,
                placement_step=2
            )
            if placement:
                await bot.build(building, placement)
                return

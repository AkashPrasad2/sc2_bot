from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
import actions


class ProtossBot(BotAI):
    async def on_step(self, iteration: int):
        # Auto-distribute workers (probes return to mining after building)
        await self.distribute_workers()

        # Test actions
        if self.supply_used == self.supply_cap:
            await actions.execute_action(2, self)  # await since its async

        if self.can_afford(UnitTypeId.PROBE) and self.supply_used < 15:
            await actions.execute_action(1, self)

        if self.can_afford(UnitTypeId.ASSIMILATOR) and self.structures(UnitTypeId.ASSIMILATOR).amount < 1:
            await actions.execute_action(5, self)


run_game(
    maps.get("AbyssalReefLE"),
    [Bot(Race.Protoss, ProtossBot()), Computer(Race.Zerg, Difficulty.VeryEasy)],
    realtime=False,
)

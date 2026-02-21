import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer

from bot import WorkerRushBot

if __name__ == "__main__":
    sc2.run_game(
        sc2.maps.get("AbyssalReefLE"),   # any ladder map you have installed
        [
            Bot(Race.Terran, WorkerRushBot()),
            Computer(Race.Zerg, Difficulty.Easy)
        ],
        realtime=False,
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import sys
from typing import Dict, Optional, Tuple

from game_constants import FoodType, ShopCosts
from item import Food
from robot_controller import RobotController

# Ensure this bot can import sibling modules when loaded via importlib from a file path.
_BOT_DIR = os.path.dirname(__file__)
if _BOT_DIR not in sys.path:
    sys.path.append(_BOT_DIR)

# Need to add library before submission

from efficient_bot_tools import MapStatic, Navigator, Pos, is_adjacent  # noqa: E402


@dataclass(frozen=True)
class Stations:
    shop: Pos
    counter: Pos
    cooker: Pos
    submit: Pos
    trash: Optional[Pos]


class Phase(Enum):
    BUY_MEAT = auto()
    PLACE_MEAT = auto()
    CHOP_MEAT = auto()
    PICKUP_MEAT = auto()
    START_COOK = auto()

    BUY_PLATE = auto()
    PLACE_PLATE = auto()
    BUY_NOODLES = auto()
    ADD_NOODLES = auto()

    WAIT_TAKE_MEAT = auto()
    ADD_MEAT = auto()
    PICKUP_PLATE = auto()
    SUBMIT = auto()


class BotPlayer:
    """
    Stage-1 "efficient_bot" implementation:

    - Single bot.
    - Manual scripted execution for the canonical map1 order: NOODLES + (CHOPPED+COOKED) MEAT.
    - Efficient movement: pre-index map stations + cached BFS distance fields for repeated targets.
    """

    def __init__(self, map_copy):
        self.static = MapStatic.from_map(map_copy)
        self.nav = Navigator(self.static)

        self.stations: Optional[Stations] = None
        # Store phase per bot_id so multiple bots can run independently.
        self.phase_by_bot: Dict[int, Phase] = {}
        self.debug: bool = bool(int(os.environ.get("EFFICIENT_BOT_DEBUG", "0")))

    def _get_phase(self, bot_id: int) -> Phase:
        return self.phase_by_bot.get(bot_id, Phase.BUY_MEAT)

    def _set_phase(self, bot_id: int, phase: Phase) -> None:
        self.phase_by_bot[bot_id] = phase

    def _pick_stations(self, from_pos: Pos) -> Optional[Stations]:
        shop = self.static.nearest_tile("SHOP", from_pos)
        counter = self.static.nearest_tile("COUNTER", from_pos)
        cooker = self.static.nearest_tile("COOKER", from_pos)
        submit = self.static.nearest_tile("SUBMIT", from_pos)
        trash = self.static.nearest_tile("TRASH", from_pos)

        if shop is None or counter is None or cooker is None or submit is None:
            return None
        return Stations(
            shop=shop, counter=counter, cooker=cooker, submit=submit, trash=trash
        )

    def _move_towards_adjacent(
        self, controller: RobotController, bot_id: int, pos: Pos, target: Pos
    ) -> Tuple[Pos, bool]:
        """
        Move at most one step toward being adjacent to `target`.
        Returns (new_pos, adjacent_now).
        """
        if is_adjacent(pos, target):
            return pos, True

        step = self.nav.step_towards_adjacent(pos, target)
        if step is None:
            return pos, False

        dx, dy = step
        moved = controller.move(bot_id, dx, dy)
        if moved:
            pos = (pos[0] + dx, pos[1] + dy)
        return pos, is_adjacent(pos, target)

    def _trash_holding_if_needed(
        self, controller: RobotController, bot_id: int, pos: Pos, stations: Stations
    ) -> Tuple[Pos, bool]:
        """
        If we have any unexpected item, trash it (best-effort).
        Returns (pos, trashed_or_noop_this_turn).
        """
        st = controller.get_bot_state(bot_id)
        holding = (st or {}).get("holding")
        if holding is None:
            return pos, True
        if stations.trash is None:
            return pos, False

        pos, adj = self._move_towards_adjacent(controller, bot_id, pos, stations.trash)
        if adj:
            controller.trash(bot_id, stations.trash[0], stations.trash[1])
            return pos, True
        return pos, False

    def play_turn(self, controller: RobotController):
        bot_ids = controller.get_team_bot_ids()
        if not bot_ids:
            return

        bot_id = bot_ids[0]
        phase = self._get_phase(bot_id)
        st = controller.get_bot_state(bot_id)
        if st is None:
            return
        pos: Pos = (int(st["x"]), int(st["y"]))

        if self.stations is None:
            self.stations = self._pick_stations(pos)
        if self.stations is None:
            return
        stations = self.stations

        holding = st.get("holding")
        holding_type = holding.get("type") if isinstance(holding, dict) else None
        holding_food_name = (
            holding.get("food_name") if isinstance(holding, dict) else None
        )
        holding_chopped = (
            bool(holding.get("chopped"))
            if isinstance(holding, dict) and holding_type == "Food"
            else False
        )

        def plate_matches_noodles_meat(plate_dict: dict) -> bool:
            foods = plate_dict.get("food")
            if not isinstance(foods, list):
                return False
            if len(foods) != 2:
                return False
            sig = sorted(
                (
                    (
                        f.get("food_name"),
                        bool(f.get("chopped")),
                        int(f.get("cooked_stage", 0)),
                    )
                    for f in foods
                    if isinstance(f, dict)
                )
            )
            return sig == [
                (FoodType.MEAT.food_name, True, 1),
                (FoodType.NOODLES.food_name, False, 0),
            ]

        # Stage-1 behavior: this bot is scripted for a single order type.
        # If there are no active orders, don't spam actions; just clean up hands.
        if not any(bool(o.get("is_active")) for o in controller.get_orders()):
            if holding is None:
                return
            self._trash_holding_if_needed(controller, bot_id, pos, stations)
            return

        # --- Phase logic (manual script) ---
        # 
        if phase == Phase.BUY_MEAT:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.shop
            )
            if adj and controller.buy(
                bot_id, FoodType.MEAT, stations.shop[0], stations.shop[1]
            ):
                self._set_phase(bot_id, Phase.PLACE_MEAT)
            return

        if phase == Phase.PLACE_MEAT:
            if holding_type != "Food" or holding_food_name != FoodType.MEAT.food_name:
                # lost state (or got blocked), restart safely
                self._set_phase(bot_id, Phase.BUY_MEAT)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.place(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.CHOP_MEAT)
            return

        if phase == Phase.CHOP_MEAT:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.chop(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.PICKUP_MEAT)
            return

        if phase == Phase.PICKUP_MEAT:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.pickup(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.START_COOK)
            return

        if phase == Phase.START_COOK:
            # Expect chopped meat in hand.
            if holding_type != "Food" or holding_food_name != FoodType.MEAT.food_name:
                self._set_phase(bot_id, Phase.BUY_MEAT)
                return
            if not holding_chopped:
                self._set_phase(bot_id, Phase.CHOP_MEAT)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.cooker
            )
            if adj and controller.place(bot_id, stations.cooker[0], stations.cooker[1]):
                self._set_phase(bot_id, Phase.BUY_PLATE)
            return

        if phase == Phase.BUY_PLATE:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.shop
            )
            if adj and controller.buy(
                bot_id, ShopCosts.PLATE, stations.shop[0], stations.shop[1]
            ):
                self._set_phase(bot_id, Phase.PLACE_PLATE)
            return

        if phase == Phase.PLACE_PLATE:
            if holding_type != "Plate":
                self._set_phase(bot_id, Phase.BUY_PLATE)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.place(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.BUY_NOODLES)
            return

        if phase == Phase.BUY_NOODLES:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.shop
            )
            if adj and controller.buy(
                bot_id, FoodType.NOODLES, stations.shop[0], stations.shop[1]
            ):
                self._set_phase(bot_id, Phase.ADD_NOODLES)
            return

        if phase == Phase.ADD_NOODLES:
            if (
                holding_type != "Food"
                or holding_food_name != FoodType.NOODLES.food_name
            ):
                self._set_phase(bot_id, Phase.BUY_NOODLES)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.add_food_to_plate(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.WAIT_TAKE_MEAT)
            return

        if phase == Phase.WAIT_TAKE_MEAT:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return

            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.cooker
            )
            if not adj:
                return

            tile = controller.get_tile(
                controller.get_team(), stations.cooker[0], stations.cooker[1]
            )
            pan = getattr(tile, "item", None)
            food = getattr(pan, "food", None)
            if not isinstance(food, Food):
                return  # keep waiting

            if int(food.cooked_stage) == 1:
                if controller.take_from_pan(
                    bot_id, stations.cooker[0], stations.cooker[1]
                ):
                    if self.debug:
                        print(
                            f"[simple_efficient_bot] took meat: chopped={bool(getattr(food,'chopped',False))} "
                            f"cooked_stage={int(getattr(food,'cooked_stage',-1))} turn={controller.get_turn()}"
                        )
                    self._set_phase(bot_id, Phase.ADD_MEAT)
                return

            if int(food.cooked_stage) >= 2:
                # burnt: trash it and restart the cycle
                if controller.take_from_pan(
                    bot_id, stations.cooker[0], stations.cooker[1]
                ):
                    self._set_phase(bot_id, Phase.BUY_MEAT)
                return

            return  # still raw, wait

        if phase == Phase.ADD_MEAT:
            if holding_type != "Food" or holding_food_name != FoodType.MEAT.food_name:
                self._set_phase(bot_id, Phase.BUY_MEAT)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.add_food_to_plate(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.PICKUP_PLATE)
            return

        if phase == Phase.PICKUP_PLATE:
            if holding is not None:
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.counter
            )
            if adj and controller.pickup(
                bot_id, stations.counter[0], stations.counter[1]
            ):
                self._set_phase(bot_id, Phase.SUBMIT)
            return

        if phase == Phase.SUBMIT:
            if holding_type != "Plate":
                self._set_phase(bot_id, Phase.BUY_MEAT)
                return
            pos, adj = self._move_towards_adjacent(
                controller, bot_id, pos, stations.submit
            )
            if not adj:
                return

            # Avoid spamming invalid submits; if our plate isn't correct, trash and restart.
            if not (isinstance(holding, dict) and plate_matches_noodles_meat(holding)):
                self._trash_holding_if_needed(controller, bot_id, pos, stations)
                self._set_phase(bot_id, Phase.BUY_MEAT)
                return

            if controller.submit(bot_id, stations.submit[0], stations.submit[1]):
                self._set_phase(bot_id, Phase.BUY_MEAT)
            elif self.debug:
                print(
                    f"[simple_efficient_bot] submit failed turn={controller.get_turn()} holding={holding}"
                )
                print(f"[simple_efficient_bot] orders={controller.get_orders()}")
            return

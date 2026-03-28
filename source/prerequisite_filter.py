"""
prerequisite_filter.py
======================
Enforces SC2 build-order prerequisites at inference time.

Why this exists
---------------
The LSTM was trained on replays where humans NEVER issue an action without
its prerequisites in place. At inference, when the model predicts e.g.
"train_voidray" with no stargate, the action fails silently and the hidden
state advances as if something happened — a trajectory the model was never
trained to recover from.

This layer intercepts those predictions and substitutes the *deepest unmet
prerequisite* in the build chain. The model's prediction is treated as
"intent" and the filter navigates toward it using the correct build order.

Design principles
-----------------
- No build order is hardcoded. Only prerequisite relationships are encoded.
- The filter never overrides the model's intent — it just ensures the path
  to that intent is structurally valid.
- Supply-block detection added: if supply_used >= supply_cap, inject
  build_pylon before any unit-training action.
- Affordability is NOT checked here. The bot already handles that via
  silent failure + retry on the next cooldown tick.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from actions import ACTIONS


# ---------------------------------------------------------------------------
# Prerequisite graph
# Each action maps to a list of (check_fn, fallback_action_name) tuples.
# Checks are evaluated in order; the first failing check provides the
# substituted action. All check_fns take a single BotAI argument.
# ---------------------------------------------------------------------------

def _has(bot: BotAI, unit_type: UnitTypeId) -> bool:
    return bool(bot.structures(unit_type).ready)


def _pending_or_has(bot: BotAI, unit_type: UnitTypeId) -> bool:
    """True if the structure is ready OR currently being built."""
    return (
        bool(bot.structures(unit_type).ready) or
        bool(bot.structures(unit_type).not_ready)
    )


# Each entry: action_name -> list of (prerequisite_check, fallback_action_name)
# Checks are evaluated in order. First failing check wins.
PREREQUISITES: dict[str, list[tuple]] = {

    # --- Unit training: Gateway units ---
    "train_zealot": [
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY),   "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),                "build_pylon"),
    ],
    "train_stalker": [
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY),         "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "train_high_templar": [
        (lambda b: _has(b, UnitTypeId.TEMPLARARCHIVE),      "build_templar_archive"),
        (lambda b: _pending_or_has(b, UnitTypeId.TWILIGHTCOUNCIL),
         "build_twilight_council"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY),         "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],

    # --- Unit training: Warp-in variants ---
    "warp_in_zealot": [
        (lambda b: _has(b, UnitTypeId.WARPGATE),   "research_warp_gate"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY),         "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "warp_in_stalker": [
        (lambda b: _has(b, UnitTypeId.WARPGATE),
         "research_warp_gate"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY),         "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "warp_in_high_templar": [
        (lambda b: _has(b, UnitTypeId.WARPGATE),
         "research_warp_gate"),
        (lambda b: _has(b, UnitTypeId.TEMPLARARCHIVE),
         "build_templar_archive"),
        (lambda b: _pending_or_has(b, UnitTypeId.TWILIGHTCOUNCIL),
         "build_twilight_council"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY),         "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],

    # --- Archon: requires 2 idle HT ---
    "archon_warp_selecton": [
        (lambda b: b.units(UnitTypeId.HIGHTEMPLAR).idle.amount >=
         2, "train_high_templar"),
    ],

    # --- Robotics ---
    "train_immortal": [
        (lambda b: _pending_or_has(b, UnitTypeId.ROBOTICSFACILITY),
         "build_robotics_facility"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                         "build_pylon"),
    ],

    # --- Stargate ---
    "train_voidray": [
        (lambda b: _pending_or_has(b, UnitTypeId.STARGATE),           "build_stargate"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                         "build_pylon"),
    ],
    "train_carrier": [
        (lambda b: _has(b, UnitTypeId.FLEETBEACON),
         "build_fleet_beacon"),
        (lambda b: _pending_or_has(b, UnitTypeId.STARGATE),           "build_stargate"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                         "build_pylon"),
    ],

    # --- Structures: everything needs power (pylon) ---
    "build_gateway": [
        (lambda b: _has(b, UnitTypeId.PYLON), "build_pylon"),
    ],
    "build_cyberneticscore": [
        (lambda b: _pending_or_has(b, UnitTypeId.GATEWAY), "build_gateway"),
        (lambda b: _has(b, UnitTypeId.PYLON),              "build_pylon"),
    ],
    "build_stargate": [
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "build_robotics_facility": [
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "build_twilight_council": [
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "build_photon_cannon": [
        (lambda b: _pending_or_has(b, UnitTypeId.FORGE), "build_forge"),
        (lambda b: _has(b, UnitTypeId.PYLON),            "build_pylon"),
    ],
    "build_fleet_beacon": [
        (lambda b: _pending_or_has(b, UnitTypeId.STARGATE),        "build_stargate"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "build_templar_archive": [
        (lambda b: _pending_or_has(b, UnitTypeId.TWILIGHTCOUNCIL),
         "build_twilight_council"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "build_forge": [
        (lambda b: _has(b, UnitTypeId.PYLON), "build_pylon"),
    ],

    # --- Research ---
    "research_charge": [
        (lambda b: _has(b, UnitTypeId.TWILIGHTCOUNCIL),
         "build_twilight_council"),
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "research_warp_gate": [
        (lambda b: _pending_or_has(b, UnitTypeId.CYBERNETICSCORE),
         "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),                      "build_pylon"),
    ],
    "upgrade_ground_weapons": [
        (lambda b: _has(b, UnitTypeId.FORGE), "build_forge"),
        (lambda b: _has(b, UnitTypeId.PYLON), "build_pylon"),
    ],
    "upgrade_air_weapons": [
        (lambda b: _has(b, UnitTypeId.CYBERNETICSCORE), "build_cyberneticscore"),
        (lambda b: _has(b, UnitTypeId.PYLON),           "build_pylon"),
    ],
    "upgrade_shields": [
        (lambda b: _has(b, UnitTypeId.FORGE), "build_forge"),
        (lambda b: _has(b, UnitTypeId.PYLON), "build_pylon"),
    ]
}

# Actions involving unit production that should be gated on supply
UNIT_TRAINING_ACTIONS = {
    "train_probe", "train_zealot", "train_stalker", "train_immortal",
    "train_voidray", "train_carrier", "train_high_templar",
    "warp_in_zealot", "warp_in_stalker", "warp_in_high_templar",
}


def resolve_action(action_id: int, bot: BotAI) -> tuple[int, bool]:
    """
    Given a model-predicted action_id, walk the prerequisite graph and
    return (resolved_action_id, was_overridden).

    The returned action_id may differ from the input if prerequisites are
    unmet. The caller can log this for debugging / analysis.

    Supply block is detected first: if supply_used >= supply_cap - 2 and
    the predicted action trains a unit, we inject build_pylon immediately.
    """
    action_name = ACTIONS[action_id]

    # --- Supply block check (unit training while near cap) ---
    if action_name in UNIT_TRAINING_ACTIONS:
        if bot.supply_used >= bot.supply_cap - 2 and bot.supply_cap < 200:
            pylon_id = ACTIONS.index("build_pylon")
            return pylon_id, True

    # --- Prerequisite chain walk ---
    checks = PREREQUISITES.get(action_name, [])
    for check_fn, fallback_name in checks:
        try:
            if not check_fn(bot):
                fallback_id = ACTIONS.index(fallback_name)
                return fallback_id, True
        except Exception:
            # If the check itself errors (e.g. empty collection), treat as
            # prerequisite not met and substitute the fallback.
            fallback_id = ACTIONS.index(fallback_name)
            return fallback_id, True

    # All checks passed — execute as predicted.
    return action_id, False

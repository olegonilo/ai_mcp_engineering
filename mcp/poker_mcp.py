#!/usr/bin/env python3
"""Poker Strategy Decision Tree MCP Server.

Exposes tools and resources for poker training:
  - GTO preflop/postflop recommendations
  - EV calculations
  - Hand range matrix visualization (13x13)
  - Decision tree from preflop → river
  - Study mode questions
"""
from __future__ import annotations

import random
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Poker Strategy Decision Tree")

# ─── Knowledge Base ───────────────────────────────────────────────────────────

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
POSITIONS = ["BTN", "CO", "HJ", "MP", "UTG", "SB", "BB"]
RANK_ORDER = {r: i for i, r in enumerate(RANKS)}

# Simplified GTO 6-max 100bb opening ranges per position
PREFLOP_RANGES: dict[str, dict[str, list[str]]] = {
    "BTN": {
        "raise": [
            "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
            "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","K9s","K8s","K7s",
            "QJs","QTs","Q9s",
            "JTs","J9s","J8s",
            "T9s","T8s","98s","87s","76s","65s","54s",
            "AKo","AQo","AJo","ATo","A9o","A8o",
            "KQo","KJo","KTo",
            "QJo","QTo","JTo",
        ],
    },
    "CO": {
        "raise": [
            "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
            "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","K9s",
            "QJs","QTs","Q9s",
            "JTs","J9s",
            "T9s","T8s","98s","87s","76s","65s",
            "AKo","AQo","AJo","ATo",
            "KQo","KJo",
            "QJo",
        ],
    },
    "HJ": {
        "raise": [
            "AA","KK","QQ","JJ","TT","99","88","77","66","55","44",
            "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s",
            "KQs","KJs","KTs",
            "QJs","QTs",
            "JTs","J9s",
            "T9s","T8s","98s","87s","76s",
            "AKo","AQo","AJo",
            "KQo",
        ],
    },
    "MP": {
        "raise": [
            "AA","KK","QQ","JJ","TT","99","88","77","66",
            "AKs","AQs","AJs","ATs","A9s","A8s","A5s","A4s",
            "KQs","KJs","KTs",
            "QJs","QTs",
            "JTs",
            "T9s","98s","87s",
            "AKo","AQo",
            "KQo",
        ],
    },
    "UTG": {
        "raise": [
            "AA","KK","QQ","JJ","TT","99","88",
            "AKs","AQs","AJs","ATs","A9s","A5s",
            "KQs","KJs",
            "QJs",
            "JTs",
            "T9s","98s",
            "AKo","AQo",
            "KQo",
        ],
    },
    "SB": {
        "raise": [
            "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
            "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","K9s","K8s",
            "QJs","QTs","Q9s",
            "JTs","J9s","J8s",
            "T9s","T8s","T7s",
            "98s","97s","87s","86s","76s","75s","65s","64s","54s",
            "AKo","AQo","AJo","ATo","A9o","A8o","A7o",
            "KQo","KJo","KTo",
            "QJo","QTo",
            "JTo",
        ],
    },
    "BB": {
        "defend": [
            "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
            "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
            "KQs","KJs","KTs","K9s","K8s","K7s",
            "QJs","QTs","Q9s","Q8s",
            "JTs","J9s","J8s","J7s",
            "T9s","T8s","T7s",
            "98s","97s","96s","87s","86s","76s","75s","65s","64s","54s",
            "AKo","AQo","AJo","ATo","A9o","A8o","A7o","A6o","A5o",
            "KQo","KJo","KTo","K9o",
            "QJo","QTo","Q9o",
            "JTo","J9o",
            "T9o","98o","87o","76o",
        ],
    },
}

# Postflop hand tiers
HAND_TIERS: dict[str, list[str]] = {
    "monster":  ["straight flush","quads","full house"],
    "strong":   ["flush","straight","set","trips","top two pair"],
    "medium":   ["top pair top kicker","overpair","two pair (bottom)"],
    "marginal": ["middle pair","top pair weak kicker","second pair"],
    "weak":     ["bottom pair","underpair","ace high"],
    "draw":     ["flush draw","open-ended straight draw","gutshot"],
}

# Board texture descriptions
BOARD_TEXTURES = {
    "dry":      "Few draws, disconnected (e.g. K72r). Favor high-frequency small c-bets.",
    "wet":      "Many draws, connected (e.g. JT9 two-tone). Lower c-bet frequency, check-raise more.",
    "monotone": "All same suit. Strong draw density. Mixed strategy; check very often.",
    "paired":   "Two same-rank cards (e.g. KK3). Polarized — trips possible.",
}

STUDY_QUESTIONS = [
    {
        "topic": "preflop", "difficulty": "beginner",
        "question": "You are UTG in a 6-handed game (100bb) and hold AJo. What is the GTO action?",
        "answer": "Fold. AJo is outside UTG's opening range. It plays poorly out of position and is dominated too often by AQo/AKo in calling ranges.",
        "concept": "Positional range construction",
    },
    {
        "topic": "preflop", "difficulty": "beginner",
        "question": "On the BTN vs a CO open (2.5bb) you hold KTo. Call, 3-bet, or fold?",
        "answer": "Call. KTo has enough equity to call in position but not enough to profitably 3-bet into a CO opening range.",
        "concept": "IP cold-calling ranges",
    },
    {
        "topic": "postflop", "difficulty": "intermediate",
        "question": "You opened BTN, BB calls. Board: K72 rainbow. BB checks. What is your c-bet strategy?",
        "answer": "High-frequency small c-bet (33% pot, ~80% of hands). K72r heavily favors BTN range (more Kx, more overpairs). Use small size with near-polar range.",
        "concept": "C-bet sizing on dry boards",
    },
    {
        "topic": "postflop", "difficulty": "intermediate",
        "question": "You 3-bet IP, villain calls. Flop: JT9 two-tone. Villain checks. Your strategy?",
        "answer": "Check back ~50-60% of range. JT9 connects heavily with the caller's range (QQ, KQ, JTo, T9s). When betting, use mixed large and medium sizes with a polarized range.",
        "concept": "Range disadvantage on wet boards",
    },
    {
        "topic": "concepts", "difficulty": "advanced",
        "question": "What is range advantage vs nut advantage, and why does the distinction matter?",
        "answer": "Range advantage = more strong hands overall. Nut advantage = more of the very strongest hands. Nut advantage enables aggressive strategies even without range advantage — e.g. 3-bettor often has nut advantage on Axx boards (more AA/AK) even if caller has more middling pairs.",
        "concept": "Range analysis",
    },
    {
        "topic": "concepts", "difficulty": "advanced",
        "question": "What is MDF (Minimum Defense Frequency) and how do you calculate it?",
        "answer": "MDF = 1 − bet/(pot+bet). It is the minimum fraction of your range to defend so opponent's bluffs break even. Example: 1/2 pot bet → MDF = 1 − 0.5/1.5 = 67%. Apply to ranges, not individual hands.",
        "concept": "Game theory / bluff defense",
    },
    {
        "topic": "preflop", "difficulty": "intermediate",
        "question": "In a 3-bet pot as PFR (OOP), board comes A83 rainbow. What is your strategy?",
        "answer": "Check at high frequency (~60%). Despite having many Ax, your opponent also called with Ax. Range advantage is marginal — avoid over-betting. Check-raise your strong Ax; check-call medium Ax.",
        "concept": "3-bet pot OOP strategy",
    },
    {
        "topic": "concepts", "difficulty": "beginner",
        "question": "What does 'equity realization' mean and how does position affect it?",
        "answer": "Equity realization is how much of your theoretical equity you actually capture in practice. Position dramatically improves it — you act last, control pot size, and find better spots to bluff or extract value.",
        "concept": "Equity realization",
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_hand(hand: str) -> str:
    """Convert 'Ah Kh', 'AhKh', or 'AK' into standard form 'AKs'/'AKo'/'AK'."""
    hand = hand.strip().replace(" ", "")
    upper = hand.upper()
    # Already in standard notation: pair (2 chars) or suited/offsuit (3 chars)
    if len(upper) in (2, 3) and upper[0] in RANK_ORDER:
        # Preserve lowercase s/o suffix (e.g. 'AKs' not 'AKS')
        return upper if len(upper) == 2 else upper[:2] + upper[2].lower()
    if len(upper) == 4:
        r1, s1, r2, s2 = upper[0], upper[1], upper[2], upper[3]
        suited = "s" if s1 == s2 else "o"
        if RANK_ORDER.get(r1, 99) > RANK_ORDER.get(r2, 99):
            r1, r2 = r2, r1
        return f"{r1}{r2}" if r1 == r2 else f"{r1}{r2}{suited}"
    return upper


def _in_range(hand: str, range_list: list[str]) -> bool:
    return _normalize_hand(hand) in range_list


def _board_texture(board: str) -> str:
    cards = board.upper().replace(",", " ").split()
    if len(cards) < 3:
        return "unknown"
    suits = [c[-1] for c in cards if len(c) >= 2]
    ranks = [c[:-1] for c in cards if len(c) >= 2]
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    if max(rank_counts.values(), default=1) >= 2:
        return "paired"
    max_suit = max((suits.count(s) for s in set(suits)), default=0)
    if max_suit >= 3:
        return "monotone"
    rank_indices = sorted(RANK_ORDER.get(r, 12) for r in ranks)
    span = rank_indices[-1] - rank_indices[0] if rank_indices else 99
    return "wet" if span <= 4 or max_suit == 2 else "dry"


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_preflop_recommendation(
    hand: str,
    position: str,
    action_facing: str = "none",
    stack_bb: float = 100.0,
) -> dict:
    """GTO preflop recommendation for a given hand and position.

    Args:
        hand: Hole cards in standard notation (e.g. 'AKs', 'QJo', 'TT', or 'Ah Kh')
        position: BTN | CO | HJ | MP | UTG | SB | BB
        action_facing: 'none' (first in) | 'open' (vs raise) | '3bet' | 'allin'
        stack_bb: Effective stack in big blinds (default 100)
    """
    hand = _normalize_hand(hand)
    position = position.upper()

    if position not in PREFLOP_RANGES:
        return {"error": f"Unknown position '{position}'. Valid: {', '.join(POSITIONS)}"}

    raise_range = PREFLOP_RANGES[position].get("raise", [])
    in_open_range = _in_range(hand, raise_range)
    is_premium = hand in {"AA","KK","QQ","AKs","AKo"}
    is_strong = hand in {"JJ","TT","AQs","AQo","AJs","KQs"}

    if action_facing == "none":
        action = "RAISE" if in_open_range else "FOLD"
        sizing = "2.5bb" if position in ("BTN","CO","HJ") else "3bb"
        note = (
            f"{'In' if in_open_range else 'NOT in'} {position} GTO opening range. "
            f"{'Open to ' + sizing + '.' if in_open_range else 'Fold — preserve equity for stronger spots.'}"
        )
    elif action_facing == "open":
        if is_premium:
            action, sizing, note = "3-BET", "~9bb", f"{hand} is premium — 3-bet for value and to build the pot."
        elif is_strong and position in ("BTN","CO"):
            action, sizing, note = "CALL or 3-BET (mixed)", "9bb if 3-betting", f"{hand} in {position}: mix calls and 3-bets to stay balanced."
        elif in_open_range:
            action, sizing, note = "CALL", "—", f"{hand} has playability to call but not enough equity for a profitable 3-bet from {position}."
        else:
            action, sizing, note = "FOLD", "—", f"{hand} lacks sufficient equity to continue vs an open from {position}."
    elif action_facing == "3bet":
        if hand in {"AA","KK"}:
            action, sizing, note = "4-BET / SHOVE", "~22-25bb or all-in", f"{hand}: top of range — 4-bet or shove for maximum value."
        elif is_premium or is_strong:
            action, sizing, note = "4-BET or CALL (mixed)", "22bb if 4-betting", f"{hand}: strong enough to mix 4-bets and calls to stay unexploitable."
        elif in_open_range:
            action, sizing, note = "FOLD or CALL (position-dependent)", "—", f"{hand}: borderline vs 3-bet. Prefer calling in position, folding OOP."
        else:
            action, sizing, note = "FOLD", "—", f"{hand}: not strong enough to continue vs a 3-bet."
    else:
        action, sizing, note = "FOLD", "—", "Default to fold vs unspecified action."

    stack_note = ""
    if stack_bb < 40:
        stack_note = " Short stack (<40bb): consider shove/fold strategy over minraise."
    elif stack_bb > 150:
        stack_note = " Deep stack (>150bb): suited connectors and pocket pairs gain relative value."

    return {
        "hand": hand,
        "position": position,
        "action_facing": action_facing,
        "stack_bb": stack_bb,
        "recommendation": action,
        "sizing": sizing,
        "reasoning": note + stack_note,
        "in_gto_range": in_open_range,
        "range_size": f"{len(raise_range)} combos in {position} opening range",
    }


@mcp.tool()
def get_postflop_recommendation(
    hand: str,
    board: str,
    position: str,
    street: str,
    pot_bb: float,
    stack_bb: float,
    hand_strength: str = "auto",
    in_3bet_pot: bool = False,
) -> dict:
    """GTO postflop recommendation for a given street and board.

    Args:
        hand: Your hole cards (e.g. 'AhKd' or 'AKs')
        board: Community cards (e.g. 'Ah 7d 2c')
        position: 'IP' (in position) or 'OOP'
        street: 'flop' | 'turn' | 'river'
        pot_bb: Pot size in big blinds before action
        stack_bb: Effective stack in big blinds
        hand_strength: 'monster' | 'strong' | 'medium' | 'marginal' | 'weak' | 'draw' | 'auto'
        in_3bet_pot: True if the hand is in a 3-bet pot
    """
    position = position.upper()
    street = street.lower()
    texture = _board_texture(board)
    spr = round(stack_bb / pot_bb, 2) if pot_bb > 0 else 99.0

    texture_desc = BOARD_TEXTURES.get(texture, "Unknown texture")

    if street == "flop":
        if texture == "dry":
            sizes = {"value": "25-33% pot", "bluff": "25-33% pot"}
            freq = "High (70-85%)" if not in_3bet_pot else "Mixed (50-60%)"
        elif texture == "wet":
            sizes = {"value": "50-75% pot", "bluff": "50% pot"}
            freq = "Low-medium (40-55%)"
        else:
            sizes = {"value": "40-50% pot", "bluff": "Rare"}
            freq = "Low (30-40%)"
    elif street == "turn":
        sizes = {"value": "60-75% pot", "bluff": "60-75% pot (semi-bluffs only)"}
        freq = "Selective — barrel with strong value or strong equity"
    else:
        sizes = {"value": "75-125% pot", "bluff": "75-125% pot"}
        freq = "Polarized — bet strong value or bluff; check everything else"

    if hand_strength == "auto":
        action = "Provide hand_strength for specific guidance: monster | strong | medium | marginal | weak | draw"
    elif hand_strength == "monster":
        action = "Bet or raise for value on all streets. Consider a flop check-raise to trap aggressive opponents."
    elif hand_strength == "strong":
        action = f"Bet for value ({sizes['value']}). Build the pot aggressively."
    elif hand_strength == "medium":
        action = f"Check mostly for pot control (SPR={spr}). Call 1 bet if opponent bets small; fold to large bets."
    elif hand_strength == "marginal":
        action = "Check for pot control. Bluff-catch once on earlier streets; fold to multi-street aggression."
    elif hand_strength == "weak":
        action = "Check and fold to aggression. No showdown value — only bluff with strong blockers."
    elif hand_strength == "draw":
        action = f"Semi-bluff ({sizes['bluff']}) in position. OOP: prefer check-raise to build pot with equity."
    else:
        action = "Evaluate hand vs board and apply GTO fundamentals."

    pos_note = (
        "IP: control pace, check back medium hands for cheap showdowns, bet value/bluffs."
        if position == "IP"
        else "OOP: check-raise > donk-bet. Avoid leading into IP player unless polarized."
    )

    return {
        "hand": hand,
        "board": board,
        "street": street,
        "position": position,
        "spr": spr,
        "board_texture": texture,
        "texture_note": texture_desc,
        "recommended_action": action,
        "bet_sizes": sizes,
        "c_bet_frequency": freq,
        "positional_note": pos_note,
        "hand_strength": hand_strength,
        "tier_examples": HAND_TIERS.get(hand_strength, []),
    }


@mcp.tool()
def calculate_ev(
    pot_bb: float,
    bet_bb: float,
    fold_equity: float,
    equity_vs_callers: float,
) -> dict:
    """Calculate the expected value (EV) of a bet or raise.

    Args:
        pot_bb: Current pot in big blinds before the bet
        bet_bb: Size of the bet/raise in big blinds
        fold_equity: Probability opponent folds (0.0–1.0)
        equity_vs_callers: Your equity vs opponent's calling range (0.0–1.0)
    """
    if pot_bb <= 0:
        return {"error": "pot_bb must be greater than 0"}
    if bet_bb <= 0:
        return {"error": "bet_bb must be greater than 0"}
    for name, val in [("fold_equity", fold_equity), ("equity_vs_callers", equity_vs_callers)]:
        if not 0.0 <= val <= 1.0:
            return {"error": f"{name} must be between 0.0 and 1.0"}

    ev_folds = fold_equity * pot_bb
    new_pot = pot_bb + 2 * bet_bb
    ev_calls = (1 - fold_equity) * (equity_vs_callers * new_pot - bet_bb)
    total_ev = ev_folds + ev_calls

    breakeven_fe = bet_bb / (pot_bb + bet_bb)
    mdf = 1 - breakeven_fe
    label = "PROFITABLE" if total_ev > 0.01 else ("LOSING" if total_ev < -0.01 else "BREAKEVEN")

    return {
        "pot_bb": pot_bb,
        "bet_bb": bet_bb,
        "bet_pct_pot": f"{round(bet_bb / pot_bb * 100)}%",
        "fold_equity": fold_equity,
        "equity_vs_callers": equity_vs_callers,
        "ev_when_fold": round(ev_folds, 3),
        "ev_when_call": round(ev_calls, 3),
        "total_ev_bb": round(total_ev, 3),
        "verdict": label,
        "breakeven_fold_equity": round(breakeven_fe, 3),
        "opponent_mdf": round(mdf, 3),
        "summary": (
            f"Bet {bet_bb}bb into {pot_bb}bb ({round(bet_bb/pot_bb*100)}% pot). "
            f"EV = {round(total_ev, 2)}bb → {label}. "
            f"Opponent must defend {round(mdf*100)}% to prevent profitable bluffs."
        ),
    }


@mcp.tool()
def get_hand_range_matrix(
    position: str = "BTN",
    range_type: str = "open",
) -> dict:
    """Visual 13×13 hand range matrix for a position.

    Args:
        position: BTN | CO | HJ | MP | UTG | SB | BB
        range_type: 'open' (raising range) | 'defend' (BB defense vs BTN open)

    Matrix legend: R = in range, . = fold
    Upper-right triangle = suited, lower-left = offsuit, diagonal = pairs.
    """
    position = position.upper()
    if position not in PREFLOP_RANGES:
        return {"error": f"Unknown position. Choose from: {', '.join(POSITIONS)}"}

    pos_data = PREFLOP_RANGES[position]
    if range_type == "defend" and position == "BB":
        active = pos_data.get("defend", [])
        label = "BB Defense Range vs BTN Open"
    else:
        active = pos_data.get("raise", pos_data.get("defend", []))
        label = f"{position} Opening Range"

    combo_count = 0
    rows: list[str] = []
    rows.append("    " + "".join(f"{r:4}" for r in RANKS))
    rows.append("    " + "─" * (len(RANKS) * 4))

    for i, r1 in enumerate(RANKS):
        row = f"{r1:3}│"
        for j, r2 in enumerate(RANKS):
            if i == j:
                hand, combos = f"{r1}{r2}", 6
            elif i < j:
                hand, combos = f"{r1}{r2}s", 4
            else:
                hand, combos = f"{r2}{r1}o", 12
            in_r = hand in active
            if in_r:
                combo_count += combos
            row += " R  " if in_r else " .  "
        rows.append(row)

    return {
        "position": position,
        "label": label,
        "matrix": "\n".join(rows),
        "combos_in_range": combo_count,
        "range_coverage_pct": round(combo_count / 1326 * 100, 1),
        "legend": "R=in range  .=fold | diagonal=pairs  upper-right=suited  lower-left=offsuit",
        "hands": active,
    }


@mcp.tool()
def get_decision_tree(
    hand: str,
    position: str,
    stack_bb: float = 100.0,
) -> dict:
    """Full preflop → flop → turn → river decision tree for a hand.

    Args:
        hand: Hole cards (e.g. 'AKs', 'JTs', 'QQ')
        position: Your position (BTN | CO | HJ | MP | UTG | SB)
        stack_bb: Effective stack in big blinds
    """
    hand = _normalize_hand(hand)
    position = position.upper()

    in_range = _in_range(hand, PREFLOP_RANGES.get(position, {}).get("raise", []))
    is_pair = len(hand) == 2 and hand[0] == hand[1]
    is_suited = hand.endswith("s")
    is_broadway = len(hand) >= 2 and all(r in "AKQJT" for r in hand[:2])
    is_premium = hand in {"AA","KK","QQ","AKs","AKo"}
    is_connector = (
        len(hand) == 3
        and abs(RANK_ORDER.get(hand[0], 0) - RANK_ORDER.get(hand[1], 0)) == 1
    )

    return {
        "hand": hand,
        "position": position,
        "stack_bb": stack_bb,
        "preflop": {
            "first_in": {
                "action": "RAISE" if in_range else "FOLD",
                "sizing": "2.5–3bb",
                "note": f"{'In' if in_range else 'NOT in'} {position} GTO opening range",
            },
            "vs_3bet": {
                "action": (
                    "4-BET / SHOVE" if hand in {"AA","KK"} else
                    "4-BET or CALL (mixed)" if is_premium or hand in {"JJ","TT","AQs"} else
                    "CALL (IP) / FOLD (OOP)" if in_range else "FOLD"
                ),
                "sizing": "22–25bb if 4-betting",
            },
        },
        "flop": {
            "as_pfr": {
                "dry_board": "High-freq small c-bet (33% pot) — exploit range advantage",
                "wet_board": "Check-heavy (~50%) — avoid bloating pot without equity",
                "hand_note": (
                    "Overpair: bet/check mix for protection"
                    if is_pair
                    else "TPTK: bet for medium value. Draw: semi-bluff or check-raise OOP."
                ),
            },
            "as_caller_ip": "Float wide, raise strong hands and combo draws. Use position on later streets.",
            "spr_guidance": (
                f"SPR ~{round(stack_bb / 3.5, 1)} (typical 3bb open, 1 caller). "
                f"{'Stack-off range: overpair+.' if stack_bb < 50 else 'Deep: avoid committing without two-pair+.'}"
            ),
        },
        "turn": {
            "barrel_criteria": [
                "Strong value hand (two pair or better)",
                "Semi-bluff equity (flush draw or OESD)",
                "Board card improves your range more than theirs",
                "Opponent showed weakness on flop",
            ],
            "sizing": "60–75% pot",
            "hand_note": (
                "Pair: re-evaluate equity before firing second barrel"
                if is_pair
                else "Connected/suited: check if draw improved; if not, evaluate bluff viability"
            ),
        },
        "river": {
            "value_threshold": "Bet if hand beats >50% of opponent's calling range",
            "bluff_selection": "Busted draws with blockers to villain's strong hands (e.g. Ace blocker to AA/AK)",
            "sizing": "75–125% pot (polarized)",
            "hand_note": (
                "Made hand: go for full value vs calling range"
                if not is_suited
                else "Missed flush draw: prime bluff candidate — your suit blocks villain's flushes"
            ),
        },
        "key_concepts": _concepts_for_hand(hand, is_pair, is_suited, is_broadway, is_premium, is_connector),
    }


def _concepts_for_hand(
    hand: str,
    is_pair: bool,
    is_suited: bool,
    is_broadway: bool,
    is_premium: bool,
    is_connector: bool,
) -> list[str]:
    tips: list[str] = []
    if is_premium:
        tips.append("Premium: 3-bet/4-bet for value. Play for stacks vs most stack depths.")
    if is_pair and not is_premium:
        tips.append("Pair: set-mine at SPR ≥ 15 (rule of 15). Overpair: protect equity but avoid over-commitment deep.")
    if is_suited:
        tips.append("Suited: gains ~3–4% equity over offsuit equivalent. Flush potential adds implied odds.")
    if is_connector:
        tips.append("Connector: playability-driven hand. Shines in multi-way pots with position.")
    if is_broadway and not is_premium:
        tips.append("Broadway: strong but vulnerable to two-pair/sets on coordinated boards.")
    if not tips:
        tips.append("Apply core fundamentals: position, SPR, range advantage, and equity realization.")
    return tips


@mcp.tool()
def get_study_question(
    topic: str = "any",
    difficulty: str = "any",
) -> dict:
    """Get a GTO study-mode question for active learning.

    Args:
        topic: 'preflop' | 'postflop' | 'concepts' | 'any'
        difficulty: 'beginner' | 'intermediate' | 'advanced' | 'any'
    """
    pool = STUDY_QUESTIONS
    if topic != "any":
        pool = [q for q in pool if q["topic"] == topic.lower()]
    if difficulty != "any":
        pool = [q for q in pool if q["difficulty"] == difficulty.lower()]

    if not pool:
        return {
            "error": f"No questions for topic='{topic}' difficulty='{difficulty}'",
            "available_topics": ["preflop","postflop","concepts"],
            "available_difficulties": ["beginner","intermediate","advanced"],
        }

    q = random.choice(pool)
    return {
        "topic": q["topic"],
        "difficulty": q["difficulty"],
        "concept": q["concept"],
        "question": q["question"],
        "gto_answer": q["answer"],
        "hint": f"Think about: {q['concept']}.",
        "pool_size": len(pool),
    }


# ─── Resources ────────────────────────────────────────────────────────────────

@mcp.resource("poker://ranges/{position}")
def position_range_resource(position: str) -> str:
    """Full GTO range for a position as plain text."""
    position = position.upper()
    if position not in PREFLOP_RANGES:
        return f"Unknown position '{position}'. Valid: {', '.join(POSITIONS)}"
    data = PREFLOP_RANGES[position]
    lines = [f"# {position} GTO Range (6-max, 100bb)\n"]
    for label, hands in data.items():
        lines.append(f"## {label.title()} ({len(hands)} hands)\n{', '.join(hands)}\n")
    return "\n".join(lines)


@mcp.resource("poker://hand-rankings")
def hand_rankings_resource() -> str:
    """Poker hand rankings and key probability reference."""
    return """\
# Poker Hand Rankings

| Rank | Hand            | 5-card Probability | Example         |
|------|-----------------|--------------------|-----------------|
| 1    | Royal Flush     | 0.000154%          | A K Q J T (s)   |
| 2    | Straight Flush  | 0.00139%           | 9 8 7 6 5 (s)   |
| 3    | Four of a Kind  | 0.0240%            | A A A A K       |
| 4    | Full House      | 0.1441%            | A A A K K       |
| 5    | Flush           | 0.1965%            | A J 8 4 2 (s)   |
| 6    | Straight        | 0.3925%            | A K Q J T       |
| 7    | Three of a Kind | 2.1128%            | A A A K Q       |
| 8    | Two Pair        | 4.7539%            | A A K K Q       |
| 9    | One Pair        | 42.2569%           | A A K Q J       |
| 10   | High Card       | 50.1177%           | A K Q J 9       |

## Key Equity Matchups
- AA vs KK: ~82% favourite
- AA vs random hand: ~85%
- Flush draw vs made hand (by river): ~35%
- Open-ended straight draw vs made hand: ~32%
- Gutshot vs made hand: ~17%
- Set vs overpair: ~80%
- AKs vs QQ (coin flip): ~50/50

## Starting Hand Combos (1326 total)
- Pocket pair: 6 combos (e.g. 6 × AA)
- Suited: 4 combos (e.g. 4 × AKs)
- Offsuit: 12 combos (e.g. 12 × AKo)
"""


@mcp.resource("poker://concepts/gto-basics")
def gto_basics_resource() -> str:
    """Core GTO concepts cheat-sheet."""
    return """\
# GTO Poker Concepts

## Fundamental Principles
1. Balance — mix value bets and bluffs so opponent cannot exploit you
2. Range-vs-range thinking — consider your entire range, not just your specific hand
3. Position — in-position players realize more equity; widen ranges accordingly
4. SPR (Stack-to-Pot Ratio) — determines commitment thresholds (low SPR = more committed)
5. MDF (Minimum Defense Frequency) — 1 − bet/(pot+bet); defend this % to prevent profitable bluffs

## Preflop Range Sizes (6-max, 100bb)
| Position | RFI Range | Key Additions Over Tighter Pos. |
|----------|-----------|---------------------------------|
| UTG      | ~14%      | TT+, ATs+, AQo+                 |
| MP       | ~18%      | 88+, A9s+, KQs                  |
| HJ       | ~22%      | 66+, A7s+, KJs                  |
| CO       | ~28%      | 44+, A4s+, KTo+                 |
| BTN      | ~45%      | 22+, A2s+, K7s+                 |
| SB       | ~40%      | raise or fold — never cold-call |
| BB       | ~65% def. | vs BTN open                     |

## Postflop Sizing Reference
| Scenario           | Bet Size        |
|--------------------|-----------------|
| Dry flop c-bet     | 25–33% pot      |
| Wet flop c-bet     | 50–75% pot      |
| Turn barrel        | 60–75% pot      |
| River value/bluff  | 75–125% pot     |
| River overbet      | 125–200% pot    |

## Common Mistakes
- C-betting too often on wet/monotone boards
- Not adjusting bet sizes for SPR on turns and rivers
- Bluffing hands that retain showdown value
- Failing to check-raise on favorable boards OOP
- Ignoring blockers when selecting river bluffs
"""


if __name__ == "__main__":
    mcp.run()

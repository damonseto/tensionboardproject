import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PAD, BOS, EOS = 0, 1, 2
ROLE_IDS = [5, 6, 7, 8]

# Layout 10 is Tension Board 2 Mirror
LAYOUT_ID = 10

# Anything longer than 20 holds is not a board problem (~2 percent)
MAX_HOLDS = 20

# Grades are community averages, so few ascents means an unreliable grade, so we filter for min of 5 ascents.
MIN_ASCENTS = 5

ROLES = {5: 'start', 6: 'hand', 7: 'finish', 8: 'foot'}
COLORS = {5: '#00DD00', 6: '#0066FF', 7: '#FF0000', 8: '#FF00FF'}

role_to_token = {r: 3 + i for i, r in enumerate(ROLE_IDS)}
token_to_role = {v: k for k, v in role_to_token.items()}


def decode_frames(frames):
    """Split a frames string like 'p466r8p477r6' into [(466, 8), (477, 6)]."""
    pairs = re.findall(r'p(\d+)r(\d+)', frames)
    return [(int(p), int(r)) for p, r in pairs]


def load_holds(conn):
    """Every placement on our layout with its x/y position on the board.

    Placements live in one table and physical coordinates in another, so this
    joins them. Needed to turn a hold ID into somewhere to draw a dot.
    """
    return pd.read_sql_query(f"""
        SELECT p.id AS placement_id, h.x, h.y, h.name
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        WHERE p.layout_id = {LAYOUT_ID}
    """, conn)


def load_data(conn):
    """The training set: one row per (climb, angle) pair, filtered."""
    d = pd.read_sql_query(f"""
        SELECT c.uuid, c.name, c.frames,
               s.angle, s.difficulty_average, s.ascensionist_count,
               s.quality_average
        FROM climbs c
        JOIN climb_stats s ON c.uuid = s.climb_uuid
        WHERE c.layout_id = {LAYOUT_ID} AND s.ascensionist_count >= {MIN_ASCENTS}
    """, conn)
    d['n_holds'] = d['frames'].apply(lambda f: len(decode_frames(f)))
    return d[d['n_holds'] <= MAX_HOLDS].reset_index(drop=True)


def build_tokenizer(holds):
    """Map placement IDs to contiguous token indices, and back.

    Placement IDs are sparse (690 of them scattered across 304-1491), but a
    model needs indices starting at 0. Layout 0-2 are PAD/BOS/EOS, 3-6 are
    roles, 7+ are holds.
    """
    placement_ids = sorted(holds['placement_id'].tolist())
    hold_to_token = {p: 7 + i for i, p in enumerate(placement_ids)}
    token_to_hold = {v: k for k, v in hold_to_token.items()}
    return hold_to_token, token_to_hold, 7 + len(placement_ids)


def encode(frames, hold_to_token):
    """Frames string -> token list, interleaved as [BOS, hold, role, ..., EOS].
    'p466r8p477r6' -> [1, 169, 6, 180, 4, 2]
                       |   |   |   |   |  |
                     BOS  p466 r8 p477 r6 EOS
    """
    tokens = [BOS]
    for pid, role in decode_frames(frames):
        tokens.append(hold_to_token[pid])
        tokens.append(role_to_token[role])
    tokens.append(EOS)
    return tokens


def decode(tokens, token_to_hold):
    """Token list -> [(placement_id, role), ...].

    Skips anything that isn't a valid hold/role pair, so this is safe to run
    on model output that hasn't respected the alternating structure.
    """
    pairs = []
    for i in range(1, len(tokens) - 1, 2):
        if i + 1 < len(tokens) and tokens[i] in token_to_hold and tokens[i+1] in token_to_role:
            pairs.append((token_to_hold[tokens[i]], token_to_role[tokens[i+1]]))
    return pairs


def plot_climb(frames, holds, title=''):
    """Draw a climb on the board"""
    coords = holds.set_index('placement_id')
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(holds['x'], holds['y'], s=8, c='lightgray')
    pairs = decode_frames(frames) if isinstance(frames, str) else frames
    for pid, role in pairs:
        row = coords.loc[pid]
        ax.scatter(row['x'], row['y'], s=200, facecolors='none',
                   edgecolors=COLORS[role], linewidths=3)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()
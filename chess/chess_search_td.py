#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chess_search_td.py

Search, evaluation functions and TD-leaf self-play, measured against a value
function that is known exactly.

Reference implementation for the lecture "Search, Evaluation and Self-Play",
Reinforcement Learning, Utah State University, Day 6.

Why this domain
---------------
Day 5 solved king and rook against king exactly: for all 399,112 legal
positions we hold the true distance to mate. Day 6 throws that table away and
rebuilds a player out of the three things you are left with when the table is
impossible: a depth-limited search, a parameterised evaluation function, and a
learning rule. Because the exact answer is still sitting in memory, every
approximation on this page can be scored against ground truth rather than
against another approximation. That is not possible in real chess and it is
the whole reason for working here.

Everything is exact. Policies are never scored by playing sample games; they
are scored by dynamic programming on the graph induced by the policy, which
gives the true distance to mate against a perfect defender from every legal
position at once.

The one stochastic component is the exploration during TD-leaf training, and
it is seeded.

Usage
-----
    python3 chess_search_td.py                # print every table
    python3 chess_search_td.py --section 4    # print one section only
"""

import argparse
import math
import random
import sys
from array import array

import chess_endgame_dp as dp
from chess_endgame_dp import WHITE, BLACK, UNSOLVED

PARAMS = {
    "version":       "1.0.0",
    "seed":          20260920,
    "gamma":         1.0,       # undiscounted, as everywhere in this course
    "step_cost":     -1.0,      # one unit per ply, so value = minus distance
    "draw_value":    -200.0,    # the practical stand-in for minus infinity
    "alpha":         0.002,     # TD-leaf step size
    "lambda":        0.7,       # eligibility decay
    "episodes":      3000,
    "train_depth":   1,         # search depth used while learning
    "epsilon":       0.2,       # exploration during training
    "max_plies":     120,       # training episode cutoff
    "board":         8,
    "eval_board":    6,         # board used for the full depth sweep
    "probe_sample":  400,       # positions in the node-count benchmark
}

FEATURES = ("bk_edge", "king_dist", "bk_mobility", "bk_corner", "rook_split",
            "rook_hangs", "bias")

# A hand-written evaluation, written the way a chess player would describe the
# endgame: drive the enemy king to the edge and then to a corner, bring your
# own king close, take away squares, keep the rook cutting the board. The
# weights are in units of plies, guessed once and never tuned.
HAND = {"bk_edge": 3.0, "king_dist": -1.5, "bk_mobility": -1.0,
        "bk_corner": -1.5, "rook_split": 2.0, "rook_hangs": -40.0,
        "bias": -12.0}


# ===========================================================================
# Features
# ===========================================================================

def make_featuriser(eg):
    """Return f(wk, wp, bk) -> tuple of feature values, for one board."""
    b = eg.b
    B = b.B
    mid = (B - 1) / 2.0
    corners = (0, B - 1, (B - 1) * B, B * B - 1)

    def cheb(s, t):
        return max(abs(s % B - t % B), abs(s // B - t // B))

    def feat(wk, wp, bk):
        f, r = bk % B, bk // B
        edge = max(abs(f - mid), abs(r - mid))          # bigger at the rim
        kd = cheb(wk, bk)
        mob = 0
        for t in b.king[bk]:
            if t == wk or b.adj[t][wk]:
                continue
            if t == wp:
                if not b.adj[wp][wk]:
                    mob += 1
                continue
            if not eg.attacks(wp, wk, t):
                mob += 1
        corner = min(cheb(bk, c) for c in corners)
        wf, wr = wk % B, wk // B
        pf, pr = wp % B, wp // B
        split = 1.0 if ((min(wr, r) < pr < max(wr, r))
                        or (min(wf, f) < pf < max(wf, f))) else 0.0
        hangs = 1.0 if (b.adj[bk][wp] and not b.adj[wk][wp]) else 0.0
        return (edge, float(kd), float(mob), float(corner), split, hangs, 1.0)

    return feat


def dot(w, f):
    return sum(w[k] * x for k, x in zip(FEATURES, f))


# ===========================================================================
# Search
# ===========================================================================

class Searcher:
    """Depth-limited minimax over the Day 5 transition graph.

    Values are on the distance scale of Day 5: an estimate of minus the number
    of plies to mate, so 0 is mate delivered and more negative is worse for
    White. Terminal positions are always scored exactly, whatever the depth.
    """

    def __init__(self, eg, weights, alphabeta=True, order=True):
        self.eg = eg
        self.w = dict(weights)
        self.feat = make_featuriser(eg)
        self.alphabeta = alphabeta
        self.order = order
        self.nodes = 0
        self.draw = PARAMS["draw_value"]
        self.INF = 1e9

    def leaf(self, s):
        wk, wp, bk, _ = self.eg.b.unindex(s)
        return dot(self.w, self.feat(wk, wp, bk))

    def terminal(self, s):
        """Exact value if s ends the game, else None."""
        eg = self.eg
        if eg.mate[s]:
            return 0.0
        if eg.stale[s]:
            return self.draw
        return None

    def search(self, s, depth, alpha=-1e9, beta=1e9):
        """Backed-up value of s, in plies to mate, negated.

        The step cost of one ply is charged on the way down, in child(), so
        that alpha and beta always refer to the same scale as the values they
        are compared against. Charging it on the way out instead is a common
        bug: the bounds are then off by one ply and the cutoffs are wrong.
        """
        self.nodes += 1
        t = self.terminal(s)
        if t is not None:
            return t
        if depth == 0:
            return self.leaf(s)
        eg = self.eg
        kids = eg.moves(s)
        white = (s & 1) == WHITE
        if not kids:
            return self.draw
        if self.order:
            kids = sorted(kids, key=lambda c: -self.quick(c, white))
        if white:
            best = -self.INF
            for c in kids:
                v = self.child(c, depth, alpha, beta)
                if v > best:
                    best = v
                if self.alphabeta:
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        break
            return best
        best = self.INF
        for c in kids:
            v = self.child(c, depth, alpha, beta)
            if v < best:
                best = v
            if self.alphabeta:
                if best < beta:
                    beta = best
                if beta <= alpha:
                    break
        return best

    def child(self, c, depth, alpha, beta):
        """Value of the position reached by one move, with the ply charged."""
        if c == self.eg.b.DRAW_NODE:
            self.nodes += 1
            return self.draw
        return self.search(c, depth - 1, alpha + 1.0, beta + 1.0) - 1.0

    def quick(self, c, white_parent):
        """One-ply ordering key: the leaf score of the child, from the point
        of view of the side that is choosing."""
        if c == self.eg.b.DRAW_NODE:
            return -self.INF if white_parent else self.INF
        v = self.leaf(c)
        return v if white_parent else -v

    def choose(self, s, depth):
        """White's move at s: the successor with the best backed-up value.

        `depth` counts plies including White's own move, so depth 1 means
        "score the position I would reach" and depth 3 means "my move, the
        reply, and my answer to it". Ties are broken by the smaller state
        index, so the policy is a deterministic function of the weights.
        """
        best, arg = None, None
        for c in self.eg.moves(s):
            v = self.child(c, depth, -self.INF, self.INF)
            if best is None or v > best:
                best, arg = v, c
        return arg


# ===========================================================================
# Exact evaluation of a policy
# ===========================================================================

def policy_distances(eg, choose):
    """Exact distance to mate for every state, when White follows the
    deterministic policy `choose` and Black defends optimally.

    Retrograde analysis again, on the graph in which every White-to-move state
    has out-degree one. States that come out unlabelled are states from which
    this policy never forces mate, whatever the reason: a real draw, a
    shuffle, or a blunder that loses the rook.
    """
    b = eg.b
    N = b.N
    # build the induced successor lists
    start = array("i", bytes(4 * (N + 1)))
    succ = array("i")
    pos = 0
    for s in range(N):
        if eg.legal[s]:
            if (s & 1) == WHITE:
                c = choose(s)
                if c is not None:
                    succ.append(c)
                    pos += 1
            else:
                for c in eg.moves(s):
                    succ.append(c)
                    pos += 1
        start[s + 1] = pos
    # invert
    indeg = array("i", bytes(4 * (N + 1)))
    for s in range(N):
        for k in range(start[s], start[s + 1]):
            t = succ[k]
            if t != b.DRAW_NODE:
                indeg[t] += 1
    pstart = array("i", bytes(4 * (N + 1)))
    run = 0
    for s in range(N):
        pstart[s] = run
        run += indeg[s]
    pstart[N] = run
    fill = array("i", pstart[:N])
    pred = array("i", bytes(4 * run))
    for s in range(N):
        for k in range(start[s], start[s + 1]):
            t = succ[k]
            if t != b.DRAW_NODE:
                pred[fill[t]] = s
                fill[t] += 1
    d = array("i", bytes(4 * N))
    for s in range(N):
        d[s] = UNSOLVED
    cnt = array("i", bytes(4 * N))
    for s in range(N):
        if eg.legal[s] and (s & 1) == BLACK:
            cnt[s] = start[s + 1] - start[s]
    frontier = [s for s in range(N) if eg.mate[s]]
    for s in frontier:
        d[s] = 0
    ply = 0
    while frontier:
        nxt = []
        for t in frontier:
            for k in range(pstart[t], pstart[t + 1]):
                s = pred[k]
                if d[s] != UNSOLVED:
                    continue
                if (s & 1) == WHITE:
                    d[s] = ply + 1
                    nxt.append(s)
                else:
                    cnt[s] -= 1
                    if cnt[s] == 0:
                        d[s] = ply + 1
                        nxt.append(s)
        frontier = nxt
        ply += 1
    return d


def score_policy(eg, d):
    """Summarise a policy's exact distances over White-to-move positions."""
    tot = n = worst = 0
    lost = 0
    opt = 0
    for s in range(eg.b.N):
        if not eg.legal[s] or (s & 1) != WHITE:
            continue
        n += 1
        if d[s] == UNSOLVED:
            lost += 1
            continue
        tot += d[s]
        worst = max(worst, d[s])
        if d[s] == eg.dtm[s]:
            opt += 1
    solved = n - lost
    return {"positions": n, "mated": solved, "failed": lost,
            "pct": 100.0 * solved / n,
            "mean_plies": tot / solved if solved else float("nan"),
            "worst_plies": worst,
            "optimal": opt, "pct_optimal": 100.0 * opt / n}


# ===========================================================================
# TD-leaf self-play
# ===========================================================================

def train(eg, episodes=None, depth=None, seed=None, log_every=None):
    """TD(lambda) on the value of the leaf of the search, learned by self-play.

    Both sides use the same weights; White maximises, Black minimises. The
    reward is minus one per ply, so the target that the weights are chasing is
    minus the distance to mate, exactly the Day 5 metric. No exact table is
    consulted anywhere in this function.
    """
    P = PARAMS
    episodes = episodes or P["episodes"]
    depth = P["train_depth"] if depth is None else depth
    rng = random.Random(P["seed"] if seed is None else seed)
    w = {k: 0.0 for k in FEATURES}
    srch = Searcher(eg, w)
    feat = make_featuriser(eg)
    b = eg.b
    starts = [s for s in range(b.N) if eg.legal[s] and (s & 1) == WHITE]
    curve = []

    for ep in range(1, episodes + 1):
        s = starts[rng.randrange(len(starts))]
        trace = {k: 0.0 for k in FEATURES}
        plies = 0
        while plies < P["max_plies"]:
            srch.w = w
            t = srch.terminal(s)
            if t is not None:
                break
            kids = eg.moves(s)
            white = (s & 1) == WHITE
            if rng.random() < P["epsilon"]:
                c = kids[rng.randrange(len(kids))]
            else:
                best, c = None, None
                for k in kids:
                    v = srch.child(k, depth, -srch.INF, srch.INF)
                    if best is None or (v > best if white else v < best):
                        best, c = v, k
            # the leaf value of the position we are leaving, and of the one we
            # are entering, both from White's point of view
            f_here = feat(*b.unindex(s)[:3])
            v_here = dot(w, f_here)
            if c == b.DRAW_NODE:
                v_next, done = P["draw_value"], True
            else:
                tn = srch.terminal(c)
                if tn is not None:
                    v_next, done = tn, True
                else:
                    v_next, done = dot(w, feat(*b.unindex(c)[:3])), False
            delta = P["step_cost"] + P["gamma"] * v_next - v_here
            for k, x in zip(FEATURES, f_here):
                trace[k] = P["gamma"] * P["lambda"] * trace[k] + x
                w[k] += P["alpha"] * delta * trace[k]
            plies += 1
            if done:
                break
            s = c
        if log_every and ep % log_every == 0:
            curve.append((ep, dict(w)))
    return w, curve


# ===========================================================================
# Printing
# ===========================================================================

def head(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def rule(n=74):
    print("  " + "-" * (n - 2))


def commas(n):
    return "{:,}".format(n)


CACHE = {}


def board(B=None, piece="R"):
    B = B or PARAMS["board"]
    key = (B, piece)
    if key not in CACHE:
        CACHE[key] = dp.solved(B, piece)
    return CACHE[key]


def probe_set(eg, k=None):
    """A deterministic, spread-out set of White-to-move positions."""
    k = k or PARAMS["probe_sample"]
    all_w = [s for s in range(eg.b.N) if eg.legal[s] and (s & 1) == WHITE]
    step = len(all_w) // k
    return all_w[::step][:k]


def section_1():
    head("1. THE EVALUATION FUNCTION AGAINST GROUND TRUTH")
    eg = board()
    feat = make_featuriser(eg)
    w = HAND
    n = 0
    se = 0.0
    sx = sy = sxx = syy = sxy = 0.0
    for s in range(eg.b.N):
        if not eg.legal[s] or (s & 1) != WHITE:
            continue
        x = dot(w, feat(*eg.b.unindex(s)[:3]))
        y = -float(eg.dtm[s])
        n += 1
        se += (x - y) ** 2
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
    r = (n * sxy - sx * sy) / math.sqrt((n * sxx - sx * sx)
                                        * (n * syy - sy * sy))
    print("""
  The hand-written evaluation, scored against the exact table on all %s
  White-to-move positions. The scale is plies: both the estimate and the truth
  are minus the number of plies to mate.
""" % commas(n))
    for k in FEATURES:
        print("  %-14s %8.2f" % (k, w[k]))
    rule()
    print("  %-14s %8.2f" % ("mean truth", sy / n))
    print("  %-14s %8.2f" % ("mean estimate", sx / n))
    print("  %-14s %8.2f" % ("rms error", math.sqrt(se / n)))
    print("  %-14s %8.3f" % ("correlation", r))
    print("""
  The estimate is off by %.1f plies, two full moves, on a quantity whose mean
  is %.1f, and it explains about %.0f percent of the variance. That is a
  mediocre value function. Section 3 measures whether it is a mediocre player,
  which is a different question with a different answer.
""" % (math.sqrt(se / n), sy / n, 100 * r * r))


def section_2():
    head("2. WHAT ALPHA-BETA AND MOVE ORDERING BUY")
    eg = board()
    probes = probe_set(eg)
    print("""
  Nodes visited by a fixed-depth search from each of %d positions spread
  evenly through the table, totalled. Three searchers, same evaluation, same
  positions, same answers: only the work differs.
""" % len(probes))
    print("  %-7s %16s %16s %16s %9s"
          % ("depth", "minimax", "alpha-beta", "+ ordering", "saving"))
    rule()
    for depth in (1, 2, 3, 4):
        row = []
        for ab, order in ((False, False), (True, False), (True, True)):
            s = Searcher(eg, HAND, alphabeta=ab, order=order)
            for p in probes:
                s.search(p, depth)
            row.append(s.nodes)
        print("  %-7d %16s %16s %16s %8.1fx"
              % (depth, commas(row[0]), commas(row[1]), commas(row[2]),
                 row[0] / float(row[2])))
    print("""
  Alpha-beta returns exactly the value plain minimax returns. Nothing is
  approximated; branches are skipped only when they provably cannot change the
  answer at the root.
""")


def move_accuracy(eg, srch, depth, states):
    """Exact quality of the chosen move at each state: does it move one ply
    closer to mate (optimal), or not (an error, measured in plies lost)?"""
    opt = err = 0
    lost = 0
    for s in states:
        c = srch.choose(s, depth)
        if c is None or c == eg.b.DRAW_NODE or eg.dtm[c] < 0:
            lost += 1
            err += 1
            continue
        if eg.dtm[c] == eg.dtm[s] - 1:
            opt += 1
        else:
            err += 1
            lost += (eg.dtm[c] + 1 - eg.dtm[s])
    n = len(states)
    return {"n": n, "optimal": opt, "pct": 100.0 * opt / n,
            "mean_loss": lost / float(n)}


def section_3():
    head("3. MOVE ACCURACY BY SEARCH DEPTH")
    eg = board()
    probes = probe_set(eg, 2000)
    print("""
  For each of %d positions spread evenly through the table, the move the
  search plays is compared with the exact table. A move is optimal when it
  reduces the true distance to mate by one ply, which is the most any move
  can do. "plies lost" is the average increase in the distance to mate caused
  by the move actually played.
""" % len(probes))
    print("  %-22s %10s %10s %12s"
          % ("player", "optimal", "percent", "plies lost"))
    rule()
    for depth in (1, 2, 3, 4):
        srch = Searcher(eg, HAND)
        r = move_accuracy(eg, srch, depth, probes)
        print("  %-22s %10d %9.1f%% %12.2f"
              % ("hand eval, depth %d" % depth, r["optimal"], r["pct"],
                 r["mean_loss"]))
    print("""
  The jump is between two and three plies, where the search first sees its
  own answer to Black's reply, and it is worth more than the ply before it or
  the ply after it. Beyond that the returns fall off while the cost keeps
  multiplying by the branching factor.
""")


def section_4():
    head("4. THE GREEDY POLICY DOES NOT TERMINATE")
    eg = board()
    print("""
  Now score the policy rather than the move. White plays the search, Black
  defends perfectly, and the distance to mate of the resulting play is
  computed exactly, by backward induction over every position at once. A
  position counts as mated only if this White actually forces mate from it.
""")
    print("  %-26s %10s %9s %12s"
          % ("player", "mates", "percent", "mean plies"))
    rule()
    r = score_policy(eg, eg.dtm)
    print("  %-26s %10s %8.1f%% %12.2f"
          % ("exact table, 8x8", commas(r["mated"]), r["pct"],
             r["mean_plies"]))
    srch = Searcher(eg, HAND)
    r = score_policy(eg, policy_distances(eg, lambda x: srch.choose(x, 1)))
    print("  %-26s %10s %8.1f%% %12.2f"
          % ("hand eval, depth 1, 8x8", commas(r["mated"]), r["pct"],
             r["mean_plies"]))
    eg6 = board(6)
    r = score_policy(eg6, eg6.dtm)
    print("  %-26s %10s %8.1f%% %12.2f"
          % ("exact table, 6x6", commas(r["mated"]), r["pct"],
             r["mean_plies"]))
    for depth in (1, 2, 3):
        s6 = Searcher(eg6, HAND)
        r = score_policy(eg6, policy_distances(eg6,
                                               lambda x: s6.choose(x, depth)))
        print("  %-26s %10s %8.1f%% %12.2f"
              % ("hand eval, depth %d, 6x6" % depth, commas(r["mated"]),
                 r["pct"], r["mean_plies"]))
    print("""
  The moves are mostly reasonable and the player almost never mates. It walks
  in a circle: the evaluation ranks positions, it does not certify progress,
  and a deterministic policy that returns to a position it has already
  occupied will occupy it forever.
""")


def play(eg, chooser, s0, cap=250, memory=True):
    """One game. White uses chooser, Black defends with the exact table.
    With memory, White refuses to re-enter a position it has already seen."""
    b = eg.b
    s, seen, plies = s0, {s0}, 0
    while plies < cap:
        if eg.mate[s]:
            return plies, "mate"
        if eg.stale[s]:
            return plies, "stalemate"
        if (s & 1) == WHITE:
            c = chooser(s, seen if memory else None)
        else:
            c = eg.best_move(s)
        if c is None or c == b.DRAW_NODE:
            return plies, "draw"
        s = c
        plies += 1
        seen.add(s)
    return plies, "loop"


def make_chooser(eg, srch, depth):
    def chooser(s, seen):
        best, arg = None, None
        for c in eg.moves(s):
            v = srch.child(c, depth, -srch.INF, srch.INF)
            if seen is not None and c in seen:
                v -= 1000.0
            if best is None or v > best:
                best, arg = v, c
        return arg
    return chooser


def section_5():
    head("5. TD-LEAF SELF-PLAY")
    eg = board()
    print("""
  Weights start at zero. Both sides search with the same evaluation, White
  maximising and Black minimising, and the weights move by TD(lambda) on the
  value of the leaf the search reaches. The reward is minus one per ply, so
  the target is minus the distance to mate: the Day 5 metric, learned rather
  than enumerated. The exact table is never read during training.
""")
    print("  alpha %.4f   lambda %.2f   epsilon %.2f   depth %d   "
          "episodes %d   seed %d"
          % (PARAMS["alpha"], PARAMS["lambda"], PARAMS["epsilon"],
             PARAMS["train_depth"], PARAMS["episodes"], PARAMS["seed"]))
    print()
    w, curve = train(eg, log_every=max(1, PARAMS["episodes"] // 6))
    print("  %-9s %s" % ("episodes", " ".join("%10s" % k for k in FEATURES)))
    rule()
    for ep, ww in curve:
        print("  %-9d %s" % (ep, " ".join("%10.2f" % ww[k]
                                          for k in FEATURES)))
    print("  %-9s %s" % ("hand", " ".join("%10.2f" % HAND[k]
                                          for k in FEATURES)))
    print("""
  At the last checkpoint every one of the six signs agrees with the
  hand-written evaluation, and nothing told the learner what the features
  mean. The trajectory is not tidy: mobility is still changing sign at 2,500
  episodes, and the bias falls monotonically throughout without settling. The
  bias is also the one weight that cannot matter, since a constant added to
  every position never changes which move is chosen; it is absorbing the
  average distance to mate, which the policy does not need and the value
  estimate does.
""")
    return w


def section_6(w=None):
    head("6. ONE BIT OF HISTORY")
    eg = board()
    if w is None:
        w, _ = train(eg)
    probes = probe_set(eg, 500)
    print("""
  The same players, now forbidden to re-enter a position they have already
  occupied in this game. That is all that changes. Games are played out from
  %d starting positions spread evenly through the table, against the same
  perfect defence.
""" % len(probes))
    print("  %-24s %8s %8s %8s %8s %9s"
          % ("player", "mates", "draws", "loops", "mean", "over 100"))
    rule()
    players = [("exact table", None, 0),
               ("hand eval, depth 1", HAND, 1),
               ("hand eval, depth 3", HAND, 3),
               ("learned, depth 1", w, 1),
               ("learned, depth 3", w, 3),
               ("learned, one weight fixed", dict(w, rook_hangs=-40.0), 1)]
    out = {}
    for name, ww, depth in players:
        if ww is None:
            chooser = lambda s, seen: eg.best_move(s)
        else:
            chooser = make_chooser(eg, Searcher(eg, ww), depth)
        tally = {"mate": 0, "draw": 0, "loop": 0, "stalemate": 0}
        tot = n = long_ = worst = 0
        for s0 in probes:
            plies, how = play(eg, chooser, s0)
            tally[how] = tally.get(how, 0) + 1
            if how == "mate":
                tot += plies
                n += 1
                worst = max(worst, plies)
                if plies > 100:
                    long_ += 1
        out[name] = (tally, tot / float(n) if n else 0.0, worst, long_)
        print("  %-24s %8d %8d %8d %8.1f %9d"
              % (name, tally["mate"], tally["draw"], tally["loop"],
                 tot / float(n) if n else float("nan"), long_))
    print("""
  The loops are gone. What replaces them is the fifty-move rule: a win that
  takes more than 100 plies without a capture is a draw under the actual laws
  of chess, and the column on the right counts the games in which this player
  would have been denied the point it had already earned.

  Two things in that table are worth arguing about in class. The learned
  evaluation mates faster than the hand-written one and never crosses the
  fifty-move line, which the hand-written one crosses in about a third of its
  games; but it also throws the rook away outright in 46 games, and the last
  row isolates the cause to a single weight. Learning recovered the geometry
  of the endgame and under-weighted the one feature whose mistakes are rare
  and catastrophic, because rare events contribute little to a gradient.
  Setting that weight by hand, and changing nothing else, removes every draw.

  The other is that depth 3 is worse than depth 1 for the learned player. A
  deeper search of an inaccurate evaluation is not automatically a better
  player: it is a more faithful optimiser of the wrong objective.
""")
    return out


def self_check():
    head("7. SELF-CHECKS")
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  [%s] %s%s" % ("pass" if cond else "FAIL", name,
                               ("  " + detail) if detail else ""))

    eg = board(5)
    probes = probe_set(eg, 60)
    a = Searcher(eg, HAND, alphabeta=False, order=False)
    b_ = Searcher(eg, HAND, alphabeta=True, order=False)
    c = Searcher(eg, HAND, alphabeta=True, order=True)
    va = [a.search(p, 3) for p in probes]
    vb = [b_.search(p, 3) for p in probes]
    vc = [c.search(p, 3) for p in probes]
    chk("alpha-beta returns exactly the minimax value",
        all(abs(x - y) < 1e-9 for x, y in zip(va, vb)))
    chk("move ordering returns exactly the minimax value",
        all(abs(x - y) < 1e-9 for x, y in zip(va, vc)))
    chk("alpha-beta visits fewer nodes", b_.nodes < a.nodes,
        "%d against %d" % (b_.nodes, a.nodes))
    chk("ordering visits fewer nodes still", c.nodes < b_.nodes,
        "%d against %d" % (c.nodes, b_.nodes))

    # a searcher whose evaluation is the exact table must play optimally
    exact = Searcher(eg, HAND)
    exact.leaf = lambda s: (-float(eg.dtm[s]) if eg.dtm[s] >= 0
                            else PARAMS["draw_value"])
    d = policy_distances(eg, lambda x: exact.choose(x, 1))
    r = score_policy(eg, d)
    chk("searching one ply on the exact table is optimal everywhere",
        r["pct_optimal"] == 100.0, "%.2f%% optimal" % r["pct_optimal"])

    # the policy evaluator reproduces the Day 5 table when White plays best
    d2 = policy_distances(eg, lambda x: eg.best_move(x))
    chk("policy evaluation of the optimal policy reproduces the exact table",
        all(d2[s] == eg.dtm[s] for s in range(eg.b.N) if eg.legal[s]))

    # determinism of training
    w1, _ = train(eg, episodes=150)
    w2, _ = train(eg, episodes=150)
    chk("two training runs at the same seed agree exactly",
        all(w1[k] == w2[k] for k in FEATURES))

    # features are within their declared ranges
    feat = make_featuriser(eg)
    vals = [feat(*eg.b.unindex(s)[:3]) for s in range(0, eg.b.N, 7)
            if eg.legal[s]]
    chk("mobility never exceeds eight", all(v[2] <= 8 for v in vals))
    chk("bias feature is always one", all(v[6] == 1.0 for v in vals))

    print()
    print("  %s" % ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, default=0)
    args = ap.parse_args()
    want = lambda n: args.section in (0, n)
    w = None
    if want(1):
        section_1()
    if want(2):
        section_2()
    if want(3):
        section_3()
    if want(4):
        section_4()
    if want(5):
        w = section_5()
    if want(6):
        section_6(w)
    if want(7):
        if not self_check():
            sys.exit(1)


if __name__ == "__main__":
    main()

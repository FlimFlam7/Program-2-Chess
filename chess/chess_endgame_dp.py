#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chess_endgame_dp.py

Exact dynamic programming on chess endgames: king and rook against king,
king and queen against king, on boards of side 4 through 8.

Reference implementation for the lecture "Chess as a Markov Decision Process",
Reinforcement Learning, Utah State University, Day 5.

Purpose
-------
Every number quoted in the lecture page is produced by this file. Run it and
the output tables should match the page exactly. If they do not, the page is
wrong.

Two solvers are implemented and they must agree:

  1. retrograde  - backward induction from the mate positions, each state
                   settled exactly once, O(edges) total work;
  2. sweep       - synchronous minimax value iteration, the Day 2 algorithm
                   applied unchanged, O(edges x sweeps) work.

Agreement between the two is the central self-check. The first is what a
tablebase generator does; the second is what the course has called value
iteration since Day 2. They compute the same fixed point.

No third-party dependencies. Python 3.8+. Fully deterministic; no random
numbers are used anywhere in this file.

Usage
-----
    python3 chess_endgame_dp.py                # print every table
    python3 chess_endgame_dp.py --section 3    # print one section only

Whole-file runtime is about two minutes; every table is recomputed from
scratch, and both solvers are run on every board.
"""

import argparse
import sys
from array import array

# ============================================================================
# Versioned parameter block. Cite this version when you cite a number.
# ============================================================================

PARAMS = {
    "version":      "1.0.0",
    "gamma":        1.0,        # undiscounted, as everywhere in this course
    "step_cost":    -1.0,       # one unit of cost per ply until mate
    "mate_value":    0.0,       # terminal: Black is mated, nothing left to pay
    "draw_value":   None,       # draws are not finite-valued under this metric
    "boards":       (4, 5, 6, 7, 8),
    "pieces":       ("R", "Q"),
    "sweep_boards": (4, 5, 6, 7, 8),   # every board is also solved by sweeping
}

WHITE, BLACK = 0, 1
FILES = "abcdefgh"
UNSOLVED = -1


# ============================================================================
# Board geometry. B is the side of the board; squares are numbered
# s = rank * B + file, with file 0 = "a" and rank 0 = rank 1.
# ============================================================================

class Board:
    def __init__(self, B):
        self.B = B
        self.NS = B * B                     # squares
        self.N = self.NS ** 3 * 2           # index space: (wk, wp, bk, stm)
        self.DRAW_NODE = self.N             # sentinel: king against king
        self.king = [[] for _ in range(self.NS)]
        self.adj = [bytearray(self.NS) for _ in range(self.NS)]
        for s in range(self.NS):
            f, r = s % B, s // B
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    nf, nr = f + df, r + dr
                    if 0 <= nf < B and 0 <= nr < B:
                        t = nr * B + nf
                        self.king[s].append(t)
                        self.adj[s][t] = 1
        self.rook_rays = self._rays(((1, 0), (-1, 0), (0, 1), (0, -1)))
        self.queen_rays = self._rays(((1, 0), (-1, 0), (0, 1), (0, -1),
                                      (1, 1), (1, -1), (-1, 1), (-1, -1)))
        self.bishop_rays = self._rays(((1, 1), (-1, 1), (-1, -1), (1, -1)))
        self.knight_rays = self._rays(((2, 1), (1, 2), (-2, 1), (1, -2),
                                       (2, -1), (-1, 2), (-2, -1), (-1, -2)),
                                       single = True)

    def _rays(self, dirs, single = False):
        out = []
        for s in range(self.NS):
            f, r = s % self.B, s // self.B
            lines = []
            for df, dr in dirs:
                line = []
                nf, nr = f + df, r + dr
                while 0 <= nf < self.B and 0 <= nr < self.B:
                    line.append(nr * self.B + nf)
                    if single:
                        break
                    nf += df
                    nr += dr
                lines.append(line)
            out.append(lines)
        return out

    def name(self, s):
        return FILES[s % self.B] + str(s // self.B + 1)

    def index(self, wk, wp, bk, stm):
        return ((wk * self.NS + wp) * self.NS + bk) * 2 + stm

    def unindex(self, i):
        stm = i & 1
        i >>= 1
        bk = i % self.NS
        i //= self.NS
        wp = i % self.NS
        wk = i // self.NS
        return wk, wp, bk, stm


# ============================================================================
# The Markov decision process: legality, moves, terminal states
# ============================================================================

class Endgame:
    """King and one white piece against a lone black king.

    White is the side with the extra piece and is the side that is trying to
    mate. Black moves only the king. The state is the quadruple

        s = (square of the white king, square of the white piece,
             square of the black king, side to move)

    which is everything the rules need in order to decide what happens next:
    the position is Markov. Castling, en passant, the fifty-move counter and
    the repetition history are all absent here, which is exactly why this
    reduced game is finite enough to solve.
    """

    def __init__(self, board, piece):
        self.b = board
        self.piece = piece

        # Problem 1
        if piece == "R":
            self.rays = board.rook_rays
        elif piece == "B":
            self.rays = board.bishop_rays
        elif piece == "N":
            self.rays = board.knight_rays
        elif piece == "Q":
            self.rays = board.queen_rays
        else:
            raise ValueError(piece)

        self.build()

    # -- attack test ---------------------------------------------------------
    def attacks(self, wp, wk, target):
        """Does the white piece on wp hit target? The white king is the only
        possible blocker; the black king is deliberately not a blocker, so a
        king that steps along a checking ray is still seen as in check."""
        for line in self.rays[wp]:
            for t in line:
                if t == target:
                    return True
                if t == wk:
                    break
        return False

    # -- successor lists -----------------------------------------------------
    def build(self):
        b = self.b
        N, NS, DRAW = b.N, b.NS, b.DRAW_NODE
        legal = bytearray(N)
        mate = bytearray(N)
        stale = bytearray(N)
        start = array("i", bytes(4 * (N + 1)))
        succ = array("i")
        pos = 0

        for wk in range(NS):
            kmov = b.king[wk]
            adjwk = b.adj[wk]
            for wp in range(NS):
                if wp == wk:
                    continue
                for bk in range(NS):
                    if bk == wk or bk == wp or adjwk[bk]:
                        continue
                    incheck = self.attacks(wp, wk, bk)

                    # ---- White to move --------------------------------
                    i0 = b.index(wk, wp, bk, WHITE)
                    if not incheck:
                        # a position with the side not to move in check is
                        # not reachable and not legal
                        legal[i0] = 1
                        for t in kmov:
                            if t == wp or t == bk or b.adj[t][bk]:
                                continue
                            succ.append(b.index(t, wp, bk, BLACK))
                            pos += 1
                        for line in self.rays[wp]:
                            for t in line:
                                if t == wk or t == bk:
                                    break
                                succ.append(b.index(wk, t, bk, BLACK))
                                pos += 1
                        start[i0 + 1] = pos

                    # ---- Black to move --------------------------------
                    i1 = b.index(wk, wp, bk, BLACK)
                    legal[i1] = 1
                    n = 0
                    for t in b.king[bk]:
                        if adjwk[t] or t == wk:
                            continue
                        if t == wp:
                            succ.append(DRAW)       # captures it: K v k
                            pos += 1
                            n += 1
                            continue
                        if self.attacks(wp, wk, t):
                            continue
                        succ.append(b.index(wk, wp, t, WHITE))
                        pos += 1
                        n += 1
                    start[i1 + 1] = pos
                    if n == 0:
                        if incheck:
                            mate[i1] = 1
                        else:
                            stale[i1] = 1

        # states never touched above have no successors; carry the running
        # offset forward through the gaps
        run = 0
        for s in range(N + 1):
            if start[s] < run:
                start[s] = run
            else:
                run = start[s]

        self.legal, self.mate, self.stale = legal, mate, stale
        self.start, self.succ = start, succ
        self.edges = len(succ)

    def moves(self, s):
        return self.succ[self.start[s]:self.start[s + 1]]

    # -- solver 1: retrograde analysis --------------------------------------
    def retrograde(self):
        """Backward induction. Each state is settled once, at the moment its
        distance to mate becomes known, and never revisited. Returns the
        array dtm of distances to mate in plies, UNSOLVED where White cannot
        force mate, and the number of levels the backward pass needed."""
        b = self.b
        N, DRAW = b.N, b.DRAW_NODE
        start, succ, legal = self.start, self.succ, self.legal

        # invert the successor relation
        indeg = array("i", bytes(4 * (N + 1)))
        for s in range(N):
            if legal[s]:
                for k in range(start[s], start[s + 1]):
                    t = succ[k]
                    if t != DRAW:
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
            if legal[s]:
                for k in range(start[s], start[s + 1]):
                    t = succ[k]
                    if t != DRAW:
                        pred[fill[t]] = s
                        fill[t] += 1

        dtm = array("i", bytes(4 * N))
        for s in range(N):
            dtm[s] = UNSOLVED
        # Black-to-move states need every move to lose; count them down
        cnt = array("i", bytes(4 * N))
        for s in range(N):
            if legal[s] and (s & 1) == BLACK:
                cnt[s] = start[s + 1] - start[s]

        frontier = [s for s in range(N) if self.mate[s]]
        for s in frontier:
            dtm[s] = 0
        ply, visits = 0, 0
        while frontier:
            nxt = []
            for t in frontier:
                for k in range(pstart[t], pstart[t + 1]):
                    visits += 1
                    s = pred[k]
                    if dtm[s] != UNSOLVED:
                        continue
                    if (s & 1) == WHITE:
                        dtm[s] = ply + 1        # one good move is enough
                        nxt.append(s)
                    else:
                        cnt[s] -= 1
                        if cnt[s] == 0:         # every move loses
                            dtm[s] = ply + 1
                            nxt.append(s)
            frontier = nxt
            ply += 1
        self.dtm = dtm
        self.levels = ply
        self.retro_visits = visits
        return dtm, ply

    # -- solver 2: synchronous minimax value iteration -----------------------
    def sweep_solve(self, cap=10 ** 9):
        """The Day 2 algorithm, unchanged except that the expectation over the
        opponent is replaced by a minimum, because the opponent chooses rather
        than rolls dice. Every state is re-examined on every sweep. Returns
        (dtm, sweeps, edge_visits)."""
        b = self.b
        N, DRAW = b.N, b.DRAW_NODE
        start, succ, legal = self.start, self.succ, self.legal
        INF = 10 ** 6
        v = array("i", bytes(4 * N))
        for s in range(N):
            v[s] = 0 if self.mate[s] else INF
        sweeps, visits = 0, 0
        while sweeps < cap:
            changed = 0
            nv = array("i", v)
            for s in range(N):
                if not legal[s] or self.mate[s]:
                    continue
                lo, hi = start[s], start[s + 1]
                if lo == hi:                       # stalemate: never mate
                    continue
                visits += hi - lo
                if (s & 1) == WHITE:
                    best = INF
                    for k in range(lo, hi):
                        t = succ[k]
                        x = INF if t == DRAW else v[t]
                        if x < best:
                            best = x
                else:
                    best = -1
                    for k in range(lo, hi):
                        t = succ[k]
                        x = INF if t == DRAW else v[t]
                        if x > best:
                            best = x
                cand = INF if best >= INF else best + 1
                if cand != nv[s]:
                    nv[s] = cand
                    changed += 1
            v = nv
            sweeps += 1
            if changed == 0:
                break
        dtm = array("i", bytes(4 * N))
        for s in range(N):
            dtm[s] = UNSOLVED if (v[s] >= INF or not legal[s]) else v[s]
        return dtm, sweeps, visits

    # -- reading the solved table -------------------------------------------
    def best_move(self, s):
        """The optimal successor of s under the solved table, or None."""
        dtm = self.dtm
        b = self.b
        best, arg = None, None
        for t in self.moves(s):
            x = None if t == b.DRAW_NODE else (dtm[t] if dtm[t] >= 0 else None)
            if (s & 1) == WHITE:
                if x is not None and (best is None or x < best):
                    best, arg = x, t
            else:
                if x is None:                    # a draw: Black takes it
                    return t
                if best is None or x > best:
                    best, arg = x, t
        return arg

    def pv(self, s, limit=200):
        """The principal variation from s, as a list of states."""
        line = [s]
        while self.dtm[s] > 0 and len(line) < limit:
            t = self.best_move(s)
            if t is None or t == self.b.DRAW_NODE:
                break
            line.append(t)
            s = t
        return line

    def notation(self, s, t):
        b = self.b
        wk, wp, bk, stm = b.unindex(s)
        wk2, wp2, bk2, _ = b.unindex(t)
        if stm == WHITE:
            if wk2 != wk:
                return "K%s-%s" % (b.name(wk), b.name(wk2))
            return "%s%s-%s" % (self.piece, b.name(wp), b.name(wp2))
        if bk2 == wp:
            return "Kx%s%s" % (self.piece, b.name(wp))
        return "K%s-%s" % (b.name(bk), b.name(bk2))


# ============================================================================
# Counting helpers
# ============================================================================

def counts(eg):
    """Outcome census of one solved endgame."""
    b, dtm, legal = eg.b, eg.dtm, eg.legal
    out = {"legal": 0, "legalW": 0, "legalB": 0, "winW": 0, "winB": 0,
           "drawW": 0, "drawB": 0, "mate": sum(eg.mate),
           "stalemate": sum(eg.stale), "edges": eg.edges,
           "index": b.N, "maxW": 0, "maxB": 0}
    for s in range(b.N):
        if not legal[s]:
            continue
        out["legal"] += 1
        w = (s & 1) == WHITE
        out["legalW" if w else "legalB"] += 1
        if dtm[s] >= 0:
            out["winW" if w else "winB"] += 1
            key = "maxW" if w else "maxB"
            if dtm[s] > out[key]:
                out[key] = dtm[s]
        else:
            out["drawW" if w else "drawB"] += 1
    return out


def histogram(eg, side):
    h = {}
    for s in range(eg.b.N):
        if eg.legal[s] and (s & 1) == side and eg.dtm[s] >= 0:
            h[eg.dtm[s]] = h.get(eg.dtm[s], 0) + 1
    return h


def orbit_count(B):
    """Legal (wk, wp, bk) triples up to the eight symmetries of the square."""
    NS = B * B
    seen = set()
    n = 0
    for wk in range(NS):
        for wp in range(NS):
            if wp == wk:
                continue
            for bk in range(NS):
                if bk in (wk, wp):
                    continue
                fw, rw = wk % B, wk // B
                fb, rb = bk % B, bk // B
                if max(abs(fw - fb), abs(rw - rb)) <= 1:
                    continue
                key = min(canon(B, wk, wp, bk, g) for g in range(8))
                if key not in seen:
                    seen.add(key)
                    n += 1
    return n


def canon(B, wk, wp, bk, g):
    def t(s):
        f, r = s % B, s // B
        if g & 1:
            f = B - 1 - f
        if g & 2:
            r = B - 1 - r
        if g & 4:
            f, r = r, f
        return r * B + f
    return (t(wk) * B * B + t(wp)) * B * B + t(bk)


# ============================================================================
# Printing
# ============================================================================

def head(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def rule(n=74):
    print("  " + "-" * (n - 2))


def commas(n):
    return "{:,}".format(n)


SOLVED = {}


def solved(B, piece):
    key = (B, piece)
    if key not in SOLVED:
        eg = Endgame(Board(B), piece)
        eg.retrograde()
        SOLVED[key] = eg
    return SOLVED[key]


# ---------------------------------------------------------------------------

def section_1():
    head("1. HOW BIG IS THE TABLE")
    print("""
  Naive indexing of a k-man pawnless endgame on 64 squares: one slot for each
  placement of each man, times two for the side to move. Most slots are not
  legal positions, but a table indexed this way must still reserve them.
""")
    print("  %-6s %22s %20s" % ("men", "index slots", "bytes at 1 per slot"))
    rule()
    for k in range(2, 9):
        slots = 2 * 64 ** k
        print("  %-6d %22s %20s" % (k, commas(slots), human(slots)))
    print("""
  For comparison the whole game, from the same arithmetic, is not in this
  table at all: the number of legal chess positions is about 4.8 x 10^44.
""")


def human(n):
    for unit, size in (("PB", 2 ** 50), ("TB", 2 ** 40), ("GB", 2 ** 30),
                       ("MB", 2 ** 20), ("kB", 2 ** 10)):
        if n >= size:
            return "%.1f %s" % (n / size, unit)
    return "%d B" % n


def section_2():
    head("2. THE LEGAL POSITIONS OF K+R v K ON THE FULL BOARD")
    eg = solved(8, "R")
    c = counts(eg)
    print("""
  A quadruple (wk, wr, bk, stm) is a legal position when the three squares are
  distinct, the kings are not adjacent, and the side not to move is not in
  check. The last clause is what makes the two sides of the move unequal.
""")
    print("  %-40s %12s" % ("quantity", "count"))
    rule()
    rows = (("index slots, 64^3 x 2", c["index"]),
            ("legal positions", c["legal"]),
            ("  White to move", c["legalW"]),
            ("  Black to move", c["legalB"]),
            ("checkmates (Black to move, no move)", c["mate"]),
            ("stalemates (Black to move, no move)", c["stalemate"]),
            ("edges in the transition graph", c["edges"]))
    for n, v in rows:
        print("  %-40s %12s" % (n, commas(v)))
    print("""
  The count of Black-to-move positions is exact arithmetic and can be checked
  by hand: 64 x 63 ordered king pairs, less 420 adjacent ones, times the 62
  squares left for the rook, is 223,944.
""")


def section_3():
    head("3. THE SOLVED TABLE: K+R v K")
    eg = solved(8, "R")
    c = counts(eg)
    print("""
  Backward induction from the 216 mate positions. Every state is settled once.
""")
    print("  %-34s %12s %12s" % ("", "White to move", "Black to move"))
    rule()
    print("  %-34s %12s %12s" % ("legal positions",
                                 commas(c["legalW"]), commas(c["legalB"])))
    print("  %-34s %12s %12s" % ("White wins",
                                 commas(c["winW"]), commas(c["winB"])))
    print("  %-34s %12s %12s" % ("drawn",
                                 commas(c["drawW"]), commas(c["drawB"])))
    print("  %-34s %12s %12s" % ("longest, plies",
                                 c["maxW"], c["maxB"]))
    print("  %-34s %12s %12s" % ("longest, moves",
                                 (c["maxW"] + 1) // 2, (c["maxB"] + 1) // 2))
    print("""
  Every legal position with White to move is a win. Nothing about the rook
  endgame is conditional: the drawn positions all have Black to move, and they
  are exactly the stalemates and the positions where the rook can be taken.
""")
    hW = histogram(eg, WHITE)
    print("  distance to mate, White to move (plies, then full moves)")
    print("  %-8s %-8s %12s %12s" % ("plies", "moves", "positions", "cumulative"))
    rule()
    run = 0
    for d in sorted(hW):
        run += hW[d]
        print("  %-8d %-8d %12s %12s"
              % (d, (d + 1) // 2, commas(hW[d]), commas(run)))
    print()
    print("  worst case: %s positions need %d moves"
          % (commas(hW[c["maxW"]]), (c["maxW"] + 1) // 2))


def section_4():
    head("4. ONE LINE OF PERFECT PLAY")
    eg = solved(8, "R")
    c = counts(eg)
    worst = [s for s in range(eg.b.N)
             if eg.legal[s] and (s & 1) == WHITE and eg.dtm[s] == c["maxW"]]
    s = worst[0]
    wk, wp, bk, _ = eg.b.unindex(s)
    print("""
  The first maximal position in index order, and the whole game from it. Both
  sides play optimally: White mates as fast as the position allows, Black
  postpones mate as long as the position allows.
""")
    print("  White: K%s R%s      Black: K%s      White to move, mate in %d"
          % (eg.b.name(wk), eg.b.name(wp), eg.b.name(bk), (c["maxW"] + 1) // 2))
    print()
    line = eg.pv(s)
    out, movenum = [], 1
    for i in range(len(line) - 1):
        n = eg.notation(line[i], line[i + 1])
        if i % 2 == 0:
            out.append("%d. %s" % (movenum, n))
        else:
            out[-1] += " %s" % n
            movenum += 1
    for i in range(0, len(out), 4):
        print("   " + "   ".join("%-18s" % x for x in out[i:i + 4]))
    print()
    print("  final position: %s, mate" % position_string(eg, line[-1]))
    print("  plies played: %d   (dtm at the root: %d)"
          % (len(line) - 1, c["maxW"]))


def position_string(eg, s):
    wk, wp, bk, stm = eg.b.unindex(s)
    return "K%s %s%s v K%s, %s to move" % (
        eg.b.name(wk), eg.piece, eg.b.name(wp), eg.b.name(bk),
        "White" if stm == WHITE else "Black")


def section_5():
    head("5. THE SAME TABLE FOR K+Q v K")
    r, q = solved(8, "R"), solved(8, "Q")
    cr, cq = counts(r), counts(q)
    print("""
  Identical state space, identical algorithm, one line changed in the move
  generator. The queen mates faster and stalemates far more often.
""")
    print("  %-34s %14s %14s" % ("", "K+R v K", "K+Q v K"))
    rule()
    for lab, kr, kq in (
            ("legal positions", cr["legal"], cq["legal"]),
            ("  White to move", cr["legalW"], cq["legalW"]),
            ("  Black to move", cr["legalB"], cq["legalB"]),
            ("edges", cr["edges"], cq["edges"]),
            ("checkmates", cr["mate"], cq["mate"]),
            ("stalemates", cr["stalemate"], cq["stalemate"]),
            ("wins, White to move", cr["winW"], cq["winW"]),
            ("wins, Black to move", cr["winB"], cq["winB"]),
            ("draws, Black to move", cr["drawB"], cq["drawB"])):
        print("  %-34s %14s %14s" % (lab, commas(kr), commas(kq)))
    print("  %-34s %14d %14d" % ("longest mate, moves",
                                 (cr["maxW"] + 1) // 2, (cq["maxW"] + 1) // 2))
    print("  %-34s %14d %14d" % ("backward levels", r.levels, q.levels))
    print("""
  The stalemate count is the whole lesson of the queen ending: %d drawn
  positions against %d for the rook, in the same number of legal positions
  with Black to move. A stronger piece buys speed and buys new ways to fail.
""" % (cq["stalemate"], cr["stalemate"]))


def section_6():
    head("6. RETROGRADE ANALYSIS AGAINST SYNCHRONOUS VALUE ITERATION")
    print("""
  Two algorithms, one fixed point. Sweeping re-examines every state on every
  sweep and needs one sweep per ply of the deepest mate, plus one to notice
  that nothing changed. Backward induction touches each edge once. Both
  columns of work below are measured, and the two tables are compared entry
  by entry: they agree everywhere.
""")
    print("  %-5s %8s %10s %8s %8s %14s %14s"
          % ("board", "legal", "edges", "sweeps", "mate in",
             "sweep visits", "retro visits"))
    rule()
    for B in PARAMS["sweep_boards"]:
        eg = solved(B, "R")
        c = counts(eg)
        dtm2, sweeps, visits = eg.sweep_solve()
        agree = all(eg.dtm[s] == dtm2[s] for s in range(eg.b.N) if eg.legal[s])
        print("  %-5s %8s %10s %8d %8d %14s %14s%s"
              % ("%dx%d" % (B, B), commas(c["legal"]), commas(c["edges"]),
                 sweeps, (c["maxW"] + 1) // 2,
                 commas(visits), commas(eg.retro_visits),
                 "" if agree else "   TABLES DISAGREE"))
    print("""
  The ratio of the last two columns on the 8x8 board is the reason tablebase
  generators are written backwards and not forwards.
""")


def section_7():
    head("7. BOARD SIZE, AND WHY THE FULL GAME IS NOT ON THIS TABLE")
    print("""
  The same endgame on smaller boards. Note the last column: the depth of the
  problem grows slowly while its width grows as the sixth power of the side.
""")
    print("  %-6s %10s %12s %10s %10s %8s"
          % ("board", "index", "legal", "wins W", "orbits", "mate in"))
    rule()
    for B in PARAMS["boards"]:
        eg = solved(B, "R")
        c = counts(eg)
        print("  %-6s %10s %12s %10s %10s %8d"
              % ("%dx%d" % (B, B), commas(c["index"]), commas(c["legal"]),
                 commas(c["winW"]), commas(orbit_count(B)),
                 (c["maxW"] + 1) // 2))
    print("""
  "orbits" counts positions up to the eight symmetries of the square, which is
  the first thing a real tablebase generator exploits. It saves a factor of
  about eight. Eight is not the factor that matters when the target is 10^44.
""")


# ============================================================================
# Self-checks
# ============================================================================

def self_check():
    head("8. SELF-CHECKS")
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  [%s] %s%s" % ("pass" if cond else "FAIL", name,
                               ("  " + detail) if detail else ""))

    r = solved(8, "R")
    q = solved(8, "Q")
    cr, cq = counts(r), counts(q)

    chk("K+R v K, Black to move, legal count is 223,944",
        cr["legalB"] == 223944, "got %s" % commas(cr["legalB"]))
    chk("same count from closed form 3612 x 62",
        (64 * 63 - 420) * 62 == 223944)
    chk("K+R v K, White to move, legal count is 175,168",
        cr["legalW"] == 175168, "got %s" % commas(cr["legalW"]))
    chk("every legal K+R v K position with White to move is won",
        cr["winW"] == cr["legalW"] and cr["drawW"] == 0)
    chk("every legal K+Q v K position with White to move is won",
        cq["winW"] == cq["legalW"] and cq["drawW"] == 0)
    chk("longest K+R v K mate is 16 moves",
        (cr["maxW"] + 1) // 2 == 16, "got %d" % ((cr["maxW"] + 1) // 2))
    chk("longest K+Q v K mate is 10 moves",
        (cq["maxW"] + 1) // 2 == 10, "got %d" % ((cq["maxW"] + 1) // 2))
    chk("White-to-move distances are all odd",
        all(r.dtm[s] % 2 == 1 for s in range(r.b.N)
            if r.legal[s] and (s & 1) == WHITE and r.dtm[s] >= 0))
    chk("Black-to-move distances are all even",
        all(r.dtm[s] % 2 == 0 for s in range(r.b.N)
            if r.legal[s] and (s & 1) == BLACK and r.dtm[s] >= 0))
    chk("stalemates are drawn, not won",
        all(r.dtm[s] == UNSOLVED for s in range(r.b.N) if r.stale[s]))
    chk("mates are won in zero plies",
        all(r.dtm[s] == 0 for s in range(r.b.N) if r.mate[s]))

    # the Bellman optimality equation, checked state by state on the 5x5 board
    small = solved(5, "R")
    bad = bellman_residual(small)
    chk("Bellman optimality holds at every solved 5x5 state",
        bad == 0, "violations: %d" % bad)

    # the two solvers agree
    for B in PARAMS["sweep_boards"]:
        eg = solved(B, "R")
        d2, sweeps, visits = eg.sweep_solve()
        same = all(eg.dtm[s] == d2[s] for s in range(eg.b.N) if eg.legal[s])
        chk("%dx%d: retrograde and value iteration agree" % (B, B), same)
        chk("%dx%d: sweeps needed = deepest mate in plies + 1" % (B, B),
            sweeps == max(eg.dtm) + 1,
            "sweeps %d, deepest %d" % (sweeps, max(eg.dtm)))
        chk("%dx%d: sweep work = edges x sweeps" % (B, B),
            visits == eg.edges * sweeps,
            "%d against %d" % (visits, eg.edges * sweeps))

    # the principal variation really ends in mate, and really is that long
    worst = next(s for s in range(r.b.N)
                 if r.legal[s] and (s & 1) == WHITE and r.dtm[s] == cr["maxW"])
    line = r.pv(worst)
    chk("the maximal line is 31 plies and ends in mate",
        len(line) - 1 == 31 and r.mate[line[-1]] == 1,
        "%d plies" % (len(line) - 1))

    # legality of every position along that line
    chk("every position in the line is legal",
        all(r.legal[s] for s in line))

    print()
    print("  %s" % ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return ok


def bellman_residual(eg):
    """Count states where the solved table fails its own backup."""
    bad = 0
    b, dtm = eg.b, eg.dtm
    for s in range(b.N):
        if not eg.legal[s] or eg.mate[s]:
            continue
        vals = []
        for t in eg.moves(s):
            vals.append(None if t == b.DRAW_NODE
                        else (dtm[t] if dtm[t] >= 0 else None))
        if not vals:
            if dtm[s] != UNSOLVED:
                bad += 1
            continue
        if (s & 1) == WHITE:
            fin = [v for v in vals if v is not None]
            want = min(fin) + 1 if fin else UNSOLVED
        else:
            want = UNSOLVED if any(v is None for v in vals) else max(vals) + 1
        if want != dtm[s]:
            bad += 1
    return bad


# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, default=0,
                    help="print one section only (1-8); 0 prints everything")
    args = ap.parse_args()
    want = lambda n: args.section in (0, n)

    if want(1):
        section_1()
    if want(2):
        section_2()
    if want(3):
        section_3()
    if want(4):
        section_4()
    if want(5):
        section_5()
    if want(6):
        section_6()
    if want(7):
        section_7()
    if want(8):
        if not self_check():
            sys.exit(1)

if __name__ == "__main__":
    main()

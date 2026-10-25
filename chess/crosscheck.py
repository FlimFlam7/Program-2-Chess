#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscheck.py

Independent verification of day05-chess-mdp-lecture.html and
chess_endgame_dp.py.

Why this file exists
--------------------
chess_endgame_dp.py is fast because it is specialised: positions are packed
into integers, attacks are computed with precomputed rays, and the black king
is deliberately excluded from the blocker set so that a king stepping along a
checking ray is still seen as in check. Every one of those tricks is a place
where a bug could hide, and a bug there would be invisible to the script's own
self-checks, since both of its solvers read the same move generator.

So this file does not import that move generator. It reimplements the rules of
the endgame in the dullest possible way:

  - a position is a tuple of coordinate pairs, not a packed integer;
  - moves are generated pseudo-legally and then filtered by making the move on
    a copy of the board and asking whether the mover is now in check, which is
    how a real engine does it and which never needs the blocker trick;
  - a position is legal when the three squares differ and the side not to move
    is not in check, with no separate rule about adjacent kings at all.

If the two move generators agree on every state and every edge, and the two
solvers agree on every value, the remaining ways to be wrong are small.

Three comparisons are made.

  A. rules      every legal flag and every successor set, both boards
  B. solver     an independent dictionary-based value iteration on 5x5
  C. page       every figure printed in the lecture HTML

Usage
-----
    python3 crosscheck.py
    python3 crosscheck.py --page /path/to/day05-chess-mdp-lecture.html
    python3 crosscheck.py --quick      # skip the 8x8 rule comparison
"""

import argparse
import os
import re
import sys

import chess_endgame_dp as dp

PAGE_DEFAULT = "day05-chess-mdp-lecture.html"

checks = 0
failures = []


def cmp(name, got, want):
    global checks
    checks += 1
    if got != want:
        failures.append("%s: got %r, want %r" % (name, got, want))
        return False
    return True


def report(title):
    print("  %-58s %s" % (title, "ok" if not failures else "FAILED"))


# ===========================================================================
# A generic, slow, obviously correct endgame engine
# ===========================================================================

class Generic:
    """Rules of K+piece v K, written without any of the fast solver's tricks."""

    def __init__(self, B, piece):
        self.B = B
        self.piece = piece
        self.dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        if piece == "Q":
            self.dirs += ((1, 1), (1, -1), (-1, 1), (-1, -1))

    def on(self, f, r):
        return 0 <= f < self.B and 0 <= r < self.B

    def attacked(self, target, pos, by_white):
        """Is square target attacked, given the occupancy in pos?

        pos is (wk, wp, bk, stm) with squares as (file, rank) pairs, or None
        for a captured white piece. Occupied squares block sliding pieces.
        """
        wk, wp, bk, _ = pos
        occ = set(x for x in (wk, wp, bk) if x is not None)
        if by_white:
            if wk is not None and self.adjacent(wk, target):
                return True
            if wp is None:
                return False
            for df, dr in self.dirs:
                f, r = wp[0] + df, wp[1] + dr
                while self.on(f, r):
                    if (f, r) == target:
                        return True
                    if (f, r) in occ:
                        break
                    f, r = f + df, r + dr
            return False
        return bk is not None and self.adjacent(bk, target)

    def adjacent(self, a, b):
        return a != b and max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1

    def in_check(self, pos, white):
        wk, wp, bk, _ = pos
        king = wk if white else bk
        return king is not None and self.attacked(king, pos, not white)

    def legal_position(self, pos):
        wk, wp, bk, stm = pos
        if len({wk, wp, bk}) != 3:
            return False
        # the side that just moved must not have left its king attacked
        return not self.in_check(pos, white=(stm == 1))

    def pseudo(self, pos):
        wk, wp, bk, stm = pos
        out = []
        if stm == 0:
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df or dr:
                        t = (wk[0] + df, wk[1] + dr)
                        if self.on(*t) and t != wp:
                            out.append((t, wp, bk, 1))
            for df, dr in self.dirs:
                f, r = wp[0] + df, wp[1] + dr
                while self.on(f, r):
                    t = (f, r)
                    if t == wk:
                        break
                    if t == bk:
                        break          # the king cannot be captured
                    out.append((wk, t, bk, 1))
                    f, r = f + df, r + dr
        else:
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df or dr:
                        t = (bk[0] + df, bk[1] + dr)
                        if not self.on(*t) or t == wk:
                            continue
                        if t == wp:
                            out.append((wk, None, t, 0))   # captures it
                        else:
                            out.append((wk, wp, t, 0))
        return out

    def moves(self, pos):
        """Legal moves: make each pseudo-legal move, reject self-check."""
        mover_white = pos[3] == 0
        return [m for m in self.pseudo(pos)
                if not self.in_check(m, white=mover_white)]


def to_pair(B, s):
    return (s % B, s // B)


def to_sq(B, p):
    return p[1] * B + p[0]


# ===========================================================================
# A. rules
# ===========================================================================

def check_rules(B, piece, label):
    g = Generic(B, piece)
    eg = dp.solved(B, piece)
    b = eg.b
    NS = b.NS
    nlegal = nedge = 0
    for wk in range(NS):
        for wp in range(NS):
            for bk in range(NS):
                for stm in (0, 1):
                    i = b.index(wk, wp, bk, stm)
                    pos = (to_pair(B, wk), to_pair(B, wp), to_pair(B, bk), stm)
                    want = g.legal_position(pos)
                    if not cmp("%s legal(%d)" % (label, i),
                               bool(eg.legal[i]), want):
                        return
                    if not want:
                        continue
                    nlegal += 1
                    mine = sorted(eg.moves(i))
                    theirs = []
                    for m in g.moves(pos):
                        if m[1] is None:
                            theirs.append(b.DRAW_NODE)
                        else:
                            theirs.append(b.index(to_sq(B, m[0]),
                                                  to_sq(B, m[1]),
                                                  to_sq(B, m[2]), m[3]))
                    theirs.sort()
                    if not cmp("%s moves(%d)" % (label, i), mine, theirs):
                        return
                    nedge += len(mine)
                    if stm == 1 and not theirs:
                        want_mate = g.in_check(pos, white=False)
                        cmp("%s mate(%d)" % (label, i),
                            bool(eg.mate[i]), want_mate)
                        cmp("%s stalemate(%d)" % (label, i),
                            bool(eg.stale[i]), not want_mate)
    print("     %s: %s legal positions, %s edges, all matched"
          % (label, "{:,}".format(nlegal), "{:,}".format(nedge)))


# ===========================================================================
# B. an independent solver
# ===========================================================================

def check_solver(B, piece):
    """Dictionary-based value iteration over tuple states, no arrays, no
    backward induction, no shared code with either solver in the main file."""
    g = Generic(B, piece)
    eg = dp.solved(B, piece)
    b = eg.b
    NS = b.NS
    INF = float("inf")
    states = []
    for wk in range(NS):
        for wp in range(NS):
            for bk in range(NS):
                for stm in (0, 1):
                    pos = (to_pair(B, wk), to_pair(B, wp), to_pair(B, bk), stm)
                    if g.legal_position(pos):
                        states.append(pos)
    succ = {p: g.moves(p) for p in states}
    d = {}
    for p in states:
        d[p] = 0.0 if (not succ[p] and g.in_check(p, white=False)) else INF
    while True:
        changed = 0
        nd = dict(d)
        for p in states:
            ms = succ[p]
            if not ms:
                continue
            vals = []
            for m in ms:
                vals.append(INF if m[1] is None else d.get(m, INF))
            best = min(vals) if p[3] == 0 else max(vals)
            x = INF if best == INF else best + 1
            if x != nd[p]:
                nd[p] = x
                changed += 1
        d = nd
        if not changed:
            break
    same = 0
    for p in states:
        i = b.index(to_sq(B, p[0]), to_sq(B, p[1]), to_sq(B, p[2]), p[3])
        want = -1 if d[p] == INF else int(d[p])
        cmp("solver dtm(%d)" % i, eg.dtm[i], want)
        same += 1
    print("     %dx%d %s: %s states agree with the packed solver"
          % (B, B, piece, "{:,}".format(same)))


# ===========================================================================
# C. the page
# ===========================================================================

def figures():
    """Recompute every number the lecture prints, keyed by where it appears."""
    f = {}
    r8 = dp.solved(8, "R")
    q8 = dp.solved(8, "Q")
    cr, cq = dp.counts(r8), dp.counts(q8)

    for k in range(2, 9):                                   # Table 5.1
        f["5.1 men %d" % k] = 2 * 64 ** k

    f["5.2 index"] = cr["index"]                            # Table 5.2
    f["5.2 legal"] = cr["legal"]
    f["5.2 legalW"] = cr["legalW"]
    f["5.2 legalB"] = cr["legalB"]
    f["5.2 mates"] = cr["mate"]
    f["5.2 stalemates"] = cr["stalemate"]
    f["5.2 edges"] = cr["edges"]
    f["5.2 difference"] = cr["legalB"] - cr["legalW"]
    f["5.2 kingpairs"] = 64 * 63 - 420
    f["5.2 product"] = (64 * 63 - 420) * 62

    f["7.1 winB"] = cr["winB"]                              # Table 7.1
    f["7.1 drawB"] = cr["drawB"]
    f["7.1 maxW"] = cr["maxW"]
    f["7.1 maxB"] = cr["maxB"]

    cap = capture_draws(r8)                                 # Table 7.2
    f["7.2 captures"] = cap
    f["7.2 total"] = cap + cr["stalemate"]

    h = dp.histogram(r8, dp.WHITE)                          # Table 7.3
    run = 0
    for d in sorted(h):
        run += h[d]
        f["7.3 count %d" % d] = h[d]
        f["7.3 cum %d" % d] = run

    f["9.1 legalQ"] = cq["legal"]                           # Table 9.1
    f["9.1 legalWQ"] = cq["legalW"]
    f["9.1 edgesQ"] = cq["edges"]
    f["9.1 matesQ"] = cq["mate"]
    f["9.1 staleQ"] = cq["stalemate"]
    f["9.1 winBQ"] = cq["winB"]
    f["9.1 drawBQ"] = cq["drawB"]
    f["9.1 capturesQ"] = capture_draws(q8)

    for B in PARAM_BOARDS:                                  # Tables 10.1, 11.1
        eg = dp.solved(B, "R")
        c = dp.counts(eg)
        _, sweeps, visits = eg.sweep_solve()
        f["10.1 %d legal" % B] = c["legal"]
        f["10.1 %d edges" % B] = c["edges"]
        f["10.1 %d sweeps" % B] = sweeps
        f["10.1 %d sweepvisits" % B] = visits
        f["10.1 %d retrovisits" % B] = eg.retro_visits
        f["11.1 %d index" % B] = c["index"]
        f["11.1 %d winW" % B] = c["winW"]
        f["11.1 %d orbits" % B] = dp.orbit_count(B)
        f["11.1 %d matein" % B] = (c["maxW"] + 1) // 2
    return f


PARAM_BOARDS = (4, 5, 6, 7, 8)


def capture_draws(eg):
    n = 0
    for s in range(eg.b.N):
        if (eg.legal[s] and (s & 1) == dp.BLACK
                and eg.b.DRAW_NODE in eg.moves(s)):
            n += 1
    return n


def check_page(path):
    text = open(path, encoding="utf-8").read()
    body = re.sub(r"<[^>]+>", " ", text)
    body = body.replace("&minus;", "-").replace("&times;", "x")
    present = set(re.findall(r"[\d,]+", body))
    f = figures()
    missing = []
    for key, val in sorted(f.items()):
        global checks
        checks += 1
        if "{:,}".format(val) not in present and str(val) not in present:
            missing.append("%s = %s" % (key, "{:,}".format(val)))
            failures.append("page is missing %s = %s"
                            % (key, "{:,}".format(val)))
    print("     %d figures recomputed, %d missing from the page"
          % (len(f), len(missing)))
    for m in missing[:12]:
        print("       missing: %s" % m)

    # the principal variation, regenerated and compared token by token
    r8 = dp.solved(8, "R")
    c = dp.counts(r8)
    s = next(i for i in range(r8.b.N)
             if r8.legal[i] and (i & 1) == dp.WHITE and r8.dtm[i] == c["maxW"])
    line = r8.pv(s)
    moves = [r8.notation(line[i], line[i + 1]) for i in range(len(line) - 1)]
    for mv in moves:
        cmp("pv contains %s" % mv, mv in body, True)
    cmp("pv length", len(moves), 31)
    print("     principal variation: %d plies, every move found in the page"
          % len(moves))


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=PAGE_DEFAULT)
    ap.add_argument("--quick", action="store_true",
                    help="skip the full-board rule comparison")
    args = ap.parse_args()

    print("crosscheck.py against chess_endgame_dp.py v%s"
          % dp.PARAMS["version"])

    print("\n  A. rules, against an independently written engine")
    for B in (4, 5):
        check_rules(B, "R", "%dx%d R" % (B, B))
        check_rules(B, "Q", "%dx%d Q" % (B, B))
    if not args.quick:
        check_rules(8, "R", "8x8 R")
        check_rules(8, "Q", "8x8 Q")
    else:
        print("     8x8 skipped (--quick)")
    report("rules")

    print("\n  B. values, against an independent value iteration")
    check_solver(5, "R")
    check_solver(4, "Q")
    report("values")

    print("\n  C. the lecture page")
    if os.path.exists(args.page):
        check_page(args.page)
        report("page")
    else:
        print("     page not found at %s, skipped" % args.page)

    print("\n  %s comparisons, %d failures" % ("{:,}".format(checks),
                                               len(failures)))
    for line in failures[:20]:
        print("    FAIL  %s" % line)
    if failures:
        sys.exit(1)
    print("  everything agrees")


if __name__ == "__main__":
    main()

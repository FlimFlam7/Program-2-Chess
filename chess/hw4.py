# Korben Cox
# Garrett Cox
# ECE 5510 Chess Program
# hw4.py

import argparse
import chess_endgame_dp as dp
import chess_search_td as td
import math

# ============================================================
# Homework 4 Parameters
# ============================================================

PARAMS = {
    "seed": 1000,
    "board": 8,
    "pieces": ("R", "Q", "B", "N"),
}

# Problem 6
PARAMS.update({
    "p6_seeds": [20260920, 20260921, 20260922,
                 20260923, 20260924],
    "p6_alphas": [0.001, 0.002, 0.004],
    "p6_lambdas": [0.3, 0.7, 0.9],
    "p6_episodes": 3000,
    "p6_probe_count": 2000,
    "p6_hang_start_fraction": 0.5,
})

# ============================================================
# Classes
# ============================================================

# Problem 5
class IterativeSearcher(td.Searcher):
    def __init__(self, eg, weights):
        super().__init__(
            eg, weights, alphabeta=True, order=True
        )
        self.root_best = None

    def search_root(self, state, depth):
        """Search root, trying the previous best move first."""
        self.nodes += 1

        terminal = self.terminal(state)
        if terminal is not None:
            return terminal

        if depth == 0:
            return self.leaf(state)

        white = (state & 1) == dp.WHITE
        moves = list(self.eg.moves(state))

        moves.sort(
            key=lambda c: -self.quick(c, white)
        )

        if self.root_best in moves:
            moves.remove(self.root_best)
            moves.insert(0, self.root_best)

        alpha, beta = -self.INF, self.INF
        best = -self.INF if white else self.INF
        best_move = None

        for move in moves:
            value = self.child(move, depth, alpha, beta)

            if (
                best_move is None
                or (white and value > best)
                or (not white and value < best)
            ):
                best = value
                best_move = move

            if white:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)

        self.root_best = best_move
        return best

    def iterative_search(self, state, depth):
        """Search depths 1, 2, ..., depth."""
        self.root_best = None
        self.nodes = 0

        value = None

        for d in range(1, depth + 1):
            value = self.search_root(state, d)

        return value

# Problem 5
class TranspositionSearcher(IterativeSearcher):
    def __init__(self, eg, weights):
        super().__init__(eg, weights)
        self.table = {}
        self.lookups = 0
        self.hits = 0

    def search(self, s, depth, alpha=-1e9, beta=1e9):
        self.nodes += 1
        self.lookups += 1

        entry = self.table.get(s)

        if entry is not None:
            stored_depth, stored_value = entry

            if stored_depth >= depth:
                self.hits += 1
                return stored_value

        terminal = self.terminal(s)
        if terminal is not None:
            self.table[s] = (depth, terminal)
            return terminal

        if depth == 0:
            value = self.leaf(s)
            self.table[s] = (depth, value)
            return value

        white = (s & 1) == dp.WHITE
        moves = list(self.eg.moves(s))

        if not moves:
            return self.draw

        moves.sort(key=lambda c: -self.quick(c, white))

        best = -self.INF if white else self.INF
        cutoff = False

        for move in moves:
            value = self.child(move, depth, alpha, beta)

            if white:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)

            if alpha >= beta:
                cutoff = True
                break

        # Only store exact results.
        # A cutoff result is generally just a bound.
        if not cutoff:
            self.table[s] = (depth, best)

        return best

    def hit_rate(self):
        if self.lookups == 0:
            return 0.0
        return 100.0 * self.hits / self.lookups

# Problem 5
class BuggySearcher(td.Searcher):
    def child(self, c, depth, alpha, beta):
        if c == self.eg.b.DRAW_NODE:
            self.nodes += 1
            return self.draw

        # Deliberate bug:
        # Alpha and beta are not adjusted for step cost.
        return self.search(
            c, depth - 1, alpha, beta
        ) - 1.0


# ============================================================
# Extra Helpers
# ============================================================

# Problem 2
def canonical_index(board, state):
    # Return the smallest state index among the eight symmetric versions of a position.
    wk, wp, bk, stm = board.unindex(state)

    return min(
        2 * dp.canon(board.B, wk, wp, bk, g) + stm
        for g in range(8)
    )

# Problem 4
def solve_linear_system(A, b):
    n = len(b)
    A = [row[:] for row in A]
    b = b[:]

    for i in range(n):
        pivot = max(
            range(i, n),
            key=lambda j: abs(A[j][i])
        )

        if abs(A[pivot][i]) < 1e-10:
            raise ValueError("Singular least-squares system")

        A[i], A[pivot] = A[pivot], A[i]
        b[i], b[pivot] = b[pivot], b[i]

        divisor = A[i][i]

        for j in range(i + 1, n):
            factor = A[j][i] / divisor

            for k in range(i, n):
                A[j][k] -= factor * A[i][k]

            b[j] -= factor * b[i]

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        x[i] = (
            b[i] - sum(
                A[i][j] * x[j]
                for j in range(i + 1, n)
            )
        ) / A[i][i]

    return x

# Problem 4
def fit_weights(eg):
    names = td.FEATURES
    n = len(names)

    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    feat = td.make_featuriser(eg)

    for s in range(eg.b.N):
        if not eg.legal[s] or (s & 1) != dp.WHITE:
            continue

        wk, wp, bk, _ = eg.b.unindex(s)

        x = feat(wk, wp, bk)
        y = -float(eg.dtm[s])

        for i in range(n):
            b[i] += x[i] * y

            for j in range(n):
                A[i][j] += x[i] * x[j]

    weights = solve_linear_system(A, b)

    return dict(zip(names, weights))

# Problem 4
def evaluation_metrics(eg, weights):
    feat = td.make_featuriser(eg)

    n = 0
    squared_error = 0.0

    sx = sy = 0.0
    sxx = syy = sxy = 0.0

    for s in range(eg.b.N):
        if not eg.legal[s] or (s & 1) != dp.WHITE:
            continue

        wk, wp, bk, _ = eg.b.unindex(s)

        x = td.dot(weights, feat(wk, wp, bk))
        y = -float(eg.dtm[s])

        n += 1
        squared_error += (x - y) ** 2

        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y

    rms = math.sqrt(squared_error / n)

    correlation = (
        (n * sxy - sx * sy)
        /
        math.sqrt(
            (n * sxx - sx * sx) *
            (n * syy - sy * sy)
        )
    )

    return rms, correlation

# Problem 5
def benchmark_searchers(eg):
    print("\nPART 1-2: SEARCH NODE COUNTS")
    print("-" * 78)

    probes = td.probe_set(eg, 400)

    print(
        f"{'Depth':<7}"
        f"{'Alpha-beta':>14}"
        f"{'AB + ordering':>16}"
        f"{'Iterative':>14}"
        f"{'Iterative + TT':>17}"
    )

    print("-" * 78)

    for depth in (2, 3, 4, 5):
        ab = td.Searcher(
            eg, td.HAND, alphabeta=True, order=False
        )

        ordered = td.Searcher(
            eg, td.HAND, alphabeta=True, order=True
        )

        iterative = IterativeSearcher(eg, td.HAND)
        transposition = TranspositionSearcher(eg, td.HAND)

        for state in probes:
            ab.search(state, depth)
            ordered.search(state, depth)
            iterative.iterative_search(state, depth)

            # Separate searches share one TT within this benchmark.
            transposition.root_best = None
            transposition.iterative_search(state, depth)

        print(
            f"{depth:<7}"
            f"{ab.nodes:>14,}"
            f"{ordered.nodes:>16,}"
            f"{iterative.nodes:>14,}"
            f"{transposition.nodes:>17,}"
        )

        print(
            f"  Transposition-table hit rate: "
            f"{transposition.hit_rate():.2f}%"
        )

# Problem 5
def verify_searchers(eg):
    print("\nPART 3: EXACT VALUE VERIFICATION")
    print("-" * 65)

    probes = td.probe_set(eg, 200)

    wrong_ab = 0
    wrong_ordered = 0
    wrong_iterative = 0
    wrong_tt = 0

    for state in probes:
        reference = td.Searcher(
            eg, td.HAND, alphabeta=False, order=False
        ).search(state, 4)

        ab = td.Searcher(
            eg, td.HAND, alphabeta=True, order=False
        ).search(state, 4)

        ordered = td.Searcher(
            eg, td.HAND, alphabeta=True, order=True
        ).search(state, 4)

        iterative = IterativeSearcher(
            eg, td.HAND
        ).iterative_search(state, 4)

        tt = TranspositionSearcher(
            eg, td.HAND
        ).iterative_search(state, 4)

        wrong_ab += (ab != reference)
        wrong_ordered += (ordered != reference)
        wrong_iterative += (iterative != reference)
        wrong_tt += (tt != reference)

    print(f"Positions tested: {len(probes)}")
    print(f"Alpha-beta mismatches: {wrong_ab}")
    print(f"Ordered mismatches: {wrong_ordered}")
    print(f"Iterative mismatches: {wrong_iterative}")
    print(f"Transposition mismatches: {wrong_tt}")

    if any((
        wrong_ab, wrong_ordered,
        wrong_iterative, wrong_tt
    )):
        print("FAILED: Search implementations disagree.")
    else:
        print("PASSED: All search values match minimax.")

# Problem 5
def test_deliberate_bug(eg):
    print("\nPART 4: DELIBERATE BUG")
    print("-" * 65)

    probes = td.probe_set(eg, 200)
    wrong = 0

    for state in probes:
        correct = td.Searcher(
            eg, td.HAND,
            alphabeta=False,
            order=False
        ).search(state, 4)

        buggy = BuggySearcher(
            eg, td.HAND,
            alphabeta=True,
            order=True
        ).search(state, 4)

        if buggy != correct:
            wrong += 1

    print(f"Positions tested: {len(probes)}")
    print(f"Incorrect results: {wrong}")
    print(f"Error rate: {100 * wrong / len(probes):.2f}%")

# Problem 6
def evaluate_learned_weights(eg, weights):
    # Evaluate move accuracy on the same 2,000 probes.
    probes = td.probe_set(
        eg, PARAMS["p6_probe_count"]
    )

    searcher = td.Searcher(eg, weights)

    accuracy = td.move_accuracy(
        eg, searcher, 1, probes
    )

    # Exact policy evaluation across all positions.
    policy_searcher = td.Searcher(eg, weights)

    distances = td.policy_distances(
        eg,
        lambda s: policy_searcher.choose(s, 1)
    )

    score = td.score_policy(eg, distances)

    return {
        "move_accuracy": accuracy["pct"],
        "optimal_policy": score["pct_optimal"],
        "mate_rate": score["pct"],
        "mean_plies": score["mean_plies"],
        "rook_hangs": weights["rook_hangs"],
    }

# Problem 6
def train_one(eg, seed, alpha, trace_decay):
    # Temporarily change training hyperparameters.
    old_alpha = td.PARAMS["alpha"]
    old_lambda = td.PARAMS["lambda"]

    try:
        td.PARAMS["alpha"] = alpha
        td.PARAMS["lambda"] = trace_decay

        weights, _ = td.train(
            eg,
            episodes=PARAMS["p6_episodes"],
            seed=seed
        )

        return weights

    finally:
        td.PARAMS["alpha"] = old_alpha
        td.PARAMS["lambda"] = old_lambda

# Problem 6
def summarize_runs(results):
    print(
        f"{'Metric':<20}"
        f"{'Minimum':>12}"
        f"{'Maximum':>12}"
        f"{'Spread':>12}"
    )
    print("-" * 56)

    for key in (
        "move_accuracy",
        "optimal_policy",
        "mate_rate",
        "mean_plies",
        "rook_hangs",
    ):
        values = [r[key] for r in results]

        minimum = min(values)
        maximum = max(values)

        print(
            f"{key:<20}"
            f"{minimum:>12.3f}"
            f"{maximum:>12.3f}"
            f"{maximum - minimum:>12.3f}"
        )


# ============================================================
# Problem 1: A Piece That Cannot Mate
# ============================================================

def p1():
    print("=" * 75)
    print("PROBLEM 1: A PIECE THAT CANNOT MATE")
    print("=" * 75)

    print(
        f"{'Piece':<10}"
        f"{'Legal':>12}"
        f"{'White':>12}"
        f"{'Black':>12}"
        f"{'Mates':>12}"
        f"{'Stalemates':>14}"
    )

    print("-" * 75)

    results = {}

    for piece in PARAMS["pieces"]:

        # Generate and solve the endgame
        eg = dp.solved(PARAMS["board"], piece)

        # Count positions and terminal states
        c = dp.counts(eg)

        results[piece] = c

        print(
            f"{piece:<10}"
            f"{c['legal']:>12,}"
            f"{c['legalW']:>12,}"
            f"{c['legalB']:>12,}"
            f"{c['mate']:>12,}"
            f"{c['stalemate']:>14,}"
        )

    print("\nVERIFICATION")
    print("-" * 75)

    rook = results["R"]

    assert rook["legal"] == 399112
    assert rook["legalW"] == 175168
    assert rook["mate"] == 216
    assert rook["stalemate"] == 68

    print("Rook baseline counts: PASSED")

    knight = results["N"]
    bishop = results["B"]

    print(f"\nKnight checkmates: {knight['mate']:,}")

    if knight["stalemate"] > 0:
        ratio = bishop["stalemate"] / knight["stalemate"]
        print(f"Bishop/knight stalemate ratio: {ratio:.2f}")
    else:
        print("Bishop/knight ratio: undefined")


# ============================================================
# Problem 2: Symmetry
# ============================================================

def p2():
    print("=" * 75)
    print("PROBLEM 2: SYMMETRY")
    print("=" * 75)

    eg = dp.solved(8, "R")
    board = eg.b

    legality_checks = 0
    distance_checks = 0
    orbit_representatives = set()

    for state in range(board.N):
        wk, wp, bk, stm = board.unindex(state)

        # Find the canonical representative
        representative = canonical_index(board, state)

        # Record unique legal state orbits
        if eg.legal[state]:
            orbit_representatives.add(representative)

        # Check all eight symmetries
        for g in range(8):
            transformed = (
                2 * dp.canon(board.B, wk, wp, bk, g) + stm
            )

            # Legality must be identical
            assert eg.legal[state] == eg.legal[transformed]
            legality_checks += 1

            # Distance to mate must also be identical
            if eg.legal[state]:
                assert eg.dtm[state] == eg.dtm[transformed]
                distance_checks += 1

    print(f"Index slots: {board.N:,}")
    print(f"Legality comparisons: {legality_checks:,}")
    print(f"DTM comparisons: {distance_checks:,}")
    print(f"Unique legal state orbits: {len(orbit_representatives):,}")

    print("\nAll symmetry invariance tests PASSED")


# ============================================================
# Problem 3: The Fifty-Move Rule
# ============================================================

def p3():
    print("=" * 75)
    print("PROBLEM 3: THE FIFTY-MOVE RULE")
    print("=" * 75)

    eg = dp.solved(8, "R")

    # Collect all legal White-to-move positions.
    white_states = [
        s for s in range(eg.b.N)
        if eg.legal[s] and (s & 1) == dp.WHITE
    ]

    print("\nPART 1: MATE WITHIN K MOVES")
    print("-" * 65)
    print(f"Total White-to-move positions: {len(white_states):,}")

    print(f"\n{'k':<8}{'Maximum plies':<18}{'Positions won':>18}")
    print("-" * 44)

    for k in range(1, 17):
        max_plies = 2 * k - 1

        won = sum(
            1 for s in white_states
            if 0 <= eg.dtm[s] <= max_plies
        )

        print(f"{k:<8}{max_plies:<18}{won:>18,}")

    print("\nPART 2: SMALLEST K")
    print("-" * 65)

    max_dtm = max(eg.dtm[s] for s in white_states)
    smallest_k = (max_dtm + 2) // 2

    print(f"Maximum White-to-move DTM: {max_dtm} plies")
    print(f"Smallest k: {smallest_k} moves")

    print("\nPART 3: DISTANCE TO ZEROING")
    print("-" * 65)
    print("See written explanation in report.")

    print("\nPART 4: DIFFERENT PLY LIMITS")
    print("-" * 65)

    probes = td.probe_set(eg, 500)

    searcher = td.Searcher(eg, td.HAND)
    chooser = td.make_chooser(eg, searcher, 1)

    # Use the same 500 games for all three limits.
    games = [
        td.play(eg, chooser, s, cap=250)
        for s in probes
    ]

    print(f"{'Ply limit':<15}{'Games exceeding limit':>25}")
    print("-" * 40)

    for limit in (60, 100, 200):
        exceeded = sum(
            1 for plies, result in games
            if result == "mate" and plies > limit
        )

        print(f"{limit:<15}{exceeded:>25}")

    print("\nGame outcome totals:")
    for outcome in ("mate", "draw", "stalemate", "loop"):
        count = sum(1 for _, result in games if result == outcome)
        print(f"  {outcome:<12}{count:>5}")


# ============================================================
# Problem 4: A Better Evaluation
# ============================================================

def p4():
    print("=" * 75)
    print("PROBLEM 4: A BETTER EVALUATION")
    print("=" * 75)

    eg = dp.solved(8, "R")

    # --------------------------------------------------------
    # Part 1: Feature descriptions
    # --------------------------------------------------------

    print("\nPART 1: NEW FEATURES")
    print("-" * 65)
    print("king_opposition: Kings separated by two squares")
    print("                 along the same rank or file.")
    print("rook_protected: White king protects the rook.")

    # --------------------------------------------------------
    # Part 2: Fit weights
    # --------------------------------------------------------

    print("\nPART 2: LEAST-SQUARES WEIGHTS")
    print("-" * 65)

    fitted = fit_weights(eg)

    print(f"{'Feature':<20}{'Hand':>12}{'Fitted':>12}")
    print("-" * 44)

    for name in td.FEATURES:
        print(
            f"{name:<20}"
            f"{td.HAND[name]:>12.4f}"
            f"{fitted[name]:>12.4f}"
        )

    hand_rms, hand_corr = evaluation_metrics(
        eg, td.HAND
    )

    fit_rms, fit_corr = evaluation_metrics(
        eg, fitted
    )

    print("\nEVALUATION COMPARISON")
    print("-" * 65)

    print(f"{'Evaluation':<20}{'RMS':>12}{'Correlation':>15}")
    print("-" * 47)

    print(
        f"{'Hand':<20}"
        f"{hand_rms:>12.4f}"
        f"{hand_corr:>15.4f}"
    )

    print(
        f"{'Fitted':<20}"
        f"{fit_rms:>12.4f}"
        f"{fit_corr:>15.4f}"
    )

    # --------------------------------------------------------
    # Part 3: Move accuracy
    # --------------------------------------------------------

    print("\nPART 3: OPTIMAL MOVE ACCURACY")
    print("-" * 65)

    probes = td.probe_set(eg, 2000)

    print(
        f"{'Depth':<10}"
        f"{'Hand accuracy':>18}"
        f"{'Fitted accuracy':>20}"
    )
    print("-" * 48)

    for depth in (1, 2, 3):

        hand_search = td.Searcher(eg, td.HAND)
        fitted_search = td.Searcher(eg, fitted)

        hand_result = td.move_accuracy(
            eg, hand_search, depth, probes
        )

        fitted_result = td.move_accuracy(
            eg, fitted_search, depth, probes
        )

        print(
            f"{depth:<10}"
            f"{hand_result['pct']:>17.1f}%"
            f"{fitted_result['pct']:>19.1f}%"
        )

    # --------------------------------------------------------
    # Part 4: Hand-built examples
    # --------------------------------------------------------

    print("\nPART 4: VALUE ERROR VS MOVE QUALITY")
    print("-" * 65)
    print("Requires two hand-constructed chess positions.")
    print("See written analysis in report.")


# ============================================================
# Problem 5: Making the Search Cheaper
# ============================================================
def p5():
    print("=" * 75)
    print("PROBLEM 5: MAKING THE SEARCH CHEAPER")
    print("=" * 75)

    eg = dp.solved(8, "R")

    benchmark_searchers(eg)
    verify_searchers(eg)
    test_deliberate_bug(eg)


# ============================================================
# Problem 6: Self-Play, Repeated
# ============================================================
def p6():
    print("=" * 75)
    print("PROBLEM 6: SELF-PLAY, REPEATED")
    print("=" * 75)

    eg = dp.solved(8, "R")

    seeds = PARAMS["p6_seeds"]
    alphas = PARAMS["p6_alphas"]
    lambdas = PARAMS["p6_lambdas"]

    # Part 1: Baseline across five seeds
    print("\nPART 1: FIVE-SEED BASELINE")
    print("-" * 76)

    baseline = []

    for seed in seeds:
        weights = train_one(
            eg, seed, 0.002, 0.7
        )

        result = evaluate_learned_weights(
            eg, weights
        )
        result["seed"] = seed
        baseline.append(result)

        print(
            f"Seed {seed}: "
            f"accuracy={result['move_accuracy']:.2f}%  "
            f"mean={result['mean_plies']:.2f}  "
            f"rook_hangs={result['rook_hangs']:.3f}"
        )

    print("\nBASELINE SPREAD")
    summarize_runs(baseline)

    # Part 2: Hyperparameter grid
    print("\nPART 2: ALPHA/LAMBDA EXPERIMENT")
    print("-" * 76)

    print(
        f"{'Alpha':<10}"
        f"{'Lambda':<10}"
        f"{'Accuracy range':<22}"
        f"{'Mean plies range':<22}"
    )

    grid_results = {}

    for alpha in alphas:
        for trace_decay in lambdas:
            # Reuse the baseline results rather than
            # training that combination again.
            if alpha == 0.002 and trace_decay == 0.7:
                results = baseline
            else:
                results = []

                for seed in seeds:
                    weights = train_one(
                        eg, seed, alpha, trace_decay
                    )

                    result = evaluate_learned_weights(
                        eg, weights
                    )
                    result["seed"] = seed
                    results.append(result)

            grid_results[(alpha, trace_decay)] = results

            accuracies = [
                r["move_accuracy"] for r in results
            ]
            plies = [
                r["mean_plies"] for r in results
            ]

            print(
                f"{alpha:<10.4f}"
                f"{trace_decay:<10.2f}"
                f"{min(accuracies):.1f}–{max(accuracies):.1f}%"
                f"{'':<8}"
                f"{min(plies):.1f}–{max(plies):.1f}"
            )

    # Part 3: Hanging-rook weight
    print("\nPART 3: HANGING-ROOK WEIGHT")
    print("-" * 76)

    rook_weights = [
        r["rook_hangs"] for r in baseline
    ]

    print(f"Minimum: {min(rook_weights):.3f}")
    print(f"Maximum: {max(rook_weights):.3f}")
    print(
        f"Average: "
        f"{sum(rook_weights) / len(rook_weights):.3f}"
    )
    print(
        f"Positive weights: "
        f"{sum(w > 0 for w in rook_weights)}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--problem",
        type=int,
        choices=range(1, 8),
        help="Run a specific problem"
    )

    args = parser.parse_args()

    if args.problem is None or args.problem == 1:
        p1()

    if args.problem is None or args.problem == 2:
        p2()

    if args.problem is None or args.problem == 3:
        p3()

    if args.problem is None or args.problem == 4:
        p4()

    if args.problem is None or args.problem == 5:
        p5()

    if args.problem is None or args.problem == 6:
        p6()

if __name__ == "__main__":
    main()

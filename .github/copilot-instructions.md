# Copilot Instructions

## Repository Overview

Java DSA problem solutions repository covering LeetCode, Codeforces, and advanced algorithm implementations. No build system, test framework, or external dependencies — plain Java files organized by package.

## Package Structure

| Package | Purpose |
|---|---|
| `com.company` | LeetCode-style DSA solutions (arrays, trees, DP, strings, etc.) |
| `AdvancedRevision` | Advanced algorithm implementations (Segment Tree, Fenwick Tree, Bitmask DP, Digit DP, Rolling Hash, etc.) |
| `CFProblems` | Codeforces problem solutions |
| `CF_Templates` | Master competitive programming template (`B.java`) with reusable utilities |
| `Graph` | Graph algorithms (BFS, DFS, DSU, Bellman-Ford, Dijkstra, TSP, etc.) |
| `SQL` | SQL query practice (window functions, aggregations) |

## Compiling & Running

There is no build tool. Compile and run directly from the `src/` directory:

```bash
# Compile a single file (from repo root)
javac -cp src src/com/company/Solution.java

# Compile a CF problem
javac -cp src src/CFProblems/CF.java

# Run (standard input from terminal)
java -cp src CFProblems.CF
```

## Competitive Programming Template

`CF_Templates/B.java` is the master template for new Codeforces solutions. It includes:
- **Fast I/O**: `MyScanner` (BufferedReader + StringTokenizer) + `PrintWriter out`
- **Constants**: `mod = 1e9+7`, direction arrays `dx[]/dy[]` (4-dir), `dx8[]/dy8[]` (8-dir), `dx9[]/dy9[]` (9-dir), `eps = 1e-10`
- **Utilities**: Graph, DSU, modular exponentiation (`powerMODe`), Prime Sieve, NCR, Binomial Coefficient, `Pair`, `Triplet`, `lcm`, `gcd`, `nextPermutation`, `sort` overloads for `int[]` and `long[]`

Always use this template as the base for new CF solutions; add problem logic inside `solve()`.

## Key Conventions

- **LeetCode solutions**: Standalone methods in classes under `com.company`, no `main()` required. Classes frequently contain multiple related problems grouped together (e.g., `Solution.java`, `Tree.java`, `SubArrays.java`).
- **CF solutions**: Each file has `main()` → reads `T` test cases → calls `solve()`. Always use `MyScanner` for input and `PrintWriter out` for output; never `Scanner` or `System.out.println` in hot paths.
- **DP memoization**: 2D/3D `long[][]` or `int[][]` arrays initialized to `-1` (not `null`), iterated with `Arrays.fill` inside nested loops.
- **Bitmask DP**: State typically `dp[index][mask][tightness]`; `mask` encodes which elements/digits have been used.
- **Custom Pair**: Use the top-level `Pair<K,V>` class in `src/Pair.java` rather than `javafx.util.Pair` or `Map.Entry`.
- **Modular arithmetic**: Use `mod = (int)(1e9 + 7)`. Intermediate multiplications cast to `long` before taking mod.
- **Comments**: Inline comments explain the algorithm's invariant or step; problem statement/examples are pasted at the top of the method as a block comment.
- **Author tag**: `//Author: Anand` appears on non-trivial solutions.

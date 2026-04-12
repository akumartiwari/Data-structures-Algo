package AdvancedRevision;

import java.util.*;

public class DPBFS {

    /*
     * ============================================================
     * Problem: Ways to Arrive at Destination (LC 1976) — Medium
     * ============================================================
     * DESCRIPTION:
     *   You are in a city with n intersections (0 to n-1) and bidirectional
     *   roads with travel times. Find the number of ways to travel from
     *   intersection 0 to n-1 using ONLY the shortest time paths.
     *   Return the count modulo 1_000_000_007.
     *
     * EXAMPLE:
     *   n=7, roads=[[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,2],[0,5,6]]
     *   Shortest time = 7
     *   Paths with time 7: 0→6, 0→1→2→5→6  → Output: 4 (total shortest paths)
     *
     * ALGORITHM (Modified Dijkstra):
     *   1. Build adjacency list from roads.
     *   2. Use min-heap (priority queue) on (cost, node); start at node 0 with cost 0.
     *   3. Maintain:
     *      - costs[i]  → minimum cost to reach node i (init: INF)
     *      - ways[i]   → number of shortest paths to node i (init: 0, ways[0]=1)
     *   4. For each popped node:
     *      - If new cost < current known cost  → update cost, reset ways[i] = ways[current]
     *      - If new cost == current known cost → add ways[current] to ways[i]
     *   5. Return ways[n-1] % MOD.
     *
     * TC: O(E log V)  — E = number of roads, V = number of intersections
     * SC: O(V + E)    — graph + distance/ways arrays
     * ============================================================
     */
    // TODO: Solve it again
    class WaysToArriveAtDestnation {
        public int countPaths(int n, int[][] roads) {
            final List<List<Node>> graph = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList());
            }
            for (final int[] arr : roads) {
                graph.get(arr[0]).add(new Node(arr[1], arr[2]));
                graph.get(arr[1]).add(new Node(arr[0], arr[2]));
            }
            return this.dfs(graph, n);
        }

        public int dfs(final List<List<Node>> adj, int n) {
            final int mod = 1_000_000_007;
            final Queue<Node> queue = new PriorityQueue<>(n);
            final long[] costs = new long[n];
            final long[] ways = new long[n];
            final boolean[] cache = new boolean[n];
            queue.add(new Node(0, 0));
            Arrays.fill(costs, Long.MAX_VALUE);
            costs[0] = 0;
            //one way to visit first node
            ways[0] = 1;
            while (!queue.isEmpty()) {
                final Node currentNode = queue.poll();
                if (currentNode.cost > costs[currentNode.position] || cache[currentNode.position]) {
                    continue;
                }
                for (final Node vertex : adj.get(currentNode.position)) {
                    if (costs[currentNode.position] + vertex.cost < costs[vertex.position]) {
                        costs[vertex.position] = costs[currentNode.position] + vertex.cost;
                        ways[vertex.position] = ways[currentNode.position] % mod;
                        queue.add(new Node(vertex.position, costs[vertex.position]));
                    } else if (costs[currentNode.position] + vertex.cost == costs[vertex.position]) {
                        ways[vertex.position] = (ways[vertex.position] + ways[currentNode.position]) % mod;
                    }
                }
            }
            return (int) ways[n - 1];
        }

        @SuppressWarnings("ALL")
        private class Node implements Comparable<Node> {
            int position;
            long cost;

            public Node(int dis, long val) {
                this.position = dis;
                this.cost = val;
            }


            @Override
            public int compareTo(final Node o) {
                return Long.compare(this.cost, o.cost);
            }
        }
    }

    /*
     * ============================================================
     * Problem: Maximum Strictly Increasing Cells in a Matrix (LC 2713) — Hard
     * ============================================================
     * DESCRIPTION:
     *   Given an m×n integer matrix, start at any cell. At each step you can
     *   move to any cell in the SAME row OR SAME column, but only if the
     *   destination cell's value is STRICTLY GREATER than the current cell.
     *   Return the maximum number of cells you can visit (including the start).
     *
     * EXAMPLE:
     *   mat = [[3,1],[3,4]]
     *   Start at (0,0)=3 → (1,1)=4  → 2 cells visited
     *   Start at (0,1)=1 → (0,0)=3 → (1,1)=4  → 3 cells visited  → Output: 3
     *
     * ALGORITHM (DP + Value-sorted grouping):
     *   1. Group all cells by their value into a sorted map (TreeMap).
     *   2. Process groups from smallest value to largest:
     *      - For each cell (i,j) in the current group:
     *          dp[i][j] = max(bestInRow[i], bestInCol[j]) + 1
     *          (best reachable sequence length ending at this value in that row/col)
     *      - After computing all dp values in the group (to avoid same-value
     *        interference), update:
     *          bestInRow[i] = max(bestInRow[i], dp[i][j])
     *          bestInCol[j] = max(bestInCol[j], dp[i][j])
     *   3. Answer is the maximum over all bestInRow and bestInCol entries.
     *
     * KEY INSIGHT: Two-pass per group (read then write) ensures cells with
     *   equal values do NOT update each other — only strictly smaller cells
     *   contribute to the current group's dp values.
     *
     * TC: O(m*n * log(m*n))  — sorting/grouping all cells
     * SC: O(m*n)             — dp array + row/col best arrays
     * ============================================================
     */
    public int maxIncreasingCells(int[][] mat) {
        int m = mat.length, n = mat[0].length;

        Map<Integer, List<int[]>> A = new TreeMap<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int val = mat[i][j];
                A.computeIfAbsent(val, k -> new ArrayList<int[]>()).add(new int[]{i, j});
            }
        }

        int[][] dp = new int[m][n];
        int[] res = new int[m + n];

        for (int a : A.keySet()) {
            // Pass 1: compute dp for all cells of this value (read-only from res)
            for (int[] pos : A.get(a)) {
                int i = pos[0], j = pos[1];
                dp[i][j] = Math.max(res[i], res[j + m]) + 1;
            }

            // Pass 2: update row/col bests (write — separated to avoid same-value influence)
            for (int[] pos : A.get(a)) {
                int i = pos[0], j = pos[1];
                res[m + j] = Math.max(res[m + j], dp[i][j]); // max nr of jumps on same column
                res[i] = Math.max(res[i], dp[i][j]); // max nr of jumps on same row
            }
        }

        int ans = 0;
        for (int a : res) {
            ans = Math.max(ans, a);
        }

        return ans;
    }
}

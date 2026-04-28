package com.company;

import common.Pair;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;

public class DPBFS {
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

    // TODO: Solve it again
    class WaysToArriveAtDestnation {
        /*
         * PROBLEM: Number of Ways to Arrive at Destination (LeetCode 1976)
         * Count the number of ways to travel from node 0 to node n-1 in minimum time on a weighted undirected graph.
         *
         * ALGORITHM: Modified Dijkstra – builds adjacency list then delegates to dfs() for shortest-path counting
         * TC: O((V + E) log V) | SC: O(V + E)
         */
        public int countPaths(int n, int[][] roads) {
            final List<List<Node>> graph = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList<>());
            }
            for (final int[] arr : roads) {
                graph.get(arr[0]).add(new Node(arr[1], arr[2]));
                graph.get(arr[1]).add(new Node(arr[0], arr[2]));
            }
            return this.dfs(graph, n);
        }

        /*
         * PROBLEM: Number of Ways to Arrive at Destination – Dijkstra core (Helper)
         * Runs modified Dijkstra on the pre-built adjacency list, simultaneously tracking minimum cost
         * and the number of distinct minimum-cost paths to each node.
         *
         * ALGORITHM: Priority-queue Dijkstra with parallel ways[] array (modular arithmetic)
         * TC: O((V + E) log V) | SC: O(V)
         */
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

    }

    /*
     * PROBLEM: Maximum Strictly Increasing Cells in a Matrix (LeetCode 2713)
     * Find the maximum number of cells that can be visited in a matrix by always moving to a strictly larger value
     * in the same row or column.
     *
     * ALGORITHM: DP + sorted grouping by cell value (TreeMap) – process cells in non-decreasing value order,
     *            updating per-row and per-column best reachable counts to avoid re-visiting equal values.
     * TC: O(M * N * log(M * N)) | SC: O(M * N)
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
            for (int[] pos : A.get(a)) {
                int i = pos[0], j = pos[1];
                dp[i][j] = Math.max(res[i], res[j + m]) + 1;
            }

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

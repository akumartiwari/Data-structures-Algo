package com.company;

import java.util.HashMap;import java.util.*;

public class DPBFS {
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

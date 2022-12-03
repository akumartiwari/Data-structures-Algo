package com.company;

import java.util.*;

public class Dijisktra {
    // Author : Anand
    List<int[]>[] nextgraph, pregraph;

    public long minimumWeight(int n, int[][] edges, int src1, int src2, int dest) {
        buildGraph(n, edges);

        // To fetch the shortest path from all possible nodes
        long[] src1To = new long[n], src2To = new long[n], destTo = new long[n];
        Arrays.fill(src1To, -1);
        Arrays.fill(src2To, -1);
        Arrays.fill(destTo, -1);

        shortestPath(src1, src1To, nextgraph);
        shortestPath(src2, src2To, nextgraph);
        shortestPath(dest, destTo, pregraph);

        long res = -1;
        for (int i = 0; i < n; i++) {
            long d1 = src1To[i];
            long d2 = src2To[i];
            long d3 = destTo[i];

            if (d1 >= 0 && d2 >= 0 && d3 >= 0) {
                long d = d1 + d2 + d3;
                if (res == -1 || d < res) {
                    res = d;
                }
            }
        }
        return res;
    }

    // Dijkstra algorithm to find the shortest distance b/w each node
    private void shortestPath(int src, long[] srcTo, List<int[]>[] graph) {
        // min PQ to find SD b/w src, dest
        PriorityQueue<long[]> queue = new PriorityQueue<>((a, b) -> Long.compare(a[1], b[1]));

        queue.add(new long[]{src, 0});

        while (!queue.isEmpty()) {
            long[] node = queue.poll();

            int to = (int) node[0];
            long dist = node[1];
            if (srcTo[to] != -1 && srcTo[to] <= dist) continue;
            srcTo[to] = dist;
            // For all adjacent nodes continue the process;
            for (int[] next : graph[to]) {
                queue.offer(new long[]{next[0], dist + next[1]});
            }
        }
    }

    private void buildGraph(int n, int[][] edges) {
        nextgraph = new ArrayList[n];
        pregraph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            nextgraph[i] = new ArrayList<int[]>();
            pregraph[i] = new ArrayList<int[]>();
        }

        for (int[] edge : edges) {
            int from = edge[0];
            int to = edge[1];
            int wt = edge[2];
            nextgraph[from].add(new int[]{to, wt});
            pregraph[to].add(new int[]{from, wt});
        }
    }


    /*
    Input: grid = [[0,1,1],[1,1,0],[1,1,0]]
    Output: 2
    Explanation: We can remove the obstacles at (0, 1) and (0, 2) to create a path from (0, 0) to (2, 2).
    It can be shown that we need to remove at least 2 obstacles, so we return 2.
    Note that there may be other ways to remove 2 obstacles to create a path.
     */
    // TC = O(mn), SC = O(mn)
    // Dijkstra on graph
    //  The idea is to find the path with min number of obstacles need to be removed
    public int minimumObstacles(int[][] grid) {

        int m = grid.length;
        int n = grid[0].length;

        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingLong(a -> a[2]));

        int[][] count = new int[m][n];// tells us the min nr of obstacle need to remove to reach at coordinate {x,y}
        count[0][0] = 0;
        queue.offer(new int[]{0, 0, 0});
        int[][] dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};

        // DFS for all guards and marked coordinate as 'P'
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int x = point[0];
            int y = point[1];
            int cost = point[2];

            if (x == (m - 1) && y == (n - 1)) return cost;

            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];

                if (!(nx >= 0 && nx < grid.length && ny >= 0 && ny < grid[0].length)) continue;

                int ncost = cost;
                // obstacle
                if (grid[nx][ny] == 1) ncost++;

                if (count[nx][ny] < ncost) continue;
                count[nx][ny] = ncost;
                queue.offer(new int[]{nx, ny, ncost});
            }
        }

        return -1;
    }

    /*
    Input: edges = [2,2,3,-1], node1 = 0, node2 = 1
    Output: 2
    Explanation: The distance from node 0 to node 2 is 1, and the distance from node 1 to node 2 is 1.
    The maximum of those two distances is 1. It can be proven that we cannot get a node with a smaller maximum distance than 1, so we return node 2.
     */
    List<List<Integer>> graph;

    public int closestMeetingNode(int[] edges, int node1, int node2) {
        int n = edges.length;
        buildGraph(edges);

        // To fetch the shortest path from all possible nodes
        long[] src1To = new long[n], src2To = new long[n];
        Arrays.fill(src1To, Long.MAX_VALUE);
        Arrays.fill(src2To, Long.MAX_VALUE);

        shortestPath(node1, src1To, graph);
        shortestPath(node2, src2To, graph);

        long res = Long.MAX_VALUE;
        int ans = -1;
        for (int i = 0; i < n; i++) {
            long d1 = src1To[i];
            long d2 = src2To[i];
            if (Math.max(d1, d2) < res) {
                ans = i;
                res = Math.max(d1, d2);
            }
        }
        return ans;
    }

    // Dijkstra algorithm to find the shortest distance b/w each node
    private void shortestPath(int src, long[] srcTo, List<List<Integer>> graph) {
        // min PQ to find SD b/w src, dest
        PriorityQueue<long[]> queue = new PriorityQueue<>((a, b) -> Long.compare(a[1], b[1]));

        queue.add(new long[]{src, 0});

        while (!queue.isEmpty()) {
            long[] node = queue.poll();
            int to = (int) node[0];
            long dist = node[1];
            if (srcTo[to] != Long.MAX_VALUE && srcTo[to] <= dist) continue;
            srcTo[to] = dist;
            // For all adjacent nodes continue the process;
            for (int next : graph.get(to)) {
                queue.offer(new long[]{next, dist + 1});
            }
        }
    }

    private void buildGraph(int[] edges) {
        graph = new ArrayList<>();
        for (int e : edges) graph.add(new ArrayList<>());
        for (int i = 0; i < edges.length; i++) if (edges[i] != -1) graph.get(i).add(edges[i]);
    }

    /*
    Input: edges = [2,2,3,-1], node1 = 0, node2 = 1
    Output: 2
    Explanation: The distance from node 0 to node 2 is 1, and the distance from node 1 to node 2 is 1.
    The maximum of those two distances is 1. It can be proven that we cannot get a node with a smaller maximum distance than 1, so we return node 2.
     */
    public int longestCycle(int[] edges) {
        int n = edges.length;
        Set<Integer> visited = new HashSet<>();

        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < edges.length; i++) {
            graph.computeIfAbsent(edges[i], k -> new HashSet<>());
            graph.get(edges[i]).add(i);
        }

        int max = -1;
        for (int i = 0; i < n; i++) {
            if (visited.contains(i))
                continue;

            // cycleSize & cycleEntryPoint
            int[] tableMeta = findCycle(i, edges, visited);
            if (tableMeta[0] > 0) max = Math.max(max, tableMeta[0]);
        }

        return max;
    }

    // return : new int[] {cycleSize, entryPoint}
    private int[] findCycle(int startPoint, int[] edges, Set<Integer> visited) {
        int next = startPoint;
        int entryPoint = -1;
        int cycleSize = 0;

        // find entry point of cycle
        while (entryPoint == -1) {
            visited.add(next);
            cycleSize++;
            if (next != -1) next = edges[next];
            else return new int[]{Integer.MIN_VALUE, entryPoint};

            if (visited.contains(next))
                entryPoint = next;
        }

        // remove the segment from startPoint to entryPoint
        next = startPoint;
        while (next != entryPoint) {
            cycleSize--;
            next = edges[next];
        }

        return new int[]{cycleSize, entryPoint};
    }
}

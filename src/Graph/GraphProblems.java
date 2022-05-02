package com.company;

import java.util.*;

public class GraphProblems {

    /*
    For each edge (i, j) in edges,
    we find a neighbour ii of node i,
    we find a neighbour jj of node i,
    If ii, i, j,jj has no duplicate, then that's a valid sequence.

    Ad the intuition mentioned,
    we don't have to enumearte all neignbours,
    but only some nodes with big value.

    But how many is enough?
    I'll say 3.
    For example, we have ii, i, j now,
    we can enumerate 3 of node j biggest neighbour,
    there must be at least one node different node ii and node i.

    So we need to iterate all edges (i, j),
    for each node we keep at most 3 biggest neighbour, which this can be done in O(3) or O(log3).

     */
    // Author: Anand
    // TC = O(n+m); n = length of edges, m = max neighbours of a node
    // SC = O(n)
    public int maximumScore(int[] scores, int[][] edges) {

        int maxSum = -1;
        int n = scores.length;
        int maxscore = 0;
        Node[] list = new Node[n];

        for (int i = 0; i < list.length; i++) {
            list[i] = new Node(scores[i]);
            maxscore = Math.max(scores[i], maxscore);
        }
        for (int[] edge : edges) {
            Node start = list[edge[0]];
            Node end = list[edge[1]];
            start.add(end);
        }

        for (int[] edge : edges) {
            Node start = list[edge[0]];
            Node end = list[edge[1]];

            if (start.value + end.value + maxscore + maxscore <= maxSum) continue;
            Queue<Node> queue = new PriorityQueue<>((a, b) -> b.value - a.value);

            for (Node node : start.next) {
                if (node != end) queue.offer(node);
            }

            if (queue.size() == 0) continue;
            // pull 2 adjacent neighbours out of queue
            Node start1 = queue.poll();
            Node start2 = queue.poll();

            queue.clear();

            for (Node node : end.next) {
                if (node != start) queue.offer(node);
            }

            if (queue.size() == 0) continue;
            // pull 2 adjacent neighbours out of queue
            Node end1 = queue.poll();
            Node end2 = queue.poll();

            int sum = start.value + end.value + start1.value;

            if (start1 != end1) {
                sum += end1.value;
            } else if (start2 == null && end2 == null) {
                continue;
            } else if (start2 != null && end2 != null) {
                sum += Math.max(start2.value, end2.value);
            } else if (start2 == null) {
                sum += end2.value;
            } else {
                sum += start2.value;
            }
            maxSum = Math.max(sum, maxSum);

        }
        return maxSum;
    }

    static class Node {
        int value;
        Set<GraphProblems.Node> next = new HashSet<>();

        Node(int val) {
            this.value = val;
        }

        void add(GraphProblems.Node node) {
            next.add(node); // a ---> [b]
            node.next.add(this); // b --> [a]
        }
    }

    // TC = O(mn), SC = O(mn)
    // Author : Anand
    // DFS on graph
    public int countUnguarded(int m, int n, int[][] guards, int[][] walls) {
        char[][] grid = new char[m][n];
        Queue<int[]> queue = new LinkedList<>();
        int[][] dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};

        for (int[] guard : guards) {
            grid[guard[0]][guard[1]] = 'G';
            queue.offer(new int[]{guard[0], guard[1]});
        }

        for (int[] wall : walls) grid[wall[0]][wall[1]] = 'W';

        // DFS for all guards and marked coordinate as 'P'
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            for (int[] dir : dirs) {
                int x = point[0] + dir[0];
                int y = point[1] + dir[1];
                // check for boundary/obstacle condition
                while (safe(x, y, grid)) {
                    grid[x][y] = 'P';
                    x += dir[0];
                    y += dir[1];
                }
            }
        }

        int cnt = 0;
        // count cells that are not blocker
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] != 'W' && grid[i][j] != 'G' && grid[i][j] != 'P') cnt++;
            }
        }
        return cnt;
    }

    private boolean safe(int x, int y, char[][] grid) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] != 'W' && grid[x][y] != 'G';
    }


    /*
     // Algorithm
     - Store locations of fire Queue
     - while queue is not empty keep on spreading all fires  for a min
     - Apply bfs  and check if user is reachable
     - If yes then increment min and again spread fire
     - Else return min.
     // TC = O(mn/4(m+n)), SC = O(m+n)
    */
    private static final int[][] DIRS = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    public int maximumMinutes(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        List<int[]> fires = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) fires.add(new int[]{i, j});
            }
        }

        boolean moved = false;
        int min = 0;
        while(true){
            boolean[] traversal = bfs(grid, new int[]{0, 0}, new int[]{m - 1, n - 1}, false);
            moved = traversal[1] || moved;
            if (traversal[0] && traversal[1]) {
                // spread fire
                min++;
                for (int[] fire : fires) {
                    for (int[] dir : DIRS) {
                        int newx = dir[0] + fire[0];
                        int newy = dir[1] + fire[1];
                        if (isSafe(grid, newx, newy)) {
                            grid[newx][newy] = 1;
                        }
                    }
                }
            }
            break;
        }

        return !moved ? 1_000_000_000 : min;
    }

    // BFS ---> To check if  user is reachable
    private boolean[] bfs(int[][] grid, int[] start, int[] dest, boolean state) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(start);

        boolean[][] vis = new boolean[m][n];
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            for (int[] dir : DIRS) {
                int newx = dir[0] + point[0];
                int newy = dir[1] + point[1];
                if (newx == m - 1 && newy == n - 1) return new boolean[]{true, state};
                if (newx >= 0 && newx < grid.length && newy >= 0 && newy < grid[0].length && grid[newx][newy] != 2 && grid[newx][newy] != 1) {
                    if (vis[newx][newy]) continue;
                    vis[newx][newy] = true;
                    grid[newx][newy] = 1;
                    state = true;
                    queue.offer(new int[]{newx, newy});
                }
            }
        }
        return new boolean[]{false, state};
    }

    // This will tell us if coordinate is within extrimities OR safe to visit the point
    private boolean isSafe(int[][] grid, int x, int y) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] != 2;
    }


//  Shortest path in an unweighted graph
    public class pathUnweighted {

    public static void main(String[] args) {
        // No of vertices
        int v = 8;

    }

    }

}

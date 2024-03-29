package Graph;

import javafx.util.Pair;

import java.util.HashMap;
import java.util.*;

public class GraphProblems {

    /*
    ###BFS+BS###
   // Algorithm
   - Store locations of fire Queue
   - Spread fire upto mn time
   - Apply bfs+BS and check if user is reachable with t time
   - If yes then  l = mid
   - Else  r = mid-1
   - return l;
   // TC = O(mnlognm), SC = O(mn)
  */
    private static final int[][] DIRS = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    /*
    For each edge (i, j) in edges,
    we find a neighbour ii of node i,
    we find a neighbour jj of node i,
    If ii, i, j,jj has no duplicate, then that's a valid sequence.

    Ad the intuition mentioned,
    we don't have to enumerate all neighbours,
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

    public int maximumMinutes(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        List<int[]> fires = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) fires.add(new int[]{i, j});
            }
        }


        int l = -1, r = m * n;
        while (l < r) {
            int mid = l + (r - l) / 2 + 1;
            if (reachable(grid, fires, mid)) l = mid;
            else r = mid - 1;
        }
        return (int) (l == m * n ? 1e9 : l);
    }

    // BFS ---> To check if  user is reachable after t min of fire spread
    private boolean reachable(int[][] grid, List<int[]> fires, int moves) {
        int[][] copy = clone(grid);

        Queue<int[]> fire = new LinkedList<>(fires);

        while (!fire.isEmpty() && moves-- > 0) {
            if (spread(fire, copy)) return false;
        }

        //check if person is reachable to dest
        Queue<int[]> person = new LinkedList<>();

        person.add(new int[]{0, 0});
        while (!person.isEmpty()) {
            boolean onFire = spread(fire, copy);
            boolean dest = spread(person, copy);

            if (dest) return true;
            if (onFire) return false;
        }
        return false;
    }

    private boolean spread(Queue<int[]> queue, int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int sz = queue.size();

        while (sz-- > 0) {
            int[] point = queue.poll();
            for (int[] dir : DIRS) {
                assert point != null;
                int newx = dir[0] + point[0];
                int newy = dir[1] + point[1];
                if (newx == m - 1 && newy == n - 1) return true;
                if (isSafe(grid, newx, newy)) {
                    grid[newx][newy] = -1;
                    queue.offer(new int[]{newx, newy});
                }
            }
        }
        return false;
    }

    // This will tell us if coordinate is within extrimities OR safe to visit the point
    private boolean isSafe(int[][] grid, int x, int y) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] == 0;
    }

    private int[][] clone(int[][] grid) {
        int[][] copy = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                copy[i][j] = grid[i][j];
            }
        }

        return copy;
    }

    // TODO
    public int mostProfitablePath(int[][] edges, int bob, int[] amount) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Map<Pair<Integer, Integer>, Integer> dm = new HashMap<>(); // wt of each edge

        int n = amount.length;
        for (int[] edge : edges) {
            int start = edge[0];
            int end = edge[1];

            dm.put(new Pair<>(start, end), amount[end]);
            dm.put(new Pair<>(end, start), amount[start]);

            if (graph.containsKey(edge[0])) {
                List<Integer> exist = graph.get(edge[0]);
                exist.add(edge[1]);
                graph.put(edge[0], exist);
            } else graph.put(edge[0], new ArrayList<>(Collections.singletonList(edge[1])));

            if (graph.containsKey(edge[1])) {
                List<Integer> exist = graph.get(edge[1]);
                exist.add(edge[0]);
                graph.put(edge[1], exist);
            } else graph.put(edge[1], new ArrayList<>(Collections.singletonList(edge[0])));
        }
        // To fetch the shortest path from all possible nodes
        TreeMap<Integer, Pair<Long, Long>> tm = new TreeMap<>();

        for (int i = 0; i < n; i++) {
            tm.put(i, new Pair<>(Long.MAX_VALUE, 0L));
        }

        // Bob
        shortestPath(bob, tm, graph, dm);

        // To fetch the shortest path from all possible nodes
        TreeMap<Integer, Pair<Long, Long>> atm = new TreeMap<>();

        for (int i = 0; i < n; i++) {
            atm.put(i, new Pair<>(Long.MAX_VALUE, 0L));
        }


        //Alice
        shortestPath(0, atm, graph, dm);

        int cost = 0;
        for (Map.Entry<Integer, Pair<Long, Long>> entry : atm.entrySet()) {
            if (tm.containsKey(entry.getKey()) && tm.get(entry.getKey()).getValue() == entry.getValue().getValue()) {
                cost += entry.getValue().getValue() / 2;
            } else if (tm.containsKey(entry.getKey()) && tm.get(entry.getKey()).getValue() >= entry.getValue().getValue())
                cost += entry.getValue().getKey();
        }

        return cost;
    }

    // Dijkstra algorithm to find the shortest distance b/w each node
    private void shortestPath(int src, TreeMap<Integer, Pair<Long, Long>> tm, Map<Integer, List<Integer>> graph, Map<Pair<Integer, Integer>, Integer> dm) {
        // min PQ to find SD b/w src, dest
        PriorityQueue<long[]> queue = new PriorityQueue<>(Comparator.comparingLong(a -> a[1]));

        queue.add(new long[]{src, Long.MAX_VALUE, 0});

        while (!queue.isEmpty()) {
            long[] node = queue.poll();
            int to = (int) node[0];
            long dist = node[1];
            if (tm.get(to).getKey() != Long.MAX_VALUE && tm.get(to).getKey() <= dist) continue;
            tm.put(to, new Pair<>(dist, tm.get(to).getValue()));
            // For all adjacent nodes continue the process;
            for (int next : graph.get(to)) {
                queue.add(new long[]{next, dist + dm.get(new Pair<>(to, next)), tm.get(to).getValue() + 1});
            }
        }

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
}

package Graph;

import java.util.Arrays;

public class TSP {
    // It shows distance/wt of each vertices for start vertex
    static int[][] dist = {{0, 20, 42, 25},
            {20, 0, 30, 34},
            {42, 30, 0, 10},
            {25, 34, 10, 0}};
    static int s = 0;
    static int n = 4; //  no of vertices in graph

    // DP Based solution
    public static void main(String[] args) {
        int[][] dp = new int[1 << n][n];
        for (int[] d : dp) Arrays.fill(d, -1);
        boolean[] visited = new boolean[n];
        System.out.println("Minimum weight hamiltonion path vian backtracking is " + tspBacktracking(0, 0, visited));
        System.out.println("Minimum weight hamiltonion path via DP is " + tsp(0, s, dp));
    }

    // TC = O(n!) --> Basically its a DFS ie. traverse all possible paths
    private static int tspBacktracking(int count, int pos, boolean[] visited) {
        // base case
        if (count == dist.length) {
            return dist[pos][s];
        }

        int ans = Integer.MAX_VALUE;
        // For all adj vertices of curr node visit them and compute cost
        for (int city = 0; city < n; city++) {
            if (!visited[city]) {
                visited[city] = true;
                int newAns = dist[pos][city] + tspBacktracking(count + 1, city, visited);
                ans = Math.min(ans, newAns);
                visited[city] = false; // backtrack
            }
        }
        return ans;
    }

    private static final int VISITED_ALL = (1 << n) - 1;

    // TC = O(2N*N), SC = O((2N*N) + recursion stack space)
    private static int tsp(int mask, int pos, int[][] dp) {
        // base case
        if (mask == VISITED_ALL) return dist[pos][s];
        if (dp[mask][pos] != -1) return dp[mask][pos];
        int ans = Integer.MAX_VALUE;
        for (int city = 0; city < dist.length; city++) {
            // if the city is not visited then  visit and compute cost of the path
            if ((mask & 1 << city) == 0) {
                int cost = dist[pos][city] + tsp((mask | 1 << city), city, dp);
                ans = Math.min(ans, cost);
            }
        }
        return dp[mask][pos] = ans;
    }


}


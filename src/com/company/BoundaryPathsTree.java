package com.company;

import java.util.Arrays;

public class BoundaryPathsTree {

    int[] dx = {0, 1, 0, -1};
    int[] dy = {1, 0, -1, 0};
    int M = 1000000000 + 7;
    int m, n;

    public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        this.m = m;
        this.n = n;
        int[][][] dp = new int[m][n][maxMove + 1];
        for (int[][] l : dp)
            for (int[] sl : l) Arrays.fill(sl, -1);

        return recursive(m, n, startRow, startColumn, dp, maxMove);
    }

    // this function will tell that the current node is safe to be visited or not
    private boolean isSafe(int r, int c) {
        return (r < m && c < n && r >= 0 && c >= 0);
    }

    private int recursive(int m, int n, int r, int c, int[][][] dp, int maxMove) {
        // base cases
        if (!isSafe(r, c)) return 1;
        if (maxMove == 0) return 0;

        if (dp[r][c][maxMove] >= 0) return dp[r][c][maxMove];

        dp[r][c][maxMove] = (
                (recursive(m, n, r + 1, c, dp, maxMove - 1)
                        + recursive(m, n, r - 1, c, dp, maxMove - 1)) % M
                        + (recursive(m, n, r, c + 1, dp, maxMove - 1)
                        + recursive(m, n, r, c - 1, dp, maxMove - 1)) % M
        ) % M;

        return dp[r][c][maxMove];
    }
}

package com.company;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Matrix {
    //Author: Anand
    /*
    Diagonal elements of matrix are found for below cases :-
    # i == j
    # i+j == n-1
     */
    public boolean checkXMatrix(int[][] grid) {
        int n = grid.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j || (i + j) == (n - 1)) {
                    if (grid[i][j] == 0) return false;
                } else if (grid[i][j] != 0) return false;
            }
        }
        return true;
    }

    /*
    Input: mat = [[1,2,3],[4,5,6],[7,8,9]]
    Output: [1,2,4,7,5,3,6,8,9]

    The idea is to change directions when touching the extreme rows and cols
     */
    public int[] findDiagonalOrder(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[] ans = new int[m * n];

        int row = 0, col = -1; // As ++col before actually using it
        int ind = 0;
        while (ind < m * n) {
            // Last row: move right and keep mv to upper right
            if (row == m - 1) {
                ans[ind++] = matrix[row][++col];
                while (row > 0 && col < n - 1) {
                    ans[ind++] = matrix[--row][++col];
                }
            }

            // Last col: move down and and keep moving lower left
            else if (col == n - 1) {
                ans[ind++] = matrix[++row][col];
                while (col > 0 && row < m - 1) {
                    ans[ind++] = matrix[++row][--col];
                }
            }

            // First row:  move right and keep moving lower left
            else if (row == 0) {
                ans[ind++] = matrix[row][++col];
                while (col > 0 && row < m - 1) {
                    ans[ind++] = matrix[++row][--col];
                }
            }
            // First col: move down and keep mv to upper right
            else if (col == 0) {
                ans[ind++] = matrix[++row][col];
                while (row > 0 && col < n - 1) {
                    ans[ind++] = matrix[--row][++col];
                }
            }
        }

        return ans;
    }


    public long maxMatrixSum(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int mini = Integer.MAX_VALUE;
        int cnt = 0;
        long sum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum += Math.abs(matrix[i][j]);
                if (matrix[i][j] < 0) cnt++;
                mini = Math.min(mini, Math.abs(matrix[i][j]));
            }
        }
        if (cnt % 2 == 0) return sum;
        return sum - 2 * mini;
    }

    /*


    Input: n = 3, queries = [[1,1,2,2],[0,0,1,1]]
    Output: [[1,1,0],[1,2,1],[0,1,1]]
    Explanation: The diagram above shows the initial matrix, the matrix after the first query, and the matrix after the second query.
    - In the first query, we add 1 to every element in the submatrix with the top left corner (1, 1) and bottom right corner (2, 2).
    - In the second query, we add 1 to every element in the submatrix with the top left corner (0, 0) and bottom right corner (1, 1).

     */
    public int[][] rangeAddQueries(int n, int[][] queries) {
        int[][] ans = new int[n][n];

        for (int[] query : queries) {
            int r1 = query[0];
            int c1 = query[1];
            int r2 = query[2];
            int c2 = query[3];

            for (int i = r1; i <= r2; i++) {
                for (int j = c1; j <= c2; j++) {
                    ans[i][j]++;
                }
            }
        }
        return ans;
    }

    public int[][] differenceOfDistinctValues(int[][] grid) {

        int m = grid.length, n = grid[0].length;
        int[][] ans = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ans[i][j] = helper(i, j, grid);
            }
        }

        return ans;
    }

    private int helper(int i, int j, int[][] grid) {
        int m = grid.length, n = grid[0].length;

        int ci = i, cj = j;
        Set<Integer> tl = new HashSet<>(), tr = new HashSet<>();
        i--;
        j--;

        while (i >= 0 && i < m && j >= 0 && j < n) {
            tl.add(grid[i][j]);
            i--;
            j--;
        }


        i = ci;
        j = cj;
        i++;
        j++;

        while (i >= 0 && i < m && j >= 0 && j < n) {
            tr.add(grid[i][j]);
            i++;
            j++;
        }

        return Math.abs(tl.size() - tr.size());
    }
}

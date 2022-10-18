package com.company;

import java.util.Arrays;

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

}

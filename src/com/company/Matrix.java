package com.company;

import java.util.ArrayList;
import java.util.List;

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
    - Create an array list
    - Place elements in all 4 directions diagonally
    */
    class Solution {
        public int[] findDiagonalOrder(int[][] matrix) {
            int m = matrix.length, n = matrix[0].length;
            List<Integer> ans = new ArrayList<>();

            // R DD D UD
            int[][] dirs1 = {{0, 1}, {1, -1}, {1, 0}, {-1, 1}};

            // R DD R UD
            int[][] dirs2 = {{0, 1}, {1, -1}, {0, 1}, {-1, 1}};

            int x = 0, y = 0;

            int[][] dirs = new int[4][2];
            ans.add(matrix[x][y]);
            while (true) {

                if (x == 0 && y == n - 1) {
                    dirs = dirs2;
                } else dirs = dirs1;

                for (int[] dir : dirs) {

                    boolean d = false;
                    int newx = dir[0] + x;
                    int newy = dir[1] + y;

                    System.out.println(newx + "," + newy);
                    if (newx == m - 1 && newy == n - 1) {
                        ans.add(matrix[newx][newy]);
                        return ans.stream().mapToInt(e -> e).toArray();
                    }


                    while (safe(newx, newy, matrix) && ((dir[0] == 1 && dir[1] == -1) || (dir[0] == -1 && dir[1] == 1))) {
                        d = true;
                        ans.add(matrix[newx][newy]);

                        if (newx == 0 && newy == n - 1) {
                            break;
                        }

                        x = newx;
                        y = newy;
                        newx = dir[0] + x;
                        newy = dir[1] + y;
                        System.out.println(newx + "," + newy);
                    }


                    if (!d && safe(newx, newy, matrix)) {
                        ans.add(matrix[newx][newy]);
                        x = newx;
                        y = newy;
                    }

                }
            }
        }

        private boolean safe(int x, int y, int[][] matrix) {
            return x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length;
        }
    }
}

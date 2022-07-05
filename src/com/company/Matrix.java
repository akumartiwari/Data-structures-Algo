package com.company;

import java.util.ArrayList;
import java.util.Arrays;
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
   - Create a matrix of given dimension
   - Place elements in all 4 directions
 */
    public int[][] spiralMatrix(int m, int n, ListNode head) {
        int[][] matrix = new int[m][n];
        for (int[] mat : matrix) Arrays.fill(mat, -1);

        if (head == null) return matrix;
        ListNode temp = head;

        List<Integer> nodes = new ArrayList<>();
        while (temp != null) {
            nodes.add(temp.val);
            temp = temp.next;
        }

        // R D L U
        int[][] dirs = {{0, 1}, {-1, 0}, {0, -1}, {1, 0}};
        int idx = 0;
        int x = 0, y = 0;
        matrix[0][0] = nodes.get(idx++);
        while (true) {
            boolean done = false;
            for (int[] dir : dirs) {
                int newx = dir[0] + x;
                int newy = dir[1] + y;

                if (idx >= nodes.size()) {
                    done = true;
                    break;
                }

                while (idx < nodes.size() && safe(newx, newy, matrix)) {
                    matrix[newx][newy] = nodes.get(idx++);
                    x = newx;
                    y = newy;
                    newx = dir[0] + x;
                    newy = dir[1] + y;
                }

                if (idx >= nodes.size()) {
                    done = true;
                    break;
                }
            }

            if (done) break;
        }
        return matrix;
    }

    private boolean safe(int x, int y, int[][] matrix) {
        return x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length && (matrix[x][y] == -1);
    }
}

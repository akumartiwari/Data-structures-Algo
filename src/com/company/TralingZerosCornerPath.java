package com.company;

import javafx.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TralingZerosCornerPath {

    // Author: Anand
    public int minimumRounds(int[] tasks) {
        int ans = 0;
        Map<Integer, Integer> freq = new HashMap<>();
        for (int task : tasks) freq.put(task, freq.getOrDefault(task, 0) + 1);

        for (Map.Entry entry : freq.entrySet()) {
            int value = (int) entry.getValue();
            boolean success = false;
            int cnt3 = value / 3;
            int cnt2 = (value % 3) / 2;
            while (cnt3 >= 0) {
                if (cnt3 * 3 + (cnt2 * 2) == value) {
                    ans += cnt3 + cnt2;
                    success = true;
                    break;
                }
                cnt3--;
                cnt2 = (value - cnt3 * 3) / 2;
            }

            if (success) continue;
            return -1;
        }
        return ans;
    }

    /*
    Input: grid = [[23,17,15,3,20],[8,1,20,27,11],[9,4,6,2,21],[40,9,1,10,6],[22,7,4,5,3]]
    Output: 3
    Explanation: The grid on the left shows a valid cornered path.
    It has a product of 15 * 20 * 6 * 1 * 10 = 18000 which has 3 trailing zeros.
    It can be shown that this is the maximum trailing zeros in the product of a cornered path.

    The grid in the middle is not a cornered path as it has more than one turn.
    The grid on the right is not a cornered path as it requires a return to a previously visited cell.

     */
    // Author: Anand
    public int maxTrailingZeros(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        List<List<Pair<Integer, Integer>>> preHorizontal = new ArrayList<>();
        List<List<Pair<Integer, Integer>>> preVertical = new ArrayList<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Pair<Integer, Integer> pair;
                if (j == 0) {
                    int cnt5 = cnt(grid[i][j], 5);
                    int cnt2 = cnt(grid[i][j], 2);
                    pair = new Pair<>(cnt5, cnt2);
                } else {
                    int cnt5 = cnt(grid[i][j], 5);
                    int cnt2 = cnt(grid[i][j], 2);
                    pair = new Pair<>(preHorizontal.get(i).get(j - 1).getKey() + cnt5,
                            preHorizontal.get(i).get(j - 1).getValue() + cnt2);
                }

                List<Pair<Integer, Integer>> list = preHorizontal.size() > 0 ? preHorizontal.get(i) : new ArrayList<>();
                list.add(pair);
                preHorizontal.add(i, list);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Pair<Integer, Integer> pair;
                if (j == 0) {
                    int cnt5 = cnt(grid[j][i], 5);
                    int cnt2 = cnt(grid[j][i], 2);
                    pair = new Pair<>(cnt5, cnt2);
                } else {
                    int cnt5 = cnt(grid[j][i], 5);
                    int cnt2 = cnt(grid[j][i], 2);
                    pair = new Pair<>(preVertical.get(j - 1).get(i).getKey() + cnt5,
                            preVertical.get(j - 1).get(i).getValue() + cnt2);
                }

                List<Pair<Integer, Integer>> list = preVertical.size() > 0 ? preVertical.get(i) : new ArrayList<>();
                list.add(pair);
                preVertical.add(i, list);
            }
        }

        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {

                Pair<Integer, Integer> cntHor1 = new Pair<Integer, Integer>(0, 0),
                        cntHor2 = new Pair<Integer, Integer>(0, 0),
                        cntVer1 = new Pair<Integer, Integer>(0, 0),
                        cntVer2 = new Pair<Integer, Integer>(0, 0);

                // |-------- ie. upper L shape from l->r
                cntHor1 = new Pair<>(preHorizontal.get(i).get(m - 1).getKey() - preHorizontal.get(i).get(j).getKey()
                        , preHorizontal.get(i).get(m - 1).getValue() - preHorizontal.get(i).get(j).getValue()
                );
                // |-------- ie. upper L shape from r->l
                cntHor2 = j > 0 ? preHorizontal.get(i).get(j) : cntHor2;

                cntVer1 = i > 0 ? new Pair<>(preVertical.get(n - 1).get(j).getKey() - preVertical.get(i - 1).get(j).getKey()
                        , preVertical.get(n - 1).get(j).getValue() - preVertical.get(i - 1).get(j).getValue()
                ) : preVertical.get(n - 1).get(j);
                cntVer2 = preVertical.get(i).get(j);

                ans = Math.max(ans, solve(cntHor1, cntVer1));
                ans = Math.max(ans, solve(cntHor1, cntVer2));
                ans = Math.max(ans, solve(cntHor2, cntVer1));
                ans = Math.max(ans, solve(cntHor2, cntVer2));

            }
        }
        return ans;
    }

    private int solve(Pair<Integer, Integer> pair1, Pair<Integer, Integer> pair2) {
        return Math.max(pair1.getKey() + pair2.getKey(), pair1.getValue() + pair2.getValue());
    }

    private int cnt(int number, int factor) {
        int ans = 0;
        while (number > 0 && number % factor == 0) {
            ans++;
            number /= factor;
        }
        return ans;
    }

}

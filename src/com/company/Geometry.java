package com.company;

import java.util.HashMap;
import java.util.Map;

// Author: Anand
// TC = O(n)
public class Geometry {
    /*
    Input
    ["DetectSquares", "add", "add", "add", "count", "count", "add", "count"]
    [[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]]
    Output
    [null, null, null, null, 1, 0, null, 2]
     */
    public static void main(String[] args) {
        Geometry obj = new Geometry();
        obj.add(new int[]{3, 10});
        int param_2 = obj.count(new int[]{3, 10});
        System.out.println(param_2);
    }

    Map<Integer, Map<Integer, Integer>> row2ColCnt; // Cnt of point present on same x axis corrspond to y axis

    public Geometry() {
        row2ColCnt = new HashMap<>();
    }

    public void add(int[] point) {
        int x = point[0], y = point[1];
        Map<Integer, Integer> col2Cnt = row2ColCnt.getOrDefault(x, new HashMap<>());
        col2Cnt.put(y, col2Cnt.getOrDefault(y, 0) + 1);
        row2ColCnt.put(x, col2Cnt);
    }

    public int count(int[] point) {
        int x = point[0], y = point[1];
        int ans = 0;
        Map<Integer, Integer> col2Cnt = row2ColCnt.getOrDefault(x, new HashMap<>());

        for (int ny : col2Cnt.keySet()) {

            if (ny == y) continue;
            int c1 = col2Cnt.get(ny);
            int len = Math.abs(y - ny);
            int[] possX = new int[]{x + len, x - len};

            for (int poss : possX) {
                Map<Integer, Integer> temp = row2ColCnt.getOrDefault(poss, new HashMap<>());
                int c2 = temp.getOrDefault(y, 0), c3 = temp.getOrDefault(ny, 0);
                ans += c1 * c2 * c3;
            }
        }
        return ans;
    }
}

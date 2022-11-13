package com.company;

import java.util.*;

public class CoordinateORPointCompressionTechnique {


    /*
    9

    5 12
    17 19
    12 26
    7 21
    17 25
    16 22
    7 9
    11 14
    12 24
    4
     */

    //Follow for complete solution -
    //https://www.youtube.com/watch?v=cYROUjJUX0w
    public static int powerfullInteger(int n, int[][] interval, int k) {
        Arrays.sort(interval, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] < o2[0]) return -1;
                if (o1[0] > o2[0]) return 1;
                return Integer.compare(o1[1], o2[1]);
            }
        });

        System.out.println(Arrays.deepToString(interval));

        // STL map for storing start
        // and end points
        HashMap<Integer, Integer> points = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            int l = interval[i][0];
            int r = interval[i][1];

            points.put(l, 0);
            points.put(r + 1, 0);
        }

        for (int i = 0; i < n; i++) {
            int l = interval[i][0];
            int r = interval[i][1];

            // Increment at starting point
            if (points.containsKey(l)) {
                points.put(l, points.get(l) + 1);
            } else {
                points.put(l, 1);

                // Decrement at ending point
                if (points.containsKey(r + 1)) {
                    points.put(r + 1, points.get(r + 1) - 1);
                }
            }
        }
        // Stores current frequency
        int cur_freq = 0;

        // Store element with maximum frequency
        int ans = 0;

        // Frequency of the current ans
        int freq_ans = 0;

        for (Map.Entry<Integer, Integer> entry : points.entrySet()) {
            // x.first denotes the current
            // point and x.second denotes
            // points[x.first]
            cur_freq += entry.getValue();

            // If frequency of x.first is
            // greater that freq_ans
            if (cur_freq >= k && entry.getKey() > ans) {
                freq_ans = cur_freq;
                ans = entry.getKey();
            }
        }

        // Print Answer

        return ans;
    }

}

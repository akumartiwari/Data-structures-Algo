package com.company;

import java.util.HashMap;

class FourSumCount {

    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int n = nums1.length;
        if (n == 1) {
            return (nums1[0] + nums2[0] + nums3[0] + nums4[0] == 0) ?
                    1 : 0;
        }

        // Put nums3 + nums4 into a HashMap, where
        // key = nums3[k] + nums4[l], value = count
        HashMap<Integer, Integer> m = new HashMap<Integer, Integer>();
        for (int k = 0; k < n; k++) {
            for (int l = 0; l < n; l++) {
                int s = nums3[k] + nums4[l];
                m.put(s, m.getOrDefault(s, 0) + 1);
            }
        }

        int r = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int comp = -nums1[i] - nums2[j];
                if (m.get(comp) != null) {
                    r += m.get(comp);
                }
            }
        }

        return r;
    }
}

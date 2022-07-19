package com.company;

public class BrainStormingProblems {
    /*
    Input: nums1 = [1,2,3,4], nums2 = [2,10,20,19], k1 = 0, k2 = 0
    Output: 579
    Explanation: The elements in nums1 and nums2 cannot be modified because k1 = 0 and k2 = 0.
    The sum of square difference will be: (1 - 2)2 + (2 - 10)2 + (3 - 20)2 + (4 - 19)2 = 579.
     */
    // Author: Anand
    // TC = O(n+constant)
    public long minSumSquareDiff(int[] nums1, int[] nums2, int k1, int k2) {
        int[] diff = new int[100_001];
        int maxDiff = Integer.MIN_VALUE;
        for (int i = 0; i < nums1.length; i++) {
            int d = Math.abs(nums1[i] - nums2[i]);
            diff[d]++;
            maxDiff = Math.max(maxDiff, d);
        }
        int total = k1 + k2;
        while (maxDiff > 0 && total > 0) {
            final int count = diff[maxDiff];
            if (count <= total) {
                diff[maxDiff] -= count;
                diff[maxDiff - 1] += count;
                maxDiff--;
            } else {
                diff[maxDiff] -= total;
                diff[maxDiff - 1] += total;
            }
        }

        // calculate the sqr of all elements with count in diff array
        long res = 1L;
        while (maxDiff > 0) {
            res += (long) maxDiff * maxDiff * diff[maxDiff];
            maxDiff--;
        }

        return res;
    }
}

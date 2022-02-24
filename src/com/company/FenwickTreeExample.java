package com.company;

// TODO Study Fenwick tree
public class FenwickTreeExample {
    /*
    Input: nums1 = [2,0,1,3], nums2 = [0,1,2,3]
    Output: 1
    Explanation:
    There are 4 triplets (x,y,z) such that pos1x < pos1y < pos1z. They are (2,0,1), (2,0,3), (2,1,3), and (0,1,3).
    Out of those triplets, only the triplet (0,1,3) satisfies pos2x < pos2y < pos2z. Hence, there is only 1 good triplet.
     */
    public long goodTriplets(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] idx = new int[n], map = new int[n];         // remap:
        for (int i = 0; i < n; i++) idx[nums1[i]] = i;        // nums1 -> idx
        for (int i = 0; i < n; i++)
            map[i] = idx[nums2[i]];   // nums2 -> map   -- could be combined with the loop below, but we keep it separate for clarity

        long ans = 0;
        FenwickTree bit = new FenwickTree(n);
        for (int i = 0; i < n; i++) {
            int lSmaller = bit.ps(map[i]);     // nums < map[i] to the left of i
            int lLarger = i - lSmaller;            // nums > map[i] to the left of i
            int larger = n - 1 - map[i];            // all nums > map[i]
            int rLarger = larger - lLarger;      // nums > map[i] to the right of i
            ans += (long) lSmaller * rLarger;
            bit.inc(map[i]);
        }
        return ans;
    }

    private static class FenwickTree {
        // binary index tree, index 0 is not used, so we shift all calls by +1
        int[] bit;
        int n;

        public FenwickTree(int range) {
            this.n = range + 1;
            this.bit = new int[this.n];
        }

        // increment count of i
        public void inc(int i) {
            for (i++; i < n; i += Integer.lowestOneBit(i))
                bit[i]++;
        }

        // prefix sum query i.e. count of nums <= i added so far
        private int ps(int i) {
            int ps = 0;
            for (i++; i != 0; i -= Integer.lowestOneBit(i))
                ps += bit[i];
            return ps;
        }
    }
}

package com.company;

import java.util.*;
import java.util.stream.Collectors;

public class HashMap {

        /*

        You are given a 0-indexed integer array nums.
        There exists an array arr of length nums.length, where arr[i] is the sum of |i - j| over all j such that nums[j] == nums[i]
        and j != i. If there is no such j, set arr[i] to be 0.

        Return the array arr.

        ---------------------------------------------------------------------------------------------------

        Input: nums = [1,3,1,1,2]
        Output: [5,0,3,4,0]
        Explanation:
        When i = 0, nums[0] == nums[2] and nums[0] == nums[3]. Therefore, arr[0] = |0 - 2| + |0 - 3| = 5.
        When i = 1, arr[1] = 0 because there is no other index with value 3.
        When i = 2, nums[2] == nums[0] and nums[2] == nums[3]. Therefore, arr[2] = |2 - 0| + |2 - 3| = 3.
        When i = 3, nums[3] == nums[0] and nums[3] == nums[2]. Therefore, arr[3] = |3 - 0| + |3 - 2| = 4.
        When i = 4, arr[4] = 0 because there is no other index with value 2.

     */

    //TC = O(nlogn)
    public long[] distance(int[] nums) {

        TreeMap<Integer, TreeMap<Integer, Long>> freq = new TreeMap<>(); // element, {index, ps of remaining indexes}
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            TreeMap<Integer, Long> tm = new TreeMap<>();
            freq.putIfAbsent(num, tm);

            if (freq.get(num).size() > 0) {
                long nps = freq.get(num).lastEntry().getValue() + i;
                tm.put(i, nps);
            } else tm.put(i, (long) i);

            freq.get(num).putAll(tm);
        }

        long[] ans = new long[nums.length];

        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            TreeMap<Integer, Long> indexes = freq.get(num);

            List<Integer> keys = new ArrayList<>(indexes.keySet());

            int ind = Arrays.binarySearch(keys.toArray(), i);
            long right = (Math.abs(indexes.lastEntry().getValue() - indexes.get(i))
                    - (long) (keys.size() - 1 - ind) * i);

            long left = keys.size() > 1 ? Math.abs(((long) (ind + 1) * i) - indexes.get(i)) : 0;

            ans[i] = left + right;
        }
        return ans;
    }


    //TC = O(N)
    // HashMap Beats 100%
    public long[] distanceOptimised(int[] arr) {
        Map<Long, long[]> map = new java.util.HashMap<>();
        // [0] -> sum of indices at left of i
        // [1] -> sum of indices at right of i
        // [2] -> left freq
        // [3] -> right freq
        int i = 0;
        for (int e : arr) {
            long x = e;
            map.computeIfAbsent(x, k -> new long[4]);
            map.get(x)[1] += i++; // total sum of indices with value x
            map.get(x)[3]++;    // no. of occurences of x in arr
        }

        long[] res = new long[arr.length];
        i = 0;
        for (int e : arr) {
            long x = e;
            long[] temp = map.get(x);
            temp[1] -= i;  // sum of indices at right
            temp[3]--;   // right freq
            res[i] = Math.abs(temp[0] - i * temp[2]) + Math.abs(temp[1] - i * temp[3]);
            temp[0] += i++;  // sum of indices at left
            temp[2]++;   // left freq
        }
        return res;
    }
}

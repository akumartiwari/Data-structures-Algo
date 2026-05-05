package com.company;

import java.util.HashMap;
import java.util.Map;

public class QualityProblems {


    /*
     * PROBLEM: Minimum Number of Groups for Valid Assignment (LeetCode 2910)
     * Assign array indices to groups such that all indices in a group share the same value and
     * any two group sizes differ by at most 1; return the minimum number of groups required.
     *
     * ALGORITHM: Greedy (try largest feasible group size downward; use groupify helper to validate)
     * TC: O(n * min_freq) | SC: O(n)
     *
     * Approach: Try to form groups of size `z` (from min_frequency down to 1).
     * For each size z, check via groupify if every frequency can be split into groups of size z or z+1.
     * The first valid z gives the minimum group count.
     */
    public int minGroupsForValidAssignment(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);
        int min = nums.length;
        for (int value : freq.values()) min = Math.min(min, value);

        for (int z = min; z >= 1; --z) {
            int result = groupify(z, freq);
            if (result > 0) return result;
        }
        return nums.length;
    }

    /*
     * PROBLEM: Groupify Frequency Map (Helper)
     * Try to partition every frequency in the map into groups of exactly `size` or `size+1`;
     * return the total group count, or 0 if the partition is impossible.
     *
     * ALGORITHM: Greedy modulo check per frequency bucket
     * TC: O(n) | SC: O(1)
     */
    private int groupify(int size, Map<Integer, Integer> freq) {
        int groups = 0;
        int next = size + 1;
        for (int value : freq.values()) {
            int rem = value % next;
            int numGroups = value / next;
            if (rem == 0) {
                groups += numGroups;
            } else if (numGroups >= size - rem) {
                groups += numGroups + 1;
            } else return 0;
        }
        return groups;
    }
}

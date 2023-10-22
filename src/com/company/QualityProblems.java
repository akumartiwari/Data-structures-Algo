package com.company;

import java.util.HashMap;
import java.util.Map;

public class QualityProblems {


    /*
    Input: nums = [3,2,3,2,3]
    Output: 2
    Explanation: One way the indices can be assigned to 2 groups is as follows, where the values in square brackets are indices:
    group 1 -> [0,2,4]
    group 2 -> [1,3]
    All indices are assigned to one group.
    In group 1, nums[0] == nums[2] == nums[4], so all indices have the same value.
    In group 2, nums[1] == nums[3], so all indices have the same value.
    The number of indices assigned to group 1 is 3, and the number of indices assigned to group 2 is 2.
    Their difference doesn't exceed 1.
    It is not possible to use fewer than 2 groups because, in order to use just 1 group, all indices assigned to that group must have the same value.
    Hence, the answer is 2.

    Algorithm:-
    We need to attempt to make max size group to minimise total number of groups.
    After division we can check if remaining elements can form a group by taking 1 element
    from adjacent full group. if so then add 1 more group
    else such a division is invalid (ie.Try with smaller size).
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

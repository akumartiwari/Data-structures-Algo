package com.company;

import java.util.HashMap;
import java.util.Map;

public class CountSubArraysWithGivenMedian {

    /*
      - int pivot: where nums[pivot] == k
      - Prefix sum array smaller: where smaller[i + 1] is the number of m so that nums[m] < k and 0 <= m <= i <= nums.length.

    0 <= i <= pivot <= j <= nums.length
    smaller[j + 1] - smaller[i] == (j - i) / 2
    2 * (smaller[j + 1] - smaller[i]) == (j - i) or (j - i - 1)
    2 * smaller[j + 1] - j == 2 * smaller[i] - i or 2 * smaller[i] - i - 1
    We define statistic[i] = 2 * smaller[i] - i,
    then statistic[j + 1] == statistic[i] or statistic[i] - 1.
    Use a map to store the count of each statistic[i], where 0 <= i <= pivot.
    And for each pivot <= j <= nums.length, add map.get(statistic[j + 1]) and map.get(statistic[j + 1] + 1) to the output.
     */
    public int countSubarrays(int[] nums, int k) {
        int n = nums.length;
        int[] smaller = new int[n + 1];  // smaller[i + 1]: number of elements smaller than k in nums[0...i]
        int pivot = -1;  // nums[pivot] == k
        for (int i = 0; i < n; i++) {
            smaller[i + 1] = smaller[i];
            if (nums[i] < k) {
                smaller[i + 1]++;
            } else if (nums[i] == k) {
                pivot = i;
            }
        }
        Map<Integer, Integer> map = new HashMap<>();  // statistics count map
        for (int i = 0; i <= pivot; i++) {
            int statistic = 2 * smaller[i] - i;
            map.put(statistic, map.getOrDefault(statistic, 0) + 1);
        }
        int count = 0;
        for (int j = pivot; j < n; j++) {
            int statistic = 2 * smaller[j + 1] - j;
            count += map.getOrDefault(statistic, 0);
            count += map.getOrDefault(statistic + 1, 0);
        }
        return count;
    }
}

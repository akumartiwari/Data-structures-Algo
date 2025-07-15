package com.company;

import java.util.*;
import java.util.stream.Collectors;

public class PractiseProblems {

    /*
    Check if str2 is subsequence of str1 after increment str1[i] to the next character cyclically at most once.
    Example1-
    Input: str1 = "abc", str2 = "ad"
    Output: true
    Explanation: Select index 2 in str1.
    Increment str1[2] to become 'd'.
    Hence, str1 becomes "abd" and str2 is now a subsequence. Therefore, true is returned.

    Example 2:
    Input: str1 = "zc", str2 = "ad"
    Output: true
    Explanation: Select indices 0 and 1 in str1.
    Increment str1[0] to become 'a'.
    Increment str1[1] to become 'd'.
    Hence, str1 becomes "ad" and str2 is now a subsequence. Therefore, true is returned.

     */
    public boolean canMakeSubsequence(String str1, String str2) {
        int i, j;
        for (i = 0, j = 0; i < str1.length() && j < str2.length(); i++) {
            char curr = str1.charAt(i), other = str2.charAt(j);
            char next = (char) ('a' + (((curr - 'a') + 1) % 26));
            if (curr == other || next == other) j++;
        }
        return j == str2.length();
    }

    /*
    Based on Recursion to traverse all possibilties
    Example 1-
    Input: nums = [2,1,3,2,1]
    Output: 3
    Explanation:
    One of the optimal solutions is to remove nums[0], nums[2] and nums[3].
    Example 2-
    nums = [2,3,1,2]
     */

    public int minimumOperations(List<Integer> nums) {
        int last = nums.get(0);
        int cnt = 0;
        for (int i = 1; i < nums.size(); i++) {
            if (nums.get(i) < last) cnt++;
            last = nums.get(i);
        }
        return cnt;
    }

    public List<Integer> intersection(int[][] nums) {
        Set<Integer> set1 = new HashSet<>();
        for (int[] num : nums) {
            Set<Integer> set2 = Arrays.stream(num).boxed().collect(Collectors.toCollection(HashSet::new));
            if (set1.isEmpty()) set1.addAll(set2);
            set1.retainAll(set2); // set1 now contains only common elements
        }
        return new ArrayList<>(set1).stream().sorted().collect(Collectors.toList());
    }

}

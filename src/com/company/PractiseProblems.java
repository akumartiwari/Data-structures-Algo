package com.company;

import java.lang.reflect.Array;
import java.util.*;
import java.util.HashMap;
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

    /*

    [35,39,65,101,101,54,1,111,8,107,96,90,91,54,
    115,36,46,76,111,39,29,122,4,113,101,73,125,39,124,33,82,39]


    {65=1, 1=1, 4=1, 8=1, 73=1, 76=1, 82=1, 90=1, 91=1, 29=1,
    96=1, 33=1, 35=1, 36=1, 101=3, 39=4, 107=1, 46=1, 111=2, 113=1, 115=1, 54=2, 122=1, 124=1, 125=1}

     */
    public boolean partitionArray(int[] nums, int k) {
        int n = nums.length;
        if (k == 1) return true;
        int v = n / k;
        if (n % k != 0) return false;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) map.put(num, map.getOrDefault(num, 0) + 1);
        if (map.keySet().size() < k) return false;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) if (entry.getValue() > v) return false;
        return true;
    }

    public int getLeastFrequentDigit(int n) {
        Map<Integer, Integer> tm = new HashMap<>();
        for (char c : String.valueOf(n).toCharArray()) {
            int d = c - '0';
            tm.put(d, tm.getOrDefault(d, 0) + 1);
        }

        int min = Integer.MAX_VALUE, freq = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            if (freq > entry.getValue()) {
                freq = entry.getValue();
                min = entry.getKey();
            }
            if (freq == entry.getValue()) {
                min = Math.min(min, entry.getKey());
            }
        }

        return min;
    }

    //Not a good quality problem, Can skip it
    public int score(String[] cards, char x) {
        Set<String> bothSameSet = new HashSet<>(), leftSameSet = new HashSet<>(), rightSameSet = new HashSet<>();
        int bothSame = 0, leftSame = 0, rightSame = 0;
        for (String card : cards) {
            char l = card.charAt(0), r = card.charAt(1);
            if (l == x && r == x) {
                bothSame++;
                bothSameSet.add(card);
            } else if (l == x) {
                leftSame++;
                leftSameSet.add(card);
            } else if (r == x) {
                rightSame++;
                rightSameSet.add(card);
            }
        }

        if (bothSame == 0 && leftSameSet.size() <= 1 && rightSameSet.size() <= 1) return 0;
        if (bothSame == 0 && (leftSame == 0 || rightSame == 0)) {
            if (leftSameSet.size() == 1 || rightSameSet.size() == 1) return 0;
            return Math.max(leftSame, rightSame) / 2;
        }

        if (bothSame == leftSame + rightSame) return bothSame;
        return Math.min(bothSameSet.size(), Math.abs(rightSameSet.size() - leftSameSet.size())) + Math.min(rightSameSet.size(), leftSameSet.size());
    }


    public int[] recoverOrder(int[] order, int[] friends) {
        int[] ans = new int[friends.length];
        Arrays.fill(ans, 0);
        int idx = 0;
        Set<Integer> set = Arrays.stream(friends).boxed().collect(Collectors.toSet());
        for (int o : order) {
            if (set.contains(o)) ans[idx++] = o;
        }
        return ans;
    }

    class Solution {
        private boolean possible(int[] arr, int k) {
                boolean[] dp = new boolean[k + 1];
                int[] prev = new int[k + 1];
                dp[0] = true;

                for (int i = 0; i < arr.length; i++) {
                    for (int j = k; j >= arr[i]; j--) {
                        if (dp[j - arr[i]]) {
                            dp[j] = true;
                            prev[j] = i; // Track the index of the element contributing to the sum
                        }
                    }
                }

                if (!dp[k]) return false; // If sum k is not achievable
                return true;
            }

        public boolean[] subsequenceSumAfterCapping(int[] nums, int k) {
            int n = nums.length, idx = 0;
            boolean[] ans  = new boolean[n];
            for (int x=1; x<=n ; x++){
                List<Integer> na = new ArrayList<>();
                for (int num: nums){
                      int nn = Math.min(num, x);
                      na.add(nn);
                }
                ans[idx++] = (na.stream().mapToInt(e -> e).toArray(), k);
            }
            return ans;
        }
    }


    //TODO: Fix stack overflow
    // add DP to optimise solution

        public int minimumOperations(List<Integer> nums) {
            int[] dp = new int[nums.size()];
            Arrays.fill(dp, -1);
            return mo(nums, 0, -1, 0);
        }

        private int mo(List<Integer> nums, int ind, int prev, int max) {
            // base case
            if (ind >= nums.size()) return 0;

            int cnt = 0 ;
            prev = nums.get(ind);
            //take
            if (nums.get(ind) > prev) cnt ++;
            else max = Math.max(cnt, max);
            mo(nums, ind++, prev, max);
            return max;
        }


}

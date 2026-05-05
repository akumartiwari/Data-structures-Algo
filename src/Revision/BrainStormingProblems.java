package com.company;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BrainStormingProblems {

    /*
     * PROBLEM: Longest Beautiful Substring of All Vowels (LeetCode 1763)
     * Find the longest substring containing all 5 vowels in sorted order a<e<i<o<u.
     *
     * ALGORITHM: Sliding Window
     * TC: O(n) | SC: O(1)
     */
    public int longestBeautifulSubstringOptimial(String word) {
        int max = 0, len = 1, cnt = 1;
        for (int i = 1; i < word.length(); i++) {
            if (word.charAt(i) == word.charAt(i - 1)) len++;
            else if (word.charAt(i) > word.charAt(i - 1)) {
                ++len;
                cnt++;
            } else {
                cnt = 1;
                len = 1;
            }
            if (cnt == 5) max = Math.max(max, len);
        }

        return max;
    }


    /*
     * PROBLEM: Put Marbles in Bags (LeetCode 2551)
     * Find the difference between max and min score when splitting weights array into k bags.
     *
     * ALGORITHM: Sort + Greedy (adjacent pair sums)
     * TC: O(n log n) | SC: O(n)
     */
    public long putMarbles(int[] weights, int k) {
        int n = weights.length - 1;
        long[] adj = new long[n];
        for (int i = 0; i < n; i++) adj[i] = weights[i] + weights[i + 1];

        // Choose k max/min choices
        Arrays.sort(adj);
        long res = 0L;
        for (int i = 0; i < k - 1; i++) res += adj[n - 1 - i] - adj[i];
        return res;
    }

    /*
     * PROBLEM: Lexicographically Smallest String After Substring Operation (LeetCode 2734)
     * Minimize total cost of rotating array elements, where each rotation costs x.
     *
     * ALGORITHM: Prefix Minimum + Simulation
     * TC: O(n^2) | SC: O(n)
     */
    public long minCost(int[] A, int x) {
        int n = A.length;
        long[] res = new long[n];
        for (int i = 0; i < n; i++) {
            res[i] += 1L * i * x;
            int cur = A[i];
            for (int k = 0; k < n; k++) {
                cur = Math.min(cur, A[(i - k + n) % n]);
                res[k] += cur;
            }
        }

        long min_res = Long.MAX_VALUE;
        for (long element : res) {
            min_res = Math.min(min_res, element);
        }

        return min_res;
    }

    /*
     * PROBLEM: Make Number of Distinct Characters Equal (LeetCode 2531)
     * Check if exactly one character swap between word1 and word2 makes their distinct char counts equal.
     *
     * ALGORITHM: Brute Force (26x26 combinations)
     * TC: O(26^2) | SC: O(26)
     */
    public boolean isItPossible(String word1, String word2) {
        int[] word1Cnt = new int[26];
        int[] word2Cnt = new int[26];

        int distinct1 = 0, distinct2 = 0;

        for (char c : word1.toCharArray()) {
            word1Cnt[c - 'a']++;
            if (word1Cnt[c - 'a'] == 1) distinct1++;
        }
        for (char c : word2.toCharArray()) {
            word2Cnt[c - 'a']++;
            if (word2Cnt[c - 'a'] == 1) distinct2++;
        }


        //Try all possible combinations i.e, 25*26;

        for (int i = 0; i < 26; i++) {
            char word1ToCharSwap = (char) (i + 'a');
            if (word1Cnt[i] == 0) continue;

            for (int j = 0; j < 26; j++) {

                char word2ToCharSwap = (char) (j + 'a');
                if (word2Cnt[j] == 0) continue;

                int distinctTemp1 = distinct1;
                int distinctTemp2 = distinct2;

                // make swap
                word1Cnt[i]--;
                word2Cnt[i]++;
                word1Cnt[j]++;
                word2Cnt[j]--;

                if (word1Cnt[i] == 0) distinctTemp1--;
                if (word2Cnt[i] == 1) distinctTemp2++;
                if (word1Cnt[j] == 1) distinctTemp1++;
                if (word2Cnt[j] == 0) distinctTemp2--;

                if (distinctTemp1 == distinctTemp2) return true;

                // clean moves
                word1Cnt[i]++;
                word2Cnt[i]--;
                word1Cnt[j]--;
                word2Cnt[j]++;

            }
        }
        return false;
    }

    /*
     * PROBLEM: Minimum Sum of Squared Difference (LeetCode 2333)
     * Minimize the sum of squared differences between nums1 and nums2 using at most k1+k2 operations.
     *
     * ALGORITHM: Counting Sort + Greedy
     * TC: O(n) | SC: O(max_diff)
     */
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


    /*
     * PROBLEM: Shortest Sequence of Rolls That Cannot Be Taken (LeetCode 2336)
     * Find the length of the shortest sequence that cannot be formed as a subsequence of rolls.
     *
     * ALGORITHM: Sliding Window with Set
     * TC: O(n) | SC: O(k)
     */
    public int shortestSequence(int[] rolls, int k) {
        int len = 1;
        Set<Integer> set = new HashSet<>();

        for (int roll : rolls) {
            set.add(roll);
            if (set.size() == k) {
                set.clear();
                len++;
            }
        }

        return len;
    }

    /*
     * PROBLEM: Check Whether Array Can Be Split (LeetCode 2811)
     * Determine if the array can be split such that all elements satisfy the constraint with value m.
     *
     * ALGORITHM: Greedy (adjacent pair check)
     * TC: O(n) | SC: O(1)
     */
    public boolean canSplitArray(List<Integer> nums, int m) {
        for (int i = 0; i < nums.size() - 1; i++) {
            if (nums.get(i) + nums.get(i + 1) >= m) return true;
        }

        return nums.size() < 3;
    }

}

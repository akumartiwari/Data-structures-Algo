package com.company;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BrainStormingProblems {


    /*
    First for any of distributions,
    we will sum up A[0] and A[n - 1] in the total socre,
    so this won't make any difference between max and min.

    To split into k bags,
    we actually choose k-1 cut points:
    A[0]...A[i1]
    A[i1+1]....A[i2]
    A[i2+1]....A[i3]
    ....
    A[ik+1]....A[n-1]

    The result score is:
    (A[0] + A[i1]) + (A[i2] + A[i3]) +..... + (A[ik+1] + A[n-1])
    equals to
    A[0] + (A[i1] + A[i1+1]) + (A[i2] + A[i2+1]) + ....

    So the problem turns out to be,
    calculate the max/min sum of k - 1 numbers in
    A[0] + A[1], A[1] + A[2],..., A[n-1] + A[n].

    We can simply sort them in O(sort)
    or we can apply a priority queue in O(nlogk).

    Input: weights = [1,3,5,1], k = 2
    Output: 4
    Explanation:
    The distribution [1],[3,5,1] results in the minimal score of (1+1) + (3+1) = 6.
    The distribution [1,3],[5,1], results in the maximal score of (1+3) + (5+1) = 10.
    Thus, we return their difference 10 - 6 = 4.
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
    If we do rotate operation k times,
    we can use the min(A[i-k], .. , A[i - 1], A[i]) as the cost for type i.

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

    Input: word1 = "ac", word2 = "b"
    Output: false
    Explanation: Any pair of swaps would yield two distinct characters in the first string, and one in the second string.
    ----------------------------------------------------------------------------------------------------------
    APPROACH -
    Because we are bound to lowercase English letters,
    we know the search space is 26 letters.
    We will try all possible combinations of swaps which is 26*26.

    So with that intuition, we have the following procedure:

    Count the initial frequencies of the strings
    Try all possible letter swaps
    After every letter swap, check if we have the same amount of distinct characters

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


    /*
    Input: rolls = [4,2,1,2,3,3,2,4,1], k = 4
    Output: 3
    Explanation: Every sequence of rolls of length 1, [1], [2], [3], [4], can be taken from rolls.
    Every sequence of rolls of length 2, [1, 1], [1, 2], ..., [4, 4], can be taken from rolls.
    The sequence [1, 4, 2] cannot be taken from rolls, so we return 3.
    Note that there are other sequences that cannot be taken from rolls.
     */
    //    The idea is to think how can you form  a all sequences of len=len starting
    //    form 1 ie. find the index, Now repeat the above process for len=2
    //    and so on.

    // TC = O(n)
    //Author: Anand
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
    Most Optimal:-
    If we have at least one adjacent pair of elements whose sum >= m then we can chip (slice 1 element from left or right) from either side of array .
     */
    public boolean canSplitArray(List<Integer> nums, int m) {
        for (int i = 0; i < nums.size() - 1; i++) {
            if (nums.get(i) + nums.get(i + 1) >= m) return true;
        }

        return nums.size() < 3;
    }

}

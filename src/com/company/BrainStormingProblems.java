package com.company;

import java.util.HashSet;
import java.util.Set;

public class BrainStormingProblems {

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

        for (int i = 0; i < rolls.length; i++) {
            set.add(rolls[i]);
            if (set.size() == k) {
                set.clear();
                len++;
            }
        }

        return len;
    }
}

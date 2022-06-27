package com.company;

import java.util.Arrays;

public class KadaneAlgo {
    /*
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: [4,-1,2,1] has the largest sum = 6.
     */

    // kadan's algorithm
    public int maxSubArray(int[] arr) {
        int n = arr.length;
        if (n == 0) return 0;

        int maxSoFar = Integer.MIN_VALUE;
        int maxEndingHere = 0;

        for (int i = 0; i < n; i++) {
            maxEndingHere += arr[i];
            maxSoFar = Math.max(maxEndingHere, maxSoFar);
            if (maxEndingHere < 0) maxEndingHere = 0;
        }

        return maxSoFar;
    }


    /*
    Input: s = "aababbb"
    Output: 3
    Explanation:
    All possible variances along with their respective substrings are listed below:
    - Variance 0 for substrings "a", "aa", "ab", "abab", "aababb", "ba", "b", "bb", and "bbb".
    - Variance 1 for substrings "aab", "aba", "abb", "aabab", "ababb", "aababbb", and "bab".
    - Variance 2 for substrings "aaba", "ababbb", "abbb", and "babb".
    - Variance 3 for substring "babbb".
    Since the largest possible variance is 3, we return it.

     */
    // TC = O(26*26*n) < TC = O(10^7)
    public int largestVariance(String s) {
        int[] freq = new int[26];
        for (int i = 0; i < s.length(); i++) freq[(int) (s.charAt(i) - 'a')]++;

        int result = 0;
        for (int a = 0; a < 26; a++) {
            for (int b = 0; b < 26; b++) {

                int rema = freq[a];
                int remb = freq[b];
                if (a == b || rema == 0 || remb == 0) continue;

                // Apply kadane algo to find the maximum length
                // on each possible pair of characters ie, {a, b}
                int cfa = 0, cfb = 0;
                for (int i = 0; i < s.length(); i++) {
                    int cc = (int) (s.charAt(i) - 'a');

                    if (cc == b) {
                        cfb++;
                    } else if (cc == a) {
                        cfa++;
                        rema--;
                    }


                    if (cfa > 0) result = Math.max(result, cfb - cfa);
                    if (cfb < cfa && rema >= 1) {
                        cfa = 0;
                        cfb = 0;
                    }
                }
            }
        }

        return result;

    }


    /*
    Input: nums1 = [60,60,60], nums2 = [10,90,10]
    Output: 210
    Explanation: Choosing left = 1 and right = 1, we have nums1 = [60,90,60] and nums2 = [10,60,10].
    The score is max(sum(nums1), sum(nums2)) = max(210, 80) = 210.

    Input: nums1 = [20,40,20,70,30], nums2 = [50,20,50,40,20]
    Output: 220
    Explanation: Choosing left = 3, right = 4, we have nums1 = [20,40,20,40,20] and nums2 = [50,20,50,70,30].
    The score is max(sum(nums1), sum(nums2)) = max(140, 220) = 220.

     */
    //Author: Anand
    public int maximumsSplicedArray(int[] nums1, int[] nums2) {
        int s1 = Arrays.stream(nums1).sum();
        int s2 = Arrays.stream(nums2).sum();
        int[] v1 = new int[nums1.length], v2 = new int[nums1.length];

        for (int i = 0; i < nums1.length; i++) {
            v1[i] = nums1[i] - nums2[i];
            v2[i] = nums2[i] - nums1[i];
        }

        return Math.max(s1 + maxSubArray(v2), s2 + maxSubArray(v1));
    }


}

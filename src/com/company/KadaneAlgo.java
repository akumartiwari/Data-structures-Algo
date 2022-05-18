package com.company;

public class KadaneAlgo {
    /*
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: [4,-1,2,1] has the largest sum = 6.
     */

    // kadane algo
    public int maxSubArray(int[] nums) {
        int meh = 0, mss = Integer.MIN_VALUE;
        for (int num : nums) {
            meh += num;
            mss = Math.max(mss, meh);
            if (meh < 0) meh = 0;
        }
        return mss;
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
        for (int i = 0; i < s.length(); i++) freq[(int)(s.charAt(i) - 'a')]++;

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
                    int cc = (int)(s.charAt(i) - 'a');

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
}

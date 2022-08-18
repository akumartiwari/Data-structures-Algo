package com.company;

import java.util.*;

public class Bitmask {

    //Author: Anand
    // The idea is to count numbers that share same bit and maximise them
    public int largestCombination(int[] candidates) {
        int max = Integer.MIN_VALUE;
        for (int c : candidates) max = Math.max(max, c);

        int ans = 0;
        // check for every bit and count numbers that share same bit
        for (int b = 1; b <= max; b <<= 1) {
            int count = 0;
            for (int c : candidates) {
                if ((c & b) > 0) count++;
            }
            ans = Math.max(ans, count);
        }
        return ans;
    }

    // Author: Anand
    public int wordCount(String[] startWords, String[] targetWords) {
        Set<Integer> bitmasks = new HashSet<>();
        for (String word : startWords) {
            int mask = bitmask(word);
            for (char c = 'a'; c <= 'z'; c++) {
                int newMask = mask | 1 << c - 'a';
                if (newMask != mask) bitmasks.add(newMask);
            }
        }

        int count = 0;
        for (String word : targetWords)
            if (bitmasks.contains(bitmask(word))) count++;
        return count;
    }

    private int bitmask(String word) {
        int mask = 0;
        for (int i = 0; i < word.length(); i++)
            mask |= 1 << (word.charAt(i) - 'a');
        return mask;
    }

    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int mask = 1;
        int bits = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) == 1) bits++;
            mask <<= 1;
        }
        return bits;

        //return Integer.bitCount(n);
    }

    //Author: Anand
    public String smallestNumber(String pattern) {
        return String.valueOf(sn(pattern, 0, 0, 0));
    }

    private int sn(String pattern, int pos, int mask, int num) {
        // base case
        if (pos > pattern.length()) return num;

        int res = Integer.MAX_VALUE, last = num % 10;
        boolean increment = pos == 0 || pattern.charAt(pos - 1) == 'I';

        for (int d = 1; d <= 9; ++d)
            // if curr digit is not taken
            if ((mask & 1 << d) == 0 && d > last == increment)
                res = Math.min(res, sn(pattern, pos + 1, mask | 1 << d, num * 10 + d));

        return res;
    }


    class Solution {
        //Author: Anand
        public int countSpecialNumbers(int n) {
            int cnt = 0;
            int d = 0;
            int number = n;

            while (number > 0) {
                d++;
                number /= 10;
            }

            int[] dp = new int[d+1];
            Arrays.fill(dp, -1);
            for (int i = 1; i <= d; i++) {
                int ans = f(i, n, new StringBuilder(), 0, dp);
                cnt += ans;
                dp[i] = ans;
            }
            return cnt;
        }

        private int f(int digits, int n, StringBuilder number, int mask, int[] dp) {

            // base case
            if (digits <= 0) return 1;
            int cnt = 0;

            if (dp[digits] != -1) return dp[digits];
            for (int i = 0; i <= 9; i++) {
                int ind = i;
                if (digits == 1) {
                    if (i == 9) continue;
                    ind = i + 1;
                }

                number.insert(0, ind);
                if (number.toString().equals("") || ((Long.parseLong(number.toString()) <= n) && (mask & 1 << ind) == 0)) {
                    cnt += f(digits - 1, n, number, mask | (1 << ind), dp);
                }
                // backtrack
                number.deleteCharAt(0);
            }
            return dp[digits] = cnt;
        }
    }

}


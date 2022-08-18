package com.company;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

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
}


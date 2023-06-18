package com.company;

import java.util.HashSet;
import java.util.Set;

public class Bitmask {


    //Author:Anand
      /*
        We use the sliding window approach, tracking used bits. We can use OR to combine bits, XOR to remove bits
        If the next number has a conflicting bit (used & nums[i] != 0),
        we shrink the window until there are no conflicts.

        Why xor is working is because xor will only make those bits 1 which are different.
        So suppose we have 11001 and we want to remove 9 - 1001 so . 11001^1001 will give 10000 ,
        and we can see that only those bits are off which we wanted to remove.
       */

    int mod = (int) (Math.pow(10, 9) + 7);

    public int longestNiceSubarray(int[] nums) {
        int res = 0, j = 0, used = 0;
        for (int i = 0; i < nums.length; i++) {
            while ((used & nums[i]) != 0)
                used ^= nums[j++];

            used |= nums[i];
            res = Math.max(res, i - j + 1);
        }
        return res;
    }

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

    public int specialPerm(int[] nums) {
        return countPerm(nums, 0, 0, -1, new HashSet<>());
    }

    private int countPerm(int[] arr, int len, int ind, int le, Set<Integer> taken) {
        // base case
        if (len == arr.length) return 1;
        if (len > arr.length) return 0;
        int take = 0, nt = 0;

        for (int i = ind; i < arr.length; i++) {
            // take
            if (!taken.contains(arr[i])) taken.add(arr[i]);
            else continue;

            if (le == -1) le = arr[i];

            for (int k : arr) {
                if (arr[i] != k && (le % k == 0 || k % le == 0)) {
                    taken.add(k);
                    take = (take + countPerm(arr, (len == 0 ? len + 2 : len + 1), i + 1, k, taken) % mod);

                    // // not take
                    // taken.remove(k);
                    // nt += countPerm(arr, len, i + 1, le, taken) % mod;
                }

                // not take
                nt += countPerm(arr, len, i + 1, le, taken) % mod;
            }
        }

        return (take + nt) % mod;
    }

}


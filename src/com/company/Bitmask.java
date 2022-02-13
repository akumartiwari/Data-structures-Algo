package com.company;

import java.util.HashSet;
import java.util.Set;

public class Bitmask {
    // TODO: Understand Bitmasking
    public int wordCount(String[] startWords, String[] targetWords) {
        Set<Integer> masks = new HashSet<>();
        for (String s : startWords) {
            int mask = bitmask(s);
            for (char c = 'a'; c <= 'z'; c++) {
                int maskPlus = mask | 1 << c - 'a';
                if (maskPlus != mask) masks.add(maskPlus);
            }
        }
        int count = 0;
        for (String t : targetWords) {
            if (masks.contains(bitmask(t))) count++;
        }
        return count;
    }

    int bitmask(String s) {
        char[] ca = s.toCharArray();
        int mask = 0;
        for (char c : ca) {
            mask |= 1 << c - 'a';
        }
        return mask;
    }

    /*
    Intuition
    DP solution, but need to figure out how to define bit mask.

    I considered two options

    Use a mask on base-3
    Use a mask on base-2, and each slot takes two bits.
    I feel no big difference on code length,
    and base-3 has slightly better complexity.


    Optimization
    Usually the bitmask dp will be in format of dp(i, mask) = max(current, dp(i, mask - bit) + value)
    and we memorized dp(i, mask).
    Actually in this problem, I only memorized mask, without caring about i.
    Since mask has the information of the slots used


    Explanation
    We recursively check dp(i, mask),
    meanning we are going to assign A[i] with current bitmask mask.
    dp(i, mask) is the biggest AND sum we can get, by assign A[0] to A[i] into the remaining slots.

    We iterate all slots, the corresponding bit = 3 ** (slot - 1).
    Ther check if this slot is availble.
    If it's available, we take it and recursively check dp(i - 1, mask - bit).


    Complexity
    Time O(ns * 3^ns)
    Space O(3^ns)
     */
    public int maximumANDSum(int[] A, int ns) {
        int mask = (int) Math.pow(3, ns) - 1;
        int[] memo = new int[mask + 1];
        return dp(A.length - 1, mask, ns, memo, A);
    }

    private int dp(int i, int mask, int ns, int[] memo, int[] A) {
        if (memo[mask] > 0) return memo[mask];
        if (i < 0) return 0;
        for (int slot = 1, bit = 1; slot <= ns; ++slot, bit *= 3)
            if (mask / bit % 3 > 0)
                memo[mask] = Math.max(memo[mask], (A[i] & slot) + dp(i - 1, mask - bit, ns, memo, A));
        return memo[mask];
    }

}


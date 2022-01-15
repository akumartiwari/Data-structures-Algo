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
}

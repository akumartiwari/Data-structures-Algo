package com.company;

import java.util.HashSet;
import java.util.Set;

public class Bitmask {

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
}


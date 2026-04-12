package com.company;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Anagram {

    /*
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Find Resultant Array After Removing Anagrams (LeetCode 2273)
     * ──────────────────────────────────────────────────────────────────
     * Given a 0-indexed string array `words`, repeatedly perform the
     * following operation until no more can be applied:
     *   - Find any adjacent pair (words[i-1], words[i]) that are anagrams
     *     of each other, and DELETE words[i].
     * Return the final array after all such deletions.
     *
     * Note: Two strings are anagrams if they contain the same characters
     * with the same frequencies (e.g. "abba" and "baab" are anagrams).
     *
     * Example:
     *   Input:  words = ["abba", "baba", "bbaa", "cd", "cd"]
     *   Output: ["abba", "cd"]
     *   - "abba" & "baba" are anagrams → remove "baba" → ["abba","bbaa","cd","cd"]
     *   - "abba" & "bbaa" are anagrams → remove "bbaa" → ["abba","cd","cd"]
     *   - "cd"   & "cd"   are anagrams → remove "cd"   → ["abba","cd"]
     *   - No more adjacent anagram pairs → done.
     *
     * ────────────────────────��─────────────────────────────────────────
     * SOLUTION — Iterative Adjacent Scan with Frequency Map:
     * ──────────────────────────────────────────────────────────────────
     *   1. Convert the input array to a mutable List.
     *   2. Outer while-loop: keep scanning until a full pass finds NO removal.
     *   3. Inner for-loop: for each adjacent pair (i, i+1), check if they
     *      are anagrams using the helper method `ana()`.
     *        - If YES → remove the element at i+1, set flag=true, and
     *          break to restart the scan (indices have shifted).
     *        - If NO  → continue to the next pair.
     *   4. If a full pass completes with flag=false → no adjacent anagrams
     *      remain; exit and return the list.
     *
     * Helper `ana(word1, word2)`:
     *   - Return false immediately if lengths differ (anagrams must be same length).
     *   - Build a character-frequency map from word1.
     *   - Walk through word2: decrement each character's count in the map;
     *     if a character is missing or its count would go negative, return false.
     *   - If the map is empty after processing word2 → they are anagrams.
     *
     * Time:  O(n² · L)  — up to n passes over the list, each O(n·L),
     *                      where L = average word length
     * Space: O(L)       — character frequency map per comparison
     * ──────────────────────────────────────────────────────────────────
     */
    // Author: Anand
    public List<String> removeAnagrams(String[] words) {

        List<String> list = Arrays.stream(words).collect(Collectors.toList());
        while (list.size() > 1) {

            boolean flag = false;
            for (int i = 0; i < list.size() - 1; i++) {
                if (ana(list.get(i), list.get(i + 1))) {
                    flag = true;
                    list.remove(list.get(i + 1));
                    break;
                }
            }

            if (!flag) break;
        }


        return list;
    }

    private boolean ana(String word1, String word2) {

        if (word1.length() != word2.length()) return false;

        Map<Character, Integer> freq = new HashMap<>();
        for (int i = 0; i < word1.length(); i++) freq.put(word1.charAt(i), freq.getOrDefault(word1.charAt(i), 0) + 1);

        for (int i = 0; i < word2.length(); i++) {
            Character key = word2.charAt(i);
            if (freq.containsKey(key)) {
                freq.put(key, freq.get(key) - 1);
                if (freq.get(key) <= 0) {
                    freq.remove(key);
                }
            } else return false;
        }
        return true;
    }
}

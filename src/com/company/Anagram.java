package com.company;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Anagram {
    /*
    list = ["abba","cd","cd"]
    Input: words = ["abba","cd"]
    Output: ["abba","cd"]
    */
    //Author: Anand
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

package com.company;

import java.util.*;

public class WordLadderBFS {
    public int ladderLength(String start, String end, List<String> wordList) {
        Set<String> dict = new HashSet<>(wordList);

        dict.add(start);
        Map<String, Integer> distances = new HashMap<>();
        return bfsHelper(start, end, dict, distances);

    }

    // use BFS, and traverse each word and its neighbors
    // if we find any word equals to end, we've return the "level" -- distances -- length
    private int bfsHelper(String start,
                          String end,
                          Set<String> dict,
                          Map<String, Integer> distances) {
        Queue<String> queue = new LinkedList<>();
        queue.offer(start);
        distances.put(start, 0); // distance keep track of how far it is from start

        while (!queue.isEmpty()) {
            String word = queue.poll();
            int distance = distances.get(word);

            for (String nextWord : getNextWords(word, dict)) {
                if (!distances.containsKey(nextWord)) {
                    queue.offer(nextWord);
                    distances.put(nextWord, distance + 1);
                }
            }
        }

        printMap(distances);

        if (distances.containsKey(end)) {
            return distances.get(end) + 1;
        }
        return 0;
    }

    private void printMap(Map<String, Integer> visited) {
        for (String s : visited.keySet()) {
            System.out.println(s + " : " + visited.get(s));
        }
    }

    private List<String> getNextWords(String word, Set<String> dict) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            for (int j = 0; j < 26; j++) {
                if (ch != (char) ('a' + j)) {
                    // substitute word.charAt(i) using (char)('a' + j)
                    String nextWord = word.substring(0, i) + (char) ('a' + j) + word.substring(i + 1);
                    if (dict.contains(nextWord)) {
                        res.add(nextWord);
                    }
                }
            }
        }
        return res;
    }
}

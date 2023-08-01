package com.company;

import java.util.HashMap;import java.util.*;
import java.util.stream.Collectors;

public class TrieExample {
    class TrieNode {
        int count; // keeps count of words ending at current node;
        TrieNode[] letter;

        public TrieNode() {
            count = 0;
            letter = new TrieNode[26];
        }

        public void add(String word) {
            TrieNode p = this;
            for (int i = 0; i < word.length(); i++) {
                char ch = word.charAt(i);
                if (p.letter[ch - 97] == null) {
                    p.letter[ch - 97] = new TrieNode();
                }
                p = p.letter[ch - 97];
            }
            p.count++;
        }
    }

    class Encrypter {
        char[] keys;
        String[] values;
        String[] dictionary;
        Set<String> setd;
        Map<Character, String> map;
        Map<String, Set<Character>> revMap;
        TrieNode root;

        // TC = O(n), SC = O(n)
        public Encrypter(char[] keys, String[] values, String[] dictionary) {
            this.keys = keys;
            this.values = values;
            this.dictionary = dictionary;
            this.setd = Arrays.stream(this.dictionary).collect(Collectors.toSet());
            map = new HashMap<>();
            revMap = new HashMap<>();
            root = new TrieNode();
            for (int i = 0; i < keys.length; i++) map.put(keys[i], values[i]);
            for (int i = 0; i < values.length; i++) {
                if (revMap.containsKey(values[i])) {
                    Set<Character> exist = revMap.get(values[i]);
                    exist.add(keys[i]);
                    revMap.put(values[i], exist);
                } else {
                    Set<Character> chars = new HashSet<>();
                    chars.add(keys[i]);
                    revMap.put(values[i], chars);
                }
            }

            for (String word : dictionary) {
                root.add(encrypt(word));
            }
        }


        /*
        For each character c in the string, we find the index i satisfying keys[i] == c in keys.
        Replace c with values[i] in the string.
        */
        // TC = O(n), SC = O(n)
        public String encrypt(String word1) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < word1.length(); i++) {
                char key = word1.charAt(i);
                if (map.containsKey(key)) sb.append(map.get(key));
            }
            return sb.toString();
        }

        /*
        1.) For each substring s of length 2 occurring at an even index in the string, we find an i such that values[i] == s. If there are multiple valid i, we choose any one of them. This means a string could have multiple possible strings it can decrypt to.
        2.) Replace s with keys[i] in the string.
         */
        // TC = O(n2), SC = O(n)
        public int decrypt(String word2) {
            TrieNode p = root;
            for (int i = 0; i < word2.length(); i++) {
                p = p.letter[word2.charAt(i) - 97];
                if (p == null) return 0;
            }
            return p.count; // count of number of words ending at that node
        }
    }

/**
 * Your Encrypter object will be instantiated and called as such:
 * Encrypter obj = new Encrypter(keys, values, dictionary);
 * String param_1 = obj.encrypt(word1);
 * int param_2 = obj.decrypt(word2);
 */
}

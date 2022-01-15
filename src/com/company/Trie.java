package com.company;

import java.util.Arrays;

public class Trie {
    static Node root;

    public static void main(String[] args) {
        String[] startWords = {"ant", "act", "tack"};
        String[] targetWords = {"tack", "act", "acti"};
        System.out.println(wordCount(startWords, targetWords));
    }


    // Algorithmic based
    public static int wordCount(String[] startWords, String[] targetWords) {

        root = new Node();
        for (String str : startWords) {
            char[] arr = str.toCharArray();
            Arrays.sort(arr);
            insert(root, String.valueOf(arr));
        }

        int ans = 0;
        for (String target : targetWords) {
            for (int j = 0; j < target.length(); j++) {
                String newStr = target.substring(0, j) + target.substring(j + 1);
                char[] arr = newStr.toCharArray();
                Arrays.sort(arr);
                if (search(root, String.valueOf(arr))) {
                    ans++;
                    break;
                }
            }
        }
        return ans;
    }

    // create a Trie
    static class Node {
        boolean isEnd;
        Node[] child = new Node[26];

        Node() {
            isEnd = false;
            for (int i = 0; i < 26; i++) {
                child[i] = null;
            }
        }
    }

    public static void insert(Node root, String str) {

        int n = str.length();
        Node temp = root;

        for (int i = 0; i < n; i++) {
            int index = str.charAt(i) - 'a';// char index to be inserted
            if (temp.child[index] == null) temp.child[index] = new Node();

            temp = temp.child[index];
        }
        temp.isEnd = true;
    }

    public static boolean search(Node root, String str) {

        int n = str.length();
        int i = 0;
        Node temp = root;
        while (i < n) {
            int index = str.charAt(i) - 'a';
            if (temp.child[index] == null) return false;
            temp = temp.child[index];
            i++;
        }

        return temp.isEnd;
    }
}

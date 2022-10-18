package com.company;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

class SolutionTrie {

    // Alphabet size (# of symbols)
    static final int ALPHABET_SIZE = 26;

    static class TrieNode {
        TrieNode[] children = new TrieNode[ALPHABET_SIZE];

        // isEndOfWord is true if the node represents
        // end of a word
        boolean isEndOfWord;

        TrieNode() {
            isEndOfWord = false;
            for (int i = 0; i < ALPHABET_SIZE; i++)
                children[i] = null;
        }
    }

    static TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public SolutionTrie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String key) {
        int level;
        int length = key.length();
        int index;

        TrieNode pCrawl = root;

        for (level = 0; level < length; level++) {
            index = key.charAt(level) - 'a';
            if (pCrawl.children[index] == null)
                pCrawl.children[index] = new TrieNode();

            pCrawl = pCrawl.children[index];
        }

        // mark last node as leaf
        pCrawl.isEndOfWord = true;
    }


    /**
     * Returns if the word is in the trie.
     */
    public String search(String key) {
        int level;
        int length = key.length();
        int index;
        StringBuilder ans = new StringBuilder();
        TrieNode pCrawl = root;

        char last = '0';
        for (level = 0; level < length; level++) {

            if (last != key.charAt(level)) {
                index = key.charAt(level) - 'a';
                if (pCrawl.children[index] == null) {
                    if (pCrawl.isEndOfWord) return ans.toString();
                    else return "";
                }


                ans.append(key.charAt(level));
                pCrawl = pCrawl.children[index];
            } else return String.valueOf(last);
        }

        return ans.toString();
    }


    public String replaceWords(List<String> dictionary, String sentence) {
        if (dictionary.size() == 0) return "";
        String[] words = sentence.split(" ");

        for (String dic : dictionary) {
            insert(dic);
        }

        StringBuilder finalAns = new StringBuilder("");
        for (String word : words) {
            String root = search(word);
            if (root.length() > 0) {
                finalAns.append(root).append(" ");
            } else finalAns.append(word).append(" ");
        }

        return finalAns.toString().trim();
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String key) {
        int level;
        int length = key.length();
        int index;
        TrieNode pCrawl = root;

        for (level = 0; level < length; level++) {
            index = key.charAt(level) - 'a';

            if (pCrawl.children[index] == null)
                return false;

            pCrawl = pCrawl.children[index];
        }

        return true;
    }

    /**
     * Your Trie object will be instantiated and called as such:
     * Trie obj = new Trie();
     * obj.insert(word);
     * boolean param_2 = obj.search(word);
     * boolean param_3 = obj.startsWith(prefix);
     */


    static class WordDictionary {
        // Alphabet size (# of symbols)
        static final int ALPHABET_SIZE = 26;
        static TrieNode root;

        static class TrieNode {
            TrieNode[] children = new TrieNode[ALPHABET_SIZE];

            // isEndOfWord is true if the node represents
            // end of a word
            boolean isEndOfWord;

            TrieNode() {
                isEndOfWord = false;
                for (int i = 0; i < ALPHABET_SIZE; i++)
                    children[i] = null;
            }
        }

        /**
         * Initialize your data structure here.
         */
        public WordDictionary() {
            root = new TrieNode();
        }

        public void addWord(String word) {
            int level;
            int length = word.length();
            int index;

            TrieNode pCrawl = root;

            for (level = 0; level < length; level++) {
                if (word.charAt(level) == '.') index = pCrawl.children.length - 1;
                else index = word.charAt(level) - 'a';
                if (pCrawl.children[index] == null)
                    pCrawl.children[index] = new TrieNode();

                pCrawl = pCrawl.children[index];
            }

            // mark last node as leaf
            pCrawl.isEndOfWord = true;
        }

        public boolean searchUtil(String word, int indx, TrieNode ptr) {
            if (indx == word.length()) {
                return ptr.isEndOfWord;
            }
            if (word.charAt(indx) != '.') {
                if (ptr.children[word.charAt(indx) - 'a'] == null) {
                    return false;
                }
                if (searchUtil(word, indx + 1, ptr.children[word.charAt(indx) - 'a'])) {
                    return true;
                }
            } else {
                for (int i = 0; i < ALPHABET_SIZE; i++) {
                    if (ptr.children[i] != null) {
                        if (searchUtil(word, indx + 1, ptr.children[i])) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        public boolean search(String word) {
            int len = word.length();
            TrieNode ptr = root;
            return searchUtil(word, 0, ptr);
        }
    }

    /**
     * Your WordDictionary object will be instantiated and called as such:
     * WordDictionary obj = new WordDictionary();
     * obj.addWord(word);
     * boolean param_2 = obj.search(word);
     */


/*
class WordFilter {

    public WordFilter(String[] words) {

        int n = words.length;
        if (n == 0) return;
        for (String word : words) {




        }
    }

    public int f(String prefix, String suffix) {

    }
}
 */

        /*
        Input: dist = [1,1,2,3], speed = [1,1,1,1]
Output: 1
Input: dist = [3,2,4], speed = [5,3,2]
Output: 1

         */

    class Solution1 {
        private TrieNode root;

        public Solution1() {
            root = new TrieNode();
        }

        public String replaceWords(List<String> dictionary, String sentence) {
            for (String root : dictionary) {
                addRoot(root);
            }
            String[] words = sentence.split(" ");
            String[] result = new String[words.length];
            for (int i = 0; i < words.length; i++) {
                char[] chars = words[i].toCharArray();
                TrieNode node = root;
                StringBuilder rootWordBuilder = new StringBuilder();
                for (char c : chars) {
                    if (!node.containsKey(c) || node.isEnd()) {
                        break;
                    }
                    rootWordBuilder.append(c);
                    node = node.get(c);
                }
                result[i] = rootWordBuilder.length() <= 0 || !node.isEnd() ? words[i] : rootWordBuilder.toString();
            }
            return String.join(" ", result);
        }

        public void addRoot(String rootWord) {
            TrieNode node = root;
            char[] chars = rootWord.toCharArray();
            for (char c : chars) {
                if (!node.containsKey(c)) {
                    node.add(c);
                }
                node = node.get(c);
            }
            node.setEnd();
        }

        class TrieNode {
            private TrieNode[] children;
            private boolean isEnd;

            public TrieNode() {
                children = new TrieNode[26];
            }

            public void add(char c) {
                children[c - 'a'] = new TrieNode();
            }

            public boolean containsKey(char c) {
                return children[c - 'a'] != null;
            }

            public TrieNode get(char c) {
                return children[c - 'a'];
            }

            public boolean isEnd() {
                return this.isEnd;
            }

            public void setEnd() {
                this.isEnd = true;
            }
        }
    }

    public int eliminateMaximum(int[] dist, int[] speed) {
        int ans = 0;
        List<Integer> list = Arrays.stream(dist).boxed().sorted().collect(Collectors.toList());

        for (int i = 0; i < speed.length; i++) {
            int s = speed[i];
            int elem = list.get(0);
            if (Arrays.binarySearch(list.stream().mapToInt(Integer::intValue).toArray(), 0) == -1) {
                if (s >= elem) {
                    list.remove(0);
                    ans++;
                }
                Collections.sort(list);
                list.forEach(x -> x = x - 1);
            } else break;
        }
        return ans;
    }


   /*
    public boolean canReach(String s, int minJump, int maxJump) {
        int n = s.length();
        if (n == 0) return false;

        Set<Integer> landingPos = new HashSet<>(); // to store valid landing positions
        landingPos.add(0);

        while (landingPos.size() > 0) {
            Optional<Integer> pos = landingPos.stream().findFirst();
            for (int j = minJump; j <= Math.min(pos.get() + maxJump, n - 1) && s.charAt(j) == '0'; j++) {
                landingPos.add(j);
            }
            if (landingPos.contains(n - 1)) return true;
        }
        return false;
    }

    */

    public boolean canReach(String s, int minJump, int maxJump) {
       /*
        int n = s.length();
        if (n == 0) return false;

        Set<Integer> landingPos = new HashSet<>(); // to store valid landing positions
        landingPos.add(0);

        while (landingPos.size() > 0) {
            Optional<Integer> pos = landingPos.stream().findFirst();

            for (int j = (pos.get() + minJump); j <= (pos.get() + Math.min(pos.get() + maxJump, n - 1)) && s.charAt(j) == '0'; j++) {
                landingPos.add(j);
            }
            if (landingPos.contains(n - 1)) return true;
            landingPos.remove(pos.get());
        }
        return false;
        */

        int n = s.length();
        int[] near = new int[n + 1];
        Arrays.fill(near, 0);
        int index = 0, walker = 0, curPos = 0;

        for (int i = 1; i < n; i++) {
            // whops getting out of range, start jumping
            // at next nearest possible index
            if (i > curPos + maxJump) {
                curPos = near[++index];
                // can't make anymore jumps
                if (curPos == 0) return false;
            }
            if (s.charAt(i) == '0' && i >= curPos + minJump) {
                near[++walker] = i;
            }
        }
        return near[walker] == n - 1;

    }


}

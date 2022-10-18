package com.company;

class WordFilter {

    /*
    static class Pair implements Comparable<WordFilter.Pair> {
        int key;
        String val;

        Pair(int key, String val) {
            this.key = key;
            this.val = val;
        }

        @Override
        public int compareTo(Pair pair) {
            return this.key - pair.key;
        }
    }

    private WordFilter.TrieNode root;

    PriorityQueue<Pair> pq = new PriorityQueue<Pair>(); // max PQ
    int index = 0;

    public void addRoot(String rootWord) {
        root = new WordFilter.TrieNode();
        WordFilter.TrieNode node = root;
        char[] chars = rootWord.toCharArray();
        for (char c : chars) {
            if (!node.containsKey(c)) {
                node.add(c);
            }
            node = node.get(c);
        }
        node.setEnd();
        pq.add(new Pair(index, rootWord));
        index++;
    }


    class TrieNode {
        private WordFilter.TrieNode[] children;
        private boolean isEnd;

        public TrieNode() {
            children = new WordFilter.TrieNode[26];
        }

        public void add(char c) {
            children[c - 'a'] = new WordFilter.TrieNode();
        }

        public boolean containsKey(char c) {
            return children[c - 'a'] != null;
        }

        public WordFilter.TrieNode get(char c) {
            return children[c - 'a'];
        }

        public boolean isEnd() {
            return this.isEnd;
        }

        public void setEnd() {
            this.isEnd = true;
        }
    }

    public WordFilter(String[] words) {
        for (String word : words) {
            addRoot(word);
        }
    }

    public int f(String prefix, String suffix) {

        Optional<Pair> linkedHashSet = pq.stream().filter(x -> x.val.startsWith(prefix)
                && x.val.endsWith(suffix)).max((x, y) -> x.key - y.key);

        if (linkedHashSet.isPresent()) return linkedHashSet.get().key;
        else return 0;
    }
    */
    TrieNode trie;

    public WordFilter(String[] words) {
        trie = new TrieNode();
        for (int weight = 0; weight < words.length; ++weight) {
            String word = words[weight] + "{";
            for (int i = 0; i < word.length(); ++i) {
                TrieNode cur = trie;
                cur.weight = weight;
                for (int j = i; j < 2 * word.length() - 1; ++j) {
                    int k = word.charAt(j % word.length()) - 'a';
                    if (cur.children[k] == null)
                        cur.children[k] = new TrieNode();
                    cur = cur.children[k];
                    cur.weight = weight;
                }
            }
        }
    }

    public int f(String prefix, String suffix) {
        TrieNode cur = trie;
        for (char letter : (suffix + '{' + prefix).toCharArray()) {
            if (cur.children[letter - 'a'] == null) return -1;
            cur = cur.children[letter - 'a'];
        }
        return cur.weight;
    }

    class TrieNode {
        TrieNode[] children;
        int weight;

        public TrieNode() {
            children = new TrieNode[27];
            weight = 0;
        }
    }
}

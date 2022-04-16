package com.company;

import java.util.*;

// TODO: Study DSU + Bitmasking
public class GraphDSU {
    public static void main(String[] args) {
        System.out.println(Arrays.toString(groupStrings(new String[]{"a", "b", "ab", "cde"})));
    }


    public static int[] groupStrings(String[] words) {
        int n = words.length;
        // System.out.println(n);
        UnionFind uf = new UnionFind(n);

        // map mask -> original index
        Map<Integer, Integer> map = new HashMap<>();
        int[] mask = new int[n];

        for (int i = 0; i < n; i++) {
            int x = 0;
            char[] temp = words[i].toCharArray();
            for (int j = 0; j < temp.length; j++) {
                char c = temp[j];

                // set the (c - 'a')th digit to 1
                x = x | (1 << (c - 'a'));
            }
            map.put(x, i);
            mask[i] = x;
        }

        // start checking words one by one, if it has connected words, join them in Union Find
        for (int i = 0; i < n; i++) {
            String current = words[i];
            int len = current.length();
            int x = mask[i];

            for (int j = 0; j < len; j++) {
                char c = current.charAt(j);

                // delete char at j -> set the (c - 'a')th digit to 0
                x = x & (~(1 << (c - 'a')));
                if (map.containsKey(x)) {
                    int next = map.get(x);
                    uf.join(i, next);
                }

                // replace char at j with 'a' to 'z':
                // replace = delete(already done) + add
                for (char t = 'a'; t <= 'z'; t++) {
                    // take the bit of the (t - 'a')th digit
                    int dig = (x >> (t - 'a')) & 1;
                    if (dig == 1) {
                        // since no letter occurs more than once in words[i],
                        // if this digit is already 1, we can continue;
                        continue;
                    }

                    // set the (t - 'a')th digit to 1, complete the replacing
                    x = x | (1 << (t - 'a'));
                    if (map.containsKey(x)) {
                        int next = map.get(x);
                        uf.join(i, next);
                    }

                    // backtracking , set it back to 0
                    x = x & (~(1 << (t - 'a')));
                }

                // backtracking, add back the char we delete
                x = x | (1 << (c - 'a'));
            }
        }

        // get output from the union Find
        Set<Integer> set = new HashSet<>();
        int max = 1;
        for (int i = 0; i < n; i++) {
            int fx = uf.find(i);
            set.add(fx);
            max = Math.max(max, uf.size[i]);
        }

        return new int[]{set.size(), max};
    }

}


class UnionFind {

    int[] father;
    int[] size;

    public UnionFind(int n) {
        father = new int[n];
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        size = new int[n];
        Arrays.fill(size, 1);
    }

    public void join(int x, int y) {
        int fx = find(x);
        int fy = find(y);
        if (fx != fy) {
            father[fx] = fy;
            size[fy] += size[fx];
        }
    }

    public int find(int x) {
        int root = x;
        while (father[root] != root) {
            root = father[root];
        }
        while (x != root) {
            int fx = father[x];
            father[x] = root;
            x = fx;
        }
        return root;
    }

    public boolean isConnected(int x, int y) {
        return find(x) == find(y);
    }

    /*
    For each edge (i, j) in edges,
    we find a neighbour ii of node i,
    we find a neighbour jj of node i,
    If ii, i, j,jj has no duplicate, then that's a valid sequence.

    Ad the intuition mentioned,
    we don't have to enumearte all neignbours,
    but only some nodes with big value.

    But how many is enough?
    I'll say 3.
    For example, we have ii, i, j now,
    we can enumerate 3 of node j biggest neighbour,
    there must be at least one node different node ii and node i.

    So we need to iterate all edges (i, j),
    for each node we keep at most 3 biggest neighbour, which this can be done in O(3) or O(log3).

     */
    // Author: Anand
    // TC = O(n+m); n = length of edges, m = max neighbours of a node
    // SC = O(n)
    public int maximumScore(int[] scores, int[][] edges) {

        int maxSum = -1;
        int n = scores.length;
        int maxscore = 0;
        Node[] list = new Node[n];

        for (int i = 0; i < list.length; i++) {
            list[i] = new Node(scores[i]);
            maxscore = Math.max(scores[i], maxscore);
        }
        for (int[] edge : edges) {
            Node start = list[edge[0]];
            Node end = list[edge[1]];
            start.add(end);
        }

        for (int[] edge : edges) {
            Node start = list[edge[0]];
            Node end = list[edge[1]];

            if (start.value + end.value + maxscore + maxscore <= maxSum) continue;
            Queue<Node> queue = new PriorityQueue<>((a, b) -> b.value - a.value);

            for (Node node : start.next) {
                if (node != end) queue.offer(node);
            }

            if (queue.size() == 0) continue;
            // pull 2 adjacent neighbours out of queue
            Node start1 = queue.poll();
            Node start2 = queue.poll();

            queue.clear();

            for (Node node : end.next) {
                if (node != start) queue.offer(node);
            }

            if (queue.size() == 0) continue;
            // pull 2 adjacent neighbours out of queue
            Node end1 = queue.poll();
            Node end2 = queue.poll();

            int sum = start.value + end.value + start1.value;

            if (start1 != end1) {
                sum += end1.value;
            } else if (start2 == null && end2 == null) {
                continue;
            } else if (start2 != null && end2 != null) {
                sum += Math.max(start2.value, end2.value);
            } else if (start2 == null) {
                sum += end2.value;
            } else {
                sum += start2.value;
            }
            maxSum = Math.max(sum, maxSum);

        }
        return maxSum;
    }

    static class Node {
        int value;
        Set<Node> next = new HashSet<>();

        Node(int val) {
            this.value = val;
        }

        void add(Node node) {
            next.add(node); // a ---> [b]
            node.next.add(this); // b --> [a]
        }
    }
}

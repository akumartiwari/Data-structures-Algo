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
    
    // Topological sort
    // Author: Anand
    public int maximumScore(int[] scores, int[][] edges) {
        int n = scores.length;
        final List<List<Node>> graph = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList());
        }
        for (final int[] arr : edges) {
            graph.get(arr[0]).add(new Node(arr[1], 0));
            graph.get(arr[1]).add(new Node(arr[0], 0));
        }
        return this.dfs(graph, n);
    }

    public int dfs(final List<List<Node>> adj, int n) {
        int ans = -1;
        final int mod = 1_000_000_007;
        final Queue<Node> queue = new PriorityQueue<>(n);
        final long[] costs = new long[n];
        final long[] ways = new long[n];
        final boolean[] cache = new boolean[n];
        queue.add(new Node(0, 0));
        Arrays.fill(costs, Long.MAX_VALUE);
        costs[0] = 0;
        //one way to visit first node
        ways[0] = 1;
        int nodesCount = 1;
        while (!queue.isEmpty()) {
            final Node currentNode = queue.poll();
            if (currentNode.cost < costs[currentNode.position] || cache[currentNode.position]) {
                continue;
            }
            for (final Node vertex : adj.get(currentNode.position)) {
                if (costs[currentNode.position] + vertex.cost > costs[vertex.position]) {
                    costs[vertex.position] = costs[currentNode.position] + vertex.cost;
                    ways[vertex.position] = ways[currentNode.position] % mod;
                    queue.add(new Node(vertex.position, costs[vertex.position]));
                } else if (costs[currentNode.position] + vertex.cost == costs[vertex.position]) {
                    ways[vertex.position] = (ways[vertex.position] + ways[currentNode.position]) % mod;
                }
            }
            nodesCount++;
            if (nodesCount == 4) {
                ans = Math.max(ans, (int) costs[currentNode.position]);
                nodesCount = 0;
            }
        }
        return ans;
    }

    @SuppressWarnings("ALL")
    private class Node implements Comparable<Node> {
        int position;
        long cost;

        public Node(int dis, long val) {
            this.position = dis;
            this.cost = val;
        }


        @Override
        public int compareTo(final Node o) {
            return Long.compare(this.cost, o.cost);
        }
    }
}

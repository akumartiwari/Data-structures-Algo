package Graph;

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
}

class Eulerian {
    public int[][] validArrangement(int[][] pairs) {
        int n = pairs.length;

        int[][] ans = new int[n][2];
        for (int[] a : ans) {
            a[0] = -1;
            a[1] = -1;
        }

        Map<Integer, Integer> outdegree = new HashMap<>();
        Map<Integer, Deque<Integer>> out = new HashMap<>();

        for (int[] pair : pairs) {
            outdegree.put(pair[0], outdegree.getOrDefault(pair[0], 0) + 1);
            outdegree.put(pair[1], outdegree.getOrDefault(pair[1], 0) - 1);

            out.computeIfAbsent(pair[0], k -> new ArrayDeque<>());
            out.computeIfAbsent(pair[1], k -> new ArrayDeque<>());

            out.get(pair[0]).addLast(pair[1]);
        }

        for (Map.Entry<Integer, Integer> entry : outdegree.entrySet()) {
            if (entry.getValue() == 1) ans[0][0] = entry.getKey();
            if (entry.getValue() == -1) ans[n - 1][1] = entry.getKey();
        }

        if (ans[0][0] == -1) {
            ans[0][0] = pairs[0][0];
            ans[n - 1][1] = pairs[0][0];
        }

        int i = 0;
        int j = n - 1;
        while (i < j) {
            int from = ans[i][0];

            Deque<Integer> toList = out.get(from);

            if (toList.size() == 0) {
                ans[j][0] = ans[--i][0];
                ans[--j][1] = ans[j + 1][0];
            } else {
                ans[i++][1] = toList.removeLast();
                ans[i][0] = ans[i - 1][1];
            }
        }

        return ans;
    }
}
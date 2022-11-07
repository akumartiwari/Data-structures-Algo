package com.company;

import java.util.*;

public class BinaryTree {
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     * int val;
     * TreeNode left;
     * TreeNode right;
     * TreeNode() {}
     * TreeNode(int val) { this.val = val; }
     * TreeNode(int val, TreeNode left, TreeNode right) {
     * this.val = val;
     * this.left = left;
     * this.right = right;
     * }
     * }
     */
    // Definition for a Node.
    class Node {
        public int val;
        public List<Node> children;


        public Node() {
            children = new ArrayList<Node>();
        }

        public Node(int _val) {
            val = _val;
            children = new ArrayList<Node>();
        }

        public Node(int _val, ArrayList<Node> _children) {
            val = _val;
            children = _children;
        }
    }

    // Author: Anand
    public int diameterOfBinaryTree(TreeNode root) {
        int ans = 0;
        if (root == null) return 0;
        int lh = height(root.left);
        int rh = height(root.right);
        int ld = diameterOfBinaryTree(root.left);
        int rd = diameterOfBinaryTree(root.right);

        return Math.max(lh + rh, Math.max(ld, rd));
    }

    private int height(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(height(root.left), height(root.right));
    }

    // Author: Anand
    int ans = 0;

    public int diameter(Node root) {
        if (root == null || root.children.size() == 0) return 0;
        maxDiamterNArray(root);
        return ans;
    }

    private int maxDiamterNArray(Node root) {
        // base case
        if (root.children.size() == 0) return 0;
        int mH1 = -1, mH2 = -1;

        for (int i = 0; i < root.children.size(); i++) {
            int ch = maxDiamterNArray(root.children.get(i));

            if (ch > mH1) {
                mH2 = mH1;
                mH1 = ch;
            } else if (ch > mH2) {
                mH2 = ch;
            }
        }

        ans = Math.max(ans, mH1 + mH2 + 2);

        return 1 + mH1;
    }


    // TC = O(nlogn) where logn factor is introduced due to PQ
    /*
    dfs(i) return the the longest path starting at the node i.

    dfs on each node i, and for each node i,
    we iterate on all its children,
    and find all the path startint at the node i.
    The 2 longest path can make up the longest path in subtree of node i,
    we update the global result res,
    and return the longest path length as result for dfs.
     */
    int res;

    // Author: Anand
    public int longestPath(int[] parent, String s) {
        int n = parent.length;
        List<Integer>[] graph = new ArrayList[n];
        int ans = 0;
        for (int i = 0; i < parent.length; i++) graph[i] = new ArrayList<>();
        for (int i = 1; i < parent.length; i++) graph[parent[i]].add(i);

        dfs(graph, s, 0);
        return res;
    }


    private int dfs(List<Integer>[] graph, String s, int idx) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder()); //max-pq
        for (int j : graph[idx]) {

            // current height of children
            int curr = dfs(graph, s, j);

            // if curr node is not intercepted in its subtree path, consider it as a valid path
            if (s.charAt(j) != s.charAt(idx)) pq.offer(curr);

        }
        int mH1 = pq.size() > 0 ? pq.poll() : 0;
        int mH2 = pq.size() > 0 ? pq.poll() : 0;

        res = Math.max(res, mH1 + mH2 + 1);
        return mH1 + 1;
    }


    /*
    Input: root = [10,3,4,2,1]
    Output: 2
    Explanation:
    For the node with value 10: The sum of its descendants is 3+4+2+1 = 10.
    For the node with value 3: The sum of its descendants is 2+1 = 3.
     */
    // Author: Anand
    public int equalToDescendants(TreeNode root) {
        return helper(root).nodes;
    }

    private Tuple helper(TreeNode root) {
        // base case
        if (root == null) return new Tuple(0, 0);
        Tuple left = helper(root.left);
        Tuple right = helper(root.right);
        int cs = left.cs + right.cs;
        int nodesCount = left.nodes + right.nodes;
        if (cs == root.val) nodesCount++;

        return new Tuple(cs + root.val, nodesCount);
    }

    static class Tuple {
        int cs;
        int nodes;

        Tuple(int cs, int nodes) {
            this.cs = cs;
            this.nodes = nodes;
        }

        Tuple() {
            this.cs = 0;
            this.nodes = 0;
        }
    }

    /*
    Input: root = [4,8,5,0,1,null,6]
    Output: 5
    Explanation:
    For the node with value 4: The average of its subtree is (4 + 8 + 5 + 0 + 1 + 6) / 6 = 24 / 6 = 4.
    For the node with value 5: The average of its subtree is (5 + 6) / 2 = 11 / 2 = 5.
    For the node with value 0: The average of its subtree is 0 / 1 = 0.
    For the node with value 1: The average of its subtree is 1 / 1 = 1.
    For the node with value 6: The average of its subtree is 6 / 1 = 6.
     */
    // Author: Anand
    public int averageOfSubtree(TreeNode root) {
        return helperAvg(root).avgCount;
    }

    private Tuple3 helperAvg(TreeNode root) {
        // base case
        if (root == null) return new Tuple3(0, 0, 0);
        Tuple3 left = helperAvg(root.left);
        Tuple3 right = helperAvg(root.right);
        int cs = root.val + left.cs + right.cs;
        int cc = 1 + left.cc + right.cc;

        int avg = cs / cc;

        int avgCount = left.avgCount + right.avgCount;
        if (avg == root.val) avgCount++;

        return new Tuple3(cs, cc, avgCount);
    }

    static class Tuple3 {
        int cs;
        int cc;
        int avgCount;

        Tuple3(int cs, int cc, int avgCount) {
            this.cs = cs;
            this.cc = cc;
            this.avgCount = avgCount;
        }

        Tuple3() {
            this.cs = 0;
            this.cc = 0;
            this.avgCount = 0;
        }
    }

    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     * int val;
     * TreeNode left;
     * TreeNode right;
     * TreeNode() {}
     * TreeNode(int val) { this.val = val; }
     * TreeNode(int val, TreeNode left, TreeNode right) {
     * this.val = val;
     * this.left = left;
     * this.right = right;
     * }
     * }
     */
    public TreeNode createBinaryTree(int[][] descriptions) {
        Map<Integer, TreeNode> map = new HashMap<>();
        Set<Integer> children = new HashSet<>();

        for (int[] desc : descriptions) {
            int p = desc[0];
            int c = desc[1];
            int isLeft = desc[2];

            children.add(c);
            TreeNode node = map.getOrDefault(p, new TreeNode(p));
            if (isLeft == 1) {
                node.left = map.getOrDefault(c, new TreeNode(c));
                map.put(c, node.left);
            } else {
                node.right = map.getOrDefault(c, new TreeNode(c));
                map.put(c, node.right);
            }
            map.put(p, node);
        }

        int root = -1;

        for (int[] desc : descriptions) {
            // Root parent is a node which is not a child of any node
            if (!children.contains(desc[0])) {
                root = desc[0];
                break;
            }
        }
        return map.getOrDefault(root, null);
    }

    // Check if 2 BT are same
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;

        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}


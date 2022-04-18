package com.company;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

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


}


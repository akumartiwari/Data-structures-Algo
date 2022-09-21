package com.company;

public class Tree {
    int res = 0;

    public int averageOfSubtree(TreeNode root) {
        if (root == null) return 0;
        dfs(root);
        return res;
    }

    private int[] dfs(TreeNode root) {
        // base case
        if (root == null) return new int[]{0, 0};

        int[] left = dfs(root.left);
        int[] right = dfs(root.right);

        int currentSum = left[0] + right[0] + root.val;
        int currentCount = left[1] + right[1] + 1;

        if (currentSum / currentCount == root.val) res++;

        return new int[]{currentSum, currentCount};
    }

    public int averageOfSubtreeWithoutGlobalVariable(TreeNode root) {
        return helper(root).countAvg;
    }

    private Tuple helper(TreeNode root) {
        // base case
        if (root == null) return new Tuple(0, 0, 0);

        Tuple left = helper(root.left);
        Tuple right = helper(root.right);

        int currentSum = root.val + left.sum + right.sum;

        int currentCount = left.count + right.count + 1;

        int avg = currentSum / currentCount;
        int countAvg = left.countAvg + right.countAvg;
        if (avg == root.val) countAvg++;

        return new Tuple(currentSum, currentCount, countAvg);
    }

    class Tuple {
        int sum;
        int count;
        int countAvg;

        Tuple() {
            sum = 0;
            count = 0;
            countAvg = 0;
        }


        public Tuple(int currentSum, int currentCount, int countAvg) {
            this.sum = currentSum;
            this.count = currentCount;
            this.countAvg = countAvg;
        }
    }

    public int sumOfLeftLeaves(TreeNode root) {
        //Base cases
        if (root == null) return 0;

        int res = 0;
        // traverse till leaf node of tree
        if (isLeafNode(root.left)) {
            res += root.left.val;
        } else {
            //Recursively traverse the left child
            res += sumOfLeftLeaves(root.left);
        }
        //Traverse right subtree iof root node
        res += sumOfLeftLeaves(root.right);
        return res;
    }

    public boolean isLeafNode(TreeNode node) {
        if (node == null) return false;
        return node.left == null && node.right == null;
    }

    /*
    Input: root = [2,3,5,8,13,21,34]
    Output: [2,5,3,8,13,21,34]
    Explanation:
    The tree has only one odd level.
    The nodes at level 1 are 3, 5 respectively, which are reversed and become 5, 3.
     */
    public TreeNode reverseOddLevels(TreeNode root) {
        rol(root.left, root.right, 1);
        return root;
    }

    private void rol(TreeNode left, TreeNode right, int level) {

        // base case
        if (left == null || right == null) return;

        // Odd level ie. reverse nodes
        if (level % 2 == 1) {
            // swap children
            int temp = left.val;
            left.val = right.val;
            right.val = temp;
        }
        rol(left.left, right.right, level + 1);
        rol(left.right, right.left, level + 1);
    }

    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode() {}
     *     TreeNode(int val) { this.val = val; }
     *     TreeNode(int val, TreeNode left, TreeNode right) {
     *         this.val = val;
     *         this.left = left;
     *         this.right = right;
     *     }
     * }
     */


}


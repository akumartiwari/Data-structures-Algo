package com.company;

import java.util.HashMap;
import java.util.*;

public class Tree {
    int res = 0;

    /*
     * PROBLEM: Count Nodes Equal to Average of Subtree (LeetCode 2265)
     * Count nodes whose value equals the integer average of their subtree.
     *
     * ALGORITHM: DFS Post-order (return sum and count)
     * TC: O(n) | SC: O(n)
     */
    public int averageOfSubtree(TreeNode root) {
        if (root == null) return 0;
        dfs(root);
        return res;
    }

    /*
     * PROBLEM: DFS helper for average of subtree (Helper)
     * Returns [subtreeSum, nodeCount] for each subtree and increments global counter.
     *
     * ALGORITHM: Recursive DFS Post-order
     * TC: O(n) | SC: O(n)
     */
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

    /*
     * PROBLEM: Count Nodes Equal to Average of Subtree, no global (LeetCode 2265)
     * Same as averageOfSubtree but without using a global variable.
     *
     * ALGORITHM: DFS with Tuple (sum, count, result)
     * TC: O(n) | SC: O(n)
     */
    public int averageOfSubtreeWithoutGlobalVariable(TreeNode root) {
        return helper(root).countAvg;
    }

    /*
     * PROBLEM: Tuple-based DFS helper (Helper)
     * Returns Tuple(sum, count, countAvg) for each subtree without global state.
     *
     * ALGORITHM: Recursive DFS Post-order with Tuple
     * TC: O(n) | SC: O(n)
     */
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

    /*
     * PROBLEM: Sum of Left Leaves (LeetCode 404)
     * Compute the sum of all left leaf node values in the binary tree.
     *
     * ALGORITHM: Recursive DFS
     * TC: O(n) | SC: O(n)
     */
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
        //Traverse right subtree of root node
        res += sumOfLeftLeaves(root.right);
        return res;
    }

    /*
     * PROBLEM: Check if Node is Leaf (Helper)
     * Return true if node is non-null and has no children.
     *
     * ALGORITHM: Null/child check
     * TC: O(1) | SC: O(1)
     */
    public boolean isLeafNode(TreeNode node) {
        if (node == null) return false;
        return node.left == null && node.right == null;
    }

    /*
     * PROBLEM: Reverse Odd Levels of Binary Tree (LeetCode 2415)
     * Reverse the values of nodes at every odd level of a perfect binary tree.
     *
     * ALGORITHM: DFS level traversal (mirror pairs at odd levels)
     * TC: O(n) | SC: O(log n)
     */
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

    /*
     * PROBLEM: Reverse Odd Levels helper (Helper)
     * Recursively swap node values at odd levels for mirror node pairs.
     *
     * ALGORITHM: Recursive DFS with mirror pointers
     * TC: O(n) | SC: O(log n)
     */
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


    // Algo:-

    // - if tree is null retrun empty array list
    // - Apply preorder traversal recursively and check of node had beee already taken in map or not
    // - if its taken first time for that level fetch it else skip it
    // - return the global ans

    /*

    [1,2,3,4,5,6,7]
       [(0, [1]),(-1, [2]), (-2, [4]),  ]

    TC = O(Nlogn) + O(N) ~= O(nlogn)
    SC = O(N)
    */

    public class Pair implements Comparable<Pair> {
        int key;
        int value;

        public Pair(int key, int value) //Constructor of the class
        {
            this.key = key;
            this.value = value;
        }

        public int compareTo(Pair o) {
            return this.key - o.key;
        }
    }

    // To store the vline and nodes with level on that vline
    HashMap<Integer, PriorityQueue<Pair>> map = new HashMap<>();

    /*
     * PROBLEM: Vertical Order Traversal (LeetCode 987)
     * Return vertical order traversal of binary tree sorted by column, level, then value.
     *
     * ALGORITHM: DFS with sorted HashMap (column → PriorityQueue of (level, value))
     * TC: O(n log n) | SC: O(n)
     */
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        if (root == null) return null;
        inorder(root, 0, 0);
        List<List<Integer>> ans = new ArrayList<>();
        Map<Integer, PriorityQueue<Pair>> treemap = new TreeMap<>(map);

        // Iterate through map and get the result in desired format
        for (Map.Entry<Integer, PriorityQueue<Pair>> entry : treemap.entrySet()) {
            PriorityQueue<Pair> pq = entry.getValue();
            // to store level along iwth nodes on that level
            // grouping the nodes based on level
            Map<Integer, List<Integer>> hm = new TreeMap<>();
            List<Integer> colList = new ArrayList<>();

            while (!pq.isEmpty()) {
                Pair res = pq.poll();
                if (hm.containsKey(res.key)) {
                    List<Integer> list = new ArrayList<>();
                    List<Integer> l = hm.get(res.key);
                    l.add(res.value);
                    hm.put(res.key, l);
                } else hm.put(res.key, new ArrayList<>(Arrays.asList(res.value)));
            }

            for (Map.Entry<Integer, List<Integer>> e : hm.entrySet()) {
                List<Integer> pd = e.getValue();
                Collections.sort(pd);
                for (Integer data : pd) colList.add(data);
            }
            ans.add(colList);
        }
        return ans;
    }

    /*
     * PROBLEM: Inorder DFS for vertical traversal (Helper)
     * Perform inorder DFS recording each node's level and vertical line for vertical traversal.
     *
     * ALGORITHM: Recursive DFS (inorder)
     * TC: O(n log n) | SC: O(n)
     */
    private void inorder(TreeNode root, int level, int vline) {
        if (root == null) return;

        vline -= 1;
        inorder(root.left, level + 1, vline);
        vline += 1;
        // node on this vline is already present then add new node in list of this vline
        if (map.get(vline) != null) {
            PriorityQueue<Pair> existingpq = map.get(vline);
            existingpq.add(new Pair(level, root.val));
            map.put(vline, existingpq);
        } else {
            // Else if node on this vline is found first time then simply add it
            PriorityQueue<Pair> newNode = new PriorityQueue<>();
            newNode.add(new Pair(level, root.val));
            map.put(vline, newNode);
        }
        inorder(root.right, level + 1, vline + 1);
    }


    boolean possible = false;

    /*
     * PROBLEM: Check if Tree Can be Split Equally (LeetCode 663)
     * Determine if removing one edge can split the tree into two subtrees with equal sum.
     *
     * ALGORITHM: DFS sum checking (find subtree with sum = total/2)
     * TC: O(n) | SC: O(n)
     */
    public boolean checkEqualTree(TreeNode root) {
        SumAndCount t = tuple(root);
        if (t.sum % 2 != 0) return false;
        isPossible(root, t.sum, t.count);
        return possible;
    }

    static class SumAndCount {
        long sum;
        int count;

        SumAndCount() {
            sum = 0L;
            count = 0;
        }


        public SumAndCount(long currentSum, int currentCount) {
            this.sum = currentSum;
            this.count = currentCount;
        }
    }

    /*
     * PROBLEM: Check Equal Split Feasibility (Helper)
     * Recursively check if any subtree has sum equal to half the total.
     *
     * ALGORITHM: Recursive DFS returning (subtreeSum, nodeCount)
     * TC: O(n) | SC: O(n)
     */
    private SumAndCount isPossible(TreeNode root, long sum, int cnt) {
        if (root == null) return new SumAndCount(0L, 0);

        SumAndCount left = isPossible(root.left, sum, cnt);
        SumAndCount right = isPossible(root.right, sum, cnt);

        long currsum = root.val + left.sum + right.sum;
        int nodes = 1 + left.count + right.count;

        if (currsum == sum / 2 && cnt != nodes) possible = true;
        return new SumAndCount(currsum, nodes);
    }

    /*
     * PROBLEM: Compute tree sum and count (Helper)
     * Return total sum and node count of the entire tree.
     *
     * ALGORITHM: Recursive DFS post-order
     * TC: O(n) | SC: O(n)
     */
    private SumAndCount tuple(TreeNode root) {

        if (root == null) return new SumAndCount(0L, 0);

        SumAndCount left = tuple(root.left);
        SumAndCount right = tuple(root.right);

        long currsum = root.val + left.sum + right.sum;
        int nodes = 1 + left.count + right.count;
        return new SumAndCount(currsum, nodes);
    }

    //TODO
    class Solution {
        TreeMap<Integer, Integer> nodes = new TreeMap<>();

        class Tuple {
            TreeMap<Integer, Integer> nodes;
            boolean valid;

            Tuple() {
                nodes = new TreeMap<>();
                valid = false;
            }


            public Tuple(TreeMap<Integer, Integer> tm, boolean valid) {
                this.nodes = tm;
                this.valid = valid;
            }
        }

        private Tuple helper(TreeNode root) {

            if (root == null) return new Tuple(new TreeMap<>(), true);
            nodes.put(root.val, nodes.getOrDefault(root.val, 0) + 1);

            boolean left = true, right = true;
            if (root.left != null)
                left = root.val > root.left.val;

            if (root.right != null)
                right = root.val < root.right.val;

            // validate for left and right nodes
            if (!(left && right)) return new Tuple(new TreeMap<>(), false);

            Tuple ls = helper(root.left);

            if (ls.nodes.size() > 0 && ls.nodes.ceilingKey(root.val) != null) new Tuple(ls.nodes, false);

            Tuple rs = helper(root.right);

            // validate for left and right subtrees
            if (rs.nodes.size() > 0 && rs.nodes.floorKey(root.val) != null) new Tuple(rs.nodes, false);

            nodes.put(root.val, nodes.getOrDefault(root.val, 0) + 1);
            nodes.putAll(ls.nodes);
            nodes.putAll(rs.nodes);

            return new Tuple(nodes, true);
        }


        public boolean isValidBST(TreeNode root) {
            return helper(root).valid;
        }

    }
}


package com.company;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;

class SolutionRecursion {

    Map<String, List<Integer>> map;

    /*
     * PROBLEM: Sort Array Using Recursion (Helper)
     * Sort an integer array via recursive insertion sort.
     *
     * ALGORITHM: Recursive Insertion Sort
     * TC: O(n^2) | SC: O(n)
     */
    // using Recursion
    public int[] sortArray(int[] nums) {
        int n = nums.length;
        map = new HashMap<>();
        if (n == 1) return nums;

        List<Integer> sorted = sort(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        return sorted.stream().mapToInt(x -> x).toArray();
    }

    /*
     * PROBLEM: Recursive Sort (Helper)
     * Recursively sort a list by removing the last element, sorting the remainder, then inserting.
     *
     * ALGORITHM: Recursive Insertion Sort
     * TC: O(n^2) | SC: O(n)
     */
    private List<Integer> sort(List<Integer> nums) {
        int n = nums.size();
        if (n == 0 || n == 1) return nums;

        // Hypothesis
        int temp = nums.get(n - 1);
        nums.remove(new Integer(temp));
        sort(nums);
        return insert(nums, temp);
    }

    /*
     * PROBLEM: Recursive Insert in Sorted List (Helper)
     * Insert num into a sorted list in the correct position recursively.
     *
     * ALGORITHM: Recursive insertion
     * TC: O(n) | SC: O(n)
     */
    private List<Integer> insert(List<Integer> sorted, int num) {
        // base case
        int n = sorted.size();
        if (n == 0 || sorted.get(n - 1) <= num) {
            System.out.println(num);
            sorted.add(num);
            return sorted;
        }

        String key = (n - 1) + "*" + sorted.get(n - 1);
        int val = sorted.get(n - 1); // last value is stored in a variable to place num at correct position
        sorted.remove(new Integer(val));
        List<Integer> finalSorted;

        if (!map.containsKey(key)) {
            finalSorted = insert(sorted, num);
        } else finalSorted = map.get(key);

        finalSorted.add(val);
        map.put(key, finalSorted);
        return map.get(key);
    }


    /*
     * PROBLEM: Count Colder Days (Helper)
     * For each index, count elements in the running stack smaller than current.
     *
     * ALGORITHM: Monotonic Stack
     * TC: O(n) | SC: O(n)
     */
    private static List<Integer> countcolderDays(int[] arr) {
        Stack<Integer> stk = new Stack<>();
        // used to store elements greater than the current element
        List<Integer> ans = new ArrayList<>();
        int n = arr.length;
        if (n == 0) return ans;
        for (int i = 0; i < n; i++) {
            if (arr[i] > stk.peek()) {
                // push into stack
                stk.push(arr[i]);
            }

            int count = Math.abs(stk.size() - i); // count of elements lesser than current one
            ans.add(count);
        }

        return ans;
    }
    /*
     * PROBLEM: Get Maximum Sum From Two Arrays With Common Points (LeetCode 1537)
     * Return the maximum path sum traversing two arrays, switching at common elements.
     *
     * ALGORITHM: Recursion + Binary Search for intersection
     * TC: O(n log n) | SC: O(n)
     */
    //         int[] arr = {100,10,89,40,1,80,97};
    public int maxSum(int[] nums1, int[] nums2) {
        return maxSum(nums1, nums2, 0, 0, 0);
    }

    private int maxSum(int[] nums1, int[] nums2, int num1, int num2, int max) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        if (n1 == 0 && n2 == 0) return 0;
        // base case
        if (num1 == n1 - 1 || num2 == n2 - 1) return 0;

        // hypothesis
        // starting from nums1
        for (int i = 0; i < n1; i++) {
            int index = Arrays.binarySearch(nums2, nums1[i]);
            if (index != -1) {
                int in = maxSum(nums1, nums2, i, index, max);
                int out = maxSum(nums1, nums2, i, num2, max);
                max += Math.max(in, out);
                return max;
            }
        }

        for (int i = 0; i < n2; i++) {
            int index = Arrays.binarySearch(nums1, nums2[i]);
            if (index != -1) {
                int in = maxSum(nums1, nums2, index, i, max);
                int out = maxSum(nums1, nums2, num1, i, max);
                max += Math.max(in, out);
                return max;
            }
        }

        // Induction
        return max;
    }

    /*
     * PROBLEM: All Possible Full Binary Trees (LeetCode 894)
     * Return all structurally unique full binary trees with n nodes.
     *
     * ALGORITHM: Recursion + Enumerate left/right subtree sizes
     * TC: O(2^n) | SC: O(2^n)
     */
    public List<TreeNode> allPossibleFBT(int n) {
        List<TreeNode> ans = new ArrayList<>();
        // base cases
        if (n == 1) {
            ans.add(new TreeNode());
            return ans;
        }
        for (int i = 1; i <= n - 2; i++) {
            List<TreeNode> left = allPossibleFBT(i);
            List<TreeNode> right = allPossibleFBT(n - 1 - i);

            for (TreeNode l : left) {
                for (TreeNode r : right) {
                    TreeNode tree = new TreeNode();
                    tree.left = l;
                    tree.right = r;
                    ans.add(tree);
                }
            }
        }
        return ans;
    }


    /*
     * PROBLEM: Check If All Characters Have Equal Number of Occurrences (LeetCode 1941)
     * Return true if all characters in the string have equal frequency.
     *
     * ALGORITHM: HashMap frequency count + equality check
     * TC: O(n) | SC: O(1)
     */
    public boolean areOccurrencesEqual(String s) {
        int n = s.length();
        HashMap<Character, Integer> freq = new HashMap<>();
        for (int i = 0; i < n; i++) {
            freq.put(s.charAt(i), freq.getOrDefault(s.charAt(i), 1) + 1);
        }

        int count = Integer.MIN_VALUE;
        for (Character c : freq.keySet()) {
            if (count == Integer.MIN_VALUE) count = freq.get(c);
            else if (count != freq.get(c)) {
                return false;
            }
        }
        return true;
    }

    static class Pair<I extends Number, I1 extends Number> implements Comparable<Pair<Number, Number>> {
        int key;
        int val;

        Pair(int key, int val) {
            this.key = key;
            this.val = val;
        }

        @Override
        public int compareTo(Pair<Number, Number> pair) {
            return 0;
        }
    }

    /*
     * PROBLEM: The Number of the Smallest Unoccupied Chair (LeetCode 1942)
     * Find the smallest chair number that the target friend sits in.
     *
     * ALGORITHM: TreeMap + sorted arrival/leave tracking
     * TC: O(n^2) | SC: O(n)
     */
    public int smallestChair(int[][] times, int targetFriend) {
        int row = times.length;
        // arrival and leave time
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < row; i++) {
            map.put(times[i][0], times[i][1]);
        }

        // TreeMap to store values of HashMap

        // Copy all data from hashMap into TreeMap
        TreeMap<Integer, Integer> sorted = new TreeMap<>(map);

        // Copy all data from hashMap into TreeMap
        TreeMap<Integer, Pair<Integer, Integer>> pairIndex = new TreeMap<>();
        // to store key, pair(index,value)

        Map<Integer, Integer> valueSorted = map.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));


        int[] unoccupied = new int[row];
        Arrays.fill(unoccupied, 0);
        int index = 0;

        // Display the TreeMap which is naturally sorted
        for (Map.Entry<Integer, Integer> entry : sorted.entrySet()) {
            pairIndex.put(entry.getKey(), new Pair(index, entry.getValue()));

            // check if any chair has been unoccupied
            for (Map.Entry<Integer, Integer> ent : valueSorted.entrySet()) {
                if (entry.getKey() > (ent.getValue())) {
                    // get index of chair to be unoccupied
                    unoccupied[pairIndex.get(ent.getKey()).key] = 0;
                    break;
                }
            }

            if (entry.getKey().equals(targetFriend)) {
                // return  first zero chair
                return getChair(unoccupied);
            }
            unoccupied[index] = 1;
            index++;
        }
        return 0;
    }

    /*
     * PROBLEM: Get First Unoccupied Chair (Helper)
     * Scan the unoccupied array to find the first free chair.
     *
     * ALGORITHM: Linear scan
     * TC: O(n) | SC: O(1)
     */
    private int getChair(int[] unoccupied) {
        int chair = 0;
        for (int c : unoccupied) {
            if (c == 0) return chair;
            chair++;
        }
        return 0;
    }

    public List<List<Long>> splitPainting(int[][] segments) {
        int row = segments.length;
        int col = segments[0].length;

        List<List<Long>> ans = new ArrayList<>();
        // arrival and leave time
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < row; i++) {
            map.put(segments[i][0], segments[i][1]);
        }

        return ans;
    }

    /*
     * PROBLEM: Number of Visible People in a Queue (LeetCode 1944)
     * For each person, count how many others they can see to their right (TLE solution).
     *
     * ALGORITHM: Brute force O(n^2)
     * TC: O(n^2) | SC: O(n)
     */
    // TLE

    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] ans = new int[n];

        for (int i = 0; i < n; i++) {
            int count = 0;
            int prev_max = Integer.MIN_VALUE;
            for (int j = i + 1; j < n; j++) {
                int ht = Math.min(heights[i], heights[j]);

                if (prev_max == Integer.MIN_VALUE) {
                    int max = Integer.MIN_VALUE;
                    int index = i + 1;
                    while (index < j) {
                        max = Math.max(heights[index], max);
                        index++;
                    }
                    if (max != Integer.MIN_VALUE) {
                        prev_max = max;
                    }
                } else {
                    prev_max = Math.max(heights[j - 1], prev_max);
                }


                if (prev_max != Integer.MIN_VALUE && ht > prev_max) count++;
            }
            ans[i] = count;
        }
        return ans;
    }

    /*
     * PROBLEM: Number of Visible People in a Queue Optimized (LeetCode 1944)
     * Optimized O(n^2) traversal from right to left tracking max height.
     *
     * ALGORITHM: Brute force right-to-left scan
     * TC: O(n^2) | SC: O(n)
     */
    /*
    Input: heights = [10,6,8,5,11,9]
    Output: [3,1,2,1,1,0]

     */

    public int[] canSeePersonsCountOpt(int[] heights) {
        int n = heights.length;
        int[] ans = new int[n];


//        List<Integer> list = Arrays.stream(heights).boxed().collect(Collectors.toList());
//        Collections.reverse(list);

        for (int i = n - 1; i >= 0; i--) {
            int count = 0;
            int max = Integer.MIN_VALUE;

            for (int j = i + 1; j < n; j++) {
                int ht = Math.min(heights[i], heights[j]);

                max = Math.max(heights[j - 1], max);
                if (ht > max) count++;
            }
            ans[i] = count;
        }
        return ans;
    }

    class Solution {
        PriorityQueue<Integer> res;

        /*
         * PROBLEM: Next Greater Element III (LeetCode 556)
         * Find the smallest integer greater than n using the same digits.
         *
         * ALGORITHM: Permutation brute force + PriorityQueue
         * TC: O(n! log n) | SC: O(n!)
         */
        public int nextGreaterElement(int n) {
            String str = Integer.toString(n);
            res = new PriorityQueue<>();
            permutation(str.toCharArray(), 0, 0);

            while (!res.isEmpty() && res.peek() <= n) res.poll();
            return res.peek() != null ? res.poll() : -1;
        }

        /*
         * PROBLEM: Generate Digit Permutations (Helper)
         * Recursively generate all permutations of digit array and add to priority queue.
         *
         * ALGORITHM: Backtracking
         * TC: O(n!) | SC: O(n)
         */
        private void permutation(char[] str, int l, int r) {
            if (l == r) {
                StringBuilder list = new StringBuilder();
                for (char c : str) list.append(String.valueOf(c));
                res.add(Integer.parseInt(list.toString()));
                return;
            }

            for (int i = l; i <= r; i++) {
                swap(str, l, i); // generate all permute
                permutation(str, l + 1, r);// fix position
                swap(str, l, i); // backtrack
            }
        }

        /*
         * PROBLEM: Swap Characters (Helper)
         * Swap characters at positions i and j in a char array.
         *
         * ALGORITHM: In-place swap
         * TC: O(1) | SC: O(1)
         */
        private void swap(char[] str, int i, int j) {
            char temp = str[i];
            str[i] = str[j];
            str[j] = temp;
        }
    }

    /*
     * PROBLEM: Reverse Sublist (Helper)
     * Reverse a portion of a list from start to end index.
     *
     * ALGORITHM: Two-pointer in-place reversal
     * TC: O(n) | SC: O(1)
     */
    public static void reverse(int start, int end, List<Integer> nums) {
        while (start < end) {
            int temp = nums.get(start);
            nums.set(start, nums.get(end));
            nums.set(end, temp);
            start++;
            end--;
        }
    }

    /*
     * PROBLEM: Next Greater Element III (LeetCode 556)
     * Find the next permutation of n's digits that is greater than n.
     *
     * ALGORITHM: Next permutation (find decreasing suffix, swap, reverse)
     * TC: O(n) | SC: O(n)
     */
    public int nextGreaterElement(int n) {
        List<Integer> nums = new ArrayList<>();
        while (n != 0) {
            int rem = n % 10;
            nums.add(rem);
            n = n / 10;
        }
        Collections.reverse(nums);
        int k = nums.size() - 2;
        while (k >= 0 && nums.get(k) >= nums.get(k + 1)) k--;
        if (k == -1) {
            return -1;
        }
        reverse(k + 1, nums.size() - 1, nums);
        for (int i = k + 1; i < nums.size(); i++) {
            if (nums.get(i) > nums.get(k)) {
                int temp = nums.get(i);
                nums.set(i, nums.get(k));
                nums.set(k, temp);
                break;
            }
        }
        long num = 0;
        int i = 0;
        while (i < nums.size()) {
            num = num * 10 + nums.get(i++);
        }
        return (num <= Integer.MAX_VALUE) ? (int) num : -1;
    }

    /*
     * PROBLEM: Find the Unsorted Subarray (LeetCode 581)
     * Find the length of the shortest subarray that if sorted makes the whole array sorted.
     *
     * ALGORITHM: Monotonic Stack (two passes)
     * TC: O(n) | SC: O(n)
     */
    public int findUnsortedSubarray(int[] nums) {
        Stack<Integer> s = new Stack<>(); // To store index of elements greater than current one in stack
        int n = nums.length;
        int left = n - 1;
        int right = 0;
        for (int i = 0; i <= n - 1; i++) {
            while (!s.isEmpty() && nums[i] < nums[s.peek()]) {
                left = Math.min(s.pop(), left);
            }
            s.add(i);
        }

        s = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!s.isEmpty() && nums[i] > nums[s.peek()]) {
                right = Math.max(s.pop(), right);
            }

            s.add(i);
        }

        return right - left > 0 ? right - left + 1 : 0;
    }

    /*
     * PROBLEM: Smallest Range Covering Elements from K Lists (LeetCode 632)
     * Find the smallest range that includes at least one number from each of k lists.
     *
     * ALGORITHM: Monotonic Stack on merged view (incomplete)
     * TC: O(n) | SC: O(n)
     */
    public int[] smallestRange(List<List<Integer>> nums) {
        int n = nums.size();

        new ArrayList<>(nums);


        Stack<Integer> s = new Stack<>(); // To store index of elements greater than current one in stack
        int left = n - 1;
        int right = 0;
        for (int i = 0; i <= n - 1; i++) {
            while (!s.isEmpty() && nums.get(i).get(i) < nums.get(i).get(s.peek())) {
                left = Math.min(s.pop(), left);
            }
            s.add(i);
        }

        s = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!s.isEmpty() && nums.get(i).get(i) > nums.get(i).get(s.peek())) {
                right = Math.max(s.pop(), right);
            }

            s.add(i);
        }

        return right - left > 0 ? new int[]{right - left + 1} : new int[]{0};
    }
    //count smaller on right using AVL
    // TC = O(nlogn) SC = O(n)

    public class HelloWorld {

        protected class TreeNode {
            int key;
            int height;
            int size;
            TreeNode left;
            TreeNode right;
            TreeNode parent;

            public TreeNode(final int key) {
                this.key = key;
                this.size = 1;
                this.height = 1;
                this.left = null;
                this.right = null;
            }
        }

        /*
         * PROBLEM: AVL Tree Node Size (Helper)
         * Return the size of a subtree, or 0 for null.
         *
         * ALGORITHM: Null check
         * TC: O(1) | SC: O(1)
         */
        public int size(final TreeNode node) {
            return node == null ? 0 : node.size;
        }

        /*
         * PROBLEM: AVL Tree Node Height (Helper)
         * Return the height of a subtree, or 0 for null.
         *
         * ALGORITHM: Null check
         * TC: O(1) | SC: O(1)
         */
        public int height(final TreeNode node) {
            return node == null ? 0 : node.height;
        }

        /*
         * PROBLEM: AVL Rotate Left (Helper)
         * Perform a left rotation to rebalance the AVL tree.
         *
         * ALGORITHM: AVL rotation
         * TC: O(1) | SC: O(1)
         */
        public TreeNode rotateLeft(final TreeNode root) {
            final TreeNode newRoot = root.right;
            final TreeNode leftSubTree = newRoot.left;

            newRoot.left = root;
            root.right = leftSubTree;

            root.height = max(height(root.left), height(root.right)) + 1;
            newRoot.height = max(height(newRoot.left), height(newRoot.right)) + 1;

            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

            return newRoot;
        }

        /*
         * PROBLEM: AVL Rotate Right (Helper)
         * Perform a right rotation to rebalance the AVL tree.
         *
         * ALGORITHM: AVL rotation
         * TC: O(1) | SC: O(1)
         */
        public TreeNode rotateRight(final TreeNode root) {
            final TreeNode newRoot = root.left;
            final TreeNode rightSubTree = newRoot.right;

            newRoot.right = root;
            root.left = rightSubTree;

            root.height = max(height(root.left), height(root.right)) + 1;
            newRoot.height = max(height(newRoot.left), height(newRoot.right)) + 1;

            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

            return newRoot;
        }

        /*
         * PROBLEM: Max of Two Integers (Helper)
         * Return the greater of two integers.
         *
         * ALGORITHM: Comparison
         * TC: O(1) | SC: O(1)
         */
        public int max(final int a, final int b) {
            return a >= b ? a : b;
        }

        /*
         * PROBLEM: Count Smaller Numbers on the Right (LeetCode 315) — AVL Insert (Helper)
         * Insert a key into AVL tree, updating the count of smaller elements seen so far.
         *
         * ALGORITHM: AVL Tree insertion with rank tracking
         * TC: O(log n) | SC: O(n)
         */
        public TreeNode insertIntoAVL(final TreeNode node, final int key, final int count[], final int index) {
            if (node == null) {
                return new TreeNode(key);
            }

            if (node.key > key) {
                node.left = insertIntoAVL(node.left, key, count, index);
            } else {
                node.right = insertIntoAVL(node.right, key, count, index);

                // update smaller elements count
                count[index] = count[index] + size(node.left) + 1;
            }

            // update the size and height
            node.height = max(height(node.left), height(node.right)) + 1;
            node.size = size(node.left) + size(node.right) + 1;

            // balance the tree
            final int balance = height(node.left) - height(node.right);
            // left-left
            if (balance > 1 && node.key > key) {
                return rotateRight(node);
            }
            // right-right
            if (balance < -1 && node.key > key) {
                return rotateLeft(node);
            }
            // left-right
            if (balance > 1 && node.key < key) {
                node.left = rotateLeft(node.left);
                return rotateRight(node);
            }
            // right-left
            if (balance > 1 && node.key < key) {
                node.right = rotateRight(node.right);
                return rotateLeft(node);
            }

            return node;
        }

        /*
         * PROBLEM: Count Smaller Numbers on the Right (LeetCode 315)
         * For each element, count how many elements to its right are smaller.
         *
         * ALGORITHM: AVL Tree insertion (right-to-left) with rank tracking
         * TC: O(n log n) | SC: O(n)
         */
        public int[] countSmallerOnRight(final int[] in) {
            final int[] smaller = new int[in.length];

            TreeNode root = null;
            for (int i = in.length - 1; i >= 0; i--) {
                root = insertIntoAVL(root, in[i], smaller, i);
            }

            return smaller;
        }


        /*
         * PROBLEM: HelloWorld Main Entry Point (Main)
         * Demonstrate countSmallerOnRight with sample input.
         *
         * ALGORITHM: Entry point
         * TC: O(n log n) | SC: O(n)
         */
        public void main(String[] args) {
            System.out.println("Hello World");

            int[] res = countSmallerOnRight(new int[]{100, 80, 70, 95, 10, 97});

            for (int r : res) System.out.println(r);
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

// Algo :-
/*
  - get the height of tree
  - create an array that store no.of nodes at each level
  - get number of nodes at each and store into array recursilvely
  - return max number of nodes at a level

*/


    /*
     * PROBLEM: Maximum Width of Binary Tree (LeetCode 662)
     * Return the maximum width (number of nodes between leftmost and rightmost) at any level.
     *
     * ALGORITHM: DFS with level array tracking node count per level
     * TC: O(n^2) | SC: O(n)
     */
    // TC = O(n^2), SC = O(n)
    // Fn to get max width of tree
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) return 0;
        int h = height(root);
        int level = 0;
        int[] arr = new int[h];

        getMaxWidthRecursively(root, arr, level);
        // preorder traversal of tree is needed
        return getMax(arr);
    }

    /*
     * PROBLEM: Max of Array (Helper)
     * Find the maximum value in an integer array.
     *
     * ALGORITHM: Linear scan
     * TC: O(n) | SC: O(1)
     */
    // O(n)
    private int getMax(int[] arr) {
        int max = Integer.MIN_VALUE;
        for (int n : arr) {
            max = Math.max(max, n);
        }
        return max;
    }

    /*
     * PROBLEM: Get Max Width Per Level Recursively (Helper)
     * Preorder DFS to increment node count for each level.
     *
     * ALGORITHM: DFS preorder traversal
     * TC: O(n) | SC: O(h)
     */
    // O(n^2) --> as for every pass we are getting 1 more pass for all nodes on that level
    // Preorder tree traversal to get all nodes on a level
    private void getMaxWidthRecursively(TreeNode root, int[] arr, int level) {

        if (root != null) {
            arr[level]++;
            getMaxWidthRecursively(root.left, arr, level + 1);
            getMaxWidthRecursively(root.right, arr, level + 1);
        }
    }

    /*
     * PROBLEM: Binary Tree Height (Helper)
     * Compute the height (number of levels) of a binary tree.
     *
     * ALGORITHM: DFS postorder
     * TC: O(n) | SC: O(h)
     */
    // Fn to get height of tree
    private int height(TreeNode root) {
        if (root == null) return 0;
        int lh = height(root.left);
        int rh = height(root.right);
        return 1 + Math.max(lh, rh);
    }

    /*
     * PROBLEM: Maximum Width of Binary Tree BFS (LeetCode 662)
     * Compute maximum width using BFS with virtual index labeling.
     *
     * ALGORITHM: BFS level-order with index numbering
     * TC: O(n) | SC: O(n)
     */
    private int maxWidthBfs(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        root.val = 0;
        q.add(root);

        int ans = 1;
        while (!q.isEmpty()) {
            int size = q.size();
            TreeNode start = q.peek();

            while (size-- > 0) {
                TreeNode temp = q.remove();

                if (temp.left != null) {
                    temp.left.val = temp.left.val * 2 + 1;
                    q.add(temp.left);
                }

                if (temp.right != null) {
                    temp.right.val = temp.right.val * 2 + 2;
                    q.add(temp.right);
                }

                if (size == 0) {
                    ans = Math.max(ans, temp.val - start.val + 1);
                }
            }
        }
        return ans;
    }


    /*
     * PROBLEM: Linked List in Binary Tree (LeetCode 1367)
     * Check if a linked list corresponds to a downward path in the binary tree.
     *
     * ALGORITHM: DFS + subpath matching
     * TC: O(n * m) | SC: O(h)
     */
    public boolean isSubPath(ListNode head, TreeNode root) {

        if (head == null) return true;
        if (root == null) return false;

        return isPath(head, root) && isSubPath(head, root.left) && isSubPath(head, root.right);
    }

    /*
     * PROBLEM: Is Path Matching (Helper)
     * Check if linked list starting at head matches a downward path from tree node root.
     *
     * ALGORITHM: Recursive path check
     * TC: O(m) | SC: O(m)
     */
    private boolean isPath(ListNode head, TreeNode root) {
        if (head == null) return true;
        if (root == null) return false;

        return head.val == root.val && (isPath(head.next, root.left) || isPath(head.next, root.right));
    }


    /*
     * PROBLEM: Tree Diameter (LeetCode 1245)
     * Find the diameter (longest path) in a tree represented as edge list.
     *
     * ALGORITHM: Double BFS (find farthest node, then find farthest from it)
     * TC: O(n) | SC: O(n)
     */
    // recursive soln
    public int treeDiameter(int[][] edges) {
        int n = edges.length;
        List<Set<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < n + 1; i++) graph.add(new HashSet<>());

        for (int[] e : edges) {
            int u = e[0], v = e[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }

        int[] distance = bfs(graph, 0); // used to find extremities of nodes in graph

        distance = bfs(graph, distance[0]); // distance b/w them
        return distance[1];
    }

    /*
     * PROBLEM: BFS on Graph (Helper)
     * Run BFS from start node, returning {farthest node, distance}.
     *
     * ALGORITHM: BFS level-order traversal
     * TC: O(n) | SC: O(n)
     */
    private int[] bfs(List<Set<Integer>> graph, int start) {

        boolean[] visited = new boolean[graph.size()];

        Arrays.fill(visited, false);
        visited[start] = true;
        LinkedList<Integer> queue = new LinkedList<>();
        queue.addLast(start);

        // bfs algo
        int lastNode = start, distance = -1;
        while (!queue.isEmpty()) {

            int size = queue.size();
            while (size-- > 0) {
                int nextNode = queue.removeFirst();
                for (Integer nbr : graph.get(nextNode)) {
                    if (!visited[nbr]) {
                        visited[nbr] = true;
                        queue.addLast(nbr);
                        lastNode = nbr;
                    }
                }
            }

            // after level is traversed
            distance += 1;
        }

        return new int[]{lastNode, distance};
    }

    /*
     * PROBLEM: Factorial (Helper)
     * Compute n! recursively.
     *
     * ALGORITHM: Recursion
     * TC: O(n) | SC: O(n)
     */
    public int factorial(int n) {
        // base case
        if (n == 0) return 1;

        int smallerOutput = factorial(n - 1);
        int answer = n * smallerOutput;
        return answer;
    }


    /*
        In = 512
        Ot = 1 + d(512/10) = 1+ d(51) = 1 + 1 + d(5) = 2 + 1 + d(0) = 3 + 0 = 3

        In = 102123
        Ot = 1 + d(10212) = 2 + d(1021) = 3 + d(102) = 4 + d(10) = 4 + d(10) = 5 + d(1) = 6 + d(0) = 6

     */
    /*
     * PROBLEM: Count Digits (Helper)
     * Count the number of digits in integer n recursively.
     *
     * ALGORITHM: Recursion by dividing by 10
     * TC: O(log n) | SC: O(log n)
     */
    public int digits(int n) {
        if (n <= 0) return 0;
        return 1 + digits(n / 10);
    }


    /*
       n = 10
       10 9 8  7 ....1
     */
    /*
     * PROBLEM: Print Numbers Descending (Helper)
     * Print numbers from n down to 2 recursively.
     *
     * ALGORITHM: Recursion (head recursion)
     * TC: O(n) | SC: O(n)
     */
    public void printNumbersDec(int n) {
        // base case
        if (n == 1) return;
        System.out.println(n);

        printNumbersDec(n - 1);
    }

    /*
       n = 10
       1 2  3 4 5 6 ... 10
     */
    /*
     * PROBLEM: Print Numbers Ascending (Helper)
     * Print numbers from 2 up to n recursively.
     *
     * ALGORITHM: Recursion (tail recursion)
     * TC: O(n) | SC: O(n)
     */
    public void printNumbersAsc(int n) {
        // base case
        if (n == 1) return;
        printNumbersDec(n - 1);
        System.out.println(n);
    }

    /*
     * PROBLEM: Tower of Hanoi (Helper)
     * Recursively solve Tower of Hanoi for n disks.
     *
     * ALGORITHM: Recursion
     * TC: O(2^n) | SC: O(n)
     */
    private void towerOfHanoi(int n, char source, char destination, char helper) {

        // base case
        if (n == 1) {
            System.out.println("Move 1st" + " disk from " + source + " to " + destination);
            return;
        }

        towerOfHanoi(n - 1, source, helper, destination);

        System.out.println("Move " + n + " disk from " + source + " to " + destination);


        towerOfHanoi(n - 1, helper, destination, source);
    }

    /*
     * PROBLEM: Keypad Get Options (Helper)
     * Return the letters corresponding to a phone keypad digit.
     *
     * ALGORITHM: Switch-case lookup
     * TC: O(1) | SC: O(1)
     */
    private static String getOptions(int n) {
        switch (n) {
            case 1:
                return "";
            case 2:
                return "abc";
            case 3:
                return "def";
            case 4:
                return "ghi";
            case 5:
                return "jkl";
            case 6:
                return "mno";
            case 7:
                return "pqrs";
            case 8:
                return "tuy";
            case 9:
                return "wxyz";
            default:
                return "";
        }
    }

    /*
     * PROBLEM: Letter Combinations of a Phone Number (LeetCode 17)
     * Return all possible letter combinations for the given phone number digits.
     *
     * ALGORITHM: Recursion (build combinations by appending options for each digit)
     * TC: O(4^n * n) | SC: O(4^n * n)
     */
    private static String[] keypad(int n) {
        // base-case
        if (n == 0) {
            String[] output = new String[1];
            output[0] = "";
            return output;
        }

        int lastDigit = n % 10;
        int remainingNumber = n / 10;

        String[] output = keypad(remainingNumber);
        String lastOptions = getOptions(lastDigit);


        String[] result = new String[output.length * lastOptions.length()];
        int index = 0;
        for (int i = 0; i < lastOptions.length(); i++) {
            for (int j = 0; j < output.length; j++) {
                result[index] = lastOptions.charAt(i) + output[j];
                index++;
            }
        }

        return result;
    }

    /*
     * PROBLEM: Determine Gender in Binary Tree (Helper)
     * Given a perfect binary tree depth n with alternating gender children, find gender of kth node.
     *
     * ALGORITHM: Recursion on binary tree structure
     * TC: O(n) | SC: O(n)
     */
    private static String gender(int n, int k, String rootGender) {
        // base-case
        if (n == 1) return rootGender;

        int c = (int) Math.pow(2, n - 1);
        // check in left subtree
        if (k <= c / 2) {
            return gender(n - 1, k, rootGender);
        } else {

            String child = "m";
            if (rootGender.equalsIgnoreCase("m")) child = "f";

            // check in right subtree
            return gender(n - 1, (k - c / 2), child);
        }
    }


    /*
     1->2<-3->4->5

    Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
reverseKnodes(2, 2, 3) -->
     */

    /*
     * PROBLEM: Reverse Nodes in k-Group (LeetCode 25)
     * Reverse every k consecutive nodes in a linked list.
     *
     * ALGORITHM: Recursion
     * TC: O(n) | SC: O(n/k)
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        // base case
        if (head == null) return null;
        ListNode curr = reverseKnodes(head, k);
        return reverseKGroup(curr, k);
    }

    /*
     * PROBLEM: Reverse K Nodes Helper (Helper)
     * Recursively reverse the first k nodes of a linked list.
     *
     * ALGORITHM: Recursion
     * TC: O(k) | SC: O(k)
     */
    // Fn. to revert first k nodes of list
    private ListNode reverseKnodes(ListNode head, int k) {
        // base case
        if (head == null) return null;
        if (k == 0) return head;

        // hypothsesis
        head.next.next = head.next;
        return reverseKnodes(head.next, k - 1);
    }

    /*
    Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
     Recursion:-


     */

    /*
     * PROBLEM: Reverse Linked List (LeetCode 206)
     * Reverse a singly linked list recursively.
     *
     * ALGORITHM: Recursion
     * TC: O(n) | SC: O(n)
     */
    public ListNode reverseList(ListNode head) {
        // base  case
        if (head == null || head.next == null) return head;

        // hypothesise
        ListNode p = reverseList(head.next);

        // induction
        head.next.next = head; // actual reverse step done
        head.next = null;
        return p;
    }

    /*
     * PROBLEM: Correct a Binary Tree (LeetCode 1660)
     * Remove the invalid node from a binary tree where one node points to a previously seen node.
     *
     * ALGORITHM: BFS level-order with parent tracking and duplicate detection
     * TC: O(n) | SC: O(n)
     */
    public TreeNode correctBinaryTree(TreeNode root) {

        if (root == null) return root;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean found = false;
        Map<TreeNode, TreeNode> parent = new HashMap<>();

        while (!queue.isEmpty()) {
            if (found) break;
            int size = queue.size();
            Map<TreeNode, TreeNode> map = new HashMap<>();
            for (int i = 0; i < size; i++) {
                TreeNode curr = queue.poll();
                if (map.containsKey(curr)) {
                    map.get(curr).right = null;
                    TreeNode pt = parent.get(map.get(curr));
                    if (pt.left == map.get(curr)) {
                        pt.left = null;
                    } else {
                        pt.right = null;
                    }
                    found = true;
                    break;
                } else {
                    if (curr.left != null) {
                        queue.offer(curr.left);
                        parent.put(curr.left, curr);
                    }
                    if (curr.right != null) {
                        queue.offer(curr.right);
                        map.put(curr.right, curr);
                        parent.put(curr.right, curr);
                    }
                }
            }
        }

        return root;
    }

    /*
     * PROBLEM: Time Needed to Inform All Employees (LeetCode 1376)
     * Find the total time to inform all employees in a company hierarchy.
     *
     * ALGORITHM: DFS with adjacency list
     * TC: O(n) | SC: O(n)
     */
    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<Integer>[] list = new ArrayList[n];
        for (int i = 0; i < n; i++) list[i] = new ArrayList<>();

        int src = 0;
        for (int i = 0; i < manager.length; i++) {
            if (manager[i] == -1) {
                src = i;
            } else list[manager[i]].add(i);
        }

        return helper(src, list, informTime);
    }

    /*
     * PROBLEM: Inform Employees DFS (Helper)
     * DFS from a manager node, accumulating inform time.
     *
     * ALGORITHM: DFS
     * TC: O(n) | SC: O(n)
     */
    private int helper(int src, List<Integer>[] list, int[] informTime) {
        int max = 0;
        for (int e : list[src]) {
            max = Math.max(helper(e, list, informTime), max);
        }
        return max + informTime[src];
    }

    int max = Integer.MIN_VALUE;

    /*
     * PROBLEM: Binary Tree Maximum Path Sum (LeetCode 124)
     * Find the maximum path sum in a binary tree where a path can start/end at any node.
     *
     * ALGORITHM: DFS postorder with global max tracking
     * TC: O(n) | SC: O(h)
     */
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return max;
    }

    /*
     * PROBLEM: Max Gain From Node (Helper)
     * Return maximum gain starting from this node; update global max if path through node is bigger.
     *
     * ALGORITHM: DFS postorder
     * TC: O(n) | SC: O(h)
     */
    private int maxGain(TreeNode root) {
        if (root == null) return 0;

        int left = Math.max(maxGain(root.left), 0);
        int right = Math.max(maxGain(root.right), 0);

        int priceNewPath = root.val + left + right;
        max = Math.max(max, priceNewPath);

        return root.val + Math.max(left, right);
    }

    /*
     * PROBLEM: Path Sum (LeetCode 112)
     * Check if a root-to-leaf path exists with the given target sum.
     *
     * ALGORITHM: DFS
     * TC: O(n) | SC: O(h)
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        return getTargetSum(root, new ArrayList<>(), targetSum);
    }

    /*
     * PROBLEM: Path Sum DFS (Helper)
     * Recursive DFS that tracks path nodes and checks leaf sum against target.
     *
     * ALGORITHM: DFS backtracking
     * TC: O(n) | SC: O(h)
     */
    private boolean getTargetSum(TreeNode node, ArrayList<TreeNode> nodelist, int targetSum) {
        if (node != null) {
            nodelist.add(node);
        }
        if (node.left != null) {
            getTargetSum(node.left, nodelist, targetSum);
        }

        if (node.right != null) {
            getTargetSum(node.right, nodelist, targetSum);
        } else if (node.left == null) {
            StringBuilder path = new StringBuilder();
            for (TreeNode treeNode : nodelist) {
                path.append(treeNode.val);
            }
            if (targetSum == Integer.parseInt(path.toString())) return true;
        }
        nodelist.remove(node);
        return false;
    }


    List<String> paths = new ArrayList<>();

    /*
     * PROBLEM: Sum Root to Leaf Numbers (LeetCode 129)
     * Treat each root-to-leaf path as a number and return sum of all such numbers.
     *
     * ALGORITHM: DFS path collection + sum
     * TC: O(n) | SC: O(h)
     */
    public int sumNumbers(TreeNode root) {
        printAllPossiblePath(root, new ArrayList<TreeNode>());
        int sum = 0;
        for (String num : paths) {
            System.out.println(num);
            sum += Integer.parseInt(num);
        }
        return sum;
    }


    /*
     * PROBLEM: Print All Root-to-Leaf Paths (Helper)
     * Collect all root-to-leaf number paths into the paths list.
     *
     * ALGORITHM: DFS backtracking
     * TC: O(n) | SC: O(h)
     */
    private void printAllPossiblePath(TreeNode node, ArrayList<TreeNode> nodelist) {
        if (node != null) {
            nodelist.add(node);
        }

        if (node.left != null) {
            printAllPossiblePath(node.left, nodelist);
        }

        if (node.right != null) {
            printAllPossiblePath(node.right, nodelist);
        } else if (node.left == null) {
            StringBuilder path = new StringBuilder();
            for (TreeNode treeNode : nodelist) {
                path.append(treeNode.val);
            }
            paths.add(path.toString());
        }
        nodelist.remove(node);

    }

    /*
     * PROBLEM: Remove Zeros from Number (Helper)
     * Return long n with all '0' digits removed.
     *
     * ALGORITHM: String manipulation
     * TC: O(n) | SC: O(n)
     */
    public long removeZeros(long n) {
        String s = String.valueOf(n);
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c != '0') sb.append(c);
        }
        return Long.valueOf(sb.toString());
    }

    /*
     * PROBLEM: Maximum Alternating Sum of a Subarray (LeetCode 1911)
     * Find the maximum alternating sum of any subsequence of nums.
     *
     * ALGORITHM: Sort by absolute value + alternate sign summation
     * TC: O(n log n) | SC: O(n)
     */
    public long maxAlternatingSum(int[] nums) {
        long ms = 0L;
        int n = nums.length;
        Integer[] boxedNumbers = Arrays.stream(nums).boxed().toArray(Integer[]::new);
        Arrays.sort(boxedNumbers, (a, b) -> Math.abs(b) - Math.abs(a));
        int ind = n % 2 == 0 ? Math.abs(n / 2) : Math.abs((n / 2) + 1);
        for (int num : boxedNumbers) {
            if (ind-- > 0) ms += (long) num * num;
            else ms -= (long) num * num;
        }
        return ms;
    }


    /*
     * PROBLEM: Count Stable Subarrays (Helper)
     * Count subarrays where capacity[l] == capacity[r] == sum of elements in between.
     *
     * ALGORITHM: Brute force O(n^2)
     * TC: O(n^2) | SC: O(1)
     */
    //TOOO : TLE
    public long countStableSubarrays(int[] capacity) {
        int n = capacity.length;
        int count = 0;

        // Iterate through all possible starting indices
        for (int l = 0; l < n - 2; l++) {
            long sum = 0;

            // Iterate through all possible ending indices
            for (int r = l + 2; r < n; r++) {
                // Add the element at r - 1 to the sum of elements in between
                sum += capacity[r - 1];

                // Check the stability condition
                if (capacity[l] == sum && capacity[r] == sum) {
                    count++;
                }
            }
        }
        return count;
    }
}


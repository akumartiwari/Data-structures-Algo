package com.company;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//      Definition for a binary tree node.
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode() {
    }

    public TreeNode(int val) {
        this.val = val;
    }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    /*
     * PROBLEM: Unique Binary Search Trees II (LeetCode 95)
     * Generate all structurally unique BSTs with values 1..n.
     *
     * ALGORITHM: Recursive enumeration of all BSTs
     * TC: O(4^n / n^(3/2)) | SC: O(4^n / n^(3/2))
     */
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> ans = new ArrayList<>();
        return generateTree(1, n, ans);
    }

    /*
     * PROBLEM: Generate BST Recursive (Helper)
     * Recursively build all valid BSTs for the range [start, end].
     *
     * ALGORITHM: Divide and conquer on each root choice
     * TC: O(4^n / n^(3/2)) | SC: O(4^n / n^(3/2))
     */
    private List<TreeNode> generateTree(int start, int end, List<TreeNode> ans) {

        if (start > end) {
            ans.add(null);
            return ans;
        }
        if (start == end) {
            ans.add(new TreeNode(start));
            return ans;
        }

        List<TreeNode> left, right;

        for (int i = start; i < end; i++) {
            left = generateTree(start, i - 1, ans);
            right = generateTree(i + 1, end, ans);

            for (TreeNode leftVal : left) {
                for (TreeNode rightVal : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftVal;
                    root.right = rightVal;
                    ans.add(root);
                }
            }
        }
        return ans;
    }


    /*
     * PROBLEM: Leaf-Similar Trees (LeetCode 872)
     * Check whether two binary trees have the same leaf value sequence.
     *
     * ALGORITHM: DFS leaf collection then list comparison
     * TC: O(n1 + n2) | SC: O(n1 + n2)
     */
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> leaves1 = new ArrayList<>();
        List<Integer> leaves2 = new ArrayList<>();
        dfs(root1, leaves1);
        dfs(root2, leaves2);
        return leaves1.equals(leaves2);
    }

    /*
     * PROBLEM: Leaf Collection DFS (Helper)
     * Collect all leaf values via DFS into a list.
     *
     * ALGORITHM: DFS pre-order
     * TC: O(n) | SC: O(n)
     */
    private void dfs(TreeNode root, List<Integer> leaves) {
        if (root == null) return;
        if (root.left == null && root.right == null) leaves.add(root.val);
        dfs(root.left, leaves);
        dfs(root.right, leaves);
    }

    /*
    Use Recursion to traverse subTree of a node and place braces around them
    Traverse the right subtree only if node is present and place braces in the string
     */
    /*
     * PROBLEM: Construct String from Binary Tree (LeetCode 606)
     * Create a string representation of a binary tree with brackets, omitting unnecessary empty brackets.
     *
     * ALGORITHM: DFS pre-order with recursive string building
     * TC: O(n) | SC: O(n)
     */
    public String tree2str(TreeNode root) {
        String ans = "";
        if (root == null) return "";
        ans += root.val;
        if (root.left == null && root.right == null) return ans;
        ans += "(" + tree2str(root.left) + ")";
        if (root.right != null) ans += "(" + tree2str(root.right) + ")";
        return ans;
    }

    /*
     * PROBLEM: Generate BST with Memoization (Helper)
     * Memoized recursion to generate all unique BSTs for range [start, end].
     *
     * ALGORITHM: Recursion + DP memoization
     * TC: O(4^n / n^(3/2)) | SC: O(n^2)
     */
    List<TreeNode> recursion(int start, int end, int[][] dp) {
        List<TreeNode> list = new ArrayList<>();
        if (dp[start][end] != -1) {
            list.add(new TreeNode(dp[start][end]));
            return list;
        }
        if (start > end) {
            list.add(null);
            return list;
        }
        if (start == end) {
            dp[start][end] = start;
            list.add(new TreeNode(start));
            return list;
        }
        List<TreeNode> left, right;
        for (int i = start; i <= end; i++) {
            left = recursion(start, i - 1, dp);
            right = recursion(i + 1, end, dp);
            for (TreeNode lst : left) {
                for (TreeNode rst : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = lst;
                    root.right = rst;
                    list.add(root);
                }
            }
        }
        return list;
    }

    /*
     * PROBLEM: Unique Binary Search Trees (LeetCode 96)
     * Count the number of structurally unique BSTs with n nodes.
     *
     * ALGORITHM: DP memoization (top-down)
     * TC: O(n^2) | SC: O(n^2)
     */
    public int numTrees(int n) {
        int[][] dp = new int[n + 1][n + 1];
        Arrays.fill(dp, -1);
        return recursion(1, n, dp).size();
    }

    /*
     * PROBLEM: Flatten Binary Tree to Linked List (LeetCode 114)
     * Flatten a binary tree to a linked list in-place using pre-order traversal.
     *
     * ALGORITHM: Iterative Morris-like: find rightmost node of left subtree and re-link
     * TC: O(n) | SC: O(1)
     */
    // left --> right --> root
    public void flatten(TreeNode root) {
        TreeNode curr = root;
        while (curr != null) {
            TreeNode left = curr.left;
            if (left != null) {
                TreeNode rightMost = getRightMost(left);
                rightMost.right = curr.right;
                curr.right = left;
                curr.left = null;
            }
            curr = curr.right;
        }
        List<Integer> ans = new ArrayList<>();

        ans.stream().forEach(System.out::println);
    }

    /*
     * PROBLEM: Get Rightmost Node (Helper)
     * Traverse right pointers until reaching the rightmost node in a subtree.
     *
     * ALGORITHM: Linear traversal
     * TC: O(n) | SC: O(1)
     */
    public TreeNode getRightMost(TreeNode node) {
        while (node.right != null) node = node.right;
        return node;
    }


    /*
     * PROBLEM: Next Greater Element I (LeetCode 496)
     * For each element of nums1, find the next greater element in nums2.
     *
     * ALGORITHM: HashMap lookup + linear search from position
     * TC: O(n1 * n2) | SC: O(n2)
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        int[] ans = new int[n1];
        Arrays.fill(ans, -1);
        HashMap<Integer, Integer> map = new HashMap<>();
        // store nums2 in map
        for (int num = 0; num < n2; num++) map.put(nums2[num], num);

        for (int i = 0; i < n1; i++) {
            Integer index = map.get(nums1[i]);
            if (map.get(nums1[i]) != -1) {
                int j = index;
                while (j < nums2.length && nums2[j] <= nums1[i]) j++;
                if (j != n2) ans[i] = nums2[j];
            }
        }
        return ans;
    }

    /*
     * PROBLEM: Next Greater Element II (LeetCode 503)
     * Given a circular array, find the next greater element for each position.
     *
     * ALGORITHM: Double-pass linear scan with wrap-around
     * TC: O(n) | SC: O(n)
     */
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);

        for (int i = 0; i < n; i++) {
            int counter = 0;
            int j = i + 1;
            while (counter < n && j < n && nums[j] <= nums[i]) {
                counter++;
                if (j == n - 1) {
                    j = 0;
                    continue;
                }
                j++;
            }
            if (counter != n) ans[i] = nums[j];
        }
        return ans;
    }

    /*
     * PROBLEM: Count Pythagorean Triples (Helper)
     * Count triples (a, b, c) where 1≤a,b,c≤n and a²+b²=c².
     *
     * ALGORITHM: HashMap of squares + nested loop
     * TC: O(n^2) | SC: O(n)
     */
    public int countTriples(int n) {
        if (n == 0 || n == 1) return 0;

        HashMap<Integer, Integer> hashMap = new HashMap<>();

        // hashing
        for (int i = 1; i <= n; i++) {
            hashMap.put(i * i, i);
        }

        // check for conidtion :- a2 + b2 = c2
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                int elem = i * i - j * j;
                if (hashMap.get(elem) != null) ans++;
            }
        }
        return ans;
    }

    /*
     * PROBLEM: Sum Game (LeetCode 1927)
     * Determine if Alice wins a game where she and Bob fill '?' with digits to equalize sums.
     *
     * ALGORITHM: Greedy / positional analysis
     * TC: O(n) | SC: O(1)
     */
    public boolean sumGame(String num) {
        int n = num.length();
        if (n == 0) return false;

        int a = 0;
        int b = 0;
        boolean ans = false;

        int first = 0;
        int second = 0;

        for (int i = 0; i < n; i++) {
            if (num.charAt(i) == '?') {
                // cases:-
                    /*
                        case 1  --> next place is an '?' ==> fill it with  9 (max digit)
                        case 2 --> next place is not an '?' ==> then follow the algo below
                     */
                // if total character is in odd place
                if (i % 2 == 0) {
                    for (int j = 0; j < i; j++) {
                        if (j < i / 2) first += Integer.parseInt(String.valueOf(num.charAt(j)));
                        else second += Integer.parseInt(String.valueOf(num.charAt(j)));
                    }
                    second += 9; // to replace '?' with greatest value by alice
                    return first == second;
                }
            }
        }
        return false;
    }

    /*
     * PROBLEM: Merge BSTs to Create Single BST (LeetCode 1932)
     * Merge a list of BST roots into a single valid BST.
     *
     * ALGORITHM: Iterative merge via inorder lists + sorted merge + array-to-BST
     * TC: O(n log n) | SC: O(n)
     */
    public TreeNode canMerge(List<TreeNode> trees) {
        TreeNode node = null;
        for (int i = 0; i < trees.size(); i++) {
            ArrayList<Integer> list1;
            ArrayList<Integer> list2;
            ArrayList<Integer> list3;
            if (i == 0 && trees.size() > 1) {
                //Stores Inorder of tree1 to list1
                list1 = storeInorder(trees.get(i));

                //Stores Inorder of tree2 to list2
                list2 = storeInorder(trees.get(i + 1));

                // Merges both list1 and list2 into list3
                list3 = merge(list1, list2, list1.size(), list2.size());

                //Eventually converts the merged list into resultant BST
                node = ALtoBST(list3, 0, list3.size() - 1);
                i++;
            } else {

                //Stores Inorder of tree1 to list1
                list1 = storeInorder(node);

                //Stores Inorder of tree2 to list2
                list2 = storeInorder(trees.get(i));

                // Merges both list1 and list2 into list3
                list3 = merge(list1, list2, list1.size(), list2.size());

                //Eventually converts the merged list into resultant BST
                node = ALtoBST(list3, 0, list3.size() - 1);
            }
        }
        return node;
    }

    /*
     * PROBLEM: Array List to BST (Helper)
     * Convert a sorted ArrayList into a balanced BST.
     *
     * ALGORITHM: Recursive mid-point construction
     * TC: O(n) | SC: O(log n)
     */
    // Method that converts an ArrayList to a BST
    TreeNode ALtoBST(ArrayList<Integer> list, int start, int end) {
        // Base case
        if (start > end)
            return null;

        // Get the middle element and make it root
        int mid = (start + end) / 2;
        TreeNode node = new TreeNode(list.get(mid));

        /* Recursively construct the left subtree and make it
        left child of root */
        node.left = ALtoBST(list, start, mid - 1);

        /* Recursively construct the right subtree and make it
        right child of root */
        node.right = ALtoBST(list, mid + 1, end);

        return node;
    }

    /*
     * PROBLEM: Merge Two Sorted ArrayLists (Helper)
     * Merge two sorted ArrayLists into one sorted ArrayList.
     *
     * ALGORITHM: Two-pointer merge
     * TC: O(m+n) | SC: O(m+n)
     */
    // Method that merges two ArrayLists into one.
    ArrayList<Integer> merge(ArrayList<Integer> list1, ArrayList<Integer> list2, int m, int n) {
        // list3 will contain the merge of list1 and list2
        ArrayList<Integer> list3 = new ArrayList<>();
        int i = 0;
        int j = 0;

        //Traversing through both ArrayLists
        while (i < m && j < n) {
            // Smaller one goes into list3
            if (list1.get(i) < list2.get(j)) {
                list3.add(list1.get(i));
                i++;
            } else {
                list3.add(list2.get(j));
                j++;
            }
        }

        // Adds the remaining elements of list1 into list3
        while (i < m) {
            list3.add(list1.get(i));
            i++;
        }
        // Adds the remaining elements of list2 into list3
        while (j < n) {
            list3.add(list2.get(j));
            j++;
        }
        return list3;
    }

    /*
     * PROBLEM: Store Inorder Traversal (Helper)
     * Collect all values from a BST via inorder traversal into an ArrayList.
     *
     * ALGORITHM: DFS inorder
     * TC: O(n) | SC: O(n)
     */
    // Method that stores inorder traversal of a tree
    ArrayList<Integer> storeInorder(TreeNode node) {
        ArrayList<Integer> list1 = new ArrayList<>();
        ArrayList<Integer> list2 = storeInorderUtil(node, list1);
        return list2;
    }

    /*
     * PROBLEM: Store Inorder Utility (Helper)
     * Recursive utility to collect inorder values into a provided ArrayList.
     *
     * ALGORITHM: DFS inorder
     * TC: O(n) | SC: O(n)
     */
    // A Utility Method that stores inorder traversal of a tree
    public ArrayList<Integer> storeInorderUtil(TreeNode node, ArrayList<Integer> list) {
        if (node == null)
            return list;

        //recur on the left child
        storeInorderUtil(node.left, list);

        // Adds data to the list
        list.add(node.val);

        //recur on the right child
        storeInorderUtil(node.right, list);

        return list;
    }

    /*
     * PROBLEM: Color the Grid (LeetCode 1931)
     * Count valid coloring configurations for an m x n grid with 3 colors.
     *
     * ALGORITHM: Mathematical formula (placeholder approximation)
     * TC: O(1) | SC: O(1)
     */
    public int colorTheGrid(int m, int n) {
        return (int) (Math.pow(Math.pow(m, n), 3) % 1000000007);
    }

    /*
     * PROBLEM: Find Maximum Index (Helper)
     * Find the index of the maximum element in nums[start..end].
     *
     * ALGORITHM: Linear scan
     * TC: O(n) | SC: O(1)
     */
    private int maxIndex(int start, int end, int[] nums) {
        int index = Integer.MIN_VALUE;
        int max = Integer.MIN_VALUE;

        for (int i = start; i <= end; i++) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        return index;
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
    /*
     * PROBLEM: Maximum Binary Tree (LeetCode 654)
     * Construct a maximum binary tree from an array where each node is the max of its subarray.
     *
     * ALGORITHM: Iterative insertion with right-spine traversal
     * TC: O(n) | SC: O(n)
     */
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode node = new TreeNode(val);
        // base case
        if (root == null) return node;

        // when val is greater than root node itself
        if (root.val < val) {
            node.left = root;
            node.right = null;
            return node;
        }

        TreeNode current = root;
        while (current.right != null && current.right.val > val) {
            current = current.right;
        }
        // if we have reached at last right node

        if (current.right == null) current.right = node;
            // we have found a node with value greater than val
        else {
            node.left = current.right;
            node.right = null;
            current.right = node;
        }
        return root;
    }

    /*
     * PROBLEM: Find Duplicate Subtrees (LeetCode 652)
     * Find all subtrees in the binary tree that appear more than once.
     *
     * ALGORITHM: DFS serialization + HashMap for duplicate detection
     * TC: O(n^2) | SC: O(n^2)
     */
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        List<TreeNode> ans = new ArrayList<>();
        getAllSubTrees(root, new HashMap<>(), ans);
        return ans;
    }

    /*
     * PROBLEM: Serialize Subtree and Detect Duplicates (Helper)
     * Serialize subtree to string and add to result if seen twice.
     *
     * ALGORITHM: DFS post-order serialization
     * TC: O(n^2) | SC: O(n^2)
     */
    private String getAllSubTrees(TreeNode root, HashMap<String, Integer> map, List<TreeNode> ans) {

        if (root == null) return " ";
        //inorder recursion call stack
        // check if tree is non-empty then add to ans

        String curr = "^" + getAllSubTrees(root.left, map, ans) + root.val + getAllSubTrees(root.right, map, ans);
        int val = map.getOrDefault(curr, 0);
        // if curr value already exists in map  ie. duplicate
        // check if tree is non-empty then add to ans
        if (val == 1) ans.add(root);
        map.put(curr, val + 1);
        return curr;
    }

    /*
    // Algo :-
     * Iterate through all array elements
     * For every elem split array to m equal halfs
     * Find largest amomg them
     * Update global min value from them
     * return min


    Input: nums = [7,2,5,10,8], m = 2
    Output: 18


    First half till ith position and remaining half can be found from left elements of array

    Dry-run :-


    [1,4,4]
    3
     */
    /*
     * PROBLEM: Split Array Largest Sum - Optimised (LeetCode 410)
     * Split array into at most m parts to minimise the largest subarray sum.
     *
     * ALGORITHM: Binary search on answer + greedy validation
     * TC: O(n log(sum)) | SC: O(1)
     */
    public int splitArrayOpt(int[] nums, int m) {
        int low = IntStream.of(nums).max().orElse(0);
        int high = IntStream.of(nums).sum();
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (split(nums, mid) > m) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    /*
     * PROBLEM: Split Array Validation (Helper)
     * Count minimum parts needed when no part's sum exceeds the given sum threshold.
     *
     * ALGORITHM: Greedy linear scan
     * TC: O(n) | SC: O(1)
     */
    private int split(int[] nums, int sum) {
        int ret = 1;
        int currentSum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (currentSum + nums[i] > sum) {
                ret++;
                currentSum = 0;
            }
            currentSum += nums[i];
        }
        return ret;
    }

    /*
    Input: nums = [7,2,5,10,8], m = 2
    Output: 18
     */
    /*
     * PROBLEM: Split Array Largest Sum - Brute Force (LeetCode 410)
     * Split array into m parts and minimise the largest subarray sum (brute force).
     *
     * ALGORITHM: Brute force enumeration of split positions
     * TC: O(n^2 * m) | SC: O(n)
     */
    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int min = Integer.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            List<List<Integer>> ans = new ArrayList<>();
            // Insert elements from first half
            int indexF = 0;
            while (indexF / m <= i) {
                List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(indexF, indexF + m);
                ans.add(list);
                indexF += m;
            }

            // Insert elements from halfs left till now
            int index = indexF;
            while (index / m < n) {
                List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(index, index + m);
                ans.add(list);
                index += m;
            }
            // for unequal halfs add remaining elements
            ans.add(Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(index, n));

            Optional<Integer> largestSum = ans.stream().map(x -> x.stream().mapToInt(Integer::intValue).sum())
                    .collect(Collectors.toList()).stream().max(Comparator.comparingInt(x -> x));

            if (largestSum.isPresent()) min = Math.min(min, largestSum.get());
        }
        return min;
    }

    /*
     * PROBLEM: Best Time to Buy and Sell Stock (LeetCode 121)
     * Find the maximum profit from a single buy-sell transaction.
     *
     * ALGORITHM: DP (track running minimum price)
     * TC: O(n) | SC: O(n)
     */
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] dp = new int[n];
        int min = prices[0];
        dp[0] = 0;

        // keep track of max profit at every stage
        for (int i = 1; i < n; i++) {
            min = Math.min(min, prices[i]);
            dp[i] = Math.max(dp[i - 1], prices[i] - min);
        }
        return dp[n - 1];
    }

    /*
     * PROBLEM: Reverse Linked List II (LeetCode 92)
     * Reverse nodes of a linked list from position left to right.
     *
     * ALGORITHM: Two-pointer iterative reversal in-place
     * TC: O(n) | SC: O(1)
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {

        // base case
        if (head == null) return head;

        ListNode curr = head;
        ListNode prev = null;

        while (left > 1) {
            ListNode next = curr.next;
            // move pointer ahead
            prev = curr;
            curr = next;
            left--;
            right--;
        }

        ListNode connection = prev;
        ListNode tail = curr;

        // reverse the actual list nodes
        while (right > 0) {
            // store the next of current on next node
            ListNode next = curr.next;
            // reverse the linked list
            curr.next = prev;
            // move pointer ahead
            prev = curr;
            curr = next;
            right--;
        }

        if (connection != null) {
            connection.next = prev;
        } else {
            head = prev;
        }
        tail.next = curr;
        return head;
    }

    /*
     * PROBLEM: Two Sum IV - Input is a BST (LeetCode 653)
     * Find if there exist two nodes in BST whose values sum to k.
     *
     * ALGORITHM: DFS + HashSet for complement lookup
     * TC: O(n) | SC: O(n)
     */
    public boolean findTarget(TreeNode root, int k) {
        return getAllNodes(root, new HashSet<>(), k);
    }

    /*
     * PROBLEM: Two Sum BST DFS (Helper)
     * DFS helper to check if any two nodes in the BST sum to k.
     *
     * ALGORITHM: DFS + HashSet
     * TC: O(n) | SC: O(n)
     */
    private Boolean getAllNodes(TreeNode root, Set<Integer> set, int k) {
        if (root == null) return false;
        if (set.contains(k - root.val)) return true;
        set.add(root.val);
        boolean left = getAllNodes(root.left, set, k);
        boolean right = getAllNodes(root.right, set, k);
        return left || right;
    }

    /*
     * PROBLEM: Maximum Binary Tree (LeetCode 654)
     * Construct a maximum binary tree from nums, where each root is the maximum of its subarray.
     *
     * ALGORITHM: Recursive divide and conquer
     * TC: O(n^2) | SC: O(n)
     */
    //Author: Anand
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return cmt(0, nums.length - 1, nums);
    }

    /*
     * PROBLEM: Construct Maximum Binary Tree Recursive (Helper)
     * Recursively build the maximum binary tree for range [start, end].
     *
     * ALGORITHM: Recursive max-finding + tree construction
     * TC: O(n^2) | SC: O(n)
     */
    private TreeNode cmt(int start, int end, int[] nums) {
        if (start > end) return null;
        if (start == end) return new TreeNode(nums[start]);
        int max = Integer.MIN_VALUE;
        int ind = -1;
        for (int i = start; i <= end; i++) {
            if (nums[i] > max) {
                max = Math.max(max, nums[i]);
                ind = i;
            }
        }

        TreeNode root = new TreeNode(max);
        root.left = cmt(start, ind - 1, nums);
        root.right = cmt(ind + 1, end, nums);
        return root;
    }


}

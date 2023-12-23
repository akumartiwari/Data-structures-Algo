package com.company;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//      Definition for a binary tree node.
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> ans = new ArrayList<>();
        return generateTree(1, n, ans);
    }

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
    Use Recursion to traverse subTree of a node and place braces around them
    Traverse the right subtree only of node is present and place braces in the string
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

    public int numTrees(int n) {
        int[][] dp = new int[n + 1][n + 1];
        Arrays.fill(dp, -1);
        return recursion(1, n, dp).size();
    }

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

    public TreeNode getRightMost(TreeNode node) {
        while (node.right != null) node = node.right;
        return node;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        // base cases
           /*
            int n = temperatures.length;

            int[] ans = new int[n];
            // iterate throught all element
            for (int i = 0; i < n; i++) {
                int count = 0;
                for (int j = i + 1; j < n; j++) {
                    count++;
                    if (temperatures[j] > temperatures[i]) ans[i] = count;
                }
            }

            return ans;
        }
        */


        int n = temperatures.length;
        int[] nextWarmerday = new int[n];
        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack

        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday[idx] = idx - i;
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

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

    // Method that stores inorder traversal of a tree
    ArrayList<Integer> storeInorder(TreeNode node) {
        ArrayList<Integer> list1 = new ArrayList<>();
        ArrayList<Integer> list2 = storeInorderUtil(node, list1);
        return list2;
    }

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

    public int colorTheGrid(int m, int n) {
        return (int) (Math.pow(Math.pow(m, n), 3) % 1000000007);
    }

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

    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        List<TreeNode> ans = new ArrayList<>();
        getAllSubTrees(root, new HashMap<>(), ans);
        return ans;
    }

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

    public boolean findTarget(TreeNode root, int k) {
        return getAllNodes(root, new HashSet<>(), k);
    }

    private Boolean getAllNodes(TreeNode root, Set<Integer> set, int k) {
        if (root == null) return false;
        if (set.contains(k - root.val)) return true;
        set.add(root.val);
        boolean left = getAllNodes(root.left, set, k);
        boolean right = getAllNodes(root.right, set, k);
        return left || right;
    }

    //Author: Anand
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return cmt(0, nums.length - 1, nums);
    }

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

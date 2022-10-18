package com.company;

import java.util.*;
import java.util.stream.Collectors;

class SolutionRecursion {

    Map<String, List<Integer>> map;

    // using Recursion
    public int[] sortArray(int[] nums) {
        int n = nums.length;
        map = new HashMap<>();
        if (n == 1) return nums;

        List<Integer> sorted = sort(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        return sorted.stream().mapToInt(x -> x).toArray();
    }

    private List<Integer> sort(List<Integer> nums) {
        int n = nums.size();
        if (n == 0 || n == 1) return nums;

        // Hypothesis
        int temp = nums.get(n - 1);
        nums.remove(new Integer(temp));
        sort(nums);
        return insert(nums, temp);
    }

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
    //         int[] arr = {100,10,89,40,1,80,97};

    private static List<Integer> countcolderDays1(int[] temperatures) {
        int n = temperatures.length;
        List<Integer> nextWarmerday = new ArrayList<>();

        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack
        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday.add(i - idx);
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

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

        Map<Integer, Integer> valueSorted =
                map.entrySet().stream()
                        .sorted(Map.Entry.comparingByValue())
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                                (e1, e2) -> e1, LinkedHashMap::new));


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


    // TC = O(N), SC = O(N)
    // FAB PROBLEM
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Stack<Integer> stk = new Stack<>(); // to store next greater elements in stack
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[i % n] >= nums[stk.peek()]) stk.pop();

            ans[i % n] = stk.isEmpty() ? -1 : nums[stk.peek()];
            stk.push(i % n);
        }
        return ans;
    }

    class Solution {
        PriorityQueue<Integer> res;

        public int nextGreaterElement(int n) {
            String str = Integer.toString(n);
            res = new PriorityQueue<>();
            permutation(str.toCharArray(), 0, 0);

            while (!res.isEmpty() && res.peek() <= n) res.poll();
            return res.peek() != null ? res.poll() : -1;
        }

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

        private void swap(char[] str, int i, int j) {
            char temp = str[i];
            str[i] = str[j];
            str[j] = temp;
        }
    }

    public static void reverse(int start, int end, List<Integer> nums) {
        while (start < end) {
            int temp = nums.get(start);
            nums.set(start, nums.get(end));
            nums.set(end, temp);
            start++;
            end--;
        }
    }

    public int nextGreaterElement(int n) {
        List<Integer> nums = new ArrayList<>();
        while (n != 0) {
            int rem = n % 10;
            nums.add(rem);
            n = n / 10;
        }
        Collections.reverse(nums);
        int k = nums.size() - 2;
        while (k >= 0 && nums.get(k) >= nums.get(k + 1))
            k--;
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

        public int size(final TreeNode node) {
            return node == null ? 0 : node.size;
        }

        public int height(final TreeNode node) {
            return node == null ? 0 : node.height;
        }

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

        public int max(final int a, final int b) {
            return a >= b ? a : b;
        }

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

        public int[] countSmallerOnRight(final int[] in) {
            final int[] smaller = new int[in.length];

            TreeNode root = null;
            for (int i = in.length - 1; i >= 0; i--) {
                root = insertIntoAVL(root, in[i], smaller, i);
            }

            return smaller;
        }


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

    // O(n)
    private int getMax(int[] arr) {
        int max = Integer.MIN_VALUE;
        for (int n : arr) {
            max = Math.max(max, n);
        }
        return max;
    }

    // O(n^2) --> as for every pass we are getting 1 more pass for all nodes on that level
    // Preorder tree traversal to get all nodes on a level
    private void getMaxWidthRecursively(TreeNode root, int[] arr, int level) {

        if (root != null) {
            arr[level]++;
            getMaxWidthRecursively(root.left, arr, level + 1);
            getMaxWidthRecursively(root.right, arr, level + 1);
        }
    }

    // Fn to get height of tree
    private int height(TreeNode root) {
        if (root == null) return 0;
        int lh = height(root.left);
        int rh = height(root.right);
        return 1 + Math.max(lh, rh);
    }

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


    public boolean isSubPath(ListNode head, TreeNode root) {

        if (head == null) return true;
        if (root == null) return false;

        return isPath(head, root) && isSubPath(head, root.left) && isSubPath(head, root.right);
    }

    private boolean isPath(ListNode head, TreeNode root) {
        if (head == null) return true;
        if (root == null) return false;

        return head.val == root.val && (isPath(head.next, root.left) || isPath(head.next, root.right));
    }


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
    public int digits(int n) {
        if (n <= 0) return 0;
        return 1 + digits(n / 10);
    }


    /*
       n = 10
       10 9 8  7 ....1
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
    public void printNumbersAsc(int n) {
        // base case
        if (n == 1) return;
        printNumbersDec(n - 1);
        System.out.println(n);
    }

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

    public ListNode reverseKGroup(ListNode head, int k) {
        // base case
        if (head == null) return null;
        ListNode curr = reverseKnodes(head, k);
        return reverseKGroup(curr, k);
    }

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

    private int helper(int src, List<Integer>[] list, int[] informTime) {
        int max = 0;
        for (int e : list[src]) {
            max = Math.max(helper(e, list, informTime), max);
        }
        return max + informTime[src];
    }

    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return max;
    }

    private int maxGain(TreeNode root) {
        if (root == null) return 0;

        int left = Math.max(maxGain(root.left), 0);
        int right = Math.max(maxGain(root.right), 0);

        int priceNewPath = root.val + left + right;
        max = Math.max(max, priceNewPath);

        return root.val + Math.max(left, right);
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        return getTargetSum(root, new ArrayList<>(), targetSum);
    }

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

    public int sumNumbers(TreeNode root) {
        printAllPossiblePath(root, new ArrayList<TreeNode>());
        int sum = 0;
        for (String num : paths) {
            System.out.println(num);
            sum += Integer.parseInt(num);
        }
        return sum;
    }


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
}

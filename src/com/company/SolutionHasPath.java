package com.company;

import java.util.*;

class SolutionHasPath {
    int rows;
    int cols;
    int[][] vis;
    String dir = "URLD"; // lexico order
    int[] dirx = {1, 0, -1, 0};
    int[] diry = {0, 1, 0, -1};


    public static String OrderOfFrequencies(String S) {

        //this is default OUTPUT. You can change it.
        StringBuilder result = new StringBuilder(" ");

        HashMap<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        HashMap<Integer, Character> reversedMap = new HashMap<>();

        map.keySet().forEach(elem -> reversedMap.put(map.get(elem), elem));

        Map<Integer, Character> treeMap = new TreeMap<>(reversedMap);

        for (Integer c : treeMap.keySet()) {
            result.append(treeMap.get(c));
        }

        return result.toString();
    }

    private boolean validateDest(int[][] maze, int[] destination) {
        int s = destination[0];
        int e = destination[1];

        boolean isOpen = false;
        // destination is at valid position
        if (maze[s][e] == 1) return false;

        if (s + 1 < rows && maze[s + 1][e] == 0)
            isOpen = true;

        if (s - 1 >= 0 && maze[s - 1][e] == 0) {
            if (isOpen) return false;
            isOpen = true;
        }
        if (e + 1 < cols && maze[s][e + 1] == 0) {
            if (isOpen) return false;
            isOpen = true;
        }
        if (e - 1 >= 0 && maze[s][e - 1] == 0) {
            if (isOpen) return false;
        }
        return true;
    }

    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        rows = maze.length;
        cols = maze[0].length;
        vis = new int[rows][cols];
        for (int[] elem : vis) Arrays.fill(elem, 0);

//            // validate destination
//            if (!validateDest(maze, destination)) return false;

        return hasPathRecursive(maze, start[0], start[1], destination, vis);
    }

    // Recursive solution of maze
    private boolean hasPathRecursive(int[][] maze, int i, int j, int[] destination, int[][] vis) {

        // boundary conditions
        if (i >= rows || j >= cols || i < 0 || j < 0 || vis[i][j] == 1 || maze[i][j] == 1) return false;

        // base case
        // apply case for destination  has 2 adjacent  empty spaces
        if (i == destination[0] && j == destination[1]) {
            return true;
        }

        // mark visited as true
        vis[i][j] = 1;

        for (int x = 0; x < 4; x++) {
            boolean pathFound = hasPathRecursive(maze, i + dirx[x], j + diry[x], destination, vis);
            if (pathFound) return true;
        }
        // backtrack
        vis[i][j] = 0;
        return false;
    }


    int max = Integer.MIN_VALUE;

    public int maxSumBST(TreeNode root) {
        if (root == null) return 0;
        subTree(root);
        return Math.max(max, 0);
    }

    private SubTree subTree(TreeNode root) {

        if (root == null) return null;

        SubTree currTree = new SubTree(root.val);
        SubTree leftSubTree = subTree(root.left);
        SubTree rightSubTree = subTree(root.right);

        if (leftSubTree != null) {
            currTree.bst &= leftSubTree.bst && root.val > leftSubTree.max;
            currTree.min = leftSubTree.min;
            currTree.sum += leftSubTree.sum;
        }

        if (rightSubTree != null) {
            currTree.bst &= rightSubTree.bst && root.val < rightSubTree.min;
            currTree.max = rightSubTree.max;
            currTree.sum += rightSubTree.sum;
        }

        if (currTree.bst) max = Math.max(max, currTree.sum);
        return currTree;
    }

    class SubTree {
        int max;
        int min;
        int sum;
        boolean bst;

        SubTree(int val) {
            this.max = val;
            this.min = val;
            this.sum = val;
            this.bst = true;
        }
    }


    /*

      [5,2,1,3,6]
      val  = 5
      idx = 1
      lo = check(-INF, 5, nums)
      ro = check(5,INF, nums)

     */
    int idx = 0;

    public boolean verifyPreorder(int[] preorder) {
        return check(Integer.MIN_VALUE, Integer.MAX_VALUE, preorder);
    }

    private boolean check(int lb, int up, int[] nums) {
        // base case
        if (idx == nums.length) return true;


        int val = nums[idx];
        if (val < lb || val > up) return false;

        idx++;
        boolean leftOrder = check(lb, val, nums);
        boolean rightOrder = check(val, up, nums);

        return leftOrder || rightOrder;
    }

    public int numIdenticalPairs(int[] nums) {

        int n = nums.length;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int j = i + 1; j < n; j++) {
                if (nums[j] == nums[i]) cnt++;
            }
            ans += cnt;
        }
        return ans;
    }

    List<Integer> ans = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) return ans;
        helper(root, 0);
        return ans;
    }


    private void helper(TreeNode root, int level) {
        if (level == ans.size()) ans.add(root.val);
        if (root.right != null) helper(root.right, level + 1);
        if (root.left != null) helper(root.left, level + 1);
    }

    /*
     // Algo:-
       Pseudo code :-
     - create a tree  with given preorder traversal and traverse to get all subpaths
     - check for current subpath size
        - if ss == (num-k) then add to ans
        - else keep on getting paths
     - ans array will hold all possible paths with size = nums.size()-k;
     - scan to get the min value
     - return min_value
     */

    public String removeKdigits(String num, int k) {
        int n = num.length();
        Stack<Integer> stk = new Stack<>();

        int elem = k;
        for (int i = 0; i < n; i++) {
            int curr = Integer.parseInt(String.valueOf(num.charAt(i)));
            while (!stk.isEmpty() && stk.peek() > curr && elem > 0) {
                stk.pop();
                elem--;
            }

            stk.push(curr);
        }

        while (stk.size() > n - k) stk.pop(); // It will remove all the trailing sequence of numbers

        if (stk.isEmpty()) return "0";
        StringBuilder sb = new StringBuilder("");
        while (!stk.isEmpty()) sb.append(stk.pop());

        String regex = "^0+(?!$)";
        return sb.reverse().toString().replaceAll(regex, "");
    }

    // TBD FROM 10 aug onwards

    public int[] mostCompetitive(int[] nums, int k) {
        Deque<Integer> queue = new ArrayDeque<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (!queue.isEmpty() && queue.peekLast() > nums[i] && queue.size() + n - i > k
            ) queue.pollLast();

            queue.addLast(nums[i]);
        }

        int[] ans = new int[k];
        for (int m = 0; m < k; m++) {
            ans[m] = queue.pollFirst();
        }

        return ans;
    }

    public String smallestSubsequence(String s) {
        int n = s.length();
        Stack<Character> stk = new Stack<>();

        int[] freq = new int[26];
        boolean[] exist = new boolean[26];
        Arrays.fill(exist, false);

        for (char ch : s.toCharArray()) freq[ch - 'a']++;
        for (int i = 0; i < n; i++) {

            char ch = s.charAt(i);
            freq[ch - 'a']--;
            if (exist[ch - 'a']) continue;

            while (!stk.isEmpty() && stk.peek() > ch && freq[stk.peek() - 'a'] > 0) {
                char rem = stk.pop();
                exist[rem - 'a'] = false;
            }

            stk.push(ch);
            exist[ch - 'a'] = true;
        }

        char[] ans = new char[stk.size()];
        int index = 0;
        while (!stk.isEmpty()) {
            ans[index] = stk.pop();
            index++;
        }
        return new StringBuilder(new String(ans)).reverse().toString();
    }

    public String removeDuplicateLetters(String s) {
        int n = s.length();
        Stack<Character> stk = new Stack<>();

        int[] freq = new int[26];
        boolean[] exist = new boolean[26];
        Arrays.fill(exist, false);

        for (char ch : s.toCharArray()) freq[ch - 'a']++;
        for (int i = 0; i < n; i++) {

            char ch = s.charAt(i);
            freq[ch - 'a']--;
            if (exist[ch - 'a']) continue;

            while (!stk.isEmpty() && stk.peek() > ch && freq[stk.peek() - 'a'] > 0) {
                char rem = stk.pop();
                exist[rem - 'a'] = false;
            }

            stk.push(ch);
            exist[ch - 'a'] = true;
        }

        char[] ans = new char[stk.size()];
        int index = 0;
        while (!stk.isEmpty()) {
            ans[index] = stk.pop();
            index++;
        }
        return new StringBuilder(new String(ans)).reverse().toString();
    }

    /*
    "abcd"
"aaaaa"
"aabcaabdaab"
     */


    public int longestRepeatingSubstring(String s) {
        Set<String> set = new HashSet<>();

        int max = 0;
        int n = s.length();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (set.contains(s.substring(i, j + 1))) max = Math.max(max, j + 1 - i);
                else set.add(s.substring(i, j));
            }
        }
        return max;
    }


    List<String> ansStr = new ArrayList<>();

    /*
"hello"
"ooolleoooleh"
"prosperity"
"properties"
"ab"
"eidbaooo"
"ab"
"eidboaoo"
 */
    // TC = O(N^2)
    public boolean checkInclusion(String s1, String s2) {
        substring(s2, s1.length());
        for (String a : ansStr) {
            if (exist(s1, a)) return true;
        }

        return false;
    }

    // Fn to generate all possible  substrings of a given string
    private void substring(String s, int l) {
        int n = s.length();
        for (int i = 0; i <= n - l; i++) ansStr.add(s.substring(i, i + l));
    }

    private boolean exist(String s1, String s2) {

        int[] freq = new int[26];

        // freq[] : stores the  frequency of each
        // character of a string
        for (int i = 0; i < s2.length(); i++) {
            freq[s2.charAt(i) - 'a']++;
        }

        for (int i = 0; i < s1.length(); i++) {
            char ch = s1.charAt(i);
            if (freq[ch - 'a'] > 0) freq[ch - 'a']--;
            else return false;
        }
        return true;
    }

    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        // base cases
        if (root1 == root2) return true;
        if (root1 == null || root2 == null || root1.val != root2.val) return false;

        return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) ||
                (flipEquiv(root1.right, root2.left)
                        && flipEquiv(root1.left, root2.right));
    }


    private boolean equals(TreeNode root1, TreeNode root2) {

        if (root1 == null && root2 == null) return true;

        if (root1 != null && root2 != null) {
            return root1.val == root2.val && equals(root1.left, root2.left) && equals(root1.right, root2.right);
        }

        return false;
    }


    private TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    private List<Integer> whoKnows(int[][] meetings, Integer personId) {
        HashMap<Integer, LinkedList<Integer[]>> map = new HashMap<>();

        for (int[] meeting : meetings) {
            map.put(meeting[0], map.getOrDefault(meeting[0], new LinkedList<>()));
            map.get(meeting[0]).add(new Integer[]{meeting[1], meeting[2]});
        }

        // Level order traversal to check nodes that knows the story
        Queue<Integer[]> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        queue.offer(new Integer[]{meetings[0][0], meetings[0][2]});
        visited.add(meetings[0][0]); // adding the actual person

        while (!queue.isEmpty()) {
            Integer[] meeting = queue.poll();
            LinkedList<Integer[]> children = map.getOrDefault(meeting[0], new LinkedList<>());
            for (Integer[] child : children) {
                Integer id = child[0];
                Integer time = child[1];
                if (time > meeting[1]) {
                    visited.add(id);
                    queue.offer(child);
                }
            }
        }
        return new ArrayList<>(visited);
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] ans = new int[2];
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (map.containsKey(target - nums[i])) {
                ans[0] = i;
                ans[1] = map.get(target - nums[i]);
                break;
            }
            map.put(nums[i], i);
        }
        return ans;
    }

    public int twoSumLessThanK(int[] nums, int k) {
        Arrays.sort(nums);
        int ans = Integer.MIN_VALUE;
        int n = nums.length;
        int i = 0, j = n - 1;
        while (i < j) {
            if (nums[i] + nums[j] < k) {
                ans = Math.max(ans, nums[i] + nums[j]);
                i++;
            } else {
                j--;
            }
        }
        return ans == Integer.MIN_VALUE ? -1 : ans;
    }

    // TC  = O(N), SC = O(N)
    public int trap(int[] height) {
        int n = height.length;
        int ans = 0;
        int[] left_max = new int[n], right_max = new int[n];

        left_max[0] = height[0];
        right_max[n - 1] = height[n - 1];

        for (int i = 1; i < n; i++) left_max[i] = Math.max(left_max[i - 1], height[i]);

        for (int i = n - 2; i >= 0; i--) right_max[i] = Math.max(right_max[i + 1], height[i]);

        for (int i = 1; i < n - 1; i++) {
            ans += Math.min(left_max[i], right_max[i]) - height[i];
        }
        return ans;
    }


    public int trapStack(int[] height) {

        /*
        // More simplified
        int n = height.length;
        int ans = 0, current = 0;
        Stack<Integer> stk = new Stack<>();// to store the indecies of smaller heights bars

        while (current < n) {
            while (!stk.isEmpty() && height[stk.peek()] < height[current]) {
                int top = stk.pop();
                if (stk.isEmpty()) break;
                int distance = current - stk.peek() - 1;
                int bounded_height = Math.min(height[current], height[stk.peek()]) - height[top];
                ans += bounded_height * distance;
            }

            stk.push(current++);
        }

        return ans;

         */

        int n = height.length, left = 0, right = n - 1, left_max = 0, right_max = 0, ans = 0;
        while (left < right) {
            // calculate the ans from left towards right
            if (height[left] < height[right]) {
                if (height[left] >= left_max) left_max = height[left];
                else ans += left_max - height[left];
                left++;
            }

            // calculate water concentration from right towards left
            else {
                if (height[right] >= right_max) right_max = height[right];
                else ans += right_max - height[right];
                right--;
            }
        }
        return ans;
    }

    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            ans[i] = ans[i & (i - 1)] + 1;
        }
        return ans;
    }

    /*
    // Algorithm:-
    -  get the left_max, right_max for current index
    -  Iterate towards left till ht is lesser than current and eventually fill water
    -  Iterate towards right till ht is lesser than current and eventually fill water
    -  return the updated array
    */
    public int[] pourWater(int[] heights, int volume, int k) {
        int n = heights.length;
        // build the condition for eventually fall
        heights[k] += volume;

        int[] left_max = new int[n], right_max = new int[n];

        left_max[0] = heights[0];
        right_max[n - 1] = heights[n - 1];

        for (int i = 1; i < n; i++) left_max[i] = Math.max(left_max[i - 1], heights[i]);

        for (int i = n - 2; i >= 0; i--) right_max[i] = Math.max(right_max[i + 1], heights[i]);


        int l = k, r = k;
        while (l-- >= 0) {
            if (left_max[l] - heights[l] < heights[k]) {
                heights[l]++;
                heights[k]--;
            }
        }


        while (r++ < n) {
            if (right_max[r] - heights[r] < heights[k]) {
                heights[r]++;
                heights[k]--;
            }
        }


        return heights;
    }


    // TC = O(N^2), SC = O(1)
    public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int ans = 0;
        for (int i = 0; i < n - 2; i++) ans += twoSumSmallerInRange(nums, i + 1, target - nums[i]);

        return ans;
    }

    private int twoSumSmallerInRange(int[] nums, int startIndex, int target) {
        int sum = 0;
        int left = startIndex;
        int right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                sum += right - left;
                left++;
            } else {
                right--;
            }
        }

        return sum;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<List<Integer>> res = twoSumZero(nums, i + 1, -nums[i]);
            for (List<Integer> l : res) {
                List<Integer> triplet = new ArrayList<>();
                triplet.add(nums[i]);
                triplet.addAll(l);
                ans.add(triplet);
            }
        }
        return ans;
    }

    private List<List<Integer>> twoSumZero(int[] nums, int startIndex, int target) {
        int left = startIndex;
        int right = nums.length - 1;
        List<List<Integer>> res = new ArrayList<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = left; i < right; i++) {
            if (map.containsKey(target - nums[i])) {
                List<Integer> ans = new ArrayList<>();
                ans.add(nums[i]);
                ans.add(target - nums[i]);
                res.add(ans);
            }
            map.put(nums[i], i);
        }
        return res;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] < target) twoSum(nums, j, res, target - (nums[i] + nums[j]), i);
            }
        }
        return res;
    }

    private void twoSum(int[] nums, int i, List<List<Integer>> res, int target, int startIndex) {
        Set<Integer> seen = new HashSet<>();
        int n = nums.length;
        for (int j = i + 1; j < n; ++j) {
            int complement = target - nums[j];
            if (seen.contains(complement)) {
                res.add(Arrays.asList(nums[startIndex], nums[i], nums[j], complement));
            }
            seen.add(nums[j]);
        }
    }

    class KSum {
        public List<List<Integer>> fourSum(int[] nums, int target) {
            Arrays.sort(nums);
            return kSum(nums, target, 0, 4);
        }

        public List<List<Integer>> kSum(int[] nums, int target, int start, int k) {
            List<List<Integer>> res = new ArrayList<>();
            if (start == nums.length || nums[start] * k > target || target > nums[nums.length - 1] * k)
                return res;
            if (k == 2)
                return twoSum(nums, target, start);

            for (int i = start; i < nums.length; ++i)
                if (i == start || nums[i - 1] != nums[i])
                    for (List<Integer> subset : kSum(nums, target - nums[i], i + 1, k - 1)) {
                        res.add(new ArrayList<>(Collections.singletonList(nums[i])));
                        res.get(res.size() - 1).addAll(subset);
                    }

            return res;
        }

        public List<List<Integer>> twoSum(int[] nums, int target, int start) {
            List<List<Integer>> res = new ArrayList<>();
            Set<Integer> s = new HashSet<>();

            for (int i = start; i < nums.length; ++i) {
                if (res.isEmpty() || res.get(res.size() - 1).get(1) != nums[i])
                    if (s.contains(target - nums[i]))
                        res.add(Arrays.asList(target - nums[i], nums[i]));
                s.add(nums[i]);
            }

            return res;
        }
    }


    public int findPeakElement(int[] nums) {
        int n = nums.length;
        return search(nums, 0, n - 1);
    }

    // recursive binary search
    private int search(int[] nums, int l, int r) {
        if (l == r) return l;
        int mid = l + (r - l) / 2;
        if (nums[mid] > nums[mid + 1])
            return search(nums, l, mid);
        return search(nums, mid + 1, r);
    }

    // Iterative sollution
    public int peakIndexInMountainArray(int[] nums) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (nums[m] < nums[m + 1]) l = m + 1;
            else r = m;
        }
        return l;
    }

    // TC=  O(N2) SC=O(N)
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int cnt = 0;
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int n1 : nums1) {
            for (int n2 : nums2) {
                int key = n1 + n2;
                map.put(key, map.getOrDefault(key, 0) + 1);
            }
        }

        for (int n3 : nums3) {
            for (int n4 : nums4) {
                int key = -(n3 + n4);
                cnt += map.getOrDefault(key, 0);
            }
        }
        return cnt;
    }

    public int kNumberSumCount(int[] A, int[] B, int[] C, int[] D) {
        return kSumCount(new int[][]{A, B, C, D});
    }

    public int kSumCount(int[][] lists) {
        Map<Integer, Integer> m = new HashMap<>();
        addToHash(lists, m, 0, 0);
        return countComplements(lists, m, lists.length / 2, 0);
    }

    void addToHash(int[][] lists, Map<Integer, Integer> m, int i, int sum) {
        if (i == lists.length / 2)
            m.put(sum, m.getOrDefault(sum, 0) + 1);
        else
            for (int a : lists[i])
                addToHash(lists, m, i + 1, sum + a);
    }

    int countComplements(int[][] lists, Map<Integer, Integer> m, int i, int complement) {
        if (i == lists.length)
            return m.getOrDefault(complement, 0);
        int cnt = 0;
        for (int a : lists[i])
            cnt += countComplements(lists, m, i + 1, complement - a);
        return cnt;
    }

    boolean flip = false;
    int maxOne = Integer.MIN_VALUE;

    public int findMaxConsecutiveOnes(int[] nums) {
        consective(nums, 0);
        return maxOne;
    }

    private void consective(int[] nums, int index) {
        if (index == nums.length) return;

        int cnt = 0;
        if (nums[index] == 1) {
            cnt += 1;
            consective(nums, index + 1);
        } else if (!flip) {
            flip = true;
            cnt++;
        } else {
            cnt = 1;
        }

        maxOne = Math.max(maxOne, cnt);
    }

    public int findTargetSumWays(int[] nums, int target) {
        int[][] memo = new int[nums.length][2001];
        for (int[] rows : memo) Arrays.fill(rows, -1);
        return calculate(nums, 0, 0, target, memo);
    }

    private int calculate(int[] nums, int index, int sum, int target, int[][] memo) {
        if (index == nums.length) {
            if (sum == target)
                return 1;
            return 0;
        } else {

            if (memo[index][sum + 1000] != -1) return memo[index][sum + 1000];
            int add = calculate(nums, index + 1, sum + nums[index], target, memo);
            int subtract = calculate(nums, index + 1, sum - nums[index], target, memo);
            memo[index][sum + 1000] = add + subtract;

            return memo[index][sum + 1000];
        }
    }

    // sliding window technique
    public int findMaxConsecutiveOneSliding(int[] nums) {
        int n = nums.length;
        int left = 0, right = 0, longest = 0, zeros = 0;

        while (right < n) {
            // move right ptr towards right direction
            if (nums[right] == 0) zeros++;

            // check for invalid state
            while (zeros == 2) {
                if (nums[left] == 0) zeros--;
                left++;
            }

            // update out longest sequnce
            longest = Math.max(longest, right - left + 1);

            // expand out window
            right++;
        }
        return longest;
    }

    // cheer up solution for placing operators b/w numbers
    public ArrayList<String> answer;
    public String digits;
    public long target;

    public void recurse(int index, long previousOperand, long currentOperand, long value, ArrayList<String> ops) {
        String nums = this.digits;

        // Done processing all the digits in num
        if (index == nums.length()) {

            // If the final value == target expected AND
            // no operand is left unprocessed
            if (value == this.target && currentOperand == 0) {
                StringBuilder sb = new StringBuilder();
                ops.subList(1, ops.size()).forEach(v -> sb.append(v));
                this.answer.add(sb.toString());
            }
            return;
        }

        // Extending the current operand by one digit
        currentOperand = currentOperand * 10 + Character.getNumericValue(nums.charAt(index));
        String current_val_rep = Long.toString(currentOperand);
        int length = nums.length();

        // To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
        // valid operand. Hence this check
        if (currentOperand > 0) {

            // NO OP recursion
            recurse(index + 1, previousOperand, currentOperand, value, ops);
        }

        // ADDITION
        ops.add("+");
        ops.add(current_val_rep);
        recurse(index + 1, currentOperand, 0, value + currentOperand, ops);
        ops.remove(ops.size() - 1);
        ops.remove(ops.size() - 1);

        if (ops.size() > 0) {

            // SUBTRACTION
            ops.add("-");
            ops.add(current_val_rep);
            recurse(index + 1, -currentOperand, 0, value - currentOperand, ops);
            ops.remove(ops.size() - 1);
            ops.remove(ops.size() - 1);

            // MULTIPLICATION
            ops.add("*");
            ops.add(current_val_rep);
            recurse(
                    index + 1,
                    currentOperand * previousOperand,
                    0,
                    value - previousOperand + (currentOperand * previousOperand),
                    ops);
            ops.remove(ops.size() - 1);
            ops.remove(ops.size() - 1);
        }
    }

    public List<String> addOperators(String num, int target) {

        if (num.length() == 0) {
            return new ArrayList<String>();
        }

        this.target = target;
        this.digits = num;
        this.answer = new ArrayList<String>();
        this.recurse(0, 0, 0, 0, new ArrayList<String>());
        return this.answer;
    }

    public int maxScore(int[] cardPoints, int k) {
        int n = cardPoints.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) Arrays.fill(dp[i], -1);
        return cards(cardPoints, k, 0, n - 1, dp);
    }

    private int cards(int[] cardPoints, int k, int l, int r, int[][] dp) {
        // base case
        if (l > r || k <= 0) {
            return 0;
        }

        if (dp[l][r] != -1) return dp[l][r];

        // take a card from left
        int left = cardPoints[l] + cards(cardPoints, k - 1, l + 1, r, dp);
        // take a card from right
        int right = cardPoints[r] + cards(cardPoints, k - 1, l, r - 1, dp);

        dp[l][r] = Math.max(left, right);

        return dp[l][r];
    }
    // using prefix sum approach

    public int maxScorePrefixSum(int[] cardPoints, int k) {
        int n = cardPoints.length;
        int[] first = new int[k + 1], last = new int[k + 1];

        for (int i = 0; i < k; i++) {
            first[i + 1] = first[i] + cardPoints[i];
            last[i + 1] = last[i] + cardPoints[n - 1 - i];
        }

        int maxScore = 0;
        for (int i = 0; i <= k; i++) {
            int currSum = first[i] + last[k - i];
            maxScore = Math.max(maxScore, currSum);
        }
        return maxScore;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return bst(nums, 0, nums.length - 1);
    }

    private TreeNode bst(int[] nums, int l, int r) {
        // base case
        if (l > r) return null;

        int mid = l + (r - l) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = bst(nums, l, mid - 1);
        node.right = bst(nums, mid + 1, r);
        return node;
    }

    public int maxSubArrayLen(int[] nums, int target) {
        int n = nums.length;

        int ans = 0;
        HashMap<Integer, Integer> map = new HashMap<>(); //map<sum, index>
        map.put(0, -1);

        for (int i = 0, sum = 0; i < n; i++) {
            sum += nums[i];
            if (map.containsKey(target - sum)) {
                ans = Math.max(ans, i - map.get(target - sum));
            }
            map.putIfAbsent(sum, i);
        }
        return ans;
    }

    static class Edge {
        int v;
        int nbr;
        int wt;

        Edge(int v, int nbr, int wt) {
            this.v = v;
            this.nbr = nbr;
            this.wt = wt;
        }
    }

    static class Pair {
        int v;
        int av; //acquiring vertex (As Prims works on considering next connected min edge)
        int wt;

        Pair(int v, int av, int wt) {
            this.v = v;
            this.av = av;
            this.wt = wt;
        }
    }

    public int minCostConnectPoints(int[][] points) {
        //Construct Graph
        int n = points.length;
        ArrayList<Edge> graph[] = new ArrayList[n];

        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        //Since this graph will be a complete graph so have to connect each and every vertex with each other
        for (int i = 0; i < n; i++) {
            int u = i;
            int xi = points[i][0];
            int yi = points[i][1];
            for (int j = i + 1; j < n; j++) {
                int v = j;
                int xj = points[j][0];
                int yj = points[j][1];

                int wt = Math.abs(xi - xj) + Math.abs(yi - yj);
                graph[u].add(new Edge(u, v, wt));
                graph[v].add(new Edge(v, u, wt));
            }
        }
        //Now run your Prim's Alogrithm
        PriorityQueue<Pair> q = new PriorityQueue<>((a, b) -> {
            return a.wt - b.wt;
        });
        q.add(new Pair(0, -1, 0));
        int cost = 0;
        boolean visited[] = new boolean[n];
        while (q.size() > 0) {
            Pair rem = q.poll();

            if (visited[rem.v]) continue;
            visited[rem.v] = true;

            if (rem.av != -1) {
                cost = cost + rem.wt;
            }
            for (Edge e : graph[rem.v]) {
                if (!visited[e.nbr]) {
                    q.add(new Pair(e.nbr, rem.v, e.wt));
                }
            }
        }
        return cost;
    }


    public int maxSatisfaction(int[] satisfaction) {
        Arrays.sort(satisfaction);
        int n = satisfaction.length;

        int max = 0;
        for (int i = 0; i < n; i++) {
            int sum = 0, count = 1;
            for (int j = i; j < n; j++) {
                sum += count * satisfaction[j];
            }
            max = Math.max(sum, max);
        }
        return max;
    }

    // Recursion with memoisation
// 2 choices 1 take it and skip it (overall sum will not increase)
    // TC = O(2^N)
    // O(N^2)
    public int maxSatisfactionRecurse(int[] satisfaction) {
        Arrays.sort(satisfaction);
        int n = satisfaction.length;
        int[][] dp = new int[n + 1][n + 1];
        for (int i = 0; i < n + 1; i++) Arrays.fill(dp[i], -1);
        return maxsum(satisfaction, dp, 0, 1);
    }

    private int maxsum(int[] satisfaction, int[][] dp, int index, int coef) {

        // base case
        if (index == satisfaction.length) return 0;

        if (dp[index][coef] != -1) return dp[index][coef];

        // select
        int taken = coef * satisfaction[index] + maxsum(satisfaction, dp, index + 1, coef + 1);

        // number is not taken
        int notTaken = maxsum(satisfaction, dp, index + 1, coef);

        // return max of 2 choices
        return dp[index][coef] = Math.max(taken, notTaken);

    }

    public int validSubarrays(int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            for (int j = i; j < nums.length; j++) {
                if (nums[i] <= nums[j]) {
                    count++;
                    continue;
                }
                break;
            }
        }
        return count;
    }

    public int minSwap(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;

        // initialize dp table
        int[][] memo = new int[2][n1];
        Arrays.fill(memo[0], -1);
        Arrays.fill(memo[1], -1);
        memo[0][0] = 0;
        memo[1][0] = 1;

        return Math.min(recurse(nums1, nums2, n1 - 1, 0, memo), recurse(nums1, nums2, n1 - 1, 1, memo));
    }

    private int recurse(int[] nums1, int[] nums2, int i, int swap, int[][] memo) {
        // base cases
        if (i == 0) return swap;

        //check dp table
        if (memo[swap][i] != -1)
            return memo[swap][i];


        int res = Integer.MAX_VALUE;
        // no swap
        if (nums1[i] > nums1[i - 1] && nums2[i] > nums2[i - 1])
            res = recurse(nums1, nums2, i - 1, swap, memo);

        // swap
        if (nums1[i - 1] < nums2[i] && nums2[i - 1] < nums1[i])
            res = Math.min(res, recurse(nums1, nums2, i - 1, 1 - swap, memo));

        memo[swap][i] = swap == 0 ? res : res + 1;
        return memo[swap][i];
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        bfs(new ArrayList<>(), 0, 0, ans, target, candidates);
        return ans;
    }

    private void bfs(List<Integer> num, int sum, int index, List<List<Integer>> ans, int k, int[] candidates) {
        // base case
        if (sum == k) {
            if (!ans.contains(num)) ans.add(new ArrayList<>(num));
        } else if (sum > k) {
            return;
        } else {
            // For all position place numbers
            for (int p = index; p < candidates.length; p++) {
                //skip duplicates
                if (p > index && candidates[p] == candidates[p - 1]) {
                    continue;
                }

                num.add(candidates[p]);
                sum += candidates[p];
                bfs(num, sum, p + 1, ans, k, candidates);
                num.remove(new Integer(candidates[p]));
                sum -= candidates[p];
            }
        }
    }
}

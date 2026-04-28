package com.company;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;

class SubSet2 {
    /*
     * PROBLEM: Subsets II (LeetCode 90)
     * Return all possible subsets of an array that may contain duplicates, without duplicate subsets.
     *
     * ALGORITHM: Backtracking with sorted input to skip duplicates
     * TC: O(2^n) | SC: O(2^n)
     */
    // TC = O(2^n), SC = exponential in nature
    // backtracking basesd solution
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        bfs(nums, 0, result, new ArrayList<Integer>());
        return result;
    }

        /*
         * PROBLEM: Subsets II BFS Helper (Helper)
         * Recursive BFS helper to generate unique subsets and accumulate into result.
         *
         * ALGORITHM: Backtracking
         * TC: O(2^n) | SC: O(n)
         */
        private void bfs(int[] nums, int index, List<List<Integer>> result, List<Integer> res) {
        // base case
        if (!result.contains(res)) result.add(new ArrayList<>(res));

        for (int i = index; i < nums.length; i++) {
            int num = nums[i];
            res.add(num);
            bfs(nums, i + 1, result, res);
            res.remove(new Integer(num));
        }
    }


    class Subsets {
        /*
         * PROBLEM: Subsets (LeetCode 78)
         * Return all possible subsets of a distinct integer array.
         *
         * ALGORITHM: Iterative BFS-style expansion per element
         * TC: O(n * 2^n) | SC: O(n * 2^n)
         */
        // TC = O(n), SC = O(n)
        public List<List<Integer>> subsets(int[] nums) {
            int n = nums.length;
            LinkedList<List<Integer>> result = new LinkedList<>();
            if (n == 0) return result;
            result.add(new ArrayList<>());
            permuate(nums, 0, result);
            return result;
        }

        /*
         * PROBLEM: Subsets Recursive Helper (Helper)
         * Recursively expand each existing subset with the element at the current index.
         *
         * ALGORITHM: Backtracking / BFS expansion
         * TC: O(n * 2^n) | SC: O(n)
         */
        private void permuate(int[] nums, int index, LinkedList<List<Integer>> result) {
            int n = nums.length;
            // base case
            if (index == n) {
                return;
            }

            int ch = nums[index];
            int size = result.size();
            LinkedList<Set<Integer>> list = new LinkedList<Set<Integer>>();

            while (size-- > 0) {
                List<Integer> elem = result.poll();
                list.add(new HashSet<>(elem));
                elem.add(ch);
                list.add(new HashSet<>(elem));
            }
            result.addAll(list.stream().map(x -> new ArrayList<>(x)).collect(Collectors.toList()));

            permuate(nums, index + 1, result);
        }
    }

// Iterative approach

// Algorithm :-

    /*
     - Idea is to take all paths by variation of a  single character (ie. uppercase or lowercase)
     - Intialise list with first given variation of string
     - Iterate two loops
       - i = 0 unitll n (n = size of string)
       - check for current char ie. if its character then its liable for changes else not
       - poll elements out of list untill it becomes empty
       - create both variation and update the stirng to list
       - Repeat the process untill while string gets exhausted
      - Return all possible paths
    */
// Recursive approach
    class LetterCasePermutation {
        /*
         * PROBLEM: Letter Case Permutation (LeetCode 784)
         * Generate all strings by changing each letter to uppercase or lowercase.
         *
         * ALGORITHM: Backtracking (recursive exploration of upper/lower cases)
         * TC: O(2^n * n) | SC: O(n)
         */
        // TC = O(2*n), SC = O(n!)
        public List<String> letterCasePermutation(String s) {
            int n = s.length();
            List<String> ans = new ArrayList<>();
            if (n == 0) return ans;
            StringBuilder sb = new StringBuilder("");
            permuate(s, sb, 0, ans);
            return ans;
        }

        /*
         * PROBLEM: Letter Case Permutation Recursive (Helper)
         * Recursive helper to build each permutation character by character.
         *
         * ALGORITHM: Backtracking
         * TC: O(2^n) | SC: O(n)
         */
        private void permuate(String s, StringBuilder sb, int index, List<String> ans) {
            int n = s.length();
            // base case
            if (index == n) {
                ans.add(sb.toString());
                return;
            }

            char c = s.charAt(index);

            // if char is letter
            if (Character.isLetter(c)) {
                Character[] array = new Character[]{Character.toUpperCase(c), Character.toLowerCase(c)};
                for (char ch : array) {
                    sb.append(ch);
                    permuate(s, sb, index + 1, ans);
                    sb.setLength(sb.length() - 1);
                }
            }
            // else do not make any variation
            else {
                sb.append(c);
                permuate(s, sb, index + 1, ans);
                sb.setLength(sb.length() - 1);
            }
        }

    /*

    // Iterative solution
      public List<String> letterCasePermutation(String s) {
        int n = s.length();
        LinkedList<String> result = new LinkedList<>();
        if (n == 0) return result;
        result.add(s);
        for (int i = 0; i<n; i++) {
            char ch = s.charAt(i);
            if (Character.isLetter(ch)) {
                int size = result.size();
                while (size--> 0) {
                    String str = result.poll();
                    String left = str.substring(0, i);
                    String right = str.substring(i + 1);
                    result.add(left + String.valueOf(ch).toUpperCase() + right);
                    result.add(left + String.valueOf(ch).toLowerCase() + right);
                }
            }
        }


        // String[] arr = new String[result.size()];
        // arr = result.toArray(arr);
        // // Get lexicographically sorted order results
        // Arrays.sort(arr);
        // return Arrays.stream(arr).map(x->x).collect(Collectors.toList());
        return new ArrayList<>(result);
    }
     */
    }

    class PickRandomBlacklist {
        HashMap<Integer, Integer> map;
        Random generator;
        int ul;

        public PickRandomBlacklist(int n, int[] blacklist) {
            this.generator = new Random();
            this.map = new HashMap<>();
            for (Integer num : blacklist)
                this.map.put(num, -1);

            this.ul = n - blacklist.length;
            int j = n - 1;
            for (Integer num : blacklist) {
                if (num < ul) {
                    while (map.containsKey(j)) j--;
                    map.put(num, j);
                    j--;
                }
            }
        }

        /*
         * PROBLEM: Random Pick with Blacklist (LeetCode 710)
         * Pick a random integer from [0, n) excluding blacklisted values.
         *
         * ALGORITHM: HashMap remapping of blacklist indices + random uniform selection
         * TC: O(1) per pick | SC: O(|blacklist|)
         */
        public int pick() {
            int random = generator.nextInt(ul);
            if (map.containsKey(random))
                return map.get(random);
            return random;
        }
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * Solution obj = new Solution(n, blacklist);
     * int param_1 = obj.pick();
     */

    class FindDifferentBinaryString {
        Set<String> set;
        String res;
        int len;
        String[] d = {
                "1", "0"
        };

        /*
         * PROBLEM: Find Unique Binary String (LeetCode 1980)
         * Find a binary string of length n that is not present in the given list.
         *
         * ALGORITHM: Backtracking + HashSet lookup
         * TC: O(2^n * n) | SC: O(n)
         */
        public String findDifferentBinaryString(String[] nums) {
            len = nums.length;
            set = new HashSet<>();
            for (String s : nums) set.add(s);
            diff(0, "");
            return res;
        }

        /*
         * PROBLEM: Find Unique Binary String Recursive (Helper)
         * Recursively try placing '1' or '0' at each position until a unique string is found.
         *
         * ALGORITHM: Backtracking
         * TC: O(2^n * n) | SC: O(n)
         */
        private Boolean diff(int index, String str) {
            // base case
            if (index == set.size()) {
                if (!set.contains(str)) {
                    res = str;
                    return true;
                }
                return false;
            }
            // For all position place string characters
            for (int k = 0; k < len; k++) {
                for (int i = 0; i < d.length; i++) {
                    str += d[i];
                    if (diff(index + 1, str)) return true;
                    str = str.substring(0, str.length() - 1);
                }
            }
            return false;
        }
    }

    class MaxTaxiEarningsDP {

        /*
         * PROBLEM: Maximum Earnings From Taxi (LeetCode 2008)
         * Find maximum earnings by selecting non-overlapping rides.
         *
         * ALGORITHM: DP (bottom-up) + sorted rides by start time
         * TC: O(n+k) | SC: O(n)
         */
        // DP + Iteration based solution
        public long maxTaxiEarnings(int n, int[][] rides) {
            // sort rides  matrix based on  start time
            Arrays.sort(rides, (a, b) -> a[0] - b[0]);
		/*
		   [[2,5,4],[1,5,1]]
		   sort ------> [[1 5 1], [2 5 4]]
		   dp[0]=5-1+1=5
		   dp[1]=max(7,5)=7

         - dp max for every switch to new point
         - For every next switch check wrt to prev path and curr and take
         - Return max value ie. dp[n]
		   Complexity -->  TC = O(n+k) , SC = O(n);
		*/
            long[] dp = new long[n + 1];
            int j = 0;
            for (int i = 1; i <= n; i++) {
                dp[i] = Math.max(dp[i - 1], dp[i]);
                while (j < rides.length && rides[j][0] == i) {
                    int d = rides[j][1] - rides[j][0] + rides[j][2];
                    dp[rides[j][1]] = Math.max(dp[rides[j][1]], dp[i] + d);
                    ++j;
                }
            }
            return dp[n];
        }
    }


    class FindOriginalArray {
        /*
         * PROBLEM: Find Original Array From Doubled Array (LeetCode 2007)
         * Given a doubled array, recover the original array.
         *
         * ALGORITHM: Sort + HashMap to match pairs
         * TC: O(n log n) | SC: O(n)
         */
        public int[] findOriginalArray(int[] changed) {
            int n = changed.length, cnt = 0;
            List<Integer> ans = new ArrayList<>();
            HashMap<Integer, List<Integer>> map = new HashMap<>();
            Arrays.sort(changed);

            // for (Integer ll: changed) System.out.println(ll);

            if (n == 1 || n == 0 || n % 2 != 0) return new int[]{};

            for (int i = 0; i < n; i++) {
                int e = changed[i];
                if (e % 2 == 0 && map.containsKey(e / 2) && map.get(e / 2).size() > 0) {
                    ans.add(e / 2);
                    cnt++;
                    List<Integer> list = map.get(e / 2);
                    list.remove(0);
                    if (list.size() > 0) map.put(e / 2, list);
                    else map.remove(e / 2);
                    continue;
                }
                if (map.containsKey(e)) {
                    List<Integer> l = map.get(e);
                    l.add(i);
                    map.put(e, l);
                } else {
                    List<Integer> l = new ArrayList<>();
                    l.add(i);
                    map.put(e, l);
                }
            }
            if (cnt == n / 2) return ans.stream().mapToInt(x -> x).toArray();
            return new int[]{};
        }
    }

    // TODO: Solve again
    class SplitIntoFibonacci {
        List<Integer> al = new ArrayList<>();

        /*
         * PROBLEM: Split Array into Fibonacci Sequence (LeetCode 842)
         * Split a numeric string into a valid Fibonacci sequence.
         *
         * ALGORITHM: Backtracking with brute-force first two number selection
         * TC: O(n^2) | SC: O(n)
         */
        public List<Integer> splitIntoFibonacci(String num) {
            for (int i = 0; i < num.length(); i++) {
                if (i > 0 && num.charAt(0) == '0') return new ArrayList<Integer>();
                long a = Long.parseLong(num.substring(0, i + 1));
                if (a > Integer.MAX_VALUE) break;
                for (int j = i + 1; j < num.length(); j++) {
                    long b = Long.parseLong(num.substring(i + 1, j + 1));
                    if (b > Integer.MAX_VALUE) break;
                    al.add((int) a);
                    al.add((int) b);
                    if (j > i + 1 && num.charAt(i + 1) == '0') break;
                    if (j + 1 < num.length() && solve((int) a, (int) b, num, j + 1)) return al;
                    else al.remove(al.size() - 1);
                }
                al.clear();
            }
            return new ArrayList<Integer>();
        }

        /*
         * PROBLEM: Fibonacci Solve Recursive (Helper)
         * Verify and extend the Fibonacci sequence from position index in the string.
         *
         * ALGORITHM: Recursion + string prefix matching
         * TC: O(n) | SC: O(n)
         */
        public boolean solve(int a, int b, String num, int index) {
            if (index == num.length()) return true;
            String sum = String.valueOf(a + b);
            if (index + sum.length() > num.length()) {
                al.remove(al.size() - 1);
                return false;
            }
            String actualSum = num.substring(index, index + sum.length());
            if (!sum.equals(actualSum)) {
                al.remove(al.size() - 1);
                return false;
            } else {
                al.add(a + b);
                return solve(b, a + b, num, index + sum.length());
            }
        }
    }


//TODO: Revisit

    class UniquePathsIIIBFSGrid {
        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};
        boolean[][] visited;
        int[][] obstacleGrid;
        int m, n, space = 0;
        int paths = 0;
        // find src and dest points
        Point src, dest;

        /*
         * PROBLEM: Unique Paths III (LeetCode 980)
         * Count paths from start to end that walk over every non-obstacle cell exactly once.
         *
         * ALGORITHM: Backtracking (DFS) with visited tracking
         * TC: O(4^(m*n)) | SC: O(m*n)
         */
        public int uniquePathsIII(int[][] obstacleGrid) {
            m = obstacleGrid.length;
            n = obstacleGrid[0].length;

            if (m == 1 && n == 1) {
                if (obstacleGrid[0][0] == 0) return 1;
                else return 0;
            }
            this.src = new Point();
            this.dest = new Point();

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (obstacleGrid[i][j] == 1) {
                        this.src.x = i;
                        this.src.y = j;
                    } else if (obstacleGrid[i][j] == 2) {
                        this.dest.x = i;
                        this.dest.y = j;
                    } else if (obstacleGrid[i][j] == 0) {
                        space++;
                    }
                }
            }

            this.obstacleGrid = obstacleGrid;
            visited = new boolean[m][n];
            int[][] dp = new int[m][n];
            for (int[] rows : dp) Arrays.fill(rows, -1);
            recursive(m, n, this.src.x, this.src.y, 0);
            // For a single path once found paths=0 is done to reset the path again and evaluate it
            int k = paths;
            paths = 0;
            return k;
        }

        /*
         * PROBLEM: Is Safe to Visit (Helper)
         * Check if cell (r,c) is within bounds and not an obstacle.
         *
         * ALGORITHM: Boundary and value check
         * TC: O(1) | SC: O(1)
         */
        // this function will tell that the current node is safe to be visited or not
        private boolean isSafe(int r, int c) {
            return (r < m && c < n && r >= 0 && c >= 0 && (obstacleGrid[r][c] != -1));
        }

        /*
         * PROBLEM: Unique Paths III Recursive (Helper)
         * Recursively explore all 4-directional paths, backtracking when dead ends are reached.
         *
         * ALGORITHM: DFS backtracking
         * TC: O(4^(m*n)) | SC: O(m*n)
         */
        private void recursive(int m, int n, int r, int c, int cells) {
            // base cases
            if (r == this.dest.x && c == this.dest.y && cells == space) {
                paths++;
                return;
            } else if (!isSafe(r, c) || visited[r][c]) return;
            for (int i = 0; i < 4; i++) {
                int x = r + dx[i];
                int y = c + dy[i];
                if (isSafe(x, y) && !visited[x][y]) {
                    visited[r][c] = true;
                    if (obstacleGrid[x][y] == 0) cells += 1;
                    recursive(m, n, x, y, cells);
                    if (obstacleGrid[x][y] == 0) cells -= 1;
                    visited[r][c] = false; // backtrack
                }
            }
            return;
        }

        class Point {
            int x;
            int y;

            Point(int x, int y) {
                this.x = x;
                this.y = y;
            }

            Point() {
            }
        }
    }

    //TODO: Revisit
    class SQRTBinarySeach {
        // Fields used by findAllPeople and findEvenNumbers
        Map<Integer, Map<Integer, List<Integer>>> g = new HashMap<>();
        Set<Integer> set = new TreeSet<>();
        /*
         * PROBLEM: Sqrt(x) (LeetCode 69)
         * Compute the integer square root of x without using sqrt().
         *
         * ALGORITHM: Binary Search
         * TC: O(log n) | SC: O(1)
         */
        public int mySqrt(int num) {

            if (num < 2) {
                return num;
            }

            long start = 2;
            long end = num;
            while (start <= end) {
                long mid = start + (end - start) / 2;
                if (mid * mid == num) return (int) mid;
                else if (mid * mid < num) start = mid + 1;
                else end = mid - 1;
            }
            return (int) (start - 1);
        }


        /*
         * PROBLEM: Find Peak Element (LeetCode 162)
         * Find an index of a peak element (greater than its neighbours) in O(log n).
         *
         * ALGORITHM: Binary Search
         * TC: O(log n) | SC: O(1)
         */
        // This is based on binary search of peak element
        // Input: nums = [1,2,3,1]
        // Output: 2
        // TC = O(logn), SC = O(1)
        public int findPeakElement(int[] nums) {
            int n = nums.length;
            int l = 0, r = n - 1;
            while (l < r) {
                int m = l + (r - l) / 2;
                if (nums[m] < nums[m + 1]) l = m + 1;
                else r = m;
            }
            return l;
        }

        /*
         * PROBLEM: Search in Rotated Sorted Array (LeetCode 33)
         * Search for a target in a rotated sorted array in O(log n).
         *
         * ALGORITHM: Binary Search with sorted-half detection
         * TC: O(log n) | SC: O(1)
         */
        // O(logn)
        // We have to iteratively partition array into haves and evavluate which half is sorted
        // and then apply BS for the element
        public int search(int[] nums, int target) {
            int n = nums.length;
            int l = 0, r = n - 1;

            // Find 2 halves of array
            while (l <= r) {
                int mid = l + (r - l) / 2;
                if (nums[mid] == target) return mid;
                // First half is sorted and other half is unsorted
                if (nums[l] <= nums[mid]) {
                    if (target >= nums[l] && target < nums[mid]) {
                        r = mid - 1;
                    } else l = mid + 1;
                }
                // Second half is sorted and other half is unsorted
                else {
                    if (target > nums[mid] && target <= nums[r]) {
                        l = mid + 1;
                    } else r = mid - 1;
                }
            }
            return -1;

        }

        /*
         * PROBLEM: Find All People With Secret (LeetCode 2092)
         * Find all people who ultimately learn the secret given meetings ordered by time.
         *
         * ALGORITHM: BFS + MinPriorityQueue ordered by meeting time
         * TC: O((n+m) log n) | SC: O(n+m)
         */
        // PQ + BFS
        public List<Integer> findAllPeople(int n, int[][] meetings, int firstPerson) {

            int[] notified = new int[n];
            Arrays.fill(notified, -1);
            for (int i = 0; i < n; i++) g.put(i, new HashMap<>());
            for (int[] m : meetings) getMeetingTimes(m[0], m[1]).add(m[2]);
            List<Integer> res = new ArrayList<>();

            PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
            pq.offer(new int[]{
                    0,
                    0
            });
            pq.offer(new int[]{
                    firstPerson,
                    0
            });

            while (!pq.isEmpty()) {
                int[] cur = pq.poll();
                int u = cur[0];
                if (notified[u] > -1) continue; //already visited
                notified[u] = cur[1];
                res.add(u);
                for (Map.Entry<Integer, List<Integer>> entry : g.get(u).entrySet()) {
                    int v = entry.getKey();
                    if (notified[v] > -1) continue; // already visited
                    Collections.sort(entry.getValue());
                    for (int time : entry.getValue()) {
                        //get first time u and v met after person u learned the secret
                        if (time >= notified[u]) {
                            pq.offer(new int[]{
                                    v,
                                    time
                            });
                            break;
                        }
                    }
                }
                g.remove(u); //u was fully processed
            }

            return res;
        }

        /*
         * PROBLEM: Get Meeting Times (Helper)
         * Get or create the shared list of meeting times between persons u and v.
         *
         * ALGORITHM: HashMap lookup
         * TC: O(1) | SC: O(1)
         */
        private List<Integer> getMeetingTimes(int u, int v) {
            List<Integer> times = g.get(u).get(v);
            if (times == null) {
                times = new ArrayList<>();
                g.get(u).put(v, times);
                g.get(v).put(u, times);
            }
            return times;
        }


        /*
         * PROBLEM: Find All Three-Digit Even Numbers (LeetCode 2094)
         * Return all unique 3-digit even integers formed by using 3 of the given digits.
         *
         * ALGORITHM: Backtracking with constraints (no leading zero, must be even)
         * TC: O(n^3) | SC: O(n)
         */
        // TC = O(2^n), SC = O(2^n)
        public int[] findEvenNumbers(int[] digits) {
            int n = digits.length;
            boolean[] seen = new boolean[n];
            permute(0, digits, "", seen);
            return set.stream().mapToInt(x -> x).sorted().toArray();
        }

        /*
         * PROBLEM: Find Even Numbers Backtracking (Helper)
         * Recursively build 3-digit numbers from the digit array respecting constraints.
         *
         * ALGORITHM: Backtracking
         * TC: O(n^3) | SC: O(n)
         */
        private void permute(int idx, int[] digits, String curr, boolean[] seen) {

            //base cases
            if (idx == 3) {
                set.add(Integer.parseInt(curr));
                return;
            }

            for (int i = 0; i < digits.length; i++) {
                if (seen[i] || (idx == 0 && digits[i] == 0) || (idx == 2 && digits[i] % 2 != 0)) continue;
                seen[i] = true;
                permute(idx + 1, digits, curr + digits[i], seen);
                seen[i] = false;
            }
        }


        /*
         * PROBLEM: Delete the Middle Node of a Linked List (LeetCode 2095)
         * Remove the middle node of a singly linked list.
         *
         * ALGORITHM: Two pointers (slow/fast)
         * TC: O(n/2) | SC: O(1)
         */
        // TC = (n/2), SC = O(1)
        public ListNode deleteMiddle(ListNode head) {
            if (head.next == null) return null;
            ListNode slow = head;
            ListNode fast = head.next.next;
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
            }
            slow.next = slow.next.next;
            return head;
        }


        /*
         * PROBLEM: Delete Middle Node Simple Approach (Helper)
         * Alternative approach to delete middle node tracking previous pointer.
         *
         * ALGORITHM: Two pointers (slow/fast with prev tracking)
         * TC: O(n) | SC: O(1)
         */
        private ListNode deleteMiddleSimpleApproach(ListNode head) {
            if (head == null || head.next == null) return null;
            ListNode curr = head;
            ListNode prev = null;
            ListNode slow = head;
            ListNode fast = head;
            while (fast != null && fast.next != null) {
                prev = curr;
                slow = slow.next;
                fast = fast.next.next;
            }
            prev.next = curr.next;
            return head;
        }


        /*
         * PROBLEM: Lowest Common Ancestor (LeetCode 236)
         * Find the lowest common ancestor of two nodes s and d in a binary tree.
         *
         * ALGORITHM: DFS post-order
         * TC: O(n) | SC: O(h)
         */
        // LCA based solution
        public TreeNode findLCA(TreeNode node, int s, int d) {
            if (node == null) return null;

            if (node.val == s || node.val == d) return node;

            TreeNode left = findLCA(node.left, s, d);
            TreeNode right = findLCA(node.right, s, d);
            if (left != null && right != null) return node;
            if (left == null && right != null) return right;
            else return left;
        }

        /*
         * PROBLEM: Step-By-Step Directions From Binary Tree (LeetCode 2096)
         * Return directions from startValue node to destValue node in the binary tree.
         *
         * ALGORITHM: LCA + DFS path finding
         * TC: O(n) | SC: O(n)
         */
        public String getDirections(TreeNode root, int startValue, int destValue) {
            TreeNode CA = findLCA(root, startValue, destValue);
            StringBuilder ans = new StringBuilder();
            ArrayDeque<String> q1 = new ArrayDeque<>();
            helper(CA, startValue, q1);
            ArrayDeque<String> q2 = new ArrayDeque<>();
            helper(CA, destValue, q2);

            for (int i = 0; i < q1.size(); i++) ans.append("U");
            while (!q2.isEmpty()) ans.append(q2.poll());

            return ans.toString();
        }

        /*
         * PROBLEM: Path to Node Helper (Helper)
         * DFS to find the path from a tree node to the target value, recording L/R directions.
         *
         * ALGORITHM: DFS with backtracking
         * TC: O(n) | SC: O(h)
         */
        private boolean helper(TreeNode n, int v, ArrayDeque<String> q) {
            if (n == null) return false;
            if (n.val == v) return true;

            q.offer("L");
            boolean left = helper(n.left, v, q);
            if (left) return true;
            q.removeLast();

            q.offer("R");
            boolean right = helper(n.right, v, q);
            if (right) return true;
            q.removeLast();

            return false;
        }

        //Represent the head and tail of the singly linked list
        ListNode curr = null;
        ListNode tail = null;

        /*
         * PROBLEM: Add Node to Linked List (Helper)
         * Append a new node with the given data to the end of the singly linked list.
         *
         * ALGORITHM: Linked list tail insertion
         * TC: O(1) | SC: O(1)
         */
        //addNode() will add a new node to the list
        public void addNode(int data) {
            //Create a new node
            ListNode newNode = new ListNode(data);

            //Checks if the list is empty
            if (curr == null) {
                //If list is empty, both head and tail will point to new node
                curr = newNode;
                tail = newNode;
            } else {
                //newNode will be added after tail such that tail's next will point to newNode
                tail.next = newNode;
                //newNode will become new tail of the list
                tail = newNode;
            }
        }

        /*
         * PROBLEM: Merge Nodes in Between Zeros (LeetCode 2181)
         * Sum values between consecutive 0-nodes and return new list of sums.
         *
         * ALGORITHM: Linear scan with running sum
         * TC: O(n) | SC: O(n)
         */
        // Author: Anand
        public ListNode mergeNodes(ListNode head) {
            int sum = 0;
            int zero = 0;

            while (head != null) {
                sum += head.val;
                if (head.val == 0) {
                    zero++;
                }
                if (zero == 2) {
                    addNode(sum);
                    zero = 1;
                    sum = 0;
                }
                head = head.next;
            }


            return curr;
        }
    }

    class combinationSum4BottomUpDP {
        /*
         * PROBLEM: Combination Sum IV (LeetCode 377)
         * Count the number of combinations that add up to target (order matters).
         *
         * ALGORITHM: DP bottom-up
         * TC: O(n * target) | SC: O(target)
         */
        // TC = O(n*target), SC = O(1)
        // Bottom up approach
        public int combinationSum4(int[] nums, int target) {
            int n = nums.length;
            int[] dp = new int[target + 1];
            dp[0] = 1; // Do not take any element at all

            for (int i = 1; i <= target; i++) {
                for (int num : nums)
                    if (i >= num) dp[i] += dp[i - num];
            }
            return dp[target];
        }
    }
}

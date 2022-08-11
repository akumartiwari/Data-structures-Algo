package com.company;

import javafx.util.Pair;

import java.awt.*;
import java.util.List;
import java.util.Queue;
import java.util.*;
import java.util.stream.Collectors;

public class EmployeeSolution {
    String name;
    Integer salary;

    EmployeeSolution(String n, Integer s) {
        this.name = n;
        this.salary = s;
        this.salary = s;
    }

    public static void main(String[] args) {
        List<EmployeeSolution> list = new ArrayList<>();
        list.add(new EmployeeSolution("muksh", 10));
        list.add(new EmployeeSolution("saksham", 50));

        List<EmployeeSolution> salaries = list.stream().map(x -> new EmployeeSolution(x.name, x.salary * 2)).collect(Collectors.toList());

        salaries.forEach(System.out::println);

        int[] arr = new int[]{2, -1, -3, 6, 8, -4, -8, -5, 9, 3, -3, 4};

        System.out.println(maxSum(arr));
    }

//
//         [5:20 pm] Keshav Bansal
//
//    { 2, -1, -3, 6, 8, -4, 5, -8, -5, 9, 3, -3, 4 }
//
//   dp[1] = 2
//   dp[2] = 2
//   dp[3] = 6
//   dp[4] = 8

    public static int maxSum(int[] arr) {
        int n = arr.length;
        if (n == 0) return 0;

        int maxSum = 0;
        int currSum = 0;
        for (int i = 0; i < n; i++) {
            if (currSum + arr[i] < 0) {
                currSum = 0;
            } else {
                currSum = Math.max(currSum + arr[i], arr[i]);
            }
            maxSum = Math.max(currSum, maxSum);
        }

        return maxSum;
    }

    Set<String> result = new HashSet<>();

    public List<String> addOperators(String num, int target) {
        calculate(num, 0, target, "");
        return new ArrayList<>(result);
    }

//    Input: num = 526
//    Output: true
//    Explanation: Reverse num to get 625, then reverse 625 to get 526, which equals num.
//    Input: num = 1800
//    Output: false
//    Explanation: Reverse num to get 81, then reverse 81 to get 18, which does not equal num.

    // Author: Anand
    // If a number has traling is zero and its not zero then it must return false
    public boolean isSameAfterReversals(int num) {
        if (num != 0 && num % 10 == 0) return false;
        return true;
    }

    private void calculate(String num, int index, int target, String expression) {
        if (index == num.length()) {


            String res = expression.substring(0, expression.length() - 1);
            int sum = 0;
            try {
                sum = Integer.parseInt(res);
            } catch (Exception ex) {
                if (!res.isEmpty()) sum = (int) Math.abs(calc(res));
            }

            if (sum == target) {
                result.add(res);
            }
        } else {

            calculate(num, index + 1, target, expression + Integer.parseInt(String.valueOf(num.charAt(index))) + "*");
            calculate(num, index + 1, target, expression + Integer.parseInt(String.valueOf(num.charAt(index))) + "+");
            calculate(num, index + 1, target, expression + Integer.parseInt(String.valueOf(num.charAt(index))) + "-");
            calculate(num, index + 1, target, expression + Integer.parseInt(String.valueOf(num.charAt(index))));

        }
    }

    public Double calculate(String expression) {
        if (expression == null || expression.length() == 0) {
            return null;
        }
        return calc(expression.replace(" ", ""));
    }

    public Double calc(String expression) {

        if (expression.startsWith("(") && expression.endsWith(")")) {
            return calc(expression.substring(1, expression.length() - 1));
        }
        String[] containerArr = new String[]{expression};
        double leftVal = getNextOperand(containerArr);
        expression = containerArr[0];
        if (expression.length() == 0) {
            return leftVal;
        }
        char operator = expression.charAt(0);
        expression = expression.substring(1);

        while (operator == '*' || operator == '/') {
            containerArr[0] = expression;
            double rightVal = getNextOperand(containerArr);
            expression = containerArr[0];
            if (operator == '*') {
                leftVal = leftVal * rightVal;
            } else {
                leftVal = leftVal / rightVal;
            }
            if (expression.length() > 0) {
                operator = expression.charAt(0);
                expression = expression.substring(1);
            } else {
                return leftVal;
            }
        }
        if (operator == '+') {
            return leftVal + calc(expression);
        } else {
            return leftVal - calc(expression);
        }

    }

    private double getNextOperand(String[] exp) {
        double res;
        if (exp[0].startsWith("(")) {
            int open = 1;
            int i = 1;
            while (open != 0) {
                if (exp[0].charAt(i) == '(') {
                    open++;
                } else if (exp[0].charAt(i) == ')') {
                    open--;
                }
                i++;
            }
            res = calc(exp[0].substring(1, i - 1));
            exp[0] = exp[0].substring(i);
        } else {
            int i = 1;
            if (exp[0].charAt(0) == '-') {
                i++;
            }
            while (exp[0].length() > i && isNumber((int) exp[0].charAt(i))) {
                i++;
            }
            res = Double.parseDouble(exp[0].substring(0, i));
            exp[0] = exp[0].substring(i);
        }
        return res;
    }


    private boolean isNumber(int c) {
        int zero = (int) '0';
        int nine = (int) '9';
        return (c >= zero && c <= nine) || c == '.';
    }


    // Greedy approach
    // The approach to traverse through the array and check if we get a n 'X'  character then move 3 steps ahead
    //  else move only 1 step (normal pace)

    public int minimumMoves(String s) {
        int n = s.length(), cnt = 0;
        for (int i = 0; i < n; ) {
            if (s.charAt(i) == 'X') {
                i += 3;
                cnt++;
            } else i++;
        }
        return cnt;
    }


    // KADAN's ALGO
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int max_sum_so_far = Integer.MIN_VALUE, max_ending_here = 0;

        for (int i = 0; i < n; i++) {
            max_ending_here += nums[i];
            max_sum_so_far = Math.max(max_ending_here, max_sum_so_far);
            if (max_ending_here < 0) max_ending_here = 0;
        }
        return max_sum_so_far;
    }

    // O(n+m), O(1)
	/*
     This problem is called Merge Sorted Array
     Example:-
     Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

    Algo :- We start filling the array from right end till all elements of nums1 is consumed
        After that remaining element of nums2 is utitlised

        */

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (j > -1) {
            if (i > -1) {
                nums1[k--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }

    // Thoughts:-

	/*

	   TC = O(2^n), Sc = O(n)

	   Algorithm:-
	  - The idea is to split array in two parts such that
	     avg(A) = avg(B)
	  - Iterate thrugh array elements and for each elem
	     check if we can split it in two parts with equals avg

	  -  We have choice of take or dont take in first part
	     ie. if arr(i) is taken in part1 sumA+arr(i)
	     else sumB + arr(i)

	  - Do above step recursilvely and backtrack
	  - check if sumA == sumB && (index == n-1) { that means all elements have been segregated into two parts successfuly
	  }
	     - if true return true
	      else return false and recurse further
	  - Add Memoization to improve exponential time complexity

	  total  = sumA + sumB
	  sumB = total - sumA

	  A+B=n
	  B=n-A

	  sumA/A = sumB/B

	  sumA/A = total-sumA/B
	  sumA/A = total-sumA/n-A
	  n*sumA/A  = total
	  sumA = total * lenA / n

	  problem boils down to finding a subsequence of length len1
	  with sum equals sumA
	*/

    public boolean splitArraySameAverage(int[] nums) {
        int n = nums.length, total = 0;
        for (int i = 0; i < n; i++) total += nums[i];

        HashMap<String, Boolean> map = new HashMap<>();
        for (int cnt = 1; cnt < n; cnt++) {
            if ((total * cnt) % n == 0) {
                if (isPossible(nums, 0, cnt, (total * cnt) / n, map)) return true;
            }
        }
        return false;
    }

    private boolean isPossible(int[] nums, int ind, int len, int sum, HashMap<String, Boolean> map) {
        int n = nums.length;
        // base case
        if (sum == 0 && len == 0) {
            return true;
        }

        if (ind >= n || len == 0) return false;
        String key = len + "-" + sum + "-" + ind;
        if (map.containsKey(key)) return map.get(key);
        // if number can be taken
        if (sum - nums[ind] >= 0) {
            // taken
            boolean case1 = isPossible(nums, ind + 1, len - 1, sum - nums[ind], map);

            // not taken
            boolean case2 = isPossible(nums, ind + 1, len, sum, map);

            map.put(key, (case1 || case2));
            return case1 || case2;
        }

        // Can't be taken
        boolean case2 = isPossible(nums, ind + 1, len, sum, map);

        map.put(key, case2);
        return case2;
    }

    // Min. no. of steps to  make both strings equal
    // Input: word1 = "sea", word2 = "eat"
    // Output: 2
    // TC = O(n1*n2), SC =  O(n1*n2)
    public int minDistance(String word1, String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        int[][] dp = new int[n1 + 1][n2 + 1]; // To get no of steps after removing a character from either strings

        for (int i = 0; i <= n1; i++) {
            for (int j = 0; j <= n2; j++) {
                // if no character is removed from word1 then steps = no. of characters of word2
                if (i == 0) dp[i][j] = j;
                    // if no character is removed from word2 then steps = no. of character of word1
                else if (j == 0) dp[i][j] = i;
                else {
                    // if prev chartacters are same then steps = prev steps
                    if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }

                    // Else take min of both cases
                    else {
                        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1]);
                    }
                }
            }
        }
        return dp[n1][n2];

    }

    // TC = O(V+E), SC = O(V+E)
    long maxScore;
    int count;

    public int countHighestScoreNodes(int[] parents) {
        int n = parents.length;
        // create an adjacancy list of edges of each vertex
        List<Integer> list[] = new ArrayList[n];

        for (int i = 0; i < n; i++) {
            list[i] = new ArrayList<>();
        }

        for (int i = 1; i < n; i++) {
            list[parents[i]].add(i);
        }

        maxScore = 0L;
        count = 0;
        dfs(0, list, n); // dfs to count the  number of nodes in tree with root 0

        return count;
    }


    // This function calculates the number of node in the subtree of root u
    private long dfs(int u, List<Integer> list[], int n) {

        int total = 0;
        long prod = 1L, rem, val;

        for (Integer v : list[u]) {
            val = dfs(v, list, n);
            total += val;
            prod *= val;
        }

        // if nodes  remaning beyond subtree with root u the  it will be taken into consideration
        rem = (long) (n - total - 1);
        if (rem > 0) prod *= rem;

        // Logic for maxScore
        if (prod > maxScore) {
            maxScore = prod;
            count = 1;
        } else if (prod == maxScore) {
            count++;
        }
        return total + 1;
    }

    // Input: n = 1000
    // Output: 1333
    // TC = O(n)
    // SC = O(1)
    public int nextBeautifulNumber(int n) {
        int number = n;
        while (true) {
            number++;
            String s = Integer.toString(number);

            int len = s.length();
            HashMap<Integer, Integer> freq = new HashMap<>();
            // calculate the frequency of each digit
            for (int i = 0; i < len; i++) {
                Integer digit = Integer.parseInt(String.valueOf(s.charAt(i)));
                freq.put(digit, freq.getOrDefault(digit, 0) + 1);
                if (digit < freq.get(digit)) break;
            }

            boolean isValid = true;
            // For every digit verify its freq
            for (Integer key : freq.keySet()) {
                if (freq.get(key) != key) {
                    isValid = false;
                    break;
                }
            }

            if (isValid) return number;
        }
    }

    // TOPOLOGICAL SORT
    // Use toplogical sort for indegree and pq to minisnmise the time taken to complete the course
    // TC = O(V+E) // As Simple DFS, SC = O(V) {Stack space}

    public int minimumTime(int n, int[][] relations, int[] time) {
        // create adjancy list of graph
        List<List<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        // create an indegree array for each node
        int[] indegree = new int[n];
        for (int[] e : relations) {
            // create the graph
            graph.get(e[0] - 1).add(e[1] - 1);
            indegree[e[1] - 1] += 1;
        }

        int maxTime = 0;
        // create a MIN-PQ to minimise time taken to complete all courses

        PriorityQueue<int[]> pq = new PriorityQueue<>(10, (t1, t2) -> t1[1] - t2[1]);

        //Insert all dead end nodes ie. nodes with 0 indegree
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 0) pq.offer(new int[]{
                    i, time[i]
            });
        }

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int currCourse = curr[0];
            int currTime = curr[1];

            maxTime = Math.max(maxTime, currTime);

            // Visit all adjance vertex of curr node and update the time taken to complete the courses
            for (int next : graph.get(currCourse)) {
                // reduce indegreee by 1 as node as visited
                indegree[next] -= 1;

                // if indegree=0 means all its adajancent node has been visisted , Now move on to next  parent node
                if (indegree[next] == 0) {
                    pq.offer(new int[]{
                            next, currTime + time[next]
                    });
                }
            }
        }

        // return minTime Taken to  complete all courses
        return maxTime;
    }

    // TC = O(n)
    private boolean isVowel(char ch) {
        return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u';
    }

    public long countVowels(String word) {
        long n = word.length();
        long ans = 0;
        for (long i = 0; i < n; i++) {
            if (isVowel(word.charAt((int) i))) ans += (long) ((n - i) * (i + 1) * 1L);
        }
        return ans;
    }


    // TC = O(2n) + O(2n)
    // Sliding window approach
    // We have found the  count of vowels <= k and then substracted vowels <= k to get vowels == k
    public int countVowelSubstrings(String word) {
        return cntVowelKMaxSubstrings(word, 5) - cntVowelKMaxSubstrings(word, 4);
    }

    private int cntVowelKMaxSubstrings(String word, int k) {
        int n = word.length();
        int i = 0, cnt = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        for (int j = 0; j < n; j++) {
            char ch = word.charAt(j);
            //when we have reached at a consonant then  directly jump till j
            if (!isVowel(ch)) {
                map.clear();
                i = j + 1;
                continue;
            }

            map.put(ch, map.getOrDefault(ch, 0) + 1);

            //count all possible sunstrings with at max K vowels
            while (map.size() > k) {
                map.put(word.charAt(i), map.getOrDefault(word.charAt(i), 0) - 1);
                if (map.get(word.charAt(i)) == 0) map.remove(word.charAt(i));
                i++;
            }
            cnt += (j - i + 1);
        }
        return cnt;
    }


    // TC = O(n^E) --> Exponential
    // We have done dfs to get the max path
    // As each node can be visisted any number of times and hence  do dfs along with updating  node  vis status backtracking
    int maxVal;

    public int maximalPathQuality(int[] values, int[][] edges, int maxTime) {
        int n = values.length;
        int[] vis = new int[n];
        // base case
        if (edges.length == 0) {
            if (n > 0) {
                return values[0];
            } else return 0;
        }
        // Initialise adjancy list for<Integer, List<Pair<Integer, Integer>> type of Map
        HashMap<Integer, List<Pair<Integer, Integer>>> adj = new HashMap<>();
        // create adjancy list
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int tm = edge[2];

            if (adj.containsKey((u))) {
                List<Pair<Integer, Integer>> val = adj.get(u);
                val.add(new Pair(v, tm));
                adj.put(u, val);
            } else {
                adj.put(u, new ArrayList<Pair<Integer, Integer>>() {
                    {
                        add(new Pair(v, tm));
                    }
                });
            }

            if (adj.containsKey((v))) {
                List<Pair<Integer, Integer>> val = adj.get(v);
                val.add(new Pair(u, tm));
                adj.put(v, val);
            } else {
                // For ArrayList
                adj.put(v, new ArrayList<Pair<Integer, Integer>>() {
                    {
                        add(new Pair(u, tm));
                    }
                });
            }

            maxVal = 0;
        }
        // dfs
        f(0, 0, values, adj, vis, maxTime);
        // return maxVal
        return maxVal;
    }

    private void f(int node, int val, int[] values, HashMap<Integer, List<Pair<Integer, Integer>>> adj, int[] vis, int maxTime) {
        // base case
        if (maxTime < 0) return;

        vis[node]++;
        if (vis[node] == 1) val += values[node];

        if (node == 0) {
            maxVal = Math.max(maxVal, val);
        }

        // if only node is contained
        if (adj.containsKey(node)) {
            for (Pair<Integer, Integer> p : adj.get(node)) {
                int child = p.getKey();
                int time = p.getValue();

                f(child, val, values, adj, vis, maxTime - time);
            }
        }

        vis[node]--;
    }

     /*
         arr = [3,4,3,3]
         k = 2

        pq = {4, 3, 3, 3}
        map = { (3, (0, 2,3), (4,1))}
        ans = {1,0,2,3}
        result = [4, 3]
     */

    //    TC = O(nlogn), sc = O(n)
    public int[] maxSubsequence(int[] nums, int k) {
        int n = nums.length;
        // max-pq
        // TC = O(nlogn)
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>(Collections.reverseOrder());
        for (int num : nums) {
            pq.add(num);
        }

        // elem-idx map
        HashMap<Integer, List<Integer>> map = new HashMap<>();

        // TC = O(n)
        for (int i = 0; i < n; i++) {
            if (map.containsKey(nums[i])) {
                List<Integer> exist = map.get(nums[i]);
                exist.add(i);
                map.put(nums[i], exist);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(nums[i], list);
            }
        }


        // TC = O(nlogn)
        List<Integer> ans = new ArrayList<>();
        // Fetch k largest elements
        while (!pq.isEmpty()) {
            int elem = pq.poll();

            //Poll all same elements
            while (!pq.isEmpty() && pq.peek() == elem) pq.poll();

            List<Integer> idx = map.get(elem);
            Collections.sort(idx);
            boolean exhausted = false;
            for (int e : idx) {
                if (ans.size() < k) ans.add(e);
                else {
                    exhausted = true;
                    break;
                }
            }
            if (exhausted) break;
        }

        Collections.sort(ans);
        int[] result = new int[k];
        int index = 0;

        // TC = O(n)
        // Traverse the array once again and remove the occurrence of index if present
        // In this way order can be preserved
        for (int i = 0; i < n; i++) {
            if (ans.contains(i)) {
                result[index++] = nums[i];
            }
        }

        return result;
    }

    // Sliding window
    // TC = O(n)
    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        int n = security.length;
        boolean[] left = new boolean[n];
        int slow = 0;
        // l->r traversal for sliding window for  non-increasing elements
        for (int i = 0; i < n; i++) {
            if (i == 0 || security[i - 1] < security[i]) slow = i;
            if (i - slow >= time) left[i] = true;
        }

        slow = n - 1;
        // r->l traversal for sliding window for  non-descresing elements
        List<Integer> ans = new ArrayList<>();
        for (int i = n - 1; i >= 0; i--) {
            if (i == n - 1 || security[i + 1] < security[i]) slow = i;
            if (slow - i >= time && left[i]) {
                ans.add(i);
            }
        }
        return ans;
    }

    // TC = O(n*n), SC = O(1)
    // Solution by Anand
    public String firstPalindrome(String[] words) {
        int n = words.length;
        for (String word : words) {
            if (isPalidrome(word)) return word;
        }
        return "";
    }

    private boolean isPalidrome(String word) {
        int n = word.length();
        for (int i = 0; i < Math.abs(n / 2); i++) {
            if (word.charAt(i) != word.charAt(n - i - 1)) return false;
        }
        return true;
    }


    // TC = O(n), SC = O(1)
    public String addSpaces(String s, int[] spaces) {
        int n = spaces.length;

        if (n == 0) return s;
        int idx = 0;
        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (idx < n && i == spaces[idx]) {
                ans.append(" ").append(s.charAt(i));
                idx++;
            } else ans.append(s.charAt(i));
        }
        return ans.toString();
    }


    /*

       Constraints:

    1 <= prices.length <= 105
    1 <= prices[i] <= 105


    Input: prices = [3,2,1,4]
    Output: 7
    Explanation: There are 7 smooth descent periods:
    [3], [2], [1], [4], [3,2], [2,1], and [3,2,1]
    Note that a period with one day is a smooth descent period by the definition.

    // 12,11,10,9,8,7,6,5,4,3 = 10 * 11 / 2 = 55

    [12,11,10,9,8,7,6,5,4,3,4,3,10,9,8,7]
    TC = O(2n), SC=O(1)
     */
    // TODO: Not solved with all edge cases
    public long getDescentPeriods(int[] prices) {
        boolean flag = false;
        long cnt = 0L;
        int l = 0, h = 0, n = prices.length, idx = 0;
        boolean isnincreasing = true;
        while (idx < n) {
            if (h == 0) {
                h++;
            } else if (prices[h - 1] - prices[h] == 1) {
                flag = true;
                if (!isnincreasing) {
                    isnincreasing = true;
                    l = h - 1;
                }
                h++;
            } else {
                cnt += (long) Math.abs((h - l + 1) * (h - l) / 2);
                isnincreasing = false;
                l = h;
                h++;
            }
            idx++;
        }

        if (h == idx) {
            cnt += (long) Math.abs((h - l + 1) * (h - l) / 2);
        }
        if (flag) return cnt;
        return n;
    }

    // Solved by anand
    public long getDescentPeriodsSimple(int[] prices) {
        int n = prices.length;
        int s = 0, e = 0;
        long cnt = 1;
        for (e = 1; e < n; e++) {
            if (prices[e - 1] - prices[e] == 1)
                cnt += (e - s + 1);
            else {
                s = e;
                cnt++;
            }
        }
        return cnt;
    }

    // TODO:- Solve for all edge cases
    public int numSubarrayBoundedMax(int[] nums, int left, int right) {
        int n = nums.length;
        int s = 0;
        int cnt = 0, max = Integer.MIN_VALUE;
        for (int e = 0; e < n; e++) {
            max = Math.max(nums[e], max);
            if (left <= max && max <= right) {
                if (!(left <= nums[e] && nums[e] <= right)) cnt++;
                else cnt += (e - s + 1);
                System.out.println(cnt);
            } else {
                if (left <= nums[e] && nums[e] <= right) cnt++;
                s = e;
                while (!(left <= nums[s] && nums[s] <= right)) {
                    s++;
                    e++;
                }
                e -= 1;
                max = Integer.MIN_VALUE;
            }
        }
        return cnt;
    }

    // Solved by anand
    // TC = O(n), SC = O(1)
//    Keep on moving right pointer till condition is met
//    The moment when  coniditon id broken cnt the results and shift left
//    pointer towards right by just one and repeat above process
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int n = nums.length;
        int s = 0;
        int cnt = 0, p = 1;
        for (int e = 0; e < n; e++) {
            p *= nums[e];
            while (p >= k) p /= nums[s++];
            cnt += (e - s + 1);
        }
        return cnt;
    }


    // Author : Anand
    // TC = O(2n), SC=O(n)
    public long[] getDistances(int[] arr) {
        int n = arr.length;
        HashMap<Integer, Long> cntMap = new HashMap<>();
        HashMap<Integer, Long> sumMap = new HashMap<>();
        long[] ans = new long[n];

        for (int i = 0; i < n; i++) {
            int key = arr[i];
            ans[i] += cntMap.getOrDefault(key, 0L) * i - (sumMap.getOrDefault(key, 0L));
            cntMap.put(key, cntMap.getOrDefault(key, 0L) + 1);
            sumMap.put(key, sumMap.getOrDefault(key, 0L) + i);
        }
        cntMap.clear();
        sumMap.clear();

        for (int i = n - 1; i >= 0; i--) {
            int key = arr[i];
            ans[i] += sumMap.getOrDefault(key, 0L) - (cntMap.getOrDefault(key, 0L) * i);
            cntMap.put(key, cntMap.getOrDefault(key, 0L) + 1);
            sumMap.put(key, sumMap.getOrDefault(key, 0L) + i);
        }
        return ans;
    }

    // Author : Anand
    // Simple BFS travesal must solve the problem
    // Brute-Force
    // TC = O(n^2)
    public int[] executeInstructions(int n, int[] startPos, String s) {
        int len = s.length();
        int[] ans = new int[len];
        int r = n - 1, c = n - 1;
        for (int i = 0; i < len; i++) {
            int cnt = 0;
            int x = startPos[0];
            int y = startPos[1];
            for (int j = i; j < len; j++) {
                if (s.charAt(j) == 'U') x--;
                else if (s.charAt(j) == 'R') y++;
                else if (s.charAt(j) == 'L') y--;
                else x++;

                if (isValidPos(x, y, n)) cnt++;
                else {
                    ans[i] = cnt;
                    break;
                }
            }
            ans[i] = cnt;
        }

        return ans;
    }

    private boolean isValidPos(int i, int j, int n) {
        return (i >= 0 && j >= 0 && i < n && j < n);
    }

    /*
    Input: nums = [2,10,6,4,8,12]
    Output: [3,7,11]
    Explanation:
    If arr = [3,7,11] and k = 1, we get lower = [2,6,10] and higher = [4,8,12].
    Combining lower and higher gives us [2,6,10,4,8,12], which is a permutation of nums.
    Another valid possibility is that arr = [5,7,9] and k = 3. In that case, lower = [2,4,6] and higher = [8,10,12].

    Input: nums = [1,1,3,3]
    Output: [2,2]
    Explanation:
    If arr = [2,2] and k = 1, we get lower = [1,1] and higher = [3,3].
    Combining lower and higher gives us [1,1,3,3], which is equal to nums.
    Note that arr cannot be [1,3] because in that case, the only possible way to obtain [1,1,3,3] is with k = 0.
    This is invalid since k must be positive.
     */

    // Author: Anand
    // Approach:-
    // For all possible k (integer) check if its possible to segregate array
    // into two half's then based on that value we can recover the array
    // TC = O(n2)
    public int[] recoverArray(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int lowest = nums[0];

        HashMap<Integer, Integer> map1 = new HashMap<>();// To store cnt of each element
        for (int num : nums) map1.put(num, map1.getOrDefault(num, 0) + 1);

        for (int i = 1; i < n; i++) {
            int highest = nums[i];
            int diff = highest - lowest;

            if (diff > 0 && (highest - lowest) % 2 == 0) {
                int k = Math.abs((highest - lowest) / 2);
                List<Integer> ans = new ArrayList<>();
                Map<Integer, Integer> map = new HashMap<>(map1);

                for (int low : nums) {
                    int high = low + 2 * k;
                    // That means it's possible to make a segregation at this place
                    if (map.containsKey(low) && map.containsKey(high)) {
                        ans.add(low + k);
                        map.put(low, map.get(low) - 1);
                        map.put(high, map.get(high) - 1);
                        if (map.get(low) == 0) map.remove(low);
                        if (map.containsKey(high) && map.get(high) == 0) map.remove(high);
                    }
                }
                if (ans.size() == n / 2) return ans.stream().mapToInt(x -> x).toArray();
            }
        }
        return new int[]{};
    }

    // Author : Anand
    // TC = O(n), SC = O(1)
    public int mostWordsFound(String[] sentences) {
        int max = Integer.MIN_VALUE;
        for (String bs : sentences) max = Math.max(bs.split(" ").length, max);
        return max;
    }

    // Author: Anand
    // TC = O(mn), SC = O(mn)
    public int maximalSquare(char[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;

        // To store max size square formed at coordinate {i,j}
        int[][] dp = new int[row][col];
        int maxi = 0;

        // Fill top row and col
        for (int i = 0; i < col; i++) {
            dp[0][i] = matrix[0][i] - '0';
            maxi = Math.max(maxi, dp[0][i]);
        }
        for (int i = 0; i < row; i++) {
            dp[i][0] = matrix[i][0] - '0';
            maxi = Math.max(maxi, dp[i][0]);
        }

        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if ((matrix[i][j] - '0') == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                }
                maxi = Math.max(maxi, dp[i][j]);
            }
        }

        return maxi * maxi;
    }

    /*
    recipe = ["bread","sandwich"]
    Ig = [["yeast","flour"],["bread","meat"]]
    sm = ["yeast","flour","meat"]

    ans = ["bread", "sandwich"]
     // Author: Anand
     // TC = O(mn) where m = # rows, n = # cols of ingredients
    Instead of thinking the solution from left to right data
    Think it from right to left so basically we will create list of indexes that can be formed
    from ingredients via map DS.
    Maintain a ingredientRecipeCount -> Used to check if recipe is ready to be formed
    Create a supplyQueue and iterate for all supplies.
    Suplly will act as an ingredient for the recipe, if ingredient is mapped with some recipes
    then update its ingredientRecipeCount and check if recipe is ready to me made
    if yes then add recipe to supply chain and result list
    Finally return result list
     */

    public List<String> findAllRecipes(String[] recipes, List<List<String>> ingredients, String[] supplies) {
        if (recipes == null || recipes.length == 0
                || ingredients == null || ingredients.size() == 0
                || supplies == null || supplies.length == 0
                || recipes.length != ingredients.size()
        ) return new ArrayList<>();


        // key- ingredient, value -> List of indexes of recipe
        Map<String, List<Integer>> ingredientRecipeMap = new HashMap<>();

        for (int i = 0; i < ingredients.size(); i++) {
            for (int j = 0; j < ingredients.get(i).size(); j++) {
                String ingredient = ingredients.get(i).get(j);
                List<Integer> recipeList = ingredientRecipeMap.computeIfAbsent(ingredient, k -> new ArrayList<>());
                recipeList.add(i);
            }
        }
        // Maintain an Array for count of ingredient for a recipe
        int[] recipeIngredientCount = new int[recipes.length];

        List<String> result = new ArrayList<>();
        LinkedList<String> supplyQueue = new LinkedList<String>(Arrays.asList(supplies));

        while (!supplyQueue.isEmpty()) {
            String ingredient = supplyQueue.poll();

            if (ingredientRecipeMap.containsKey(ingredient)) {
                List<Integer> recipeList = ingredientRecipeMap.get(ingredient);

                for (Integer index : recipeList) {
                    recipeIngredientCount[index]++;

                    // All ingredients are present for a recipe then it can be made and added as a suplly
                    if (recipeIngredientCount[index] == ingredients.get(index).size()) {
                        supplyQueue.offer(recipes[index]);
                        result.add(recipes[index]);
                    }
                }
            }
        }
        return result;
    }

    // ArrayDeque based approach
    public List<String> findAllRecipesV1(String[] recipes, List<List<String>> ingredients, String[] supplies) {
        List<String> result = new ArrayList<>();
        return result;
    }

    public boolean possibleToStamp(int[][] grid, int stampHeight, int stampWidth) {
        return false;
    }

    /*

    [5,4,2,1]
    len = 4

    lh = [6 6]
     */
    public int pairSum(ListNode head) {
        int maxSum = 0;
        int len = getLength(head);
        int ind = 0;
        List<Integer> lh = new ArrayList<>();

        while (head != null) {
            if (ind >= len / 2) {
                lh.set(len - ind - 1, lh.get(len - ind - 1) + head.val);
                maxSum = Math.max(maxSum, lh.get(len - ind - 1));
            } else lh.add(head.val);

            ind++;
            head = head.next;
        }
        return maxSum;
    }

    private int getLength(ListNode head) {
        int cnt = 0;
        ListNode curr = head;
        while (curr != null) {
            cnt++;
            curr = curr.next;
        }
        return cnt;
    }


    /*
      TC = O(n), SC = O(n)
      A palindrome must be mirrored over the center. Suppose we have a palindrome. If we prepend the word "ab" on the left, what must we append on the right to keep it a palindrome?
      We must append "ba" on the right. The number of times we can do this is the minimum of (occurrences of "ab") and (occurrences of "ba").
      For words that are already palindromes, e.g. "aa", we can prepend and append these in pairs as described in the previous hint. We can also use exactly one in the middle to form an even longer palindrome.
      */
    public int longestPalindrome(String[] words) {
        int lp = 0;
        Map<String, Integer> map = new HashMap<>(); // Map of word-count

        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }

        for (String word : words) {
            String reverse = new StringBuilder(word).reverse().toString();
            if (!(word.charAt(0) == word.charAt(word.length() - 1))
                    && (map.containsKey(word) && map.get(word) > 0 && map.containsKey(reverse) && map.get(reverse) > 0
            )) {
                int count = Math.min(map.get(word), map.get(reverse));
                lp += count * 4;
                while (count-- > 0) {
                    map.put(word, map.getOrDefault(word, 0) - 1);
                    if (map.get(word) <= 0) map.remove(word);
                    map.put(reverse, map.getOrDefault(reverse, 0) - 1);
                    if (map.get(reverse) <= 0) map.remove(reverse);
                }
            } else if (word.charAt(0) == word.charAt(word.length() - 1)
                    && (map.containsKey(word) && map.get(word) > 0)
            ) {
                int count = map.get(word);
                int evenCnt = count / 2;
                lp += evenCnt * 4;
                map.put(word, map.getOrDefault(word, 0) - (evenCnt * 2));
                if (map.get(word) <= 0) map.remove(word);
            }
        }

        //  We can also use exactly one in the middle to form an even longer palindrome.
        for (String word : map.keySet()) {
            if (word.charAt(0) == word.charAt(word.length() - 1)
                    && map.get(word) == 1) {
                lp += 2;
                map.put(word, map.getOrDefault(word, 0) - 1);
                if (map.get(word) <= 0) map.remove(word);
                break;
            }
        }

        return lp;
    }

    public boolean checkValid(int[][] matrix) {
        int n = matrix.length;
        HashMap<Integer, Boolean> map = new HashMap<>();
        for (int i = 1; i <= n; i++) map.put(i, false);
        // traverse row-wise
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                map.put(matrix[i][j], true);
            }
            // check if all numbers are occupied
            for (Map.Entry<Integer, Boolean> entry : map.entrySet()) {
                if (!entry.getValue()) return false;
            }
            for (int id = 1; id <= n; id++) map.put(id, false);
        }

        // traverse col-wise
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                map.put(matrix[j][i], true);
            }
            // check if all numbers are occupied
            for (Map.Entry<Integer, Boolean> entry : map.entrySet()) {
                if (!entry.getValue()) return false;
            }
            for (int id = 1; id <= n; id++) map.put(id, false);
        }
        return true;
    }

    /*
    Input: nums = [0,1,0,1,1,0,0]
    Output: 1
    Explanation: Here are a few of the ways to group all the 1's together:
    [0,0,1,1,1,0,0] using 1 swap.
    [0,1,1,1,0,0,0] using 1 swap.
    [1,1,0,0,0,0,1] using 2 swaps (using the circular property of the array).
    There is no way to group all 1's together with 0 swaps.
    Thus, the minimum number of swaps required is 1.
    */
    // TC = O(n)
    // Author: Anand
    public static int minSwaps(int[] nums) {
        int n = nums.length;

        int total = 0;
        for (int num : nums) if (num == 1) total++;

        int sbArray = Integer.MAX_VALUE;
        int ptr = 0;

        // Create all subarrays of length=total
        for (int i = 0; i <= n - total; i++) {
            if (i == 0) {
                for (int j = i; j < total; j++) {
                    if (nums[j] == 0) ptr++;
                }
            } else {
                if (nums[i - 1] == 0) ptr--;
                if (nums[i - 1 + total] == 0) ptr++;
            }
            sbArray = Math.min(sbArray, ptr);
        }

        int[] newArray = Arrays.copyOf(nums, 2 * n);
        System.arraycopy(nums, 0, newArray, n, n);

        int newN = 2 * n;
        ptr = 0;

        // Create all subarrays of length=total
        for (int i = 0; i <= newN - total; i++) {
            if (i == 0) {
                for (int j = i; j < total; j++) {
                    if (newArray[j] == 0) ptr++;
                }
            } else {
                if (newArray[i - 1] == 0) ptr--;
                if (newArray[i - 1 + total] == 0) ptr++;
            }
            sbArray = Math.min(sbArray, ptr);
        }

        return sbArray;
    }

    // TC = O(MLogM)
    // Author : Anand


    /*
    Input: n = 2, batteries = [3,3,3]
    Output: 4
    Explanation:
    Initially, insert battery 0 into the first computer and battery 1 into the second computer.
    After two minutes, remove battery 1 from the second computer and insert battery 2 instead. Note that battery 1 can still run for one minute.
    At the end of the third minute, battery 0 is drained, and you need to remove it from the first computer and insert battery 1 instead.
    By the end of the fourth minute, battery 1 is also drained, and the first computer is no longer running.
    We can run the two computers simultaneously for at most 4 minutes, so we return 4.



    2
    [3,4,5]

    sum = 3-1=2
    [5,1_0000_00000_00000L]
    level = 4 = 5 + 1  = 6
    dist = 0 -> 1 -> 2
    val = 5, nextVal = 1_0000_00000_00000L
     */
    public long maxRunTime(int n, int[] batteries) {
        long sum = 0;
        Queue<Long> pq = new PriorityQueue<>();
        for (long b : batteries) pq.offer(b);
        Arrays.sort(batteries);
        while (pq.size() != n) {
            sum += pq.poll();
        }
        pq.offer(1_000000_0000_0000L);

        long dist = 0L, level = pq.peek();
        while (pq.size() != 1 && sum >= 0) {
            dist++;
            level = pq.poll();
            long nextLevel = pq.peek();

            if ((nextLevel - level) * dist <= sum) {
                sum -= (nextLevel - level) * dist;
                level = nextLevel;
            } else {
                level += sum / dist;
                return level;
            }
        }

        return level;

    }

    // TC = O(nlogn)
    // Author : Anand
    public int findFinalValue(int[] nums, int original) {
        int n = nums.length;
        Arrays.sort(nums);
        int l = 0, r = n - 1;
        if (n == 1) {
            if (nums[0] == original) return original * 2;
            return original;
        }
        boolean isFound = false;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < original) {
                l = mid + 1;
            } else if (nums[mid] > original) {
                r = mid;
            } else {
                // Repeat the process
                original *= 2;
                l = 0;
                r = n - 1;
            }
            if (l == r) {
                if (!isFound) isFound = true;
                else break;
            }
        }
        return original;
    }

    /*
        Input: nums = [0,0,1,0]
        Output: [2,4]
        Explanation: Division at index

        Input: nums = [0,0,0]
        Output: [3]

        Input: nums = [1,1]
        Output: [0]

 [0,0,0]
 map = {(0,[0), (1,[1), (2, [2), (3,0], [3,0])}

          */
    // Tc = O(n)
    // Author : Anand
    public List<Integer> maxScoreIndices(int[] nums) {

        int n = nums.length;
        HashMap<Integer, Integer> map = new HashMap<>();// Used to store sum of 0 left and sum of 1 to right for each index

        int sum_zero = 0, i = 0;
        for (i = 0; i < n; i++) {
            map.put(i, sum_zero);
            if (nums[i] == 0) sum_zero++;
        }

        // put last_index
        map.put(i, sum_zero);

        int sum_one = 0;
        for (int j = n - 1; j >= 0; j--) {
            int zero = map.get(j);
            if (nums[j] == 1) sum_one++;
            map.put(j, sum_one + zero);
        }

        // calculate sum for each index
        // map is ready now start doing computation based logic
        int max_sum = 0;
        List<Integer> res = new ArrayList<>();
        for (Map.Entry entry : map.entrySet()) {
            int sum = (int) entry.getValue();
            if (sum > max_sum) {
                res = new ArrayList<>();
                res.add((Integer) entry.getKey());
                max_sum = sum;
            } else if (sum == max_sum)
                res.add((Integer) entry.getKey());
        }

        return res;
    }
    //--------------------------------------------------------

    public int finalValueAfterOperations(String[] operations) {

        Set<String> plus = new HashSet<>();
        plus.add("++X");
        plus.add("X++");

        Set<String> minus = new HashSet<>();
        minus.add("--X");
        minus.add("X--");
        int ans = 0;
        for (String operation : operations) {
            if (plus.contains(operation)) ans += 1;
            if (minus.contains(operation)) ans -= 1;
        }
        return ans;
    }

    /*
      2, if nums[j] < nums[i] < nums[k], for all 0 <= j < i and for all i < k <= nums.length - 1.
      1, if nums[i - 1] < nums[i] < nums[i + 1], and the previous condition is not satisfied.
      0, if none of the previous conditions holds.


       */
    // Tc = O(n)
    // Author : Anand
    public int sumOfBeauties(int[] nums) {
        int ans = 0;
        int n = nums.length;
        HashMap<Integer, List<Integer>> map = new HashMap<>();// Used ot store max to left and min to right for each index

        int max_left = nums[0];
        for (int i = 1; i <= n - 2; i++) {
            map.put(i, new ArrayList<>(Arrays.asList(max_left)));
            if (nums[i] > max_left) max_left = nums[i];
        }

        int min_right = nums[n - 1];
        for (int i = n - 2; i > 0; i--) {
            List<Integer> list = map.get(i);
            list.add(min_right);
            map.put(i, list);
            if (nums[i] < min_right) min_right = nums[i];
        }


        // map is ready now start doing computation based logic
        for (int i = 1; i <= n - 2; i++) {
            List<Integer> list = map.get(i);
            if (nums[i] > list.get(0) && nums[i] < list.get(1)) ans += 2;
            else if (nums[i - 1] < nums[i] && nums[i + 1] > nums[i]) ans++;
        }

        return ans;

    }


    public int minimumSum(int num) {
        TreeMap<Integer, Integer> freq = new TreeMap<>();  // asc order of elements freq
        while (num > 0) {
            int digit = num % 10;
            freq.put(digit, freq.getOrDefault(digit, 0) + 1);
            num /= 10;
        }
        // Freq map is created
        // Iterate through the freq map
        int num1 = 0, num2 = 0;
        Map<Integer, Integer> map = new HashMap<>();  // to store count of digits in num1 and num2

        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            int value = entry.getValue();
            int key = entry.getKey();
            while (value > 0) {
                int cnt1 = map.getOrDefault(1, 0);
                int cnt2 = map.getOrDefault(2, 0);
                if (cnt1 <= cnt2) {
                    num1 = num1 * 10 + key;
                    map.put(1, map.getOrDefault(num1, 0) + 1);
                    value--;
                    if (value-- > 0) {
                        num2 = num2 * 10 + key;
                        map.put(2, map.getOrDefault(num2, 0) + 1);
                    }
                    System.out.println("num1:" + num1 + " num2:" + num2);
                } else {
                    num2 = num2 * 10 + key;
                    map.put(2, map.getOrDefault(num2, 0) + 1);
                    value--;
                    if (value-- > 0) {
                        num1 = num1 * 10 + key;
                        map.put(1, map.getOrDefault(num1, 0) + 1);
                    }
                    System.out.println("num1:" + num1 + " num2:" + num2);
                }
            }
        }

        return num1 + num2;
    }

    public int[] pivotArray(int[] nums, int pivot) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        int freq = 0;
        for (int num : nums) {
            if (num == pivot) freq++;
            if (num < pivot) list1.add(num);
            else if (num > pivot) list2.add(num);
        }
        int[] ans = new int[nums.length];
        int ind = 0;
        for (int n : list1) {
            ans[ind++] = n;
        }
        while (freq-- > 0) ans[ind++] = pivot;
        for (int n : list2) {
            ans[ind++] = n;
        }
        return ans;
    }

    // Author: Anand
    public int minCostSetTime(int startAt, int moveCost, int pushCost, int targetSeconds) {
        int minS = Integer.MAX_VALUE;

        int mins = targetSeconds / 60;
        int seconds = targetSeconds % 60;

        if (mins > 99) {
            mins = 99;
            seconds += 60;
            if (seconds > 99) {
                seconds = 99;
            }
        }

        int digits = 0;
        int copy = targetSeconds;
        List<Integer> dig = new ArrayList<>();
        while (copy > 0) {
            digits++;
            dig.add(copy % 10);
            copy /= 10;
        }

        Collections.reverse(dig);
        if (digits <= 2) {
            int cost = 0;
            int curr = startAt;
            for (int d : dig) {
                if (curr != d) {
                    cost += moveCost; // move
                    cost += pushCost;// push
                    curr = d;
                } else {
                    cost += pushCost; // simply push
                }
            }

            minS = Math.min(minS, cost);
        }

        // 1st case
        minS = Math.min(cal(mins, seconds, startAt, moveCost, pushCost), minS);

        if (seconds == 0) {
            minS = Math.min(cal(mins - 1, 60, startAt, moveCost, pushCost), minS);
        }

        // 2nd case
        if (seconds <= 39 && mins > 0) {
            minS = Math.min(cal(mins - 1, 60 + seconds, startAt, moveCost, pushCost), minS);
        }

        return minS;
    }

    private int cal(int mins, int seconds, int startAt, int moveCost, int pushCost) {
        int cost = 0;
        int curr = startAt;

        String ms = String.valueOf(mins);

        String ss = String.valueOf(seconds);

        for (int i = 0; i < ms.length(); i++) {
            int d = Integer.parseInt(String.valueOf(ms.charAt(i)));
            if (curr != d) {
                cost += moveCost; // move
                cost += pushCost;// push
                curr = d;
            } else {
                cost += pushCost; // simply push
            }
        }


        if (seconds == 0) {
            if (curr != 0) {
                cost += moveCost; // move
            }
            cost += (2 * pushCost);// push 2 times
        } else {
            // if single digit then append zero
            if (seconds / 10 == 0) {
                if (curr != 0) {
                    cost += moveCost; // move
                }
                cost += pushCost;// push
                curr = 0;
            }

            for (int i = 0; i < ss.length(); i++) {
                int d = Integer.parseInt(String.valueOf(ss.charAt(i)));
                if (curr != d) {
                    cost += moveCost; // move
                    cost += pushCost;// push
                    curr = d;
                } else {
                    cost += pushCost; // simply push
                }
            }
        }
        return cost;
    }

    /*
    Input: nums = [3,1,2]
    Output: -1


    Input: nums = [7,9,5,8,1,3]
    Output: 1

 */
    public long minimumDifference(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();// elem-ind
        int ind = 0;
        for (int num : nums) map.putIfAbsent(ind++, num);
        long f = Long.MAX_VALUE, s = Long.MIN_VALUE;
        first(nums, 0, 0, f, 0, map);
        System.out.println("f=" + f);
        second(nums, 0, 0, s, 0, map);
        System.out.println("s=" + s);
        return f - s;
    }

    private void first(int[] nums, int ind, long sum, long f, int cnt, Map<Integer, Integer> map) {

        // base case
        if (ind >= nums.length) return;
        if (cnt == nums.length) {
            f = Math.min(f, sum);
            return;
        }
        if (cnt > nums.length) return;

        // t
        map.remove(ind);
        sum += nums[ind];
        cnt++;
        first(nums, ind + 1, sum, f, cnt, map);

        // nt
        // backtrack
        map.putIfAbsent(ind, nums[ind]);
        sum -= nums[ind];
        cnt--;
        first(nums, ind + 1, sum, f, cnt, map);

    }

    private void second(int[] nums, int ind, long sum, long s, int cnt, Map<Integer, Integer> map) {


        // base case
        if (ind >= nums.length) return;
        if (cnt == nums.length) {
            s = Math.max(s, sum);
            return;
        }
        if (cnt > nums.length) return;

        long t, nt;
        // t
        if (map.containsKey(ind)) {
            // t
            map.remove(ind);
            sum += nums[ind];
            cnt++;
            second(nums, ind + 1, sum, s, cnt, map);

            // nt
            // backtrack
            map.putIfAbsent(ind, nums[ind]);
            sum -= nums[ind];
            cnt--;
            second(nums, ind + 1, sum, s, cnt, map);
        } else second(nums, ind + 1, sum, s, cnt, map);
    }

    public int[] sortEvenOdd(int[] nums) {
        List<Integer> even = new ArrayList<>();
        List<Integer> odd = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i % 2 == 0) {
                even.add(nums[i]);
            } else odd.add(nums[i]);
        }
        Collections.sort(even);
        Collections.sort(odd);
        Collections.reverse(odd);
        int ind = 0;

        int i = 0;
        while (even.size() > 0 || odd.size() > 0) {
            if (i < even.size() - 1) {
                nums[ind++] = even.get(i);
            }
            if (i < odd.size() - 1) {
                nums[ind++] = odd.get(i);
            }
            i++;
        }

        return nums;
    }

    // Author: Anand
    public long smallestNumber(long num) {
        long abs = Math.abs(num);
        long[] digits = Arrays.stream(String.valueOf(abs).chars().toArray()).map(x -> x - '0').mapToLong(x -> x).toArray();
        StringBuilder ans = new StringBuilder();
        Arrays.sort(digits);

        if (num > 0) {
            for (long d : digits) {
                ans.append(d);
            }
            if (ans.toString().startsWith("0")) {
                int cnt = 0;
                StringBuilder newAns = new StringBuilder();
                for (char c : ans.toString().toCharArray()) {
                    if (c != '0') {
                        newAns.append(c);
                        break;
                    } else cnt++;
                }

                while (cnt-- > 0) {
                    newAns.append('0');
                }
                for (int i = newAns.length(); i < ans.length(); i++) {
                    newAns.append(ans.charAt(i));
                }
                return Long.parseLong(newAns.toString());
            }
            return Long.parseLong(ans.toString());
        } else {
            for (int i = digits.length - 1; i >= 0; i--) ans.append(digits[i]);
            return -Long.parseLong(ans.toString());
        }
    }

    public int countOperations(int num1, int num2) {
        int ans = 0;
        while (num1 != 0 && num2 != 0) {

            if (num1 >= num2) num1 -= num2;
            else num2 -= num1;
            ans++;
        }
        return ans;
    }

    public int minimumOperations(int[] nums) {
        int ans = 0;
        int n = nums.length;

        if (n == 1) return 0;
        if (n == 2) return nums[0] == nums[1] ? 1 : 0;
        Map<Integer, Integer> first = new TreeMap<>();
        Map<Integer, Integer> second = new TreeMap<>();

        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                first.put(nums[i], first.getOrDefault(nums[i], 0) + 1);
            } else {
                second.put(nums[i], second.getOrDefault(nums[i], 0) + 1);
            }
        }

        List<Map.Entry<Integer, Integer>> fl = new ArrayList<>(first.entrySet());
        fl.sort((e1, e2) -> e2.getValue().compareTo(e1.getValue()));


        List<Map.Entry<Integer, Integer>> sl = new ArrayList<>(second.entrySet());
        sl.sort((e1, e2) -> e2.getValue().compareTo(e1.getValue()));

        int c1 = fl.get(0).getKey(), c2 = sl.get(0).getKey();
        int ind = 1;

        while (c1 == c2) {
            if (fl.size() > ind && sl.size() > ind) {
                if ((int) fl.get(ind).getValue() > (int) sl.get(ind).getValue()) {
                    c1 = (int) fl.get(ind).getKey();
                } else {
                    c2 = (int) sl.get(ind).getKey();
                }
            } else {
                if (fl.size() > ind) {
                    c1 = (int) fl.get(ind).getKey();
                } else if (sl.size() > ind) {
                    c2 = (int) sl.get(ind).getKey();
                }
            }

            if (fl.size() < ind && sl.size() < ind) break;
            ind++;
        }

        if (c1 == c2) {
            c2 = Integer.MAX_VALUE;
        }
        // c1 and c2 is ready
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                if (nums[i] != c1) ans++;
            } else {
                if (nums[i] != c2) ans++;
            }
        }
        return ans;
    }

    /*
    Input: beans = [4,1,6,5]
    Output: 4
    Explanation:
    - We remove 1 bean from the bag with only 1 bean.
      This results in the remaining bags: [4,0,6,5]
    - Then we remove 2 beans from the bag with 6 beans.
      This results in the remaining bags: [4,0,4,5]
    - Then we remove 1 bean from the bag with 5 beans.
      This results in the remaining bags: [4,0,4,4]
    We removed a total of 1 + 2 + 1 = 4 beans to make the remaining non-empty bags have an equal number of beans.
    There are no other solutions that remove 4 beans or fewer.
     */

    // Author: Anand
    // TC = O(nlogn)
    public long minimumRemoval(int[] beans) {
        int n = beans.length;
        Arrays.sort(beans);
        long total = 0L;
        for (int bean : beans) total += bean;
        long ans = Long.MAX_VALUE;
        long m = n;
        for (int i = 0; i < n; i++, m--) {
            ans = Math.min(ans, total - (m * beans[i]));
        }
        return ans;
    }

    // Author: Anand
    // dfs :- Iterate all possible combinations and get best possible state after memo (to reduce recursive calls)
    // Author : Anand
    public int maximumANDSum(int[] nums, int numSlots) {
        int[] slots = new int[numSlots + 1];
        Map<List<Integer>, Integer> map = new HashMap<>();
        return f(nums, slots, numSlots, 0, map);
    }

    private int f(int[] nums, int[] slots, int numSlots, int ind, Map<List<Integer>, Integer> map) {
        // base case
        if (ind >= nums.length) return 0;

        List<Integer> slotList = Arrays.stream(slots).boxed().collect(Collectors.toList());
        slotList.add(ind);
        if (map.containsKey(slotList)) return map.get(slotList);

        // Explore all possibilities
        int ans = Integer.MIN_VALUE;
        for (int i = 1; i <= numSlots; i++) {
            // num can be placed inside the slot
            if (slots[i] < 2) {
                slots[i]++;
                ans = Math.max(ans, ((nums[ind] & i) + f(nums, slots, numSlots, ind + 1, map)));
                slots[i]--;
            }
        }
        map.put(slotList, ans);
        return ans;
    }

    // Author : Anand
    public long[] sumOfThree(long num) {
        long[] ans = new long[3];
        if ((num % 3 != 0)) return new long[]{};
        long val = num / 3;
        ans[0] = val - 1;
        ans[1] = val;
        ans[2] = val + 1;
        return ans;
    }

    public String repeatLimitedString(String s, int repeatLimit) {

        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        StringBuilder ans = new StringBuilder();
        for (int i = 25; i >= 0; i--) {
            char ch = (char) ('a' + i);
            boolean taken = false;
            if (map.containsKey(ch)) {
                while (map.containsKey(ch)) {
                    if (!taken) {
                        if (map.get(ch) < repeatLimit) {
                            int rl = map.get(ch);
                            StringBuilder perm = new StringBuilder();
                            while (rl-- > 0) {
                                perm.append(ch);
                            }
                            ans.append(perm.toString());
                            map.remove(ch);
                        } else {
                            int rl = repeatLimit;
                            if (map.get(ch) <= 0) map.remove(ch);
                            StringBuilder perm = new StringBuilder();
                            while (rl-- > 0) {
                                perm.append(ch);
                            }
                            ans.append(perm.toString());
                            map.put(ch, map.get(ch) - repeatLimit);
                        }
                        taken = true;
                    } else {
                        boolean isNextCharPresent = false;
                        for (int j = (int) ch - 1; j >= 0; j--) {
                            char nc = (char) ('a' + j);

                            if (map.containsKey(nc) && map.get(nc) > 0) {
                                map.put(nc, map.get(nc) - 1);
                                isNextCharPresent = true;
                            }
                            if (map.get(nc) <= 0) map.remove(nc);
                            ans.append(nc);
                            break;
                        }
                        if (!isNextCharPresent) return ans.toString();
                        taken = false;
                    }
                }
            }
        }
        return ans.toString();
    }

    public long countPairs(int[] nums, int k) {
        Map<Long, Long> gcdMap = new HashMap<>(); // to store gcd factors count seen so far
        long result = 0;
        for (int n : nums) {
            long gcd = __gcd(n, k);
            for (long num : gcdMap.keySet()) {
                if ((long) gcd * num % k == 0) {
                    result += gcdMap.get(num);
                }
            }
            gcdMap.put(gcd, gcdMap.getOrDefault(gcd, 0L) + 1);
        }
        return result;
    }

    //long version for gcd
    public long __gcd(long a, long b) {
        if (b == 0)
            return a;

        return __gcd(b, a % b);
    }


    // Author: Anand
    public List<Long> maximumEvenSplit(long finalSum) {
        long ind = 2;
        long sum = 0;
        LinkedList<Long> ans = new LinkedList<Long>();
        if (finalSum % 2 != 0) return ans;
        while (ind <= finalSum) {
            ans.add(ind);
            finalSum -= ind;
            ind += 2;
        }

        // set the last possible value
        ans.set(ans.size() - 1, finalSum + ans.peekLast());
        return ans;
    }

    // Author: Anand
    public int prefixCount(String[] words, String pref) {
        int cnt = 0;
        for (String word : words) {
            if (word.startsWith(pref)) cnt++;
        }
        return cnt;
    }

    // Author: Anand
    public int minSteps(String s, String t) {
        int cnt = 0;
        Map<Character, Integer> freq1 = new HashMap<>();
        Map<Character, Integer> freq2 = new HashMap<>();
        for (char c : s.toCharArray()) {
            freq1.put(c, freq1.getOrDefault(c, 0) + 1);
        }

        for (char c : t.toCharArray()) {
            freq2.put(c, freq2.getOrDefault(c, 0) + 1);
        }

        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (freq1.containsKey(c)) {
                freq1.put(c, freq1.getOrDefault(c, 0) - 1);
                if (freq1.get(c) <= 0) freq1.remove(c);
            } else cnt++;
        }

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (freq2.containsKey(c)) {
                freq2.put(c, freq2.getOrDefault(c, 0) - 1);
                if (freq2.get(c) <= 0) freq2.remove(c);
            } else cnt++;
        }
        return cnt;
    }

    // Author : Anand
    public int mostFrequent(int[] nums, int key) {

        int ans = 0;
        int n = nums.length;
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);

        Set<Integer> targets = new HashSet<>();
        for (int i = 0; i < n - 1; i++) if (nums[i] == key) targets.add(nums[i + 1]);

        for (int t : targets) {
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (nums[i] == key && nums[i + 1] == t) cnt++;
            }
            ans = Math.max(ans, cnt);
        }
        return ans;
    }

    // Author : Anand
    public int[] sortJumbled(int[] mapping, int[] nums) {
        Map<Integer, List<Integer>> freq = new TreeMap<>();
        for (int num : nums) {
            int nv = num;
            StringBuilder sb = new StringBuilder();
            if (num == 0) sb.append(mapping[0]);
            else {
                while (num > 0) {
                    sb.append(mapping[num % 10]);
                    num /= 10;
                }
            }

            int v = Integer.parseInt(sb.reverse().toString());
            if (freq.containsKey(v)) {
                List<Integer> elem = freq.get(v);
                elem.add(nv);
                freq.put(v, elem);
            } else {
                freq.put(v, Collections.singletonList(nv));
            }
        }

        int[] ans = new int[nums.length];
        int idx = 0;
        for (int e : freq.keySet()) {
            List<Integer> elem = freq.get(e);
            for (int v : elem) {
                ans[idx++] = v;
            }
        }
        return ans;
    }

    // Author : Anand
    public List<String> cellsInRange(String s) {
        String[] cells = s.split(":");
        String first = cells[0];
        String last = cells[1];
        int r = last.charAt(0) - first.charAt(0) + 1;
        int c = last.charAt(1) - first.charAt(1) + 1;
        List<String> ans = new ArrayList<>();
        int rn = 0;

        boolean isF = true;
        while (r-- > 0) {
            int cn = 0;
            while (c-- > 0) {
                String val = "";
                if (isF) val = first;
                else val = last;
                StringBuilder res = new StringBuilder(String.valueOf((char) (val.charAt(0) + rn)));
                res.append(Integer.parseInt(String.valueOf(val.charAt(1))) + cn++);
                System.out.println(res.toString());
                ans.add(res.toString());
            }
            isF = false;
            rn++;
        }
        return ans;
    }

    // Author : Anand
    public long minimalKSum(int[] nums, int k) {
        int len = nums.length;
        Arrays.sort(nums);
        long sumK = (long) k * (k + 1) / 2;
        for (int i = 0; i < len; i++) {
            if (nums[i] <= k) {
                if (i > 0 && nums[i] == nums[i - 1]) {
                } else {
                    sumK -= nums[i];
                    k++;
                    sumK += k;
                }
            } else break;
        }
        return sumK;
    }


    // Author: Anand
    // TODO :- Use factor based approach
    public List<Integer> replaceNonCoprimes(int[] nums) {
        int n = nums.length;
        int i = 0;
        Map<Integer, Integer> map = new HashMap<>();

        List<Integer> ans = new ArrayList<>();
        for (int ind = 0; ind < n; ind++) map.put(ind, nums[ind]);
        while (i < n - 1) {
            long gcd = __gcd((long) nums[i], (long) nums[i + 1]);

            if (gcd > 1) {
                int lcm;
                if (nums[i] == nums[i + 1]) lcm = nums[i];
                else lcm = (int) (nums[i] * (nums[i + 1] / gcd));
                map.remove(i);
                nums[i + 1] = lcm;
                map.put(i + 1, lcm);
            }
            i++;
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            ans.add(entry.getValue());
        }


        // check for all numbers unitl its extreme left
        // use factors based method to store factors till CI
        int k = 1;
        while (k < ans.size()) {
            long gcd = __gcd((long) ans.get(k), (long) ans.get(k - 1));
            if (gcd > 1) {
                int lcm;
                if (ans.get(k).equals(ans.get(k - 1))) lcm = ans.get(k);
                else lcm = (int) (ans.get(k) * (ans.get(k - 1) / gcd));
                ans.remove(k - 1);
                ans.set(k - 1, lcm);
                continue;
            }
            k++;
        }

        return ans;
    }


    public List<Integer> findKDistantIndices(int[] nums, int key, int k) {
        Set<Integer> ans = new HashSet<>();
        List<Integer> choice = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == key) choice.add(i);
        }

        for (int c : choice) {
            for (int i = 0; i < nums.length; i++) {
                if (Math.abs(c - i) <= k) ans.add(i);
            }
        }

        List<Integer> sortedList = new ArrayList<>(ans);
        Collections.sort(sortedList);
        return sortedList;
    }

    // Author : Anand
    public int maximumTop(int[] nums, int k) {
        int max = -1;

        if (nums.length == 1) {
            if (k % 2 != 0) {
                return -1;
            }
        }
        if (k >= nums.length) {
            for (int num : nums) {
                if (k == 1) return max;
                max = Math.max(max, num);
                k--;
            }
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (k == 1) {
                    int r = Integer.MIN_VALUE;
                    if (i + 1 < nums.length) {
                        r = Math.max(max, nums[i + 1]);
                    }
                    return Math.max(r, max);
                }
                max = Math.max(max, nums[i]);
                if (k-- == 0) return max;
            }
        }
        return max;
    }

    // Author : Anand
    public int digArtifacts(int n, int[][] artifacts, int[][] dig) {
        Set<List<Integer>> digsWell = new HashSet<>();
        for (int[] d : dig) digsWell.add(Arrays.stream(d).boxed().collect(Collectors.toList()));

        int cnt = 0;
        // For all artifacts check if its completely covered
        for (int[] art : artifacts) {
            int r1 = art[0];
            int c1 = art[1];
            int r2 = art[2];
            int c2 = art[3];

            boolean fl = true;
            for (int i = r1; i <= r2 && fl; i++) {
                for (int j = c1; j <= c2; j++) {
                    if (!digsWell.contains(new ArrayList<>(Arrays.asList(i, j)))) {
                        fl = false;
                        break;
                    }
                }
            }
            cnt += fl ? 1 : 0;
        }
        return cnt;
    }

    // Author :Anand
    public int countHillValley(int[] nums) {

        int cnt = 0;
        int last = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == last) continue;
            last = nums[i];
            boolean hill = false, valley = false;
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] > nums[j]) {
                    hill = true;
                    break;
                } else if (nums[i] < nums[j]) {
                    valley = true;
                    break;
                }
            }

            for (int j = i + 1; j < nums.length; j++) {
                if (hill && nums[i] > nums[j]) {
                    cnt++;
                    break;
                } else if (valley && nums[i] < nums[j]) {
                    cnt++;
                    break;
                } else if (nums[i] == nums[j]) continue;
                else break;
            }
        }
        return cnt;
    }

    // Author :Anand
    public int countCollisions(String s) {
        int l = 0, n = s.length(), r = n - 1, ans = 0;

        while (l < n && s.charAt(l) == 'L')
            l++;

        while (r >= 0 && s.charAt(r) == 'R')
            r--;

        while (l <= r) {
            if (s.charAt(l) != 'S')
                ans++;

            l++;
        }

        return ans;
    }

    /*
    Input: nums1 = [1,2,3], nums2 = [2,4,6]
    Output: [[1,3],[4,6]]
    Input: nums1 = [1,2,3,3], nums2 = [1,1,2,2]
    Output: [[3],[]]

     */
    // Author : Anand
    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> ans1 = Arrays.stream(nums1).boxed().distinct().collect(Collectors.toList());
        ans1.removeAll(Arrays.stream(nums2).boxed().distinct().collect(Collectors.toList()));

        List<Integer> ans2 = Arrays.stream(nums2).boxed().distinct().collect(Collectors.toList());
        ans1.removeAll(Arrays.stream(nums1).boxed().distinct().collect(Collectors.toList()));

        ans.add(ans1);
        ans.add(ans2);
        return ans;
    }

    // Author : Anand
    public int minDeletion(int[] nums) {
        int ans = 0;
        int n = nums.length;

        for (int i = 0; i < n - 1; i++) {
            int pos = i - ans;
            if (pos % 2 == 0 && nums[i] == nums[i + 1]) ans++;
        }

        return (n - ans) % 2 == 0 ? ans : ans + 1;
    }

    /*
    Input: queries = [1,2,3,4,5,90], intLength = 3
    Output: [101,111,121,131,141,999]
    Explanation:
    The first few palindromes of length 3 are:
    101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, ...
    The 90th palindrome of length 3 is 999.

     */
    // Author : Anand
    public long[] kthPalindrome(int[] queries, int intLength) {
        long[] ans = new long[queries.length];
        int pw = intLength % 2 == 0 ? (intLength / 2) - 1 : (intLength / 2);
        int start = (int) Math.pow(10, pw);

        int idx = 0;
        for (int q : queries) {
            StringBuilder number = new StringBuilder();
            number.append(start + q - 1);
            StringBuilder rev = new StringBuilder(number);
            rev.reverse();
            StringBuilder nnumber = new StringBuilder();
            nnumber.append(number);
            nnumber.append(intLength % 2 == 1 ? rev.substring(1) : rev);
            if (nnumber.length() != intLength) {
                ans[idx++] = -1;
            } else ans[idx++] = Long.parseLong(nnumber.toString());
        }
        return ans;
    }

    // Author: Anand
    public int triangularSum(int[] nums) {
        List<Integer> ans = Arrays.stream(nums).boxed().collect(Collectors.toList());
        while (ans.size() > 1) {
            for (int i = 0; i < ans.size() - 1; i++) {
                int e = (ans.get(i) + ans.get(i + 1)) % 10;
                ans.set(i, e);
            }
            ans.remove(ans.size() - 1);
        }
        return ans.get(0);
    }


    // Author: Anand
    public int largestInteger(int num) {
        List<Integer> even = new ArrayList<>();
        List<Integer> odd = new ArrayList<>();

        int numc = num;
        while (numc > 0) {
            int d = numc % 10;
            if (d % 2 == 0) even.add(d);
            else odd.add(d);
            numc /= 10;
        }

        Collections.sort(even);
        Collections.sort(odd);

        String nums = String.valueOf(num);

        StringBuilder ans = new StringBuilder();
        for (int i = 0; i < nums.length(); i++) {
            if (Integer.parseInt(String.valueOf(nums.charAt(i))) % 2 == 0 && even.size() > 0) {
                ans.append(even.get(even.size() - 1));
                even.remove(even.size() - 1);
            } else if (odd.size() > 0) {
                ans.append(odd.get(odd.size() - 1));
                odd.remove(odd.size() - 1);
            } else break;
        }

        return Integer.parseInt(ans.toString());
    }

    /*
      I = [2,-1,1]
      O = 1
     */
    // Author : Anand
    public int findClosestNumber(int[] nums) {
        int ans = Integer.MAX_VALUE;
        List<Integer> list = new ArrayList<>();
        for (int num : nums) {
            if (Math.abs(num) < ans && list.size() > 0) {
                list.remove(list.size() - 1);
                list.add(num);
            } else if (Math.abs(num) == ans && list.size() > 0) list.add(num);
            else if (list.size() == 0) list.add(num);
            ans = Math.min(ans, Math.abs(num));

        }

        Collections.sort(list);
        return list.get(list.size() - 1);
    }

    // Author : Anand

    /*
    Input: total = 20, cost1 = 10, cost2 = 5
    Output: 9
    Explanation: The price of a pen is 10 and the price of a pencil is 5.
    - If you buy 0 pens, you can buy 0, 1, 2, 3, or 4 pencils.
    - If you buy 1 pen, you can buy 0, 1, or 2 pencils.
    - If you buy 2 pens, you cannot buy any pencils.
    The total number of ways to buy pens and pencils is 5 + 3 + 1 = 9.
     */
    // Author : Anand
    public long waysToBuyPensPencils(int total, int cost1, int cost2) {
        long ans = 0;
        if (total < cost1 && total < cost2) return 1;
        int larger = Math.max(cost1, cost2);
        int smaller = Math.min(cost1, cost2);
        int ind = 0;
        while (total - larger * ind >= 0) {
            int newtotal = total - larger * ind;
            ans += newtotal / smaller + 1;
            ind++;
        }

        return ans;
    }

    // Author: Anand
    public String digitSum(String s, int k) {
        while (s.length() > k) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.length(); i += k) {
                String newString = i + k < s.length() ? s.substring(i, i + k) : s.substring(i);
                int d = 0;
                for (int j = 0; j < newString.length(); j++)
                    d += Integer.parseInt(String.valueOf(newString.charAt(j)));
                sb.append(d);
            }
            s = sb.toString();
        }
        return s;
    }


    //Author: Anand
    public List<Integer> intersection(int[][] nums) {
        List<Integer> ans = new ArrayList<>();
        for (int[] num : nums) {
            List<Integer> list = Arrays.stream(num).boxed().collect(Collectors.toList());
            if (ans.isEmpty()) ans = list;
            else ans = list.stream()
                    .distinct()
                    .filter(ans::contains)
                    .collect(Collectors.toList());
        }
        Collections.sort(ans);
        return ans;
    }

    //Author: Anand
    public int countLatticePoints(int[][] circles) {
        Set<Point> ans = new HashSet<>();

        for (int[] c : circles) {
            int x = c[0];
            int y = c[1];
            int r = c[2];
            for (int i = x - r; i <= x + r; i++) {
                for (int j = y - r; j <= y + r; j++) {
                    // calculate distance and check if its within curcumference of circle
                    if ((x - i) * (x - i) + (y - j) * (y - j) <= r * r) ans.add(new Point(i, j));
                }
            }
        }
        return ans.size();
    }

    // Author : Anand
    // Easy, can skip
    public int countPrefixes(String[] words, String s) {
        int cnt = 0;
        for (String word : words) if (s.startsWith(word)) cnt++;
        return cnt;
    }

    // Author: Anand
    // Have a look at edge case, precisely
    public int minimumAverageDifference(int[] nums) {
        int n = nums.length;
        long[] prefSum = new long[n];
        for (int i = 0; i < nums.length; i++) prefSum[i] = (i > 0 ? prefSum[i - 1] : 0) + nums[i];
        long mini = Long.MAX_VALUE;
        int idx = -1;
        for (int i = 0; i < nums.length; i++) {
            long curr = Math.abs((long) prefSum[i] / (i + 1) - (long) ((n - 1 - i) > 0 ? (prefSum[n - 1] - prefSum[i]) / (n - 1 - i) : 0));
            if (curr < mini) {
                idx = i;
                mini = curr;
            }
        }

        return idx;
    }

    //Author: Anand
    public int maxConsecutive(int bottom, int top, int[] special) {

        int ans = 0;
        Arrays.sort(special);
        int prev = bottom;
        boolean first = false;
        for (int s : special) {
            if (!first) ans = Math.max(ans, s - prev);
            else {
                if (s - prev > 1) {
                    ans = Math.max(ans, s - prev - 1);
                }
            }
            prev = s;
            first = true;
        }

        ans = Math.max(ans, top - prev);
        return ans;
    }

    // Author: Anand
    public String largestGoodInteger(String num) {

        Set<String> goodIntegers = new HashSet<>();
        for (int i = 0; i < num.length() - 2; i++) {
            if (num.charAt(i) == num.charAt(i + 1) && num.charAt(i + 1) == num.charAt(i + 2))
                goodIntegers.add(num.substring(i, i + 3));
        }
        int max = Integer.MIN_VALUE;
        for (String ge : goodIntegers) {
            max = Math.max(max, Integer.parseInt(ge));
        }
        return max == Integer.MIN_VALUE ? "" : String.format("%03d", max);
    }

    // Author: Anand
    public int divisorSubstrings(int num, int k) {

        int ans = 0;
        String numstr = String.valueOf(num);
        for (int i = 0; i < numstr.length(); i++) {
            if (numstr.substring(i, Math.min(i + k, numstr.length())).length() == k) {
                long divisor = Long.parseLong(numstr.substring(i, Math.min(i + k, numstr.length())));
                if (num != 0 && divisor != 0 && (num % divisor == 0)) {
                    ans++;
                }
            }
        }
        return ans;
    }

    /*
    Input: nums = [10,4,-8,7]
    Output: 2
    Explanation:
    There are three ways of splitting nums into two non-empty parts:
    - Split nums at index 0. Then, the first part is [10], and its sum is 10. The second part is [4,-8,7], and its sum is 3. Since 10 >= 3, i = 0 is a valid split.
    - Split nums at index 1. Then, the first part is [10,4], and its sum is 14. The second part is [-8,7], and its sum is -1. Since 14 >= -1, i = 1 is a valid split.
    - Split nums at index 2. Then, the first part is [10,4,-8], and its sum is 6. The second part is [7], and its sum is 7. Since 6 < 7, i = 2 is not a valid split.
    Thus, the number of valid splits in nums is 2.

     */
    // Author: Anand
    public int waysToSplitArray(int[] nums) {

        long[] prefixSum = new long[nums.length];
        int idx = 0;
        for (int num : nums) {
            prefixSum[idx] = idx > 0 ? (prefixSum[idx - 1] + num) : num;
            idx++;
        }

        long tt = prefixSum[nums.length - 1];

        int ans = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            if (prefixSum[i] >= (tt - prefixSum[i])) ans++;
        }
        return ans;
    }

    /*
    [[83,35],[79,51],[61,48],[54,87],[44,93],[22,5],[87,28],[64,8],[89,78],[62,83],[58,72],[48,7],[97,16],[27,100],[65,48],[11,31],[29,76],[93,29],[72,59],[73,74],[9,90],[66,81],[12,8],[86,80],[84,43],[36,63],[80,45],[81,88],[95,5],[40,59]]
    Learning: Never use double while calculating slope
    Use:- y2-y1 * x1-x0 == y1-y0 * x2-x1 to avoid precision error
     */
    // Author : Anand
    public int minimumLines(int[][] stockPrices) {
        if (stockPrices.length == 1) return 0;
        int cnt = 1;
        Arrays.sort(stockPrices, Comparator.comparingLong(a -> a[0]));

        for (int i = 2; i < stockPrices.length; i++) {

            // Check if the slopes of three consecutive points are equal then continue
            // otherwise add another line to the count.
            // check (y2 - y1) / (x2 - x1) == (y1 - y0) / (x1 - x0) => (y2 - y1) * (x1 - x0) == (y1 - y0) * (x2 - x1)
            if ((stockPrices[i][1] - stockPrices[i - 1][1]) * (stockPrices[i - 1][0] - stockPrices[i - 2][0]) ==
                    (stockPrices[i - 1][1] - stockPrices[i - 2][1]) * (stockPrices[i][0] - stockPrices[i - 1][0]))
                continue;

            cnt += 1;
        }
        return cnt;
    }


    /*
    Input: num = "1210"
    Output: true
    Explanation:
    num[0] = '1'. The digit 0 occurs once in num.
    num[1] = '2'. The digit 1 occurs twice in num.
    num[2] = '1'. The digit 2 occurs once in num.
    num[3] = '0'. The digit 3 occurs zero times in num.
    The condition holds true for every index in "1210", so return true.

     */
    //Author: Anand
    public boolean digitCount(String num) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i = 0; i < num.length(); i++) {
            int key = Integer.parseInt(String.valueOf(num.charAt(i)));
            freq.put(key, freq.getOrDefault(key, 0) + 1);
        }

        for (int i = 0; i < num.length(); i++) {
            int count = Integer.parseInt(String.valueOf(num.charAt(i)));
            if (!freq.containsKey(i) && count == 0) continue;
            if (freq.containsKey(i) && freq.get(i) == count) continue;
            return false;
        }
        return true;
    }

    /*
    Input: messages = ["Hello userTwooo","Hi userThree","Wonderful day Alice","Nice day userThree"], senders = ["Alice","userTwo","userThree","Alice"]
    Output: "Alice"
    Explanation: Alice sends a total of 2 + 3 = 5 words.
    userTwo sends a total of 2 words.
    userThree sends a total of 3 words.
    Since Alice has the largest word count, we return "Alice".

     */
    //Author: Anand
    public String largestWordCount(String[] messages, String[] senders) {
        Map<String, Integer> freq = new HashMap<>();
        int idx = 0;
        for (String sender : senders)
            freq.put(sender, freq.getOrDefault(sender, 0) + messages[idx++].split(" ").length);

        freq = sortByValue(freq);
        int max = -1;
        List<String> ans = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : freq.entrySet()) {
            if (max == -1) {
                ans.add(entry.getKey());
                max = (int) entry.getValue();
            } else if (max == (int) entry.getValue()) {
                ans.add(entry.getKey());
            } else break;
        }

        String result = "";
        for (String a : ans) {
            if (Objects.equals(result, "")) result = a;
            else if (result.compareTo(a) < 0) result = a;
        }
        return result;
    }


    // function to sort hashmap by values
    public static HashMap<String, Integer> sortByValue(Map<String, Integer> hm) {
        HashMap<String, Integer> temp
                = hm.entrySet()
                .stream()
                .sorted((i1, i2) -> i2.getValue().compareTo(i1.getValue()))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1, LinkedHashMap::new));

        return temp;
    }

    //Author: Anand
    public String discountPrices(String sentence, int discount) {
        String[] words = sentence.split(" ");
        StringBuilder sb = new StringBuilder();

        for (String word : words) {
            String nw = "";
            if (word.startsWith("$") && !word.substring(1).isEmpty() && word.substring(1).chars().allMatch(Character::isDigit)) {
                long price = Long.parseLong(word.substring(1));
                double percent = ((100 - discount) * price * 1.00) / 100;
                String nv = String.format("%.2f", percent);
                nw = "$" + nv + " ";
            } else nw = word + " ";

            sb.append(nw);
        }
        return sb.toString().trim();
    }

    //Author: Anand
    public int minMaxGame(int[] nums) {
        while (nums.length > 1) {
            int[] nn = new int[nums.length / 2];
            for (int i = 0; i < nn.length; i++) {
                if (i % 2 == 0) {
                    nn[i] = Math.min(nums[2 * i], nums[2 * i + 1]);
                } else nn[i] = Math.max(nums[2 * i], nums[2 * i + 1]);
            }
            nums = nn;
        }
        return nums[0];
    }

    /*
    Input: nums = [3,6,1,2,5], k = 2
    Output: 2
    Explanation:
    We can partition nums into the two subsequences [3,1,2] and [6,5].
    The difference between the maximum and minimum value in the first subsequence is 3 - 1 = 2.
    The difference between the maximum and minimum value in the second subsequence is 6 - 5 = 1.
    Since two subsequences were created, we return 2. It can be shown that 2 is the minimum number of subsequences needed.=
     */
    //Author: Anand
    public int partitionArray(int[] nums, int k) {
        int ans = 1;
        int min = -1;
        Arrays.sort(nums);
        for (int num : nums) {
            if (min == -1) min = num;
            else if (Math.abs(num - min) > k) {
                min = num;
                ans++;
            }
        }

        return ans;
    }

    /*
    A password is said to be strong if it satisfies all the following criteria:

    It has at least 8 characters.
    It contains at least one lowercase letter.
    It contains at least one uppercase letter.
    It contains at least one digit.
    It contains at least one special character. The special characters are the characters in the following string: "!@#$%^&*()-+".
    It does not contain 2 of the same character in adjacent positions (i.e., "aab" violates this condition, but "aba" does not).

    Input: password = "IloveLe3tcode!"
    Output: true
    Explanation: The password meets all the requirements. Therefore, we return true.

     */
    public boolean strongPasswordCheckerII(String password) {
        if (password.length() < 8) return false;
        Set<Character> special = new HashSet<>();
        String s = "!@#$%^&*()-+";
        for (int i = 0; i < s.length(); i++) special.add(s.charAt(i));

        Map<Character, Boolean> map = new HashMap<>();
        map.put('u', false);
        map.put('s', false);
        map.put('l', false);
        map.put('d', false);

        char prev = 'a';
        boolean first = true;
        for (int i = 0; i < password.length(); i++) {
            char c = password.charAt(i);
            if (!first && prev == c) return false;
            else if (Character.isUpperCase(c)) map.put('u', true);
            else if (Character.isLowerCase(c)) map.put('l', true);
            else if (Character.isDigit(c)) map.put('d', true);
            else if (special.contains(c)) map.put('s', true);
            prev = c;
            first = false;
        }

        for (Map.Entry<Character, Boolean> entry : map.entrySet()) {
            if (!(boolean) entry.getValue()) return false;
        }
        return true;

    }


    /*
    Input: s = "fool3e7bar", sub = "leet", mappings = [["e","3"],["t","7"],["t","8"]]
    Output: true
    Explanation: Replace the first 'e' in sub with '3' and 't' in sub with '7'.
    TC = O(n2)
     "fool3e7bar"
    "leet"
    [["e","3"],["t","7"],["t","8"]]
     */
    public boolean matchReplacement(String s, String sub, char[][] mappings) {
        Map<Character, Set<Character>> mappingc = new HashMap<>();
        for (char[] map : mappings) {
            if (mappingc.containsKey(map[0])) {
                Set<Character> exist = mappingc.get(map[0]);
                exist.add(map[1]);
                mappingc.put(map[0], exist);
            } else {
                Set<Character> set = new HashSet<>();
                set.add(map[1]);
                mappingc.put(map[0], set);
            }
        }

        List<String> substrings = new ArrayList<>();

        int len = sub.length();
        for (int i = 0; i < s.length(); i++) {
            if (i + len <= s.length()) substrings.add(s.substring(i, i + len));
        }

        for (String gs : substrings) {
            boolean done = true;
            for (int i = 0; i < sub.length(); i++) {
                if (gs.charAt(i) != sub.charAt(i)) {
                    if (mappingc.containsKey(sub.charAt(i)) && mappingc.get(sub.charAt(i)).contains(gs.charAt(i)))
                        continue;
                    done = false;
                    break;
                }
            }

            if (done) return true;
        }
        return false;
    }

    /*
    Input: s = "arRAzFif"
    Output: "R"
    Explanation:
    The letter 'R' is the greatest letter to appear in both lower and upper case.
    Note that 'A' and 'F' also appear in both lower and upper case, but 'R' is greater than 'F' or 'A'.
     */
    //Author: Anand
    public String greatestLetter(String s) {
        String ans = "";
        Set<Character> lowercase = new HashSet<>();

        for (int i = 0; i < s.length(); i++) {
            char key = s.charAt(i);
            if (Character.isLowerCase(key)) lowercase.add(key);
        }

        for (int i = 0; i < s.length(); i++) {
            char key = s.charAt(i);
            if (lowercase.contains(Character.toLowerCase(key))
                    && Character.isUpperCase(key)
                    && (ans.isEmpty() || (int) key > (int) ans.charAt(0))) {
                ans = String.valueOf(key);
            }
        }
        return ans;
    }

    // Author: Anand
    // Similar to Div-2 codeforces
    /*
    Observe a pattern that you can always group all zeros to left
    and then shrink to single zero via 00 -> 10 operation
    Now generate the final result use fx,z
      */
    public String maximumBinaryString(String binary) {
        int fz = Integer.MAX_VALUE, z = 0;
        for (int i = 0; i < binary.length(); i++) {
            if (binary.charAt(i) == '0') {
                z++;
                fz = Math.min(fz, i);
            }
        }

        if (!(binary.length() < 2 || fz == Integer.MAX_VALUE)) {
            StringBuilder l = new StringBuilder(), r = new StringBuilder();
            int li = fz + z - 1;
            while (li-- > 0) l.append("1");

            int ri = binary.length() - z - fz;
            while (ri-- > 0) r.append("1");

            return l + "0" + r;
        }
        return binary;
    }


    //Author: Anand
    /*
    Input: s = "yo|uar|e**|b|e***au|tifu|l"
    Output: 5
    Explanation: The considered characters are underlined: "yo|uar|e**|b|e***au|tifu|l". There are 5 asterisks considered. Therefore, we return 5.
     [0,0,2,3,0,0]
     5
     */
    public int countAsterisks(String s) {
        int ans = 0;
        List<Long> cntStar = Arrays.stream(s.split("\\|")).map(e -> e.chars().filter(x -> x == '*').count()).collect(Collectors.toList());
        for (int i = 0; i < cntStar.size(); i += 2) ans += cntStar.get(i);
        return ans;
    }

    //Author: Anand
    public String decodeMessage(String key, String message) {
        if (message.trim().isEmpty()) return message;
        Map<Character, Character> map = new HashMap<>();

        int idx = 0;
        for (char c : key.toCharArray()) {
            if (Character.isWhitespace(c) || map.containsKey(c)) continue;
            if (map.size() < 26) {
                map.put(c, (char) (idx + 'a'));
                idx++;
            } else break;
        }

        StringBuilder sb = new StringBuilder();
        for (String word : message.split(" ")) {
            for (char c : word.toCharArray()) {
                if (map.containsKey(c)) {
                    sb.append(map.get(c));
                }
            }
            sb.append(" ");
        }

        sb = new StringBuilder(sb.substring(0, sb.length() - 1));

        for (int i = message.length() - 1; i >= 0; i--) {
            char c = message.charAt(i);
            if (Character.isWhitespace(c)) sb.append(" ");
            else break;
        }

        return sb.toString();
    }

    //Author: Anand
    /*
    Input: nums = [18,43,36,13,7]
    Output: 54
    Explanation: The pairs (i, j) that satisfy the conditions are:
    - (0, 2), both numbers have a sum of digits equal to 9, and their sum is 18 + 36 = 54.
    - (1, 4), both numbers have a sum of digits equal to 7, and their sum is 43 + 7 = 50.
    So the maximum sum that we can obtain is 54.
     */
    //Author: Anand
    //TC = O(nlogn)
    public int maximumSum(int[] nums) {
        Arrays.sort(nums);
        Map<Long, List<Integer>> map = new HashMap<>(); // store sum of digits, indexes
        int ind = 0;
        for (int num : nums) {
            long key = sod(num);
            if (map.containsKey(key)) {
                List<Integer> exist = map.get(key);
                exist.add(ind++);
                map.put(key, exist);
            } else map.put(key, new ArrayList<>(Arrays.asList(ind++)));
        }

        int max = Integer.MIN_VALUE;
        boolean flag = false;
        for (Map.Entry<Long, List<Integer>> entry : map.entrySet()) {
            List<Integer> indexes = entry.getValue();
            if (indexes.size() > 1) {
                flag = true;
                int value = (nums[indexes.get(indexes.size() - 1)] + nums[indexes.get(indexes.size() - 2)]);
                max = Math.max(max, value);
            }
        }

        return flag ? max : -1;
    }

    private long sod(int num) {
        long cnt = 0L;
        while (num > 0) {
            cnt += (num % 10);
            num /= 10;
        }
        return cnt;
    }

    //Author: Anand
    public int[] smallestTrimmedNumbers(String[] nums, int[][] queries) {
        int[] ans = new int[queries.length];
        int idx = 0;
        for (int[] query : queries) {
            int smallest = query[0];
            int ld = query[1];
            int ind = 0;

            Map<Integer, String> map = new HashMap<>(); // ind, string
            for (String num : nums) map.put(ind++, num.substring(num.length() - ld));

            Map<String, List<Integer>> tm = new TreeMap<>(); // String, List<ind>

            for (Map.Entry<Integer, String> entry : map.entrySet()) {
                String key = entry.getValue();
                if (!tm.containsKey(key)) tm.put(key, new ArrayList<>());
                tm.get(key).add(entry.getKey());
            }

            int sn = 0;
            for (Map.Entry<String, List<Integer>> entry : tm.entrySet()) {
                List<Integer> numbers = entry.getValue();
                if (smallest <= 0) break;
                int ptr = 0;
                while (ptr < numbers.size()) {
                    if (smallest-- == 0) break;
                    sn = numbers.get(ptr++);
                }
            }
            ans[idx++] = sn;
        }
        return ans;
    }

    /*
    Input: root = [2,1,3,null,null,0,1]
    Output: true
    Explanation: The above diagram illustrates the evaluation process.
    The AND node evaluates to False AND True = False.
    The OR node evaluates to True OR False = True.
    The root node evaluates to True, so we return true.
     */

    //Author: Anand
    public boolean evaluateTree(TreeNode root) {
        return helper(root);
    }

    // L-> R -> root
    private boolean helper(TreeNode root) {
        // base case '
        if (root == null) return true;

        // if leaf node
        if (root.left == null || root.right == null) {
            if (root.val == 0) return false;
            return true;
        }

        boolean left = helper(root.left);
        boolean right = helper(root.right);

        if (root.val == 2) return left || right;
        if (root.val == 3) return left && right;
        return true;
    }

    /*
      Input: buses = [10,20], passengers = [2,17,18,19], capacity = 2
      Output: 16
      Explanation:
      The 1st bus departs with the 1st passenger.
      The 2nd bus departs with you and the 2nd passenger.
      Note that you must not arrive at the same time as the passengers, which is why you must arrive before the 2nd passenger to catch the bus.


       buses = [10,20], pass = [2,17,18,19]
      //Author: Anand
     */
    // TC = O(nlogn)
    public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
        Arrays.sort(buses);
        Arrays.sort(passengers);

        int prev = -1;
        int pcb = 0;
        for (int bus : buses) {
            int ind = bs(passengers, bus);
            if (prev == -1) {
                prev = Math.min(ind, capacity);
                pcb = prev;
            } else {
                pcb = (ind > (prev + capacity)) ? capacity : ind;
                prev += pcb;
            }
        }

        int value = 0;
        if (pcb >= capacity) {
            prev--;
            value = passengers[prev];
            for (int i = prev - 1; i >= 0; i--) {
                if ((value - passengers[i]) != 1) return value - 1;
                value = passengers[i];
            }
        } else {
            value = buses[buses.length - 1];
            for (int i = passengers.length - 1; i >= 0; i--) {
                if (value > passengers[i]) return value;
                value = passengers[i];
            }
        }

        return value - 1;
    }

    private int bs(int[] passengers, int bus) {
        int l = 0, h = passengers.length - 1;
        while (l < h) {
            int mid = l + (h - l) / 2;
            if (passengers[mid] <= bus) {
                l = mid + 1;
            } else h = mid;
        }
        return l;
    }


    /*
    Input: nums = [2,3,2,4,3], numsDivide = [9,6,9,3,15]
    Output: 2
    Explanation:
    The smallest element in [2,3,2,4,3] is 2, which does not divide all the elements of numsDivide.
    We use 2 deletions to delete the elements in nums that are equal to 2 which makes nums = [3,4,3].
    The smallest element in [3,4,3] is 3, which divides all the elements of numsDivide.
    It can be shown that 2 is the minimum number of deletions needed.
     */
    //Author: Anand
    //TC = O(nlogn)
    public int minOperations(int[] nums, int[] numsDivide) {
        long gcd = numsDivide[0];
        for (int i = 1; i < numsDivide.length; i++) gcd = _gcd(gcd, numsDivide[i]);
        Arrays.sort(nums);
        int cnt = 0;
        for (int num : nums) {
            if (gcd % num == 0) return cnt;
            cnt++;
        }

        return -1;
    }

    //long version for gcd
    public static long _gcd(long a, long b) {
        if (b == 0)
            return a;

        return _gcd(b, a % b);
    }

        /*
    Input: n = 6, delay = 2, forget = 4
    Output: 5
    Explanation:
    Day 1: Suppose the first person is named A. (1 person)
    Day 2: A is the only person who knows the secret. (1 person)
    Day 3: A shares the secret with a new person, B. (2 people)
    Day 4: A shares the secret with a new person, C. (3 people)
    Day 5: A forgets the secret, and B shares the secret with a new person, D. (3 people)
    Day 6: B shares the secret with E, and C shares the secret with F. (5 people)
     */

    //Author: Anand
    public int peopleAwareOfSecret(int n, int delay, int forget) {
        final int mod = 1_000_000_007;

        int i = 1;
        int discovery = 1;
        Map<Integer, String> store = new HashMap<>();

        String key = discovery + "-" + (discovery + delay) + "-" + (discovery + forget);
        store.put(i, key);

        int day = 1;
        while (day <= n) {
            List<Pair<Integer, String>> list = new ArrayList<>();
            for (Map.Entry<Integer, String> entry : store.entrySet()) {
                list.add(new Pair<>(entry.getKey(), entry.getValue()));
            }

            for (Pair<Integer, String> pair : list) {
                int person = pair.getKey();

                int[] arr = Arrays.stream(store.get(person).split("-"))
                        .mapToInt(Integer::parseInt).toArray();
                String nk = day + "-" + (day + delay) + "-" + (day + forget);

                if (day >= arr[1] && day < arr[2]) {
                    store.put(((i + 1) % mod), nk);
                }
                // if  days passed then this person will never able to generate new people
                if (day >= arr[2]) store.remove(person);
            }
            day++;
        }

        int max = -1;
        for (Map.Entry<Integer, String> entry : store.entrySet()) {
            max = Math.max(max, (int) entry.getKey());
            System.out.println(entry.getValue());
        }
        return max;
    }

//23/07/2022    -----------------------------------------------------------------------------------------


    /*
    "Flush": Five cards of the same suit.
    "Three of a Kind": Three cards of the same rank.
    "Pair": Two cards of the same rank.
    "High Card": Any single card.

    Input: ranks = [4,4,2,4,4], suits = ["d","a","a","b","c"]
    Output: "Three of a Kind"
    Explanation: The hand with the first, second, and fourth card consists of 3 cards with the same rank, so we have a "Three of a Kind".
    Note that we could also make a "Pair" hand but "Three of a Kind" is a better hand.
    Also note that other cards could be used to make the "Three of a Kind" hand.

     */
    //Author: Anand
    public String bestHand(int[] ranks, char[] suits) {
        char c = '#';
        boolean sameSuite = true;
        for (char s : suits) {
            if (c == '#') c = s;
            else if (c != s) {
                sameSuite = false;
                break;
            }
        }

        if (sameSuite) return "Flush";

        Map<Integer, Integer> freq = new HashMap<>();// Rank,Count

        for (int rank : ranks) {
            freq.put(rank, freq.getOrDefault(rank, 0) + 1);
            if (freq.get(rank) == 3) return "Three of a Kind";
        }

        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            if (entry.getValue() == 2) return "Pair";
        }

        return "High Card";
    }

    /*
    Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
    Output: 1
    Explanation: There is 1 equal row and column pair:
    - (Row 2, Column 1): [2,7,7]
     */
    //Author: Anand
    public int equalPairs(int[][] grid) {
        int ans = 0;

        List<String> allColumns = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < grid[0].length; j++) {
                sb.append(grid[j][i]).append(",");
            }
            allColumns.add(sb.toString());
        }


        for (int[] r : grid) {
            StringBuilder row = new StringBuilder();
            for (int c : r) {
                row.append(c).append(",");
            }

            ans += allColumns.stream().filter(x -> x.equals(row.toString())).count();
        }

        return ans;
    }

    /*
    Input: grades = [10,6,12,7,3,5]
    Output: 3
    Explanation: The following is a possible way to form 3 groups of students:
    - 1st group has the students with grades = [12]. Sum of grades: 12. Student count: 1
    - 2nd group has the students with grades = [6,7]. Sum of grades: 6 + 7 = 13. Student count: 2
    - 3rd group has the students with grades = [10,3,5]. Sum of grades: 10 + 3 + 5 = 18. Student count: 3
    It can be shown that it is not possible to form more than 3 groups.
     */
    // Author: Anand
    public int maximumGroups(int[] grades) {
        Arrays.sort(grades);
        int ind = 0, cnt = 0;
        while (ind < grades.length) {
            if (ind + cnt < grades.length) cnt++;
            ind += cnt;
        }
        return cnt;
    }


    /*
    Input: items1 = [[1,1],[3,2],[2,3]], items2 = [[2,1],[3,2],[1,3]]
    Output: [[1,4],[2,4],[3,4]]
    Explanation: 
    The item with value = 1 occurs in items1 with weight = 1 and in items2 with weight = 3, total weight = 1 + 3 = 4.
    The item with value = 2 occurs in items1 with weight = 3 and in items2 with weight = 1, total weight = 3 + 1 = 4.
    The item with value = 3 occurs in items1 with weight = 2 and in items2 with weight = 2, total weight = 2 + 2 = 4.
    Therefore, we return [[1,4],[2,4],[3,4]].

     //Author: Anand
     */
    public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
        Map<Integer, Integer> tm = new TreeMap<>();// v->tw

        for (int[] item : items1) tm.put(item[0], tm.getOrDefault(item[0], 0) + item[1]);
        for (int[] item : items2) tm.put(item[0], tm.getOrDefault(item[0], 0) + item[1]);
        List<List<Integer>> ans = new ArrayList<>();

        for (Map.Entry<Integer, Integer> entry : tm.entrySet())
            ans.add(new ArrayList<>(Arrays.asList(entry.getKey(), entry.getValue())));

        return ans;
    }

    /*
    Input: nums = [4,1,3,3]
    Output: 5
    Explanation: The pair (0, 1) is a bad pair since 1 - 0 != 1 - 4.
    The pair (0, 2) is a bad pair since 2 - 0 != 3 - 4, 2 != -1.
    The pair (0, 3) is a bad pair since 3 - 0 != 3 - 4, 3 != -1.
    The pair (1, 2) is a bad pair since 2 - 1 != 3 - 1, 1 != 2.
    The pair (2, 3) is a bad pair since 3 - 2 != 3 - 3, 1 != 0.
    There are a total of 5 bad pairs, so we return 5.
     */

    public long countBadPairs(int[] nums) {
        long cnt = 0L;
        TreeMap<Integer, Integer> diffMap = new TreeMap<>();
        for (int i = 0; i < nums.length; i++) diffMap.put(i, nums[i] - i);

        Map<Integer, Integer> freq = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : diffMap.entrySet())
            freq.put(entry.getValue(), freq.getOrDefault(entry.getValue(), 0) + 1);


        int idx = 0;
        for (Map.Entry<Integer, Integer> entry : diffMap.entrySet()) {
            freq.put(entry.getValue(), freq.getOrDefault(entry.getValue(), 0) - 1);
            if (freq.containsKey(entry.getValue()) && freq.get(entry.getValue()) < 0) freq.remove(entry.getValue());

            long ans = nums.length - 1 - idx++ - (freq.getOrDefault(entry.getValue(), 0));
            cnt += ans;
        }

        return cnt;
    }

    /*
    Input: tasks = [1,2,1,2,3,1], space = 3
    Output: 9
    Explanation:
    One way to complete all tasks in 9 days is as follows:
    Day 1: Complete the 0th task.
    Day 2: Complete the 1st task.
    Day 3: Take a break.
    Day 4: Take a break.
    Day 5: Complete the 2nd task.
    Day 6: Complete the 3rd task.
    Day 7: Take a break.
    Day 8: Complete the 4th task.
    Day 9: Complete the 5th task.
    It can be shown that the tasks cannot be completed in less than 9 days.
     */
    public long taskSchedulerII(int[] tasks, int space) {
        long days = 1;
        Map<Integer, Long> map = new HashMap<>();// task -> last index
        int ind = 0;
        while (ind < tasks.length) {
            if (map.containsKey(tasks[ind])) {
                if (days - map.get(tasks[ind]) > space) {
                    map.put(tasks[ind], days);
                } else days = map.get(tasks[ind]) + space;
            } else {
                map.put(tasks[ind], days);
            }
            days++;
        }

        return days;
    }


    /*
    Input: nums = [0,1,4,6,7,10], diff = 3
    Output: 2
    Explanation:
    (1, 2, 4) is an arithmetic triplet because both 7 - 4 == 3 and 4 - 1 == 3.
    (2, 4, 5) is an arithmetic triplet because both 10 - 7 == 3 and 7 - 4 == 3. 
     */
    //Author: Anand
    public int arithmeticTriplets(int[] nums, int diff) {

        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                for (int k = j + 1; k < nums.length; k++)
                    if (nums[j] - nums[i] == diff && nums[k] - nums[j] == diff) cnt++;
            }
        }
        return cnt;

    }

    /*
    Input: n = 7, edges = [[0,1],[1,2],[3,1],[4,0],[0,5],[5,6]], restricted = [4,5]
    Output: 4
    Explanation: The diagram above shows the tree.
    We have that [0,1,2,3] are the only nodes that can be reached from node 0 without visiting a restricted node.
     */
    //Author: Anand
    public int reachableNodes(int n, int[][] edges, int[] restricted) {

        int cnt = 0;
        Map<Integer, List<Integer>> graph = new HashMap<>();

        for (int[] edge : edges) {
            if (!graph.containsKey(edge[0])) graph.put(edge[0], new ArrayList<>());
            graph.get(edge[0]).add(edge[1]);

            if (!graph.containsKey(edge[1])) graph.put(edge[1], new ArrayList<>());
            graph.get(edge[1]).add(edge[0]);

        }

        Set<Integer> restricteds = Arrays.stream(restricted).boxed().collect(Collectors.toSet());

        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[n];
        queue.add(0);
        visited[0] = true;
        cnt++;
        while (!queue.isEmpty()) {
            int elem = queue.poll();
            for (int e : graph.get(elem)) {

                if (visited[e]) continue;

                if (!restricteds.contains(e)) {
                    visited[e] = true;
                    queue.offer(e);
                    cnt++;
                }
            }
        }
        return cnt;
    }


    public int numUniqueEmails(String[] emails) {
        Set<String> uniqueEmails = new HashSet<>();
        for (String email : emails) {
            String[] parts = email.split("[@]");
            uniqueEmails.add(parts[0].replaceAll("[.]", "").split("[/+/]")[0] + "@" + parts[1]);
        }
        return uniqueEmails.size();
    }

    public String licenseKeyFormatting(String s, int k) {
        int n = s.length();
        StringBuilder ans = new StringBuilder();
        int ind = n - 1;
        while (ind >= 0) {
            char ch = s.charAt(ind);
            if (ch == '-') {
                ind--;
                continue;
            }

            int cnt = k;
            StringBuilder sb = new StringBuilder();
            sb.append(Character.toUpperCase(ch));
            ind--;
            cnt--;

            while (cnt > 0 && ind >= 0) {
                if (s.charAt(ind) == '-') {
                    ind--;
                    continue;
                }

                sb.insert(0, Character.toUpperCase(s.charAt(ind--)));
                cnt--;
            }

            if (ind >= 0 && sb.length() > 0) ans.insert(0, "-" + sb);

            else if (ind < 0 && sb.length() > 0) {
                ans.insert(0, sb);
                break;
            } else if (ind < 0 && sb.length() == 0) {
                ans.deleteCharAt(0);
                break;
            }
        }

        if (ans.length() > 0 && ans.charAt(0) == '-') ans.deleteCharAt(0);
        return ans.toString();
    }


    //Author: Anand
    public int totalFruit(int[] fruits) {
        Map<Integer, Integer> map = new HashMap<>();
        int maxi = Integer.MIN_VALUE;
        int cnt = 0;

        int continous = -1;
        int lastKey = -1;
        int firstKey = -1;
        for (int fruit : fruits) {

            if (lastKey == -1) lastKey = fruit;
            if (firstKey == -1) firstKey = fruit;
            if (continous == -1) continous = 1;

            if (!map.containsKey(fruit)) {
                if (map.size() >= 2) {
                    maxi = Math.max(maxi, cnt);
                    cnt = continous;
                    map.remove(firstKey);
                    map.put(lastKey, continous);
                }
                continous = 1;
            } else if (fruit == lastKey) continous++;
            else continous = 1;

            if (lastKey != fruit) {
                firstKey = lastKey;
                lastKey = fruit;
            }
            map.put(fruit, map.getOrDefault(fruit, 0) + 1);
            cnt++;
        }


        cnt = 0;
        for (int key : map.keySet()) cnt += map.get(key);
        maxi = Math.max(maxi, cnt);
        return maxi;
    }
}

    /*
    // TODO: maxRunTime Binary search solution
    public:
    bool fun(vector<int>& a, long long x, long long k){
        long long val = x*k;
        for(int i=0; i<a.size(); i++){
            val -= min((long long)a[i],k);
        }
        return val <= 0;
    }
    long long maxRunTime(int n, vector<int>& a) {
        long long sum = 0;
        for(auto i : a){
            sum += i;
        }
        long long ans = 0;
        long long l = 0, r = sum;
        while(l <= r){
            long long mid = l + (r-l)/2;
            if(fun(a,n,mid)){
                ans = mid;
                l = mid + 1;
            }
            else{
                r = mid - 1;
            }
        }
        return ans;
    }
     */

/*
    private static final int[][] DIRS = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    public int minPushBox(char[][] grid) {
        int R = grid.length, C = grid[0].length;
        int[] box = new int[2], player = new int[2];
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (grid[i][j] == 'B') {
                    box[0] = i;
                    box[1] = j;
                } else if (grid[i][j] == 'S') {
                    player[0] = i;
                    player[1] = j;
                }
            }
        }
        Queue<Pair<int[], int[]>> queue = new LinkedList();
        queue.add(new Pair(box, player));
        boolean[][] visited = new boolean[R * C][R * C];
        // in some cases, player needs to push the box further in order to change its direction; hence, tracking the box itself isn't enough,

        // we need to track both box and player locations. for example,
        // . # T # .
        // . . . B S
        // . . . # .
        // `B` needs to land on location(1,2) twice
        visited[box[0] * C + box[1]][player[0] * C + player[1]] = true;
        int step = 0;
        while (!queue.isEmpty()) {
            step++;
            for (int i = queue.size() - 1; i >= 0; i--) {
                Pair<int[], int[]> state = queue.poll();
                int[] b = state.getKey(), p = state.getValue();
                for (int j = 0; j < DIRS.length; j++) {
                    int[] nb = new int[]{b[0] + DIRS[j][0], b[1] + DIRS[j][1]};
                    if (nb[0] >= 0 && nb[0] < R && nb[1] >= 0 && nb[1] < C && grid[nb[0]][nb[1]] != '#') {
                        // check where was it pushed from. basically, the opposite direction where the box moves to.
                        int[] np = new int[]{b[0] - DIRS[j][0], b[1] - DIRS[j][1]};
                        if (np[0] >= 0 && np[0] < R && np[1] >= 0 && np[1] < C
                                && grid[np[0]][np[1]] != '#'
                                && !visited[nb[0] * C + nb[1]][np[0] * C + np[1]]) {
                            // can the player reach to the box-pushing location
                            if (isReachable(grid, R, C, b, p, np)) {
                                if (grid[nb[0]][nb[1]] == 'T') {
                                    return step;
                                }
                                visited[nb[0] * C + nb[1]][np[0] * C + np[1]] = true;
                                queue.add(new Pair(nb, b));
                            }
                        }
                    }
                }
            }
        }
        return -1;
    }

    private boolean isReachable(char[][] grid, int R, int C, int[] box, int[] from, int[] to) {
        Queue<int[]> queue = new LinkedList();
        queue.add(from);
        boolean[][] visited = new boolean[R][C];
        visited[from[0]][from[1]] = true;
        while (!queue.isEmpty()) {
            int[] loc = queue.poll();
            if (loc[0] == to[0] && loc[1] == to[1]) {
                return true;
            }
            for (int j = 0; j < DIRS.length; j++) {
                int nr = loc[0] + DIRS[j][0], nc = loc[1] + DIRS[j][1];
                if (nr >= 0 && nr < R && nc >= 0 && nc < C && !visited[nr][nc]
                        && grid[nr][nc] != '#'
                        && (nr != box[0] || nc != box[1])) {
                    visited[nr][nc] = true;
                    queue.add(new int[]{nr, nc});
                }
            }
        }
        return false;
    }
 */
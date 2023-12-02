package com.company;

import javafx.util.Pair;

import java.awt.*;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.List;
import java.util.Queue;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class EmployeeSolution {
    String name;
    Integer salary;
    Set<String> result = new HashSet<>();
    // TC = O(V+E), SC = O(V+E)
    long maxScore;

    //
//         [5:20 pm] Keshav Bansal
//
//    { 2, -1, -3, 6, 8, -4, 5, -8, -5, 9, 3, -3, 4 }
//
//   dp[1] = 2
//   dp[2] = 2
//   dp[3] = 6
//   dp[4] = 8
    int count;
    // TC = O(n^E) --> Exponential
    // We have done dfs to get the max path
    // As each node can be visisted any number of times and hence  do dfs along with updating  node  vis status backtracking
    int maxVal;
    //Author: Anand
    int max = Integer.MIN_VALUE;

    //    Input: num = 526
//    Output: true
//    Explanation: Reverse num to get 625, then reverse 625 to get 526, which equals num.
//    Input: num = 1800
//    Output: false
//    Explanation: Reverse num to get 81, then reverse 81 to get 18, which does not equal num.
    List<List<Integer>> onePos = new ArrayList<>();
    Set<String> colP = new HashSet<>();
    int MOD = (int) 1e9 + 7;
    List<Integer> primeNumbers = new ArrayList<>();
    int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    // R -> D -> L -> U
    int n;


    // Greedy approach
    // The approach to traverse through the array and check if we get a n 'X'  character then move 3 steps ahead
    //  else move only 1 step (normal pace)
    TreeMap<Integer, Integer> affectedPowers = new TreeMap<>();

    // O(n+m), O(1)
	/*
     This problem is called Merge Sorted Array
     Example:-
     Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

    Algo :- We start filling the array from right end till all elements of nums1 is consumed
        After that remaining element of nums2 is utitlised

        */
    List<List<Integer>> subArrays = new ArrayList<>();

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
    int N = 8;

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

        System.out.println(alternatingSubarray(new int[]{2, 3, 4, 3, 4}));
        int[] nums = new int[]{1, 6, 7, 8};
        int[] moveFrom = new int[]{1, 7, 2};
        int[] moveTo = new int[]{2, 9, 5};

        System.out.println(relocateMarbles(nums, moveFrom, moveTo));
    }

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

    // function to sort hashmap by values
    public static HashMap<String, Integer> sortByValue(Map<String, Integer> hm) {
        HashMap<String, Integer> temp = hm.entrySet().stream().sorted((i1, i2) -> i2.getValue().compareTo(i1.getValue())).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        return temp;
    }

    //long version for gcd
    public static long _gcd(long a, long b) {
        if (b == 0) return a;

        return _gcd(b, a % b);
    }

    public static int alternatingSubarray(int[] nums) {

        int max = -1;
        for (int i = 0; i < nums.length; i++) {
            int len = 1;
            int flag = 1;
            for (int j = i + 1; j < nums.length; j++) {
                if ((nums[j] - nums[j - 1] == flag)) {
                    len++;
                    flag = flag > 0 ? Math.negateExact(flag) : Math.abs(flag);
                } else {
                    max = Math.max(max, len);
                    break;
                }
            }
            max = Math.max(max, len);
        }

        return max == 1 ? Math.negateExact(max) : max;
    }

    public static List<Integer> relocateMarbles(int[] nums, int[] moveFrom, int[] moveTo) {
        TreeMap<Integer, Integer> tm = new TreeMap();
        for (int num : nums) tm.put(num, tm.getOrDefault(num, 0) + 1);
        for (int i = 0; i < moveFrom.length; i++) {
            int count = tm.get(moveFrom[i]);
            tm.remove(moveFrom[i]);
            tm.put(moveTo[i], count);
        }
        return new ArrayList<>(tm.keySet());
    }

    public List<String> addOperators(String num, int target) {
        calculate(num, 0, target, "");
        return new ArrayList<>(result);
    }

    // Author: Anand
    // If a number has traling is zero and its not zero then it must return false
    public boolean isSameAfterReversals(int num) {
        return num == 0 || num % 10 != 0;
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

     /*
         arr = [3,4,3,3]
         k = 2

        pq = {4, 3, 3, 3}
        map = { (3, (0, 2,3), (4,1))}
        ans = {1,0,2,3}
        result = [4, 3]
     */

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
            while (exp[0].length() > i && isNumber(exp[0].charAt(i))) {
                i++;
            }
            res = Double.parseDouble(exp[0].substring(0, i));
            exp[0] = exp[0].substring(i);
        }
        return res;
    }

    private boolean isNumber(int c) {
        int zero = '0';
        int nine = '9';
        return (c >= zero && c <= nine) || c == '.';
    }

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

    public int countHighestScoreNodes(int[] parents) {
        int n = parents.length;
        // create an adjacancy list of edges of each vertex
        List<Integer>[] list = new ArrayList[n];

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
    private long dfs(int u, List<Integer>[] list, int n) {

        int total = 0;
        long prod = 1L, rem, val;

        for (Integer v : list[u]) {
            val = dfs(v, list, n);
            total += val;
            prod *= val;
        }

        // if nodes  remaning beyond subtree with root u the  it will be taken into consideration
        rem = n - total - 1;
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

    public long countVowels(String word) {
        long n = word.length();
        long ans = 0;
        for (long i = 0; i < n; i++) {
            if (isVowel(word.charAt((int) i))) ans += (n - i) * (i + 1);
        }
        return ans;
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
            if (!map.containsKey(nums[i])) map.put(nums[i], new ArrayList<>());
            map.get(nums[i]).add(i);
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
                cnt += Math.abs((h - l + 1) * (h - l) / 2);
                isnincreasing = false;
                l = h;
                h++;
            }
            idx++;
        }

        if (h == idx) {
            cnt += Math.abs((h - l + 1) * (h - l) / 2);
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
            if (prices[e - 1] - prices[e] == 1) cnt += (e - s + 1);
            else {
                s = e;
                cnt++;
            }
        }
        return cnt;
    }

    // TC = O(MLogM)
    // Author : Anand

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
    //--------------------------------------------------------

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

    public List<String> findAllRecipes(String[] recipes, List<List<String>> ingredients, String[] supplies) {
        if (recipes == null || recipes.length == 0 || ingredients == null || ingredients.size() == 0 || supplies == null || supplies.length == 0 || recipes.length != ingredients.size())
            return new ArrayList<>();


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
            if (!(word.charAt(0) == word.charAt(word.length() - 1)) && (map.containsKey(word) && map.get(word) > 0 && map.containsKey(reverse) && map.get(reverse) > 0)) {
                int count = Math.min(map.get(word), map.get(reverse));
                lp += count * 4;
                while (count-- > 0) {
                    map.put(word, map.getOrDefault(word, 0) - 1);
                    if (map.get(word) <= 0) map.remove(word);
                    map.put(reverse, map.getOrDefault(reverse, 0) - 1);
                    if (map.get(reverse) <= 0) map.remove(reverse);
                }
            } else if (word.charAt(0) == word.charAt(word.length() - 1) && (map.containsKey(word) && map.get(word) > 0)) {
                int count = map.get(word);
                int evenCnt = count / 2;
                lp += evenCnt * 4;
                map.put(word, map.getOrDefault(word, 0) - (evenCnt * 2));
                if (map.get(word) <= 0) map.remove(word);
            }
        }

        //  We can also use exactly one in the middle to form an even longer palindrome.
        for (String word : map.keySet()) {
            if (word.charAt(0) == word.charAt(word.length() - 1) && map.get(word) == 1) {
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
            } else if (sum == max_sum) res.add((Integer) entry.getKey());
        }

        return res;
    }

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
            map.put(i, new ArrayList<>(Collections.singletonList(max_left)));
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
                if (fl.get(ind).getValue() > sl.get(ind).getValue()) {
                    c1 = fl.get(ind).getKey();
                } else {
                    c2 = sl.get(ind).getKey();
                }
            } else {
                if (fl.size() > ind) {
                    c1 = fl.get(ind).getKey();
                } else if (sl.size() > ind) {
                    c2 = sl.get(ind).getKey();
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
                            ans.append(perm);
                            map.remove(ch);
                        } else {
                            int rl = repeatLimit;
                            if (map.get(ch) <= 0) map.remove(ch);
                            StringBuilder perm = new StringBuilder();
                            while (rl-- > 0) {
                                perm.append(ch);
                            }
                            ans.append(perm);
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
                if (gcd * num % k == 0) {
                    result += gcdMap.get(num);
                }
            }
            gcdMap.put(gcd, gcdMap.getOrDefault(gcd, 0L) + 1);
        }
        return result;
    }

    //long version for gcd
    public long __gcd(long a, long b) {
        if (b == 0) return a;

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
                System.out.println(res);
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
            long gcd = __gcd(nums[i], nums[i + 1]);

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

        while (l < n && s.charAt(l) == 'L') l++;

        while (r >= 0 && s.charAt(r) == 'R') r--;

        while (l <= r) {
            if (s.charAt(l) != 'S') ans++;

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
            else ans = list.stream().distinct().filter(ans::contains).collect(Collectors.toList());
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
            long curr = Math.abs(prefSum[i] / (i + 1) - ((n - 1 - i) > 0 ? (prefSum[n - 1] - prefSum[i]) / (n - 1 - i) : 0));
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
            if ((stockPrices[i][1] - stockPrices[i - 1][1]) * (stockPrices[i - 1][0] - stockPrices[i - 2][0]) == (stockPrices[i - 1][1] - stockPrices[i - 2][1]) * (stockPrices[i][0] - stockPrices[i - 1][0]))
                continue;

            cnt += 1;
        }
        return cnt;
    }

    /*
    Input: root = [2,1,3,null,null,0,1]
    Output: true
    Explanation: The above diagram illustrates the evaluation process.
    The AND node evaluates to False AND True = False.
    The OR node evaluates to True OR False = True.
    The root node evaluates to True, so we return true.
     */

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
                max = entry.getValue();
            } else if (max == entry.getValue()) {
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

//23/07/2022    -----------------------------------------------------------------------------------------

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
            if (lowercase.contains(Character.toLowerCase(key)) && Character.isUpperCase(key) && (ans.isEmpty() || (int) key > (int) ans.charAt(0))) {
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
            } else map.put(key, new ArrayList<>(Collections.singletonList(ind++)));
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
            return root.val != 0;
        }

        boolean left = helper(root.left);
        boolean right = helper(root.right);

        if (root.val == 2) return left || right;
        if (root.val == 3) return left && right;
        return true;
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

                int[] arr = Arrays.stream(store.get(person).split("-")).mapToInt(Integer::parseInt).toArray();
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
            max = Math.max(max, entry.getKey());
            System.out.println(entry.getValue());
        }
        return max;
    }

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
    Input: blocks = "WBBWWBBWBW", k = 7
    Output: 3
    Explanation:
    One way to achieve 7 consecutive black blocks is to recolor the 0th, 3rd, and 4th blocks
    so that blocks = "BBBBBBBWBW".
    It can be shown that there is no way to achieve 7 consecutive black blocks in less than 3 operations.
    Therefore, we return 3.
     */

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

    //Author: Anand
    public int edgeScore(int[] edges) {
        Map<Integer, List<Integer>> graph = new TreeMap<>();

        for (int i = 0; i < edges.length; i++) {
            if (!graph.containsKey(edges[i])) graph.put(edges[i], new ArrayList<>());
            graph.get(edges[i]).add(i);
        }

        int maxNode = -1;
        long sum = Long.MIN_VALUE;

        for (Map.Entry<Integer, List<Integer>> entry : graph.entrySet()) {
            long cs = entry.getValue().stream().mapToLong(x -> x).sum();
            if (cs > sum) {
                sum = cs;
                maxNode = entry.getKey();
            }
        }
        return maxNode;
    }

    public boolean canChange(String start, String target) {
        // order of insertion is maintained
        Map<Character, List<Integer>> maps = new LinkedHashMap<>(), mapt = new LinkedHashMap<>();
        List<Character> lists = new ArrayList<>(), listt = new ArrayList<>();


        for (int i = 0; i < start.length(); i++) {
            char ch = start.charAt(i);
            if (maps.containsKey(ch)) maps.get(ch).add(i);
            else maps.put(ch, new ArrayList<>(Collections.singletonList(i)));
            if (ch != '_') lists.add(ch);
        }

        for (int i = 0; i < target.length(); i++) {
            char ch = target.charAt(i);
            if (mapt.containsKey(ch)) mapt.get(ch).add(i);
            else mapt.put(ch, new ArrayList<>(Collections.singletonList(i)));
            if (ch != '_') listt.add(ch);
        }

        // To check order of insertion for L && R
        if (!lists.equals(listt)) return false;

        for (Map.Entry<Character, List<Integer>> entry : maps.entrySet()) {
            if (entry.getValue().size() != mapt.getOrDefault(entry.getKey(), new ArrayList<>()).size()) return false;

            int idx = 0;
            List<Integer> values = entry.getValue();
            for (int e : values)
                if ((entry.getKey() == 'L' && e < mapt.get(entry.getKey()).get(idx++)) || (entry.getKey() == 'R' && e > mapt.get(entry.getKey()).get(idx++)))
                    return false;
        }
        return true;
    }

    public String complexNumberMultiply(String num1, String num2) {

        String[] n1 = num1.split("\\+");
        String[] n2 = num2.split("\\+");

        int part = 0;
        StringBuilder imag = new StringBuilder();
        for (String s1 : n1) {
            for (String s2 : n2) {
                if (s2.contains("i") && s1.contains("i")) {
                    part -= Integer.parseInt(s1.replace("i", "")) * Integer.parseInt(s2.replace("i", ""));
                } else if (s1.contains("i")) {
                    int np = Integer.parseInt(s2) * Integer.parseInt(s1.replace("i", ""));
                    if (imag.length() != 0) {
                        int first = Integer.parseInt(imag.toString().replace("i", ""));
                        imag.delete(0, imag.length());
                        imag.append(first + np);
                    } else imag.append(np);

                    imag.append("i");
                } else if (s2.contains("i")) {
                    int np = Integer.parseInt(s1) * Integer.parseInt(s2.replace("i", ""));
                    if (imag.length() != 0) {
                        int first = Integer.parseInt(imag.toString().replace("i", ""));
                        imag.delete(0, imag.length());
                        imag.append(first + np);
                    } else imag.append(np);

                    imag.append("i");
                } else part += Integer.parseInt(s1) * Integer.parseInt(s2);
            }
        }
        return imag.insert(0, part + "+").toString();
    }

    public String decodeCiphertext(String encodedText, int rows) {

        int col = encodedText.length() / rows;
        char[][] mat = new char[rows][col];
        for (char[] c : mat) Arrays.fill(c, ' ');

        int r = 0, c = 0;
        for (int i = 0; i < encodedText.length(); i++) {
            if (c >= col) {
                c = 0;
                ++r;
            }

            mat[r][c++] = encodedText.charAt(i);
        }


        StringBuilder sb = new StringBuilder();
        int p, q;
        int ind = 0;
        while (ind < col) {
            p = 0;
            q = ind;

            // keep mv diagonally down
            while (p < rows && q < col) sb.append(mat[p++][q++]);
            ind++;
        }

        return sb.toString().replaceFirst("\\s++$", "");

    }

    //Author: Anand
    public int minimumRecolors(String blocks, int k) {

        if (k > blocks.length()) return 0;

        int i = 0, j = 0;
        int ans = Integer.MAX_VALUE;
        int cnt = 0;
        while (i < blocks.length() && j < blocks.length()) {

            if (j >= k) {
                if (blocks.charAt(i) == 'W') cnt--;
                if (blocks.charAt(j) == 'W') cnt++;
                j++;
                i++;
                ans = Math.min(cnt, ans);
                continue;
            }
            while (j < k) {
                if (blocks.charAt(j++) == 'W') cnt++;
            }
            ans = Math.min(cnt, ans);
        }

        return ans;
    }

    /*
    Input: s = "0110101"
    Output: 4
    Explanation:
    After one second, s becomes "1011010".
    After another second, s becomes "1101100".
    After the third second, s becomes "1110100".
    After the fourth second, s becomes "1111000".
    No occurrence of "01" exists any longer, and the process needed 4 seconds to complete,
    so we return 4.
     */
    //Author: Anand
    public int secondsToRemoveOccurrences(String s) {
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == '1') idx.add(i);

        System.out.println(idx);
        int op = 0;
        while (true) {
            List<Integer> nl = new ArrayList<>();
            nl.add(Math.max(idx.get(0) - 1, 0));
            op++;

            boolean changed = false;
            for (int i = 1; i < idx.size(); i++) {
                if (idx.get(i) != i) {
                    if (idx.get(i - 1) + 1 != idx.get(i)) {
                        changed = true;
                        nl.add(idx.get(i) - 1);
                    } else nl.add(idx.get(i));
                } else nl.add(idx.get(i));
            }

            if (!changed) return op;
            idx.clear();
            idx.addAll(nl);
        }
    }

    /*
    Input: n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
    Output: 0
    Explanation:
    - At time 0, both rooms are not being used. The first meeting starts in room 0.
    - At time 1, only room 1 is not being used. The second meeting starts in room 1.
    - At time 2, both rooms are being used. The third meeting is delayed.
    - At time 3, both rooms are being used. The fourth meeting is delayed.
    - At time 5, the meeting in room 1 finishes. The third meeting starts in room 1 for the time period [5,10).
    - At time 10, the meetings in both rooms finish. The fourth meeting starts in room 0 for the time period [10,11).
    Both rooms 0 and 1 held 2 meetings, so we return 0.
     */

    //Author: Anand
    public String shiftingLetters(String s, int[][] shifts) {
        Arrays.sort(shifts, Comparator.comparingInt(x -> x[0]));
        Map<List<Integer>, Integer> map = new LinkedHashMap<>();

        for (int i = 0; i < shifts.length; i++) {
            // overlap
            if (shifts[i][1] < shifts[i + 1][1]) {
                List<Integer> nk = new ArrayList<>(Arrays.asList(shifts[i][1], shifts[i + 1][0], shifts[i][2]));
                map.put(nk, map.getOrDefault(nk, 0) + 1);

                // same priority
                if (shifts[i][1] == shifts[i + 1][1]) {
                    List<Integer> key = new ArrayList<>(Arrays.asList(shifts[i][0], shifts[i][1], shifts[i][2]));
                    map.put(key, map.getOrDefault(key, 0) + 1);
                }
            }
        }

        return null;
    }

    /*
    Input: initialEnergy = 5, initialExperience = 3, energy = [1,4,3,2], experience = [2,6,3,1]
    Output: 8
    Explanation: You can increase your energy to 11 after 6 hours of training, and your experience to 5 after 2 hours of training.
    You face the opponents in the following order:
    - You have more energy and experience than the 0th opponent so you win.
      Your energy becomes 11 - 1 = 10, and your experience becomes 5 + 2 = 7.
    - You have more energy and experience than the 1st opponent so you win.
      Your energy becomes 10 - 4 = 6, and your experience becomes 7 + 6 = 13.
    - You have more energy and experience than the 2nd opponent so you win.
      Your energy becomes 6 - 3 = 3, and your experience becomes 13 + 3 = 16.
    - You have more energy and experience than the 3rd opponent so you win.
      Your energy becomes 3 - 2 = 1, and your experience becomes 16 + 1 = 17.
    You did a total of 6 + 2 = 8 hours of training before the competition, so we return 8.
    It can be proven that no smaller answer exists.

     */
    //Author: Anand
    public int minNumberOfHours(int initialEnergy, int initialExperience, int[] energy, int[] experience) {
        long es = Arrays.stream(energy).sum();
        long total = Math.max(es + 1 - initialEnergy, 0);
        long expn;
        for (int e : experience) {
            expn = Math.max(e - initialExperience + 1, 0);
            total += expn;
            initialExperience += e + expn;
        }
        return (int) total;
    }

    /*
    Input: num = "444947137"
    Output: "7449447"
    Explanation:
    Use the digits "4449477" from "444947137" to form the palindromic integer "7449447".
    It can be shown that "7449447" is the largest palindromic integer that can be formed.

     */
    //Author: Anand
    public String largestPalindromic(String num) {

        Map<Integer, Integer> freq = new HashMap<>();

        for (int i = 0; i < num.length(); i++) {
            int key = num.charAt(i) - '0';
            freq.put(key, freq.getOrDefault(key, 0) + 1);
        }

        TreeMap<Integer, Integer> taken = new TreeMap<>(Collections.reverseOrder());
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            if (entry.getValue() >= 2) {
                if (entry.getValue() % 2 == 0) taken.put(entry.getKey(), entry.getValue());
                else taken.put(entry.getKey(), entry.getValue() - 1);
            }
        }

        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, Integer> entry : taken.entrySet()) {
            int times = entry.getValue() / 2;
            while (times-- > 0) {
                if (sb.length() == 0 && entry.getKey() == 0) break;
                int offset = Math.max(sb.length() / 2, 0);
                sb.insert(offset, entry.getKey());
                sb.insert(offset + 1, entry.getKey());
            }
        }

        int[] arr = num.chars().boxed().mapToInt(x -> x - '0').toArray();
        Arrays.sort(arr);
        int largest = -1;

        for (int i = arr.length - 1; i >= 0; i--) {
            if (freq.get(arr[i]) % 2 != 0) {
                largest = arr[i];
                break;
            }
        }


        if (largest != -1) sb.insert(sb.length() / 2, largest);
        if (sb.length() == 0) return String.valueOf(num.charAt(0));
        return sb.toString();
    }

    //Author: Anand
    public int garbageCollection(String[] garbage, int[] travel) {
        Map<Character, Integer> map = new HashMap<>();
        int ans = 0;
        for (int i = 0; i < garbage.length; i++)
            for (char c : garbage[i].toCharArray()) map.put(c, i);
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            char t = entry.getKey();
            for (int i = 0; i < garbage.length; i++) {
                if (entry.getValue() < i) break;
                ans += (i > 0 ? travel[i - 1] : 0) + garbage[i].length() - garbage[i].replace(String.valueOf(t), "").length();
            }
        }
        return ans;
    }

    //Author: Anand
    public boolean findSubarrays(int[] nums) {
        Set<Integer> ts = new HashSet<>();
        for (int i = 0; i < nums.length - 1; i++) {
            int sum = nums[i] + nums[i + 1];
            if (ts.contains(sum)) return true;
            ts.add(sum);
        }
        return false;
    }

    public int maximumRows(int[][] mat, int cols) {

        int m = mat.length;
        int n = mat[0].length;

        for (int i = 0; i < m; i++) {
            onePos.add(new ArrayList<>());
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 1) onePos.get(i).add(j);
            }
        }

        chooseCol(m, n, cols, mat, new HashSet<>(), 0);
        return max;
    }

    private void chooseCol(int m, int n, int cols, int[][] mat, HashSet<Integer> colTaken, int x) {

        if (cols == 0) {
            if (!colP.contains(colTaken.toString())) {
                max = Math.max(max, selectRows(colTaken, mat));
                colP.add(colTaken.toString());
            }
            return;
        }


        for (int i = x; i < n; i++) {
            if (colTaken.contains(i)) continue;
            // take
            colTaken.add(i);
            chooseCol(m, n, cols - 1, mat, colTaken, x + 1);
            // not take
            colTaken.remove(i);
        }
    }

    private int selectRows(HashSet<Integer> colTaken, int[][] mat) {

        int cnt = 0;
        for (List<Integer> rows : onePos) {

            boolean present = true;
            for (int pos : rows) {
                if (!colTaken.contains(pos)) {
                    present = false;
                    break;
                }
            }

            if (present) cnt++;
        }

        return cnt;
    }

    public boolean checkDistances(String s, int[] distance) {

        Map<Character, Integer> map = new HashMap();
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                map.put(s.charAt(i), i - map.get(s.charAt(i)) - 1);
            } else map.put(s.charAt(i), i);
        }

        for (int i = 0; i < distance.length; i++) {
            char c = (char) ('a' + i);
            if (map.containsKey(c)) {
                if (map.get(c) != distance[i]) return false;
            }
        }

        return true;
    }

    //Author: Anand
    public int mostBooked(int n, int[][] meetings) {
        long t = Long.MIN_VALUE;

        TreeMap<Integer, Integer> fm = new TreeMap<>();
        for (int i = 0; i < n; i++) fm.put(i, Integer.MAX_VALUE);

        TreeMap<Long, PriorityQueue<Integer>> um = new TreeMap<>();
        Arrays.sort(meetings, Comparator.comparingInt(m -> m[0]));

        Map<Integer, Integer> freq = new HashMap<>();

        for (int[] meeting : meetings) {
            int start = meeting[0];
            int end = meeting[1];
            int room;
            long net;

            if (t == Long.MIN_VALUE) t = start;

            t = Math.max(start, t);

            int lowestRoom = Integer.MAX_VALUE;
            long key = Long.MIN_VALUE;

            if (fm.size() > 0) {
                for (Map.Entry<Long, PriorityQueue<Integer>> en : um.entrySet()) {
                    if (en.getKey() <= t) {
                        if (en.getValue().size() > 0 && en.getValue().peek() < lowestRoom) {
                            key = en.getKey();
                            lowestRoom = en.getValue().peek();
                        }
                    } else break;
                }


                if (lowestRoom < fm.firstEntry().getKey()) {
                    room = lowestRoom;
                    um.get(key).poll();
                    if (um.get(key).size() <= 0) um.remove(key);

                } else room = fm.firstEntry().getKey();

                fm.remove(room);
                net = end - start + t;
            } else {

                Map.Entry<Long, PriorityQueue<Integer>> entry = um.firstEntry();

                if (t > entry.getKey()) {
                    for (Map.Entry<Long, PriorityQueue<Integer>> en : um.entrySet()) {
                        if (en.getKey() <= t) {
                            if (en.getValue().size() > 0 && en.getValue().peek() < lowestRoom) {
                                key = en.getKey();
                                lowestRoom = en.getValue().peek();
                            }
                        } else break;
                    }
                } else lowestRoom = entry.getValue().size() > 0 ? entry.getValue().poll() : lowestRoom;


                if (key != Long.MIN_VALUE) um.get(key).poll();
                t = Math.max(t, entry.getKey());
                room = lowestRoom;
                net = end - start + t;
                if (entry.getValue().size() <= 0) um.remove(entry.getKey());
            }

            t++;
            if (!um.containsKey(net)) um.put(net, new PriorityQueue<>());
            um.get(net).add(room);


            if (room != Integer.MIN_VALUE) freq.put(room, freq.getOrDefault(room, 0) + 1);
        }

        int max = Integer.MIN_VALUE;
        int roomNo = Integer.MIN_VALUE;
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {

            if (entry.getValue() > max) {
                roomNo = entry.getKey();
                max = entry.getValue();
            }
        }

        return roomNo;
    }

    //Author: Anand
    public int mostFrequentEven(int[] nums) {

        Map<Integer, Integer> tm = new TreeMap<>();

        for (int num : nums) if (num % 2 == 0) tm.put(num, tm.getOrDefault(num, 0) + 1);

        int elem = -1;
        int count = Integer.MIN_VALUE;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {

            if (entry.getValue() > count) {
                count = entry.getValue();
                elem = entry.getKey();
            }
        }
        return elem;
    }

    //Author: Anand
    public int partitionString(String s) {
        int cnt = 0;
        Set<Character> set = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {

            if (set.contains(s.charAt(i))) {
                cnt++;
                set.clear();
            }
            set.add(s.charAt(i));
        }


        if (set.size() > 0) cnt++;
        return cnt;
    }

    public int findComplement(int num) {
        String nums = Integer.toBinaryString(num);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nums.length(); i++) sb.append(nums.charAt(i) == '1' ? '0' : '1');
        return Integer.parseInt(sb.toString(), 2);
    }

    public int magicalString(int n) {
        int cnt = 0;
        StringBuilder sb = new StringBuilder();
        sb.append('1');

        int li = 0;

        while (sb.length() < n) {
            if (sb.charAt(li) == '1' && sb.length() == 1) {
                sb.append("22");
                li += 2;
            } else if (sb.charAt(li) == '2') {
                if (sb.charAt(sb.length() - 1) == '1') sb.append("22");
                else sb.append("11");
                li++;
            } else if (sb.charAt(li) == '1') {
                if (sb.charAt(sb.length() - 1) == '2') sb.append("1");
                else sb.append("2");
                li++;
            }
        }


        int ind = 0;
        while (ind < n) if (sb.charAt(ind++) == '1') cnt++;

        return cnt;
    }

    //Author: Anand
    public int[][] reconstructQueue(int[][] people) {
        int[][] ans = new int[people.length][2];
        Arrays.sort(people, Comparator.comparingInt(a -> a[0]));
        Map<Integer, Integer> lhm = new LinkedHashMap<>();
        for (int i = people.length - 1; i >= 0; i--) lhm.put(i, -1);

        int prev = -1;
        for (int[] p : people) {
            int htG = p[1];
            int idx = lhm.keySet().size() > htG ? new ArrayList<>(lhm.keySet()).get(htG) : new ArrayList<>(lhm.keySet()).get(0);
            if (p[0] == prev && htG > 1) idx++;
            ans[idx] = p;
            lhm.remove(idx);
            prev = p[0];
        }

        int idx = 0;
        int[][] res = new int[people.length][2];
        for (int[] a : ans) res[people.length - 1 - idx++] = a;

        return res;
    }

    public String licenseKeyFormatting(String s, int k) {
        StringBuilder original = new StringBuilder();
        Arrays.stream(s.split("-")).forEach(original::append);
        int ind = 0;
        StringBuilder sb = new StringBuilder();
        int first = original.length() % k;
        if (first != 0) {
            sb.append(original.substring(0, first).toUpperCase());
            ind += first;
            if (ind < original.length()) sb.append("-");
        }

        while (ind < original.length()) {
            if (ind + k < original.length()) sb.append(original.substring(ind, ind + k).toUpperCase());
            else sb.append(original.substring(ind).toUpperCase());
            ind += k;
            if (ind < original.length()) sb.append("-");
        }

        return sb.toString();
    }

    /*
    Input: n = 10, logs = [[0,3],[2,5],[0,9],[1,15]]
    Output: 1
    Explanation:
    Task 0 started at 0 and ended at 3 with 3 units of times.
    Task 1 started at 3 and ended at 5 with 2 units of times.
    Task 2 started at 5 and ended at 9 with 4 units of times.
    Task 3 started at 9 and ended at 15 with 6 units of times.
    The task with the longest time is task 3 and the employee with id 1 is the one that worked on it, so we return 1.


     */
    public int hardestWorker(int n, int[][] logs) {
        int id = -1;
        int prev = -1;
        int time = Integer.MIN_VALUE;
        for (int[] log : logs) {
            int ei = log[0];
            int end = log[1];
            if (prev == -1 && id == -1) {
                time = Math.max(time, end);
                id = ei;
            } else if ((end - prev) > time) {
                time = end - prev;
                id = ei;
            } else if ((end - prev) == time) {
                id = Math.min(ei, id);
            }

            prev = end;
        }

        return id;
    }

    public int[] findArray(int[] pref) {
        int[] ans = new int[pref.length];
        for (int i = 0; i < pref.length; i++) ans[i] = i == 0 ? pref[i] : pref[i - 1] ^ pref[i];
        return ans;
    }

    public int countTime(String time) {
        String[] array = time.split(":");
        int cnt = 1;
        int ind = 0;
        for (String s : array) {
            if (s.charAt(0) == '?' && s.charAt(1) == '?') {
                if (ind == 0) cnt *= 24;
                else cnt *= 60;
            } else {
                if (ind == 0) {
                    if (s.charAt(0) == '?') {
                        if (Integer.parseInt(String.valueOf(s.charAt(1))) < 4) cnt *= 3;
                        else cnt *= 2;
                    } else if (s.charAt(1) == '?') {
                        if (Integer.parseInt(String.valueOf(s.charAt(0))) < 2) cnt *= 10;
                        else cnt *= 4;
                    }
                } else {
                    if (s.charAt(0) == '?') {
                        cnt *= 6;
                    } else if (s.charAt(1) == '?') {
                        cnt *= 10;
                    }
                }
            }
            ind++;
        }

        return cnt;
    }

    public int[] productQueries(int n, int[][] queries) {

        List<Integer> list = new ArrayList<>();

        String bs = Integer.toBinaryString(n);
        for (int i = bs.length() - 1; i >= 0; i--) {
            if (bs.charAt(i) == '1') list.add((int) Math.pow(2, bs.length() - 1 - i));
        }

        BigInteger[] prefix = new BigInteger[list.size()];
        int idx = 0;
        for (int e : list) {
            prefix[idx] = idx == 0 ? new BigInteger(String.valueOf(e)) : new BigInteger(String.valueOf((e))).multiply(prefix[idx - 1]);
            idx++;
        }

        List<BigInteger> bigIntegers = new ArrayList<>();
        for (int[] query : queries) {
            if (query[0] == query[1]) bigIntegers.add(new BigInteger(String.valueOf(list.get(query[0]))));
            else {
                BigInteger b1 = new BigInteger(String.valueOf(prefix[query[1]]));
                BigInteger b2 = new BigInteger(String.valueOf(query[0] == 0 ? 1 : prefix[Math.max(query[0] - 1, 0)]));

                BigInteger res = b1.divide(b2);
                bigIntegers.add(res);
            }
        }

        List<Integer> ans = new ArrayList<>();
        for (BigInteger a : bigIntegers) {
            boolean flag = false;
            int er = Integer.MAX_VALUE;
            while (!flag) {
                try {
                    er = Integer.parseInt(String.valueOf(a.mod(new BigInteger(String.valueOf(MOD)))));
                    flag = true;
                } catch (Exception ex) {
                    a = new BigInteger(String.valueOf(a.mod(new BigInteger(String.valueOf(MOD)))));
                }
            }

            ans.add(er);
        }

        return ans.stream().mapToInt(x -> x).toArray();
    }

    //Author: Anand
    public int findMaxK(int[] nums) {
        TreeMap<Integer, Integer> tm = new TreeMap<>(Collections.reverseOrder()); // number-> count

        for (int num : nums) tm.put(num, tm.getOrDefault(num, 0) + 1);

        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            if (entry.getKey() > 0 && tm.containsKey(-entry.getKey())) return entry.getKey();
        }

        return -1;
    }

    //Author: Anand
    public int countDistinctIntegers(int[] nums) {

        Set<Integer> set = new HashSet<>();

        for (int num : nums) set.add(num);

        for (int num : nums) {
            StringBuilder sb = new StringBuilder();
            int nn = Integer.parseInt(sb.append(num).reverse().toString());
            set.add(nn);
        }

        return set.size();
    }

    //Author: Anand
    public boolean sumOfNumberAndReverse(int num) {

        for (int i = 1; i <= num; i++) {
            StringBuilder sb = new StringBuilder();
            int nn = Integer.parseInt(sb.append(i).reverse().toString());
            if (i + nn == num) return true;
        }

        return false;
    }

    public long countSubarrays(int[] nums, int minK, int maxK) {
        long cnt1 = subArrays(nums, minK, maxK);
        long cnt2 = subArrays(nums, minK + 1, maxK);
        long cnt3 = subArrays(nums, minK, maxK - 1);
        long cnt4 = subArrays(nums, minK + 1, maxK - 1);

        return cnt1 - cnt2 - cnt3 + cnt4;
    }

    private long subArrays(int[] arr, int l, int u) {
        int i = 0, n = arr.length;
        long ans = 0L;
        while (i < n) {
            if (arr[i] > u || arr[i] < l) {
                i++;
                continue;
            }

            long count = 0;
            while (i < n && arr[i] <= u && arr[i] >= l) {
                count++;
                i++;
            }

            ans += (count * (count + 1)) / 2;
        }

        return ans;
    }

    public int minimizeArrayValue(int[] nums) {
        long sum = 0L;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            int avg = (int) Math.ceil((sum + i) / (i + 1));
            max = Math.max(max, avg);
        }

        return max;
    }

    public boolean checkPerfectNumber(int num) {
        if (num == 1) return false;
        int sum = 0;
        for (int i = 1; i <= Math.sqrt(num); i++) {
            if (num % i == 0) {
                // If divisors are equal, print only one
                if ((num / i) != i && i != 1) sum += (num / i);
                sum += i;
            }
        }

        return sum == num;
    }

    public boolean judgeCircle(String moves) {
        int x = 0, y = 0;
        Map<Character, List<Integer>> dirs = new HashMap<>();
        // l, r, u, d
        int[][] move = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        char[] d = new char[]{'L', 'R', 'U', 'D'};
        int idx = 0;
        for (int[] m : move) {
            dirs.put(d[idx++], new ArrayList<>(Arrays.asList(m[0], m[1])));
        }


        for (char c : moves.toCharArray()) {
            List<Integer> cord = dirs.get(c);
            x += cord.get(0);
            y += cord.get(1);
        }

        return x == 0 && y == 0;
    }

    public List<Integer> findClosestElements(int[] arr, int k, int x) {

        // [dist, element]

        PriorityQueue<int[]> pq = new PriorityQueue<>((t1, t2) -> {
            if (t1[0] < t2[0]) return -1;
            if (t1[0] > t2[0]) return 1;
            return Integer.compare(t1[1], t2[1]);
        });


        for (int a : arr) pq.add(new int[]{Math.abs(a - x), a});

        List<Integer> ans = new ArrayList<>();
        while (!pq.isEmpty()) {
            ans.add(pq.poll()[1]);
            if (--k <= 0) break;
        }

        Collections.sort(ans);
        return ans;
    }

    public boolean haveConflict(String[] event1, String[] event2) {

        List<Integer> el1 = new ArrayList<>();
        List<Integer> el2 = new ArrayList<>();

        for (String e1 : event1) {
            int time = Integer.parseInt(e1.split(":")[0]) * 60 + Integer.parseInt(e1.split(":")[1]);
            el1.add(time);
        }

        for (String e2 : event2) {
            int time = Integer.parseInt(e2.split(":")[0]) * 60 + Integer.parseInt(e2.split(":")[1]);
            el2.add(time);
        }

        if (el2.get(0) <= el1.get(1) && el2.get(0) >= el1.get(0)) return true;
        return el1.get(0) <= el2.get(1) && el1.get(0) >= el2.get(0);
    }

    public int subarrayGCD(int[] nums, int k) {
        int cnt = 0;
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            list.add(nums[i]);
            if (list.get(list.size() - 1) == k) cnt++;
            for (int j = i + 1; j < nums.length; j++) {
                int _gcd = j == i + 1 ? gcd(nums[i], nums[j]) : gcd(list.get(list.size() - 1), nums[j]);
                list.add(_gcd);
                if (list.get(list.size() - 1) == k) cnt++;
            }
        }
        return cnt;
    }

    /*
        Time Complexity: O(N * log(N))
        Space Complexity: O(1)

        Where N is the number of elements in array
    */
    public long minCost(int[] arr, int[] cost) {
        int n = arr.length;
        // Variable to contain minimum value of array
        int lowerLimit = Integer.MAX_VALUE;

        long ans = Long.MAX_VALUE;
        // Variable to contain maximum value )of array
        int upperLimit = Integer.MIN_VALUE;

        for (int a : arr) {
            lowerLimit = Math.min(lowerLimit, a);
            upperLimit = Math.max(upperLimit, a);
        }

        int diff = upperLimit - lowerLimit;
        if (diff <= 2) {
            while (upperLimit >= lowerLimit) {
                long cc = findCostWithTargetValue(arr, cost, n, upperLimit);
                ans = Math.min(ans, cc);
                upperLimit--;
            }
            return ans;
        }

        int maxCost = Integer.MIN_VALUE;
        int elem = Integer.MAX_VALUE;
        for (int i = 0; i < cost.length; i++) {
            if (cost[i] > maxCost) {
                maxCost = cost[i];
                elem = arr[i];
            }
        }

        ans = Math.min(ans, findCostWithTargetValue(arr, cost, n, elem));

        while (upperLimit - lowerLimit > 2) {
            int mid1 = lowerLimit + (upperLimit - lowerLimit) / 3;
            int mid2 = upperLimit - (upperLimit - lowerLimit) / 3;

            // Variable which contains cost with mid1 as target value
            long cost1 = findCostWithTargetValue(arr, cost, n, mid1);

            // Variable which contains cost with mid2 as target value
            long cost2 = findCostWithTargetValue(arr, cost, n, mid2);

            if (cost1 < cost2) {
                upperLimit = mid2;
            } else {
                lowerLimit = mid1;
            }

            ans = Math.min(ans, Math.min(cost1, cost2));
        }

        // Returning cost with average of lowerLimit and upperLimit as target value
        return Math.min(ans, findCostWithTargetValue(arr, cost, n, (lowerLimit + upperLimit) / 2));
    }

    private long findCostWithTargetValue(int[] arr, int[] cost, int n, int target) {
        long tc = 0L;
        // Loop to calculate cost with given target value
        for (int index = 0; index < n; index++) {
            tc += (long) Math.abs(arr[index] - target) * cost[index];
        }
        // Return cost
        return tc;
    }

    public int newInteger(int n) {
        int ans = 0;
        int base = 1;

        while (n > 0) {
            ans += n % 9 * base;
            n /= 9;
            base *= 10;
        }
        return ans;
    }

    public boolean isPossible(int[] nums) {
        int n = nums.length;

        TreeMap<Integer, List<Integer>> tm = new TreeMap<>(); // {(lastElement, List with the sizes in sorted order)}

        for (int num : nums) {
            int key = num - 1;
            if (tm.containsKey(key) && tm.get(key).size() > 0) {
                List<Integer> sizes = tm.get(key);
                Collections.sort(sizes);
                int cnt = sizes.get(0);
                sizes.remove(0);
                tm.put(key, sizes);

                if (!tm.containsKey(num)) tm.put(num, new ArrayList<>());
                tm.get(num).add(cnt + 1);
            } else {
                if (!tm.containsKey(num)) tm.put(num, new ArrayList<>());
                tm.get(num).add(1);
            }
        }
        for (int v : tm.values().stream().flatMap(Collection::stream).collect(Collectors.toList())) {
            if (v < 3) return false;
        }
        return true;
    }

    // More optimised solution exist in DP.java file
    public int maxA(int n) {
        StringBuilder sb = new StringBuilder();
        sb.append('A');

        Map<String, Integer> dp = new HashMap<>();
        return helper(sb, "", 1, n, dp);
    }

    private int helper(StringBuilder sb, String copied, int op, int n, Map<String, Integer> dp) {
        // base case
        if (op == n) return sb.length();

        String key = sb.toString() + "-" + copied + "-" + op;
        if (dp.containsKey(key)) return dp.get(key);
        //last operation was not feasible
        int len = Integer.MIN_VALUE;
        // try all possible ways

        // paste operation is possible
        if (!copied.isEmpty()) {

            // System.out.println("copied=" + copied);
            sb.append(copied);
            len = Math.max(len, helper(sb, copied, op + 1, n, dp));
            //backtrack
            sb.delete(Math.max(sb.length() - copied.length(), 0), sb.length());
            // System.out.println("bactrack=" + sb);
        }

        // ctrl+A -> ctrl+C -> ctrl+V
        if (op + 3 <= n && sb.length() >= 3) {
            String prev = sb.toString();
            sb.append(prev);

            // System.out.println("copied=" + prev);
            int cnt = helper(sb, prev, op + 3, n, dp);
            len = Math.max(len, cnt);

            //backtrack
            sb.delete(Math.max(sb.length() - prev.length(), 0), sb.length());
            // System.out.println("bactrack=" + sb);
        }

        if (op + 1 <= n) {
            sb.append("A");
            // System.out.println("copied=" + copied);
            int cnt = helper(sb, copied, op + 1, n, dp);
            len = Math.max(len, cnt);
            sb.delete(Math.max(sb.length() - 1, 0), sb.length());
            // System.out.println("backtrack=" + sb.substring(sb.length() - 1, sb.length()));
        }

        dp.put(key, len);
        return len;
    }

    public String oddString(String[] words) {

        Map<List<Integer>, List<String>> diffM = new HashMap<>();
        for (String word : words) {
            List<Integer> sb = new ArrayList<>();
            char prev = '#';
            for (char c : word.toCharArray()) {
                if (prev != '#') sb.add((int) c - prev);
                prev = c;
            }

            if (!diffM.containsKey(sb)) diffM.put(sb, new ArrayList<>());
            diffM.get(sb).add(word);
        }

        for (Map.Entry<List<Integer>, List<String>> entry : diffM.entrySet()) {
            if (entry.getValue().size() == 1) return entry.getValue().get(0);
        }
        return "";
    }

    public List<String> twoEditWords(String[] queries, String[] dictionary) {
        List<String> ans = new ArrayList<>();
        for (String query : queries) {

            for (String word : dictionary) {
                int cnt = 0;
                for (int i = 0; i < word.length(); i++) {
                    if (word.charAt(i) != query.charAt(i)) cnt++;
                    if (cnt > 2) break;
                }
                if (cnt <= 2) {
                    ans.add(query);
                    break;
                }
            }
        }

        return ans;
    }

    public int averageValue(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int num : nums) if (num % 2 == 0 && num % 3 == 0) list.add(num);
        return list.size() > 0 ? Math.abs(list.stream().mapToInt(x -> x).sum() / list.size()) : 0;
    }

    public List<List<String>> mostPopularCreator(String[] creators, String[] ids, int[] views) {
        List<List<String>> ans = new ArrayList<>();

        Map<String, List<Pair<String, Integer>>> map = new HashMap<>();

        for (int i = 0; i < creators.length; i++) {
            if (!map.containsKey(creators[i])) map.put(creators[i], new ArrayList<>());
            map.get(creators[i]).add(new Pair<>(ids[i], views[i]));
        }


        PriorityQueue<Pair<String, Long>> pq = new PriorityQueue<>(Collections.reverseOrder(Comparator.comparingInt(a -> Math.toIntExact(a.getValue()))));

        for (Map.Entry<String, List<Pair<String, Integer>>> entry : map.entrySet()) {
            long v = entry.getValue().stream().mapToInt(Pair::getValue).sum();
            pq.add(new Pair<>(entry.getKey(), v));
        }
        Set<String> pc = new HashSet<>();
        long hv = Long.MIN_VALUE;

        while (!pq.isEmpty()) {
            Pair<String, Long> entry = pq.poll();
            if (entry.getValue() >= hv) {
                pc.add(entry.getKey());
                hv = entry.getValue();
            } else break;
        }

        for (String creator : pc) {

            List<Pair<String, Integer>> value = map.get(creator);
            Collections.sort(value, (o1, o2) -> {
                if (o1.getValue() < o2.getValue()) return 1;
                if (o2.getValue() < o1.getValue()) return -1;
                return o1.getKey().compareTo(o2.getKey());
            });

            ans.add(new ArrayList<>(Arrays.asList(creator, value.get(0).getKey())));

        }

        return ans;
    }

    public long makeIntegerBeautiful(long n, int target) {

        StringBuilder ans = new StringBuilder();
        int ds = sod(n);
        String ns = String.valueOf(n);
        int carry = 0;
        if (ds > target) {
            for (int i = ns.length() - 1; i >= 0; i--) {
                int d = Integer.parseInt(String.valueOf(ns.charAt(i)));
                int complement = 10 - d - carry;
                // skip that digit in this case
                if (complement == 10) {
                    ans.insert(0, 0);
                    continue;
                }
                ans.insert(0, complement);
                carry = 1;
                ds = ds - d;
                // take carry as well for sum of digits carried from last
                if (ds + carry <= target) break;
            }
            return Long.parseLong(ans.toString());
        }
        return 0L;
    }

    private int sod(long n) {
        int s = 0;
        while (n > 0) {
            s += n % 10;
            n /= 10;
        }

        return s;
    }

    public int[] applyOperations(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
        }

        List<Integer> m = new ArrayList<>();

        m.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        move_zeros_to_right(m);

        return m.stream().mapToInt(x -> x).toArray();
    }

    // function to shift zeros
    private void move_zeros_to_right(List<Integer> m) {
        int count = 0;
        for (int i = 0; i < m.size(); i++) {
            if (m.get(i) == 0) {
                count++;
                // deleting the element from vector
                m.remove(i);
                i--;
            }
        }

        for (int i = 0; i < count; i++) {
            // inserting the zero into arraylist
            m.add(0);
        }
    }

    public int distinctAverages(int[] nums) {

        PriorityQueue<Integer> minPq = new PriorityQueue<>();
        PriorityQueue<Integer> maxPq = new PriorityQueue<>(Collections.reverseOrder());

        for (int num : nums) {
            minPq.offer(num);
            maxPq.offer(num);
        }

        int sz = 0;

        Set<Double> ans = new HashSet<>();
        while (true) {
            if (sz == nums.length) return ans.size();
            int min = minPq.poll();
            int max = maxPq.poll();
            double avg = (float) (min + max) / 2;
            ans.add(avg);
            sz += 2;
        }
    }

    public double[] convertTemperature(double celsius) {
        double[] ans = new double[2];
        ans[0] = celsius + 273.15000;
        ans[1] = celsius * 1.80000 + 32.00000;
        return ans;
    }

    public int minimumOperations(TreeNode root) {
        if (root == null) return 0;

        int cnt = 0;
        List<Integer> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<TreeNode> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();
            List<Integer> level = new ArrayList<>();
            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                TreeNode p = q.peek();
                q.remove();
                level.add(p.val);
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
                n--;
            }
            cnt += minSwaps(level.stream().mapToInt(x -> x).toArray(), level.size());
        }
        return cnt;
    }

    // Return the minimum number
    // of swaps required to sort the array
    public int minSwaps(int[] arr, int N) {

        int ans = 0;
        int[] temp = Arrays.copyOfRange(arr, 0, N);

        // Hashmap which stores the
        // indexes of the input array
        HashMap<Integer, Integer> h = new HashMap<Integer, Integer>();

        Arrays.sort(temp);
        for (int i = 0; i < N; i++) {
            h.put(arr[i], i);
        }
        for (int i = 0; i < N; i++) {

            // This is checking whether
            // the current element is
            // at the right place or not
            if (arr[i] != temp[i]) {
                ans++;
                int init = arr[i];

                // If not, swap this element
                // with the index of the
                // element which should come here
                swap(arr, i, h.get(temp[i]));

                // Update the indexes in
                // the hashmap accordingly
                h.put(init, h.get(temp[i]));
                h.put(temp[i], i);
            }
        }
        return ans;
    }

    public void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public int unequalTriplets(int[] nums) {
        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                for (int k = j + 1; k < nums.length; k++) {
                    if (nums[i] != nums[j] && nums[i] != nums[k] && nums[j] != nums[k]) cnt++;
                }
            }
        }

        return cnt;
    }

    //28th Nov---------------------------------------------------------------------------------------------------
    public int pivotInteger(int n) {
        int total = Math.abs(n * (n + 1) / 2);

        if (n == 1) return 1;
        int currSum = 0;
        for (int i = 1; i < n; i++) {
            currSum += i;
            if (currSum == (total + i - currSum)) return i;
        }

        return -1;
    }

    public int appendCharacters(String s, String t) {

        List<Character> tl = new ArrayList<>();
        for (char c : t.toCharArray()) tl.add(c);

        int ind = 0;
        for (char c : s.toCharArray()) {
            if (ind < tl.size() && c == tl.get(ind)) {
                ind++;
            }
        }

        return tl.size() - ind;

    }

    public int numberOfCuts(int n) {
        if (n == 1) return 0;
        if (n % 2 == 0 && n % 3 == 0) return n / 2;
        if (n % 2 == 0) return n / 2;
        return n;
    }

    public int[][] onesMinusZeros(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] ans = new int[m][n];

        List<Pair<Integer, Integer>> rows = new ArrayList<>();
        List<Pair<Integer, Integer>> cols = new ArrayList<>();


        for (int i = 0; i < m; i++) {
            int cnt1 = 0, cnt0 = 0;
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) cnt0++;
                else cnt1++;
            }
            rows.add(new Pair<>(cnt0, cnt1));
        }

        for (int i = 0; i < n; i++) {
            int cnt1 = 0, cnt0 = 0;
            for (int j = 0; j < m; j++) {
                if (grid[j][i] == 0) cnt0++;
                else cnt1++;
            }
            cols.add(new Pair<>(cnt0, cnt1));
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ans[i][j] = rows.get(i).getValue() + cols.get(j).getValue() - rows.get(i).getKey() - cols.get(j).getKey();
            }
        }

        return ans;
    }

    public int bestClosingTime(String customers) {

        TreeMap<Integer, Integer> tm = new TreeMap<>();
        int cntN = 0;
        for (int i = 0; i < customers.length(); i++) {
            if (customers.charAt(i) == 'Y') {
                tm.put(i, cntN + 1);
            } else {
                tm.put(i, cntN);
                cntN++;
            }
        }

        tm.put(customers.length(), cntN);
        int cntY = 0;
        for (int i = customers.length() - 1; i >= 0; i--) {
            tm.put(i, tm.getOrDefault(i, 0) + cntY);
            if (customers.charAt(i) == 'Y') cntY++;
        }

        int ans = Integer.MAX_VALUE, ind = 0;

        int ci = 0;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            if (entry.getValue() < ans) {
                ans = entry.getValue();
                ind = ci;
            }
            ci++;
        }

        return ind;
    }

    // TBC
    public String multiply(String num1, String num2) {
        List<Integer> prev = new ArrayList<>();

        StringBuilder sb = new StringBuilder();

        int pos = 1;
        for (int i = num2.length() - 1; i >= 0; i--) {
            List<Integer> list = new ArrayList<>();
            int carry = 0;
            for (int j = num1.length() - 1; j >= 0; j--) {
                int ans = Integer.parseInt(String.valueOf(num2.charAt(i))) *
                        Integer.parseInt(String.valueOf(num1.charAt(j))) + carry;
                list.add(ans % 10);
                carry = ans / 10;
            }

            if (carry > 0) list.add(carry);
            Collections.reverse(list);

            System.out.println(Arrays.toString(list.toArray()));
            if (prev.size() == 0) {
                sb.insert(0, list.get(list.size() - 1));
                prev.clear();
                prev.addAll(list);
                pos++;
            } else {
                int nc = 0;
                System.out.println("prev=" + Arrays.toString(prev.toArray()) + ", pos=" + pos);
                int ind = prev.size() - pos++;
                System.out.println("list=" + Arrays.toString(list.toArray()));


                int cut = ind;
                int pl = sb.length();
                for (int k = list.size() - 1; k >= 0; k--) {
                    int elem = (ind >= 0 ? prev.get(ind) : 0) + list.get(k);

                    if (pl > cut && cut >= 0) {
                        sb.setCharAt(ind, (char) ((elem + nc) % 10 + '0'));
                        cut--;
                    } else sb.insert(0, ((elem + nc) % 10));
                    System.out.println("sb_cal=" + sb);
                    nc = elem / 10;
                    ind--;
                }

                if (nc > 0) sb.insert(0, nc);
                prev.clear();
                for (int t = 0; t < sb.length(); t++) {
                    prev.add(Integer.parseInt(String.valueOf(sb.charAt(t))));
                }
            }

            System.out.println("sb=" + sb);
        }


        return sb.toString();

    }

    //Author: Anand
    public boolean isCircularSentence(String sentence) {
        String[] words = sentence.split(" ");
        char first = '#';
        char last = '#';
        for (String word : words) {
            if (first == '#') first = word.charAt(0);
            if (last == '#') last = word.charAt(word.length() - 1);
            else if (last != word.charAt(0)) return false;

            last = word.charAt(word.length() - 1);
        }

        return last == first;
    }

    //Author: Anand
    public long dividePlayers(int[] skill) {
        Arrays.sort(skill);
        int i = 0, j = skill.length - 1;
        long ans = 0L;

        long sum = -1L;
        while (i < j) {
            if (sum == -1) sum = (long) skill[i] + skill[j];
            else if (sum != (long) skill[i] + skill[j]) return -1;
            ans += (long) skill[i++] * skill[j--];
        }

        return ans;
    }

    public int maximumValue(String[] strs) {
        int max = Integer.MIN_VALUE;
        for (String s : strs) {
            if (s.replaceAll("\\d", "").isEmpty()) {
                max = Math.max(max, Integer.parseInt(s));
            } else {
                max = Math.max(max, s.length());
            }
        }
        return max;
    }

    //Author: Anand
    public int deleteGreatestValue(int[][] grid) {

        Map<Integer, PriorityQueue<Integer>> rowiseMax = new ConcurrentHashMap<>();

        int row = 0;
        for (int[] g : grid) {
            for (int r : g) {
                if (!rowiseMax.containsKey(row)) rowiseMax.put(row, new PriorityQueue<>(Collections.reverseOrder()));
                rowiseMax.get(row).offer(r);
            }
            row++;
        }

        int ans = 0;

        while (rowiseMax.size() > 0) {
            int max = Integer.MIN_VALUE;
            for (Map.Entry<Integer, PriorityQueue<Integer>> entry : rowiseMax.entrySet()) {
                if (entry.getValue().size() == 0) {
                    rowiseMax.remove(entry.getKey());
                } else max = Math.max(max, entry.getValue().poll());
            }

            if (max != Integer.MIN_VALUE) ans += max;
        }

        return ans;
    }

    //Author: Anand
    public int longestSquareStreak(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;

        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int num : nums) {

            if ((Math.sqrt(num) == Math.floor(Math.sqrt(num))) && !Double.isInfinite(Math.sqrt(num)) && map.containsKey((int) Math.sqrt(num))) {
                List<Integer> exist = map.get((int) Math.sqrt(num));
                map.remove((int) Math.sqrt(num));
                exist.add(num);
                map.put(num, exist);
            } else {
                if (!map.containsKey(num)) map.put(num, new ArrayList<>(Collections.singletonList(num)));
            }
        }
        int max = -1;
        for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
            if (entry.getValue().size() > 1) max = Math.max(max, entry.getValue().size());
        }

        return max;
    }

    /*
    Input: words = ["aba","aabb","abcd","bac","aabc"]
    Output: 2
    Explanation: There are 2 pairs that satisfy the conditions:
    - i = 0 and j = 1 : both words[0] and words[1] only consist of characters 'a' and 'b'.
    - i = 3 and j = 4 : both words[3] and words[4] only consist of characters 'a', 'b', and 'c'.
     */
    public int similarPairs(String[] words) {

        int cnt = 0;
        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                if (isSimilar(words[i], words[j])) cnt++;
            }
        }
        return cnt;
    }

    private boolean isSimilar(String word1, String word2) {
        Set<Character> s1 = new HashSet<>();
        for (char c : word1.toCharArray()) s1.add(c);
        Set<Character> s2 = new HashSet<>();
        for (char c : word2.toCharArray()) s2.add(c);
        return s1.equals(s2);
    }

    //prime sieve
    public void primeSieve(int n) {
        BitSet bitset = new BitSet(n + 1);
        for (long i = 0; i < n; i++) {
            if (i == 0 || i == 1) {
                bitset.set((int) i);
                continue;
            }
            if (bitset.get((int) i)) continue;
            primeNumbers.add((int) i);
            for (long j = i; j <= n; j += i)
                bitset.set((int) j);
        }
    }

    public int smallestValue(int n) {

        if (n == 2 || n == 4) return n;

        primeSieve(n);
        List<Integer> powers = new ArrayList<>();
        primeFactors(n, powers);
        while (powers.size() > 0) {
            n = powers.stream().mapToInt(x -> x).sum();
            powers.clear();
            primeNumbers.clear();
            primeSieve(n);
            primeFactors(n, powers);
        }
        return n;
    }

    // A function to print all prime factors
    // of a given number n
    public void primeFactors(int n, List<Integer> powers) {
        // Print the number of 2s that divide n
        while (n % 2 == 0) {
            powers.add(2);
            n /= 2;
        }

        // n must be odd at this point.  So we can
        // skip one element (Note i = i +2)
        for (int i : primeNumbers) {

            if (i > n) break;
            // While i divides n, print i and divide n
            while (n % i == 0) {
                powers.add(i);
                n /= i;
            }
        }
    }

    public boolean isPossible(int n, List<List<Integer>> edges) {

        Map<Integer, Set<Integer>> edgesM = new HashMap<>();
        Map<Integer, Integer> map = new HashMap<>(); // node, degree

        for (List<Integer> edge : edges) {
            int n1 = edge.get(0);
            int n2 = edge.get(1);
            map.put(n1, map.getOrDefault(n1, 0) + 1);
            map.put(n2, map.getOrDefault(n2, 0) + 1);

            if (!edgesM.containsKey(n1)) edgesM.put(n1, new HashSet<>());
            if (!edgesM.containsKey(n2)) edgesM.put(n2, new HashSet<>());
            edgesM.get(n1).add(n2);
            edgesM.get(n2).add(n1);
        }

        // collect all odd degree nodes
        Map<Integer, Integer> oddD = map.entrySet()
                .stream()
                .filter(x -> x.getValue() % 2 != 0)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        if (oddD.size() == 0) return true;
        if (oddD.size() > 4 || oddD.size() % 2 != 0) return false;

        List<Integer> keyset = new ArrayList<>(oddD.keySet());

        // If 4 nodes having odd degree then  check if there is combination to connect them with 2 edges
        if (oddD.size() == 4) {
            int cnt = 0;
            for (int i = 0; i < keyset.size(); i++) {
                for (int j = i + 1; j < keyset.size(); j++) {
                    if (oddD.get(keyset.get(i)) % 2 == 0 || oddD.get(keyset.get(j)) % 2 == 0) continue;
                    if (!edgesM.get(keyset.get(i)).contains(keyset.get(j))) {

                        edgesM.get(keyset.get(i)).add(keyset.get(j));
                        edgesM.get(keyset.get(j)).add(keyset.get(i));

                        oddD.put(keyset.get(i), oddD.get(keyset.get(i)) + 1);
                        oddD.put(keyset.get(j), oddD.get(keyset.get(j)) + 1);
                        cnt++;
                    }
                }
            }
            return cnt == 2;
        }

        // 2 odd edges
        // if 2 edges are not connected thn connect them and make degrees even
        if (!edgesM.get(keyset.get(0)).contains(keyset.get(1))) return true;

        // check if there exist a node with even degree that is not connected to either of them
        // if exist connect it to both of them 1 by 1 and all degrees will become even
        Map<Integer, Integer> un = map.entrySet()
                .stream()
                .filter(x -> (x.getValue() % 2 == 0 &&
                        !edgesM.get(keyset.get(0)).contains(x.getKey())
                        &&
                        !edgesM.get(keyset.get(1)).contains(x.getKey())))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        return un.size() > 0;
    }

    public int countDigits(int num) {

        int cnt = 0;
        for (char c : String.valueOf(num).toCharArray()) {
            int d = c - '0';
            if (num % d == 0) cnt++;
        }
        return cnt;
    }

    public int minimumPartition(String s, int k) {
        // Sanitiy check for the algorithm
        for (char c : s.toCharArray()) if ((c - '0') > k) return -1;

        int cnt = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {

            sb.append(s.charAt(i));

            // valid partition
            if (sb.toString().replaceAll("^0+", "").length() > String.valueOf(k).length()
                    || Integer.parseInt(sb.toString()) > k) {
                cnt++;
                sb.delete(0, sb.length());
                sb.append(s.charAt(i));
            }
        }

        if (sb.length() > 0 && sb.length() <= String.valueOf(k).length()) cnt++;
        return cnt;
    }

    public int captureForts(int[] forts) {
        int pos = Integer.MIN_VALUE, maxCnt = 0, cnt = 0;
        for (int fort : forts) {
            if ((fort == 1 || fort == -1) && fort != pos) {
                maxCnt = Math.max(maxCnt, cnt);
                pos = fort;
                cnt = 0;
            } else if ((fort == 1 || fort == -1)) {
                cnt = 0;
            } else if (pos != Integer.MIN_VALUE) {
                cnt++;
            }
        }

        return maxCnt;
    }

    public List<Integer> topStudents(String[] positive_feedback, String[] negative_feedback, String[] report, int[] student_id, int k) {
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(new Comparator<Pair<Integer, Integer>>() {
            @Override
            public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                if (o1.getValue() > o2.getValue()) return -1;
                if (o1.getValue() < o2.getValue()) return 1;
                return o1.getKey().compareTo(o2.getKey());
            }
        });


        Set<String> pfs = new HashSet<>();
        Collections.addAll(pfs, positive_feedback);

        Set<String> nfs = new HashSet<>();
        Collections.addAll(nfs, negative_feedback);

        for (int i = 0; i < student_id.length; i++) {
            int si = student_id[i];
            String[] r = report[i].split(" ");

            int score = 0;
            for (String s : r) {
                if (pfs.contains(s)) score += 3;
                else if (nfs.contains(s)) score -= 1;
            }

            pq.offer(new Pair<>(si, score));
        }

        List<Integer> ans = new ArrayList<>();
        while (!pq.isEmpty()) {
            if (k-- == 0) break;
            ans.add(pq.poll().getKey());
        }

        return ans;
    }

    public int closetTarget(String[] words, String target, int startIndex) {
        int n = words.length;
        int cnt = 0, maxCnt = -1;
        // left to right traversal


        if (words[startIndex].equals(target)) return 0;
        int ni = (startIndex + 1) % n;
        cnt++;
        while (!words[ni].equals(target) && ni != startIndex) {
            ni = (ni + 1) % n;
            cnt++;
        }

        if (ni != startIndex) maxCnt = Math.max(maxCnt, cnt);

        cnt = 0;
        ni = (startIndex - 1 + n) % n;
        cnt++;
        while (!words[ni].equals(target) && ni != startIndex) {
            ni = (ni - 1 + n) % n;
            cnt++;
        }

        if (ni != startIndex) {
            if (cnt == 0) return maxCnt;
            if (maxCnt == -1) return cnt;
            return Math.min(maxCnt, cnt);
        }
        return maxCnt;
    }

    public int takeCharacters(String s, int k) {
        int n = s.length();
        TreeMap<Character, List<Integer>> pos = new TreeMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (!pos.containsKey(s.charAt(i))) pos.put(s.charAt(i), new ArrayList<>());
            pos.get(s.charAt(i)).add(i);
        }

        int left = 0, right = n - 1;

        if (pos.size() != 3) return -1;
        for (Map.Entry<Character, List<Integer>> entry : pos.entrySet()) {
            if (entry.getValue().size() < k) return -1;
        }

        System.out.println(pos);

        for (Map.Entry<Character, List<Integer>> entry : pos.entrySet()) {
            List<Integer> indexes = entry.getValue();
            int ck = k;
            int ind = 0;
            while (ck-- > 0) {

                int lp = indexes.get(ind);
                int rp = indexes.get(indexes.size() - 1 - ind);
                if (lp < n - rp) {
                    left = Math.max(left, lp);
                } else if (lp > n - rp) {
                    if (n - rp > rp) {
                        left = Math.max(left, lp);
                    } else {
                        right = Math.min(right, rp);
                    }
                }
                ind++;

                System.out.println(left + ":" + right);
            }
        }

        return left + n - right + 1;
    }

    //Author: Anand
    public String categorizeBox(int length, int width, int height, int mass) {
        boolean bulky = false;
        long volume = (long) length * width * height;
        if (volume >= Math.pow(10, 9) || length >= Math.pow(10, 4)
                || width >= Math.pow(10, 4) || height >= Math.pow(10, 4)
        ) bulky = true;
        boolean heavy = mass >= 100;
        if (heavy && bulky) return "Both";
        if (!bulky && !heavy) return "Neither";
        if (bulky) return "Bulky";
        return "Heavy";
    }

    // TODO: For all test cases
    public long maxPower(int[] stations, int r, int k) {
        long ans = Long.MIN_VALUE;

        int[] prefix = new int[stations.length];
        for (int i = 0; i < stations.length; i++) {
            if (i == 0) prefix[i] = stations[i];
            else prefix[i] = prefix[i - 1] + stations[i];
        }

        int[] np = new int[stations.length];

        for (int i = 0; i < stations.length; i++) {
            np[i] = prefix[Math.min(i + r, stations.length - 1)] - (i != 0 && (i - r - 1 >= 0) ? prefix[Math.max(i - r - 1, 0)] : 0);
        }

        System.out.println(Arrays.toString(prefix));

        System.out.println(Arrays.toString(np));

        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int p : np) pq.add(p);

        TreeMap<Integer, Integer> tm = new TreeMap<>(); // {value, cnt}

        for (int p : np) tm.put(p, tm.getOrDefault(p, 0) + 1);
        ans = tm.firstKey();

        System.out.println("tm=" + tm);
        while (!pq.isEmpty() && k > 0) {
            int min = pq.poll();
            if (pq.peek() != null) {
                int cnt = 1; // cnt if items to be increased
                if (tm.containsKey(min)) cnt = tm.get(min);
                k -= (pq.peek() - min) * cnt;
                ans = Math.max(ans, pq.peek());
            } else {
                ans = Math.max(ans, (long) k / stations.length);
                k = 0;
            }
        }

        return ans;
    }

    public int differenceOfSum(int[] nums) {

        long ds = 0L, ns = 0L;
        for (int num : nums) {
            ds += digitsum(num);
            ns += num;
        }
        return (int) Math.abs(ds - ns);
    }

    private int digitsum(int num) {
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }

    public int[][] rangeAddQueries(int n, int[][] queries) {
        int[][] ans = new int[n][n];
        this.n = n;

        Set<Pair<Integer, Integer>> queue = new HashSet<>();
        for (int[] query : queries) {
            int r1 = query[0];
            int c1 = query[1];
            int r2 = query[2];
            int c2 = query[3];

            int x = r1, y = c1;
            queue.add(new Pair<>(x, y));
            ans[x][y] += 1;

            while (true) {
                boolean canMove = false;

                for (int[] dir : dirs) {
                    int nx = x + dir[0];
                    int ny = y + dir[1];

                    // check for boundary/obstacle condition
                    while (safe(nx, ny) && nx <= r2 && ny <= c2 && nx >= r1 && ny >= c1
                            && !queue.contains(new Pair<>(nx, ny))
                    ) {
                        queue.add(new Pair<>(nx, ny));
                        ans[nx][ny] += 1;

                        canMove = true;
                        x = nx;
                        y = ny;

                        nx += dir[0];
                        ny += dir[1];
                    }
                }

                if (!canMove) break;
            }
        }

        return ans;
    }

    private boolean safe(int nx, int ny) {
        return nx >= 0 && ny >= 0 && nx < this.n && ny < this.n;
    }

    public int getCommon(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums1) set.add(num);
        for (int num : nums2) if (set.contains(num)) return num;
        return -1;
    }

    public long minOperations(int[] nums1, int[] nums2, int k) {

        int[] diff = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            diff[i] = nums1[i] - nums2[i];
        }

        long pos = 0L, neg = 0L;

        for (int num : diff) {
            if (num > 0) pos += num;
            else if (num < 0) neg += num;
        }


        if (k == 0) {
            for (int i = 0; i < nums1.length; i++)
                if (nums1[i] != nums2[i]) return -1;
            return 0;
        }


        // check for each element should be mulitple of k
        boolean valid = true;

        for (int num : diff) {
            if (num % k != 0) {
                valid = false;
                break;
            }
        }

        if (valid && pos % k == 0 && neg % k == 0 && Math.abs(pos) == Math.abs(neg)) return pos / k;
        if (valid && pos == 0 && neg == 0) return 0;
        return -1;
    }

    public long maxScore(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        int[][] pairs = new int[n][2];
        for (int i = 0; i < n; i++) pairs[i] = new int[]{nums2[i], nums1[i]};
        Arrays.sort(pairs, (a, b) -> b[0] - a[0]); // Sort num2 from big  to small

        PriorityQueue<Integer> pq = new PriorityQueue<>(); // Sort num1 from small to big

        long res = 0L, sumS = 0L;

        for (int[] pair : pairs) {
            pq.add(pair[1]); // Add num2 guy to get the minimum
            sumS += pair[1];
            if (pq.size() > k) {
                // pop the minimum guy and compute the result
                sumS -= pq.poll(); // Provides minimum num1 guy
            }

            if (pq.size() == k) res = Math.max(res, sumS * pair[0]);
        }
        return res;
    }

    public int distinctIntegers(int n) {

        Set<Integer> dn = new HashSet<>();
        dn.add(n);

        for (int i = 2; i < n; i++) {
            if (n % i == 1) dn.add(i);
        }

        while (true) {
            boolean nn = false;
            Set<Integer> factors = new HashSet<>();
            factors.addAll(dn);


            for (int factor : factors) {
                for (int i = 2; i < factor; i++) {
                    if (factor % i == 1 && !dn.contains(i)) {
                        nn = true;
                        dn.add(i);
                    }
                }
            }

            if (!nn) return dn.size();
        }
    }

    public int monkeyMove(int n) {
        int nn = (int) expo(2, n, MOD) - 2;
        if (nn < 0) return (nn + MOD);
        return nn;
    }

    public long expo(long a, long b, long mod) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1L) res = (res * a) % mod;  //think about this one for a second
            a = (a * a) % mod;
            b = b >> 1;
        }
        return res;
    }

    public int[] evenOddBit(int n) {

        String binary = Integer.toBinaryString(n);
        StringBuilder sb = new StringBuilder();
        sb.append(binary).reverse();

        int odd = 0, even = 0;
        for (int i = 0; i < sb.length(); i++) {
            if (i % 2 == 0 && sb.charAt(i) == '1') even++;
            else if (i % 2 != 0 && sb.charAt(i) == '1') odd++;
        }

        return new int[]{even, odd};
    }

    public int beautifulSubsets(int[] nums, int k) {
        Arrays.sort(nums);
        Set<Integer> hSet = new HashSet<>();
        return solve(nums, k, 0, hSet);
    }

    private int solve(int[] nums, int k, int ind, Set<Integer> hSet) {
        if (ind == nums.length) {
            if (hSet.size() > 0) return 1;
            return 0;
        }

        int take = 0, nottake = 0;
        nottake += solve(nums, k, ind + 1, hSet);

        if (!hSet.contains(nums[ind] - k)) {
            hSet.add(nums[ind]);
            take += solve(nums, k, ind + 1, hSet);
            hSet.remove(nums[ind]);
        }

        return take + nottake;
    }

    //TBD
    // You can add or subtract value any number of times
    public int findSmallestInteger(int[] nums, int value) {

        int gi = -1;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            int nv = num % value;
            if (set.size() == 0) {
                gi = num % value;
                set.add(nv);
            } else {
                if (!set.contains(nv)) {
                    set.add(nv);
                } else {
                    int nnv = nv;
                    int cnt = 1;
                    while (set.contains(nnv + value * cnt)) cnt++;
                    set.add(nnv + value * cnt);
                }
                gi = Math.max(nv, gi);
            }
        }

        List<Integer> list = new ArrayList<>();
        list.addAll(set);
        Collections.sort(list);
        System.out.println(Arrays.toString(list.toArray()));

        boolean consective = false;
        int last = -1;
        for (int l : list) {
            if (last == -1) {
                last = l;
                continue;
            }
            if ((l - last) != 1 && consective) return last + 1;
            if ((l - last) == 1) consective = true;
            last = l;
        }

        if (!consective) {
            for (int l : list) {
                if (l > 0) {
                    return l - 1;
                }
            }
        }
        return list.get(list.size() - 1) + 1;
    }

    //TBD
    public int distMoney(int money, int children) {
        if (money == 20 && children >= 14 && children <= 20) return 0;

        int pm = money;
        int pc = children;
        System.out.println(money + ":" + children);
        if (money < children) return -1;
        if (money - children < 8) return 0;

        int ans = money / 8;
        int rem = money % 8;
        money = rem;
        children -= ans;

        if (children < 0) return pc - 1;
        if (rem == 4) {
            if (children == 1 || children > rem) return --ans;
            if (children <= rem && children != 0) return ans;


            boolean zero = false;
            if (children > rem) {
                money += 8;
                children++;

                int na = distMoney(money, children);
                if (pm == money && pc == children) {
                    zero = true;
                    return 0;
                }
                ans--;

                if (na == 0) {
                    zero = true;
                    return 0;
                }
                if (na == -1) {
                    if (zero) return 0;
                    int naa = distMoney(money + 8, children + 1);
                    if (naa == 0) {
                        zero = true;
                        return 0;
                    }
                    if (zero) return 0;
                    return ans + naa - 1;
                }
                if (zero) return 0;
                return ans;
            }
            return ans > 0 ? --ans : ans;
        }


        //16
        //10
        /*

         */
        if (rem == 0) {
            if (children >= 8) {
                children++;
                money += 8;
                ans--;
                return ans + distMoney(money, children);
            } else if (children > 0 && money >= children) return ans;
            else if (children == 0) return ans;
            else return --ans;
        }
        if (rem > 0 && children == 1 && ans > 0) return ans;
        if (rem > 0) {
            if (children <= rem && children != 0) return ans;
            if (children > rem) {
                money += 8;
                children++;

                if (pm == money && pc == children) return 0;
                return --ans + distMoney(money, children);
            }
            return --ans;
        }
        return ans;
    }

    public int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
        int sum = 0;
        while (k > 0) {
            if (numOnes > 0) {
                numOnes--;
                sum++;
            } else if (numZeros > 0) numZeros--;
            else {
                numNegOnes--;
                sum--;
            }
            k--;
        }
        return sum;
    }

    //TBD
    public int collectTheCoins(int[] coins, int[][] edges) {

        // No of vertices
        int v = coins.length;
        ArrayList<ArrayList<Integer>> adj = new ArrayList<ArrayList<Integer>>(v);

        for (int i = 0; i < v; i++) adj.add(new ArrayList<Integer>());
        addEdge(adj, 0, 1);
        addEdge(adj, 0, 3);
        addEdge(adj, 1, 2);
        addEdge(adj, 3, 4);
        addEdge(adj, 3, 7);
        addEdge(adj, 4, 5);
        addEdge(adj, 4, 6);
        addEdge(adj, 4, 7);
        addEdge(adj, 5, 6);
        addEdge(adj, 6, 7);
        int source = 0, dest = 7;
        printShortestDistance(adj, source, dest, v);
        return 0;
    }

    private void printShortestDistance(ArrayList<ArrayList<Integer>> adj, int source, int dest, int v) {
        // predesessor array --> used to trace path in bfs
        int[] pred = new int[v];
        int[] dist = new int[v]; // to store min dist from source
        if (!BFS(adj, source, dest, pred, dist, v)) {
            System.out.println("Source and destination are not connected");
            return;
        }


        // Fetch-path
        // crwal in reverse direction
        LinkedList<Integer> path = new LinkedList<>();
        int crawl = dest;
        while (pred[crawl] != -1) {
            path.add(pred[crawl]);
            crawl = pred[crawl];
        }

        // shortest distance
        System.out.println(dist[dest]);
        // print path
        for (int i = path.size() - 1; i >= 0; i--) System.out.print(path.get(i) + " ");
    }

    private boolean BFS(ArrayList<ArrayList<Integer>> adj, int source, int dest, int[] pred, int[] dist, int v) {
        Arrays.fill(pred, -1);
        Arrays.fill(dist, Integer.MAX_VALUE);
        boolean[] visited = new boolean[v];
        Queue<Integer> queue = new LinkedList<>();
        queue.add(source);
        visited[source] = true;
        dist[source] = 0;

        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int i = 0; i < adj.get(u).size(); i++) {
                if (visited[adj.get(u).get(i)]) continue;
                // if not visited then mark it visited and process it in queue
                visited[adj.get(u).get(i)] = true;
                queue.offer(adj.get(u).get(i));
                dist[adj.get(u).get(i)] = 1 + dist[u];
                pred[adj.get(u).get(i)] = u;
                if (adj.get(u).get(i) == dest) return true;
            }
        }
        return false;
    }

    private void addEdge(ArrayList<ArrayList<Integer>> adj, int vertex1, int vertex2) {
        adj.get(vertex1).add(vertex2);
        adj.get(vertex2).add(vertex1);
    }

    public int minNumber(int[] nums1, int[] nums2) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        for (int num : nums1) list1.add(num);
        for (int num : nums2) list2.add(num);

        Collections.sort(list1);
        Collections.sort(list2);

        for (int num : list1) if (list2.contains(num)) return num;

        int d1 = list1.get(0);
        int d2 = list2.get(0);
        return d1 < d2 ? Integer.parseInt(d1 + "" + d2) : Integer.parseInt(d2 + "" + d1);
    }

    public int findTheLongestBalancedSubstring(String s) {

        int max = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j < s.length(); j++) {
                String ss = s.substring(i, j + 1);
                if (balanced(ss)) {
                    max = Math.max(max, ss.length());
                }
            }
        }

        return max;
    }

    private boolean balanced(String ss) {


        if (ss.length() % 2 != 0) return false;
        for (int i = 0; i < ss.length() / 2; i++)
            if (ss.charAt(i) != '0' || ss.charAt(ss.length() - 1 - i) != '1') return false;
        return true;
    }

    public List<List<Integer>> findMatrix(int[] nums) {
        Map<Integer, Set<Integer>> matrix = new ConcurrentHashMap<>();

        for (int num : nums) {
            if (matrix.size() == 0) {
                Set<Integer> row = new HashSet<>();
                row.add(num);
                matrix.put(0, row);
            } else {
                boolean added = false;
                for (Map.Entry<Integer, Set<Integer>> entry : matrix.entrySet()) {
                    Set<Integer> s = entry.getValue();
                    if (s.size() == 0 || !s.contains(num)) {
                        added = true;
                        matrix.remove(entry.getKey());
                        s.add(num);
                        matrix.put(entry.getKey(), s);
                        break;
                    }
                }

                if (!added) {
                    Set<Integer> ns = new HashSet<>();
                    ns.add(num);
                    matrix.put(matrix.size(), ns);
                }
            }
        }

        List<List<Integer>> ans = new ArrayList<>();
        for (Map.Entry<Integer, Set<Integer>> entry : matrix.entrySet())
            ans.add(new ArrayList<>(entry.getValue()));
        return ans;
    }

    //TLE
    public int miceAndCheese(int[] reward1, int[] reward2, int k) {
        return helper(reward1, reward2, k, 0, new ArrayList<>());
    }

    private int helper(int[] reward1, int[] reward2, int k, int ind, List<Integer> it) {
        // base case
        if (ind >= reward1.length && k > 0) return 0;

        if (k <= 0) {
            List<Integer> indexes = new ArrayList<>();
            for (int i = 0; i < reward1.length; i++) indexes.add(i);
            indexes.removeAll(it);

            int sum = 0;
            for (int i : indexes) sum += reward2[i];
            return sum;
        }

        int take = 0, nt = 0;
        // take
        it.add(ind);
        take += reward1[ind] + helper(reward1, reward2, k - 1, ind + 1, it);
        //not take
        it.remove(new Integer(ind));
        nt += helper(reward1, reward2, k, ind + 1, it);
        return Math.max(take, nt);
    }

    //-----------------------------------------------------------------------------------------------------
    // 15th april

    public int diagonalPrime(int[][] nums) {

        int max = 0;
        for (int i = 0; i < nums.length; i++) {

            if (isPrime(nums[i][i])) {
                max = Math.max(max, nums[i][i]);
            }

            if (isPrime(nums[i][nums.length - i - 1])) {
                max = Math.max(max, nums[i][nums.length - i - 1]);
            }
        }

        return max;
    }

    public boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (int i = 5; i * i <= n; i = i + 6)
            if (n % i == 0 || n % (i + 2) == 0)
                return false;
        return true;
    }

    public java.util.HashMap<Integer, Integer> sortByValue(java.util.HashMap<Integer, Integer> hm) {


        List<Map.Entry<Integer, Integer>> list = new LinkedList<Map.Entry<Integer, Integer>>(hm.entrySet());

        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
            public int compare(Map.Entry<Integer, Integer> o1,
                               Map.Entry<Integer, Integer> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        java.util.HashMap<Integer, Integer> temp = new LinkedHashMap<Integer, Integer>();
        for (Map.Entry<Integer, Integer> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }

        return temp;
    }

    public int minimizeMax(int[] nums, int p) {
        Map<Integer, Integer> freq = new ConcurrentHashMap<>();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);
        sortByValue((java.util.HashMap<Integer, Integer>) freq);

        int ans = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {

            int values = entry.getValue();
            while (values >= 2 && p > 0) {
                p--;
                values -= 2;
            }

            if (p <= 0) return 0;
            if (values <= 0) freq.remove(entry.getKey());
            freq.put(entry.getKey(), values);
        }

        System.out.println(freq);

        // sort the map based on keys
        TreeMap<Integer, Integer> tm = new TreeMap<>(freq);

        int lastKey = -1;
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            int values = entry.getValue();
            if (values == 1) {

                if (p <= 0) return ans;
                if (lastKey == -1) lastKey = entry.getKey();
                ans = Math.min(Math.abs(lastKey - entry.getKey()), ans);
                p--;
            }
        }


        return ans;
    }

    public int[] findColumnWidth(int[][] grid) {
        // Traverse columnwise
        int m = grid.length, n = grid[0].length;

        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            int max = -1;
            for (int j = 0; j < m; j++) {
                max = Math.max(len(grid[j][i]), max);
            }

            ans[i] = max;
        }

        return ans;
    }


    // 16th april

    private int len(int num) {
        int an = Math.abs(num);
        int cnt = 0;
        while (an > 0) {
            an /= 10;
            cnt++;
        }

        return num > 0 ? cnt : cnt + 1;
    }

    public long[] findPrefixScore(int[] nums) {

        long[] conv = new long[nums.length];
        int max = Integer.MIN_VALUE, ind = 0;
        for (int num : nums) {
            max = Math.max(max, num);
            conv[ind++] = max + num;
        }
        long[] ps = new long[nums.length];

        for (int i = 0; i < conv.length; i++) {
            long num = conv[i];
            ps[i] = i == 0 ? num : num + ps[i - 1];
        }

        return ps;

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
    // This solution assumes no duplicate nodes
    // Solve for duplicate nodes case
    public TreeNode replaceValueInTree(TreeNode root) {
        Map<Integer, Map<Integer, List<Integer>>> map = new TreeMap<>(); // {level, {parent -> childs}}
        return lot(root, map);
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

    public TreeNode lot(TreeNode root, Map<Integer, Map<Integer, List<Integer>>> map) {

        if (root == null)
            return new TreeNode(0);

        Map<int[], Boolean> cp = new TreeMap<>(); // node.val,level, parent -> true
        int level = 0;
        // Standard level order traversal code
        // using queue
        Queue<TreeNode> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();

            Map<Integer, List<Integer>> pc = new java.util.HashMap<>();

            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                TreeNode p = q.poll();
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
                n--;


                if (!pc.containsKey(p.val)) pc.put(p.val, new ArrayList<>());
                if (p.left != null) {
                    pc.get(p.val).add(p.left.val);
                    cp.put(new int[]{p.left.val, p.val}, true);
                }
                if (p.right != null) {
                    pc.get(p.val).add(p.right.val);
                    cp.put(new int[]{p.right.val, p.val}, true);
                }
            }

            map.put(level++, pc);
        }
        // map is ready
        System.out.println(map);
        System.out.println(cp);


        // Standard level order traversal code
        // using queue
        level = 0;
        Queue<TreeNode> nq = new LinkedList<>(); // Create a queue
        nq.add(root); // Enqueue root
        while (!nq.isEmpty()) {
            int n = nq.size();

            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                TreeNode p = nq.poll();
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) nq.add(p.left);
                if (p.right != null) nq.add(p.right);


                // Traversal to find node in quickest way possible
                Set<Integer> possibleCousins = map.get(level).keySet();

                possibleCousins.remove(p.val);
                System.out.println(possibleCousins);
                List<Integer> actualCousins = new ArrayList<>();

                // remove all nodes on current level wtih same parent
                for (int pc : possibleCousins) {
                    if (cp.containsKey(pc) && cp.containsKey(p.val) && (pc != p.val) && (cp.get(pc) != cp.get(p.val))) {
                        actualCousins.add(pc);
                    }
                }

                System.out.println(actualCousins);

                p.val = actualCousins.stream().mapToInt(x -> x).sum();
                n--;
            }

            level++;
        }

        return root;
    }

    public int[] rowAndMaximumOnes(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] ans = new int[]{0, 0};
        int max = 0;
        for (int i = 0; i < m; i++) {
            int cnt = 0;
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) cnt++;
            }
            if (cnt > max) {
                max = cnt;
                ans = new int[]{i, max};
            }
        }
        return ans;
    }

    public int maxDivScore(int[] nums, int[] divisors) {

        Arrays.sort(divisors);
        int max = 0, ans = 0;
        for (int d : divisors) {
            int cnt = 0;
            for (int num : nums) {
                if (num % d == 0) cnt++;
            }
            if (cnt > max) {
                max = cnt;
                ans = d;
            }
        }

        return ans == 0 ? divisors[0] : ans;
    }

    public int splitNum(int num) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();

        while (num > 0) {
            int r = num % 10;
            tm.put(r, tm.getOrDefault(r, 0) + 1);
            num /= 10;
        }

        StringBuilder sb1 = new StringBuilder(), sb2 = new StringBuilder();
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            int cnt = entry.getValue();
            while (cnt-- > 0) {
                if (sb1.length() > sb2.length()) sb2.append(entry.getKey());
                else sb1.append(entry.getKey());
            }
        }

        return Integer.parseInt(sb1.toString()) + Integer.parseInt(sb2.toString());
    }

    // TC = O(mn), SC = O(mn)
    // Author : Anand
    // DFS on graph
    public int countUnguarded(int m, int n, int[][] guards, int[][] walls) {
        char[][] grid = new char[m][n];
        Queue<int[]> queue = new LinkedList<>();
        int[][] dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};

        for (int[] guard : guards) {
            grid[guard[0]][guard[1]] = 'G';
            queue.offer(new int[]{guard[0], guard[1]});
        }

        for (int[] wall : walls) grid[wall[0]][wall[1]] = 'W';

        // DFS for all guards and marked coordinate as 'P'
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            for (int[] dir : dirs) {
                int x = point[0] + dir[0];
                int y = point[1] + dir[1];
                // check for boundary/obstacle condition
                while (safe(x, y, grid)) {
                    grid[x][y] = 'P';
                    x += dir[0];
                    y += dir[1];
                }
            }
        }

        int cnt = 0;
        // count cells that are not blocker
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] != 'W' && grid[i][j] != 'G' && grid[i][j] != 'P') cnt++;
            }
        }
        return cnt;
    }

    private boolean safe(int x, int y, char[][] grid) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] != 'W' && grid[x][y] != 'G';
    }

    public long kthLargestLevelSum(TreeNode root, int k) {
        if (root == null) return 0;
        PriorityQueue<Long> pq = new PriorityQueue<>(Collections.reverseOrder()); // sum PQ
        // Standard level order traversal code
        // using queue
        Queue<TreeNode> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();
            long sum = 0L;
            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                TreeNode p = q.peek();
                q.remove();
                sum += p.val;
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
                n--;
            }

            pq.add(sum);
        }

        while (!pq.isEmpty()) {
            long ans = pq.poll();
            if (--k == 0) return ans;
        }

        return -1;
    }

    //TC = O(NlogN)
    public int findValidSplit(int[] nums) {
        TreeMap<Integer, Integer> overallPowers = new TreeMap<>();

        for (int n : nums) {
            if (n > 0) {
                while (n % 2 == 0) {
                    overallPowers.put(2, overallPowers.getOrDefault(2, 0) + 1);
                    n /= 2;
                }

                for (int i = 3; i <= Math.sqrt(n); i += 2) {
                    while (n % i == 0) {
                        overallPowers.put(i, overallPowers.getOrDefault(i, 0) + 1);
                        n /= i;
                    }
                }
                if (n > 2) {
                    overallPowers.put(n, overallPowers.getOrDefault(n, 0) + 1);
                }
            }
        }

        for (int p = 0; p < nums.length - 1; p++) {
            int n = nums[p];
            TreeMap<Integer, Integer> tm = new TreeMap<>();

            computePowers(n, tm);

            // check if valid
            if (valid(overallPowers, tm)) return p;
        }
        return -1;
    }

    private void computePowers(int n, TreeMap<Integer, Integer> tm) {
        if (n > 0) {
            while (n % 2 == 0) {
                tm.put(2, tm.getOrDefault(2, 0) + 1);
                n /= 2;
            }

            for (int i = 3; i <= Math.sqrt(n); i += 2) {
                while (n % i == 0) {
                    tm.put(i, tm.getOrDefault(i, 0) + 1);
                    n /= i;
                }
            }
            if (n > 2) {
                tm.put(n, tm.getOrDefault(n, 0) + 1);
            }
        }
    }

    private boolean valid(TreeMap<Integer, Integer> overall, TreeMap<Integer, Integer> tm) {

        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            if (overall.containsKey(entry.getKey()) && overall.get(entry.getKey()) != entry.getValue()) {
                // Put default power
                if (!affectedPowers.containsKey(entry.getKey()))
                    affectedPowers.put(entry.getKey(), overall.get(entry.getKey()));

                affectedPowers.put(entry.getKey(), affectedPowers.getOrDefault(entry.getKey(), 0) - entry.getValue());
                if (affectedPowers.get(entry.getKey()) <= 0) affectedPowers.remove(entry.getKey());
            }
        }


        return affectedPowers.size() == 0;
    }

    //is vowel function
    public boolean isVowel(char c) {
        return (c == 'a' || c == 'A' || c == 'e' || c == 'E' || c == 'i' || c == 'I' || c == 'o' || c == 'O' || c == 'u' || c == 'U');
    }


    // TOPOLOGICAL SORT
    // Use toplogical sort for indegree and pq to minisnmise the time taken to complete the course
    // TC = O(V+E) // As Simple DFS, SC = O(V) {Stack space}


    // TOPO Sort algorithm using DFS
    // The idea is to do dfs for all nodes after marking them visited,
    // after returning from recursion calls add them to stack

    public int vowelStrings(String[] words, int left, int right) {
        int cnt = 0;
        for (int i = left; i <= right; i++)
            if (isVowel(words[i].charAt(0)) && isVowel(words[i].charAt(words[i].length() - 1))) cnt++;
        return cnt;
    }

    public int maxScore(int[] nums) {
        int cnt = 0;
        List<Integer> integersList = Arrays.stream(nums).boxed().sorted(Collections.reverseOrder()).collect(Collectors.toList());

        long[] ps = new long[nums.length];
        for (int i = 0; i < integersList.size(); i++) {
            ps[i] = (i == 0 ? 0 : ps[i - 1]) + integersList.get(i);
        }

        for (long p : ps) {
            if (p > 0) cnt++;
            else break;
        }

        return cnt;
    }

    /*
      Steps:
        Build the graph using rowConditions
        Find topological sorting order for this graph
        Build one more graph using colConditions
        Find topological sorting order for this graph
        fill the matrix using the sorting order given by topological sort.
     */

    public int addMinimum(String word) {

        int cnt = 0;
        int ind = 0;
        while (ind < word.length()) {
            if (word.charAt(ind) == 'a') {
                // check next 2 characters
                if (ind + 1 < word.length() && ind + 2 < word.length()) {
                    if (word.charAt(ind + 1) == 'b' && word.charAt(ind + 2) == 'c') {
                        ind += 3;
                        continue; // Already 'abc'
                    } else if (word.charAt(ind + 1) == 'b' || word.charAt(ind + 1) == 'c') {
                        ind += 2;
                        cnt++; // add anyone b or c
                    } else {
                        cnt += 2; // add b,c
                        ind++;
                    }
                } else if (ind + 1 < word.length()) {
                    if (word.charAt(ind + 1) == 'b' || word.charAt(ind + 1) == 'c') {
                        ind += 2;
                        cnt++; // add c
                    } else {
                        ind++;
                        cnt += 2; // add b,c
                    }
                } else {
                    ind++;
                    cnt += 2; // add b,c
                }
            } else if (word.charAt(ind) == 'b') {
                // check 1 next
                if ((ind + 1 < word.length())) {
                    if (word.charAt(ind + 1) == 'c') {
                        ind += 2;
                        cnt++; // Add 'a'
                    } else {
                        ind++;
                        cnt += 2; // add b,c
                    }
                } else {
                    ind++;
                    cnt += 2; // add b,c
                }
            } else {
                ind++;
                cnt += 2; // add b,c
            }
        }

        return cnt;
    }

    public int[] topoSort(int N, List<List<Integer>> graph) {
        Stack<Integer> stk = new Stack<>();
        int[] vis = new int[N];

        for (int i = 0; i < N; i++) {
            if (vis[i] == 0) {
                findTopoSort(i, vis, graph, stk);
            }
        }

        // Pop elements out of stack to get final result
        int[] ans = new int[N];
        int ind = 0;
        while (!stk.empty()) {
            ans[ind++] = stk.pop();
        }

        return ans;
    }

    //TC = O(N+E), SC = O(N), ASS = O(N)
    private void findTopoSort(int node, int[] vis, List<List<Integer>> graph, Stack<Integer> stk) {
        vis[node] = 1;

        for (int e : graph.get(node)) {
            if (vis[e] == 0)
                findTopoSort(e, vis, graph, stk);
        }

        stk.push(node);
    }

    //TC  = O(N+E), SC = O(N) + O(N)
    // TOPO Sort using BFS algorithm
    public List<Integer> generateTopoSort(int[][] conditions, int k) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < k; i++) graph.add(new ArrayList<>());

        int[] indegree = new int[k];
        List<Integer> ans = new ArrayList<>();


        for (int[] condition : conditions) {
            graph.get(condition[0] - 1).add(condition[1] - 1);
            indegree[condition[1]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < k; i++) {
            if (indegree[i] == 0) queue.offer(i);
        }

        while (!queue.isEmpty()) {
            int element = queue.poll();

            ans.add(element + 1);
            for (int e : graph.get(element)) {
                if (--indegree[e] == 0) queue.add(e);
            }
        }
        return ans;
    }

    public int minimumTotalPrice(int n, int[][] edges, int[] price, int[][] trips) {
        return 0;
    }

    public int[] separateDigits(int[] nums) {
        List<Integer> ans = new ArrayList<>();
        for (int num : nums) ans.addAll(digits(num));
        return ans.stream().mapToInt(x -> x).toArray();
    }

    private List<Integer> digits(int num) {
        List<Integer> d = new ArrayList<>();
        while (num > 0) {
            d.add(num % 10);
            num /= 10;
        }
        Collections.reverse(d);
        return d;
    }

    public int maxCount(int[] banned, int n, int maxSum) {
        long ms = 0L;
        int cnt = 0;
        Set<Integer> set = Arrays.stream(banned).boxed().collect(Collectors.toSet());

        for (int i = 1; i <= n; i++) {
            if (!set.contains(i)) {
                ms += i;
                if (ms > maxSum) return cnt;
                cnt++;
            }
        }

        return cnt;
    }

    // Solve for edge cases
    //Author: Anand
    // TC = O(N), SC=O(N)
    public int maximizeWin(int[] prizePositions, int k) {

        int ans = 0;
        TreeMap<Integer, Integer> treeMap = new TreeMap<>();
        for (int pp : prizePositions) treeMap.put(pp, treeMap.getOrDefault(pp, 0) + 1);


        if (k == 0) {

            int f = 0, s = 0;
            for (Map.Entry<Integer, Integer> entry : treeMap.entrySet()) {
                if (entry.getValue() >= f) {
                    s = f;
                    f = entry.getValue();
                }
            }
            return f + s;
        }


        Set<Integer> keyset = treeMap.keySet();

        List<Integer> segment1 = ptr(new ArrayList<>(keyset), k, treeMap);

        System.out.println(segment1);
        HashMap<Integer, Integer> nm = new HashMap<>();
        nm.putAll(treeMap);


        if (segment1.size() == 0) {
            return treeMap.values().stream().mapToInt(x -> x).sum();
        }
        for (Map.Entry<Integer, Integer> entry : treeMap.entrySet()) {
            if (entry.getKey() >= segment1.get(0) && entry.getKey() > segment1.get(1)) {
                nm.remove(entry.getKey());
                ans += entry.getValue();
            }
        }


        treeMap = new TreeMap<>(nm);
        keyset = treeMap.keySet();
        // 2 ptr technique for optimal segment
        List<Integer> segment2 = ptr(new ArrayList<>(keyset), k, treeMap);

        nm = new HashMap<>();
        nm.putAll(treeMap);

        for (Map.Entry<Integer, Integer> entry : treeMap.entrySet()) {
            if (entry.getKey() >= segment1.get(0) && entry.getKey() <= segment1.get(1)) {
                nm.remove(entry.getKey());
                ans += entry.getValue();
            }
        }

        return ans;
    }

    // 2 ptr technique for optimal segment
    private List<Integer> ptr(List<Integer> keyset, int k, TreeMap<Integer, Integer> treeMap) {
        int i = 0, j = 0;
        int curr = 0, max = 0;
        List<Integer> ans = new ArrayList<>();

        while (i < keyset.size() && j < keyset.size() && i <= j) {
            int s = keyset.get(j);
            if (keyset.get(j) - keyset.get(i) <= k) {
                curr += treeMap.get(s);
                j++;
            } else {
                max = Math.max(max, curr);
                ans = new ArrayList<>(Arrays.asList(i, j));
                i++;
                curr -= treeMap.get(keyset.get(i));
            }
        }

        return ans;
    }

    public long pickGifts(int[] gifts, int k) {
        long ans = 0L;
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());//max pq
        for (int gift : gifts) pq.add(gift);
        while (k-- > 0) {
            int gift = pq.poll();
            pq.add((int) Math.floor(Math.sqrt(gift)));
        }
        while (!pq.isEmpty()) ans += pq.poll();
        return ans;
    }

    public int[] vowelStrings(String[] words, int[][] queries) {

        int[] ps = new int[words.length];
        int[] ans = new int[queries.length];
        Set<Character> vowel = new HashSet<>();
        vowel.add('a');
        vowel.add('e');
        vowel.add('i');
        vowel.add('o');
        vowel.add('u');

        for (int i = 0; i < words.length; i++) {
            boolean valid = (vowel.contains(words[i].charAt(0)) && vowel.contains(words[i].charAt(words[i].length() - 1)));
            if (i == 0) ps[i] += valid ? 1 : 0;
            else ps[i] += ps[i - 1] + (valid ? 1 : 0);
        }

        int ind = 0;
        for (int[] query : queries) {
            ans[ind++] = ps[query[1]] - (query[0] > 0 ? ps[query[0] - 1] : 0);
        }
        return ans;
    }

    public long findTheArrayConcVal(int[] nums) {
        long ans = 0L;
        int last = nums.length % 2 != 0 ? (int) Math.ceil((double) nums.length / 2) : nums.length / 2;
        for (int i = 0; i < last; i++) {
            String eval = "";
            if (i == last - 1 && nums.length % 2 != 0) eval += nums[i];
            else eval = nums[i] + ((nums.length - 1 - i) >= 0 ? String.valueOf(nums[nums.length - 1 - i]) : "");
            ans += eval.isEmpty() ? 0 : Long.parseLong(eval);
        }
        return ans;
    }

    public int minMaxDifference(int num) {
        String nums = String.valueOf(num);

        int maxi = -1, mini = -1;
        for (int i = 0; i < nums.length(); i++) {
            if (nums.charAt(i) != '9' && maxi == -1) maxi = i;
            if (nums.charAt(i) != '0' && mini == -1) mini = i;
        }

        int max = maxi != -1 ? Integer.parseInt(nums.replaceAll(String.valueOf(nums.charAt(maxi)), "9")) : num;
        int min = mini != -1 ? Integer.parseInt(nums.replaceAll(String.valueOf(nums.charAt(mini)), "0")) : 0;

        return max - min;
    }

    // TODO: Complete for correct answer
    public int minimizeSum(int[] nums) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();
        for (int num : nums) tm.put(num, tm.getOrDefault(num, 0) + 1);
        int avg = (int) Math.ceil((double) Arrays.stream(nums).sum() / nums.length);
        Arrays.sort(nums);

        boolean first = false;
        if (Math.abs(avg - nums[0]) > Math.abs(avg - nums[nums.length - 1])) {
            nums[0] = avg;
            first = true;
        } else nums[nums.length - 1] = avg;


        if (first) {
            if (Math.abs(avg - nums[1]) > Math.abs(avg - nums[nums.length - 1])) {
                nums[1] = avg;
            } else nums[nums.length - 1] = avg;
        } else {
            if (Math.abs(avg - nums[0]) > Math.abs(avg - nums[nums.length - 2])) {
                nums[0] = avg;
            } else nums[nums.length - 2] = avg;
        }

        PriorityQueue<Integer> pq = new PriorityQueue<>();
        PriorityQueue<Integer> maxpq = new PriorityQueue<>(Collections.reverseOrder());

        for (int num : nums) {
            pq.add(num);
            maxpq.add(num);
        }
        System.out.println(Arrays.toString(nums));
        int max = maxpq.poll();
        int min = pq.poll();
        System.out.println(max + ":" + min);
        return Math.abs(max - min);
    }

    /*
    Input: nums1 = [[1,2],[2,3],[4,5]], nums2 = [[1,4],[3,2],[4,1]]
    Output: [[1,6],[2,3],[3,2],[4,6]]
    Explanation: The resulting array contains the following:
    - id = 1, the value of this id is 2 + 4 = 6.
    - id = 2, the value of this id is 3.
    - id = 3, the value of this id is 2.
    - id = 4, the value of this id is 5 + 1 = 6.

     */
    public int[][] mergeArrays(int[][] nums1, int[][] nums2) {
        TreeMap<Integer, Integer> tm = new TreeMap<>();

        for (int i = 0; i < Math.max(nums1.length, nums2.length); i++) {
            if (i < nums1.length) tm.put(nums1[i][0], tm.getOrDefault(nums1[i][0], 0) + nums1[i][1]);
            if (i < nums2.length) tm.put(nums2[i][0], tm.getOrDefault(nums2[i][0], 0) + nums2[i][1]);
        }
        int[][] array = new int[tm.size()][2];
        int count = 0;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            array[count][0] = entry.getKey();
            array[count][1] = entry.getValue();
            count++;
        }

        return array;
    }

    public int minOperations(int n) {
        int op = 0;
        while (n != 0) {
            if (n == 1) {
                op++;
                break;
            }
            int num = 1;
            while (num <= n) num *= 2;
            num /= 2;
            n = Math.min(n - num, num * 2 - n);
            op++;
        }
        return op;
    }

    public int[] leftRigthDifference(int[] nums) {
        int[] ls = new int[nums.length];
        int[] rs = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            ls[i] += nums[i] + (i == 0 ? 0 : ls[i - 1]);
            rs[nums.length - 1 - i] += nums[nums.length - 1 - i] + (i == 0 ? 0 : rs[nums.length - i]);
        }

        int[] ans = new int[nums.length];
        for (int i = 0; i < nums.length; i++) ans[i] = Math.abs(ls[i] - rs[i]);
        return ans;
    }

    //TLE
    public int[] divisibilityArray(String word, int m) {
        int[] ans = new int[word.length()];
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < word.length(); i++) {
            sb.append(word.charAt(i));
            // Converting long to BigInteger
            BigInteger b = new BigInteger(sb.toString());
            ans[i] += b.remainder(new BigInteger(String.valueOf(m))).equals(new BigInteger(String.valueOf(0))) ? 1 : 0;
        }

        return ans;
    }

    public String makeSmallestPalindrome(String s) {
        StringBuilder ans = new StringBuilder();
        ans.append(s);
        for (int i = 0; i < s.length() / 2; i++) {
            if (s.charAt(i) != s.charAt(s.length() - 1 - i)) {
                int min = Math.min(s.charAt(i) - 'a', s.charAt(s.length() - 1 - i) - 'a');
                ans.replace(i, i + 1, String.valueOf((char) (min + 'a')));
                ans.replace(s.length() - 1 - i, s.length() - i, String.valueOf((char) (min + 'a')));
            }
        }

        return ans.toString();
    }

    public int buyChoco(int[] prices, int money) {
        Arrays.sort(prices);
        if (prices.length <= 1) return money;
        if (prices[0] + prices[1] > money) return money;
        return money - (prices[0] + prices[1]);
    }

    public long maxStrength(int[] nums) {

        List<Integer> pos = new ArrayList<>();
        List<Integer> neg = new ArrayList<>();
        long ans = 1L;

        boolean flag = false, zero = false;
        for (int num : nums) {
            if (num > 0) pos.add(num);
            else if (num < 0) neg.add(Math.abs((num)));
            else zero = true;
        }


        for (int n : pos) {
            flag = true;
            ans *= n;
        }

        if (neg.size() % 2 != 0 && neg.size() > 1) {
            neg.sort(Collections.reverseOrder());
            neg.remove(neg.size() - 1); // remove the smallest guy
        }


        if (neg.size() > 1) {
            for (int n : neg) {
                flag = true;
                ans *= n;
            }
        }

        if (neg.size() == 1 && zero && pos.size() == 0) return 0;
        if (neg.size() == 1 && pos.size() == 0) return -neg.get(0);
        if (flag) return Math.max(ans, 0);
        return 0;
    }

    public String removeTrailingZeros(String num) {
        return num.replaceAll("0+$", "");
    }


    // At most 1 consecutive pair

    /*
    Input: s = "0011"
    Output: 2
     Explanation: Apply the second operation with i = 2 to obtain s = "0000" for a cost of 2.
     It can be shown that 2 is the minimum cost to make all characters equal.
     */
    public long minimumCost(String s) {
        long ans = 0;
        int n = s.length();
        for (int i = 0; i < s.length() - 1; ++i) {
            if (s.charAt(i) != s.charAt(i + 1)) ans += Math.min(i + 1, n - i - 1);
        }
        return ans;
    }

    public boolean isFascinating(int n) {
        String number = n + "" + 2 * n + "" + 3 * n;
        Set<Integer> all = new HashSet<>();
        for (int i = 1; i <= 9; i++) all.add(i);

        for (int i = 0; i < number.length(); i++) {
            if (all.contains((int) (number.charAt(i) - '0'))) all.remove(number.charAt(i) - '0');
            else return false;
        }
        return all.isEmpty();
    }

    public int longestSemiRepetitiveSubstring(String s) {
        int maxLen = 1;
        Set<String> ss = new HashSet<>();

        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j <= s.length(); j++) {
                String sub = s.substring(i, j);
                ss.add(sub);
            }
        }

        Collections.sort(new ArrayList<String>(ss), Comparator.comparingInt(String::length).reversed());
        for (String str : ss) if (repetitive(str)) return str.length();

        return maxLen;
    }

    private boolean repetitive(String str) {
        boolean flag = false;
        for (int i = 1; i < str.length(); i++) {
            char c = str.charAt(i);
            int cd = c - '0';
            int pd = str.charAt(i - 1) - '0';
            if (cd == pd) {
                if (!flag) flag = true;
                else return false;
            }
        }

        return true;
    }

    public int sumDistance(int[] nums, String s, int d) {
        final int modulo = (int) 1e9 + 7;
        final int n = nums.length;
        long[] position = new long[n];
        for (int i = 0; i < n; i++) {
            position[i] = (s.charAt(i) == 'L' ? -1L : 1L) * d + nums[i];
        }

        Arrays.sort(position);
        long distance = 0;
        long prefix = 0;
        for (int i = 0; i < n; i++) {
            distance = (distance + (i * position[i] - prefix)) % modulo;
            prefix += position[i];
        }

        return (int) distance;
    }

    public int findNonMinOrMax(int[] nums) {
        if (nums.length <= 2) return -1;
        Arrays.sort(nums);
        return nums[1];
    }

    public String smallestString(String s) {
        boolean operation = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == 'a') {
                if (operation) {
                    sb.append(s.substring(i));
                    return sb.toString();
                } else sb.append(c);
            } else {
                int cc = (c - 'a' - 1);
                sb.append((char) ('a' + cc));
                operation = true;
            }
        }

        if (!operation) return s.substring(0, s.length() - 1) + 'z';
        return sb.toString();
    }

    // ----------------------------------------------------------------------------------------------------
    // 21st May LC
    public int minLength(String s) {
        while (s.contains("AB") || s.contains("CD")) {
            s = s.replaceAll("AB", "");
            s = s.replaceAll("CD", "");
        }
        return s.length();
    }

    //TODO: Correct solution
    public int punishmentNumber(int n) {

        int pn = 0;
        for (int i = 1; i <= n; i++) {
            int num = i * i;
            if (helper(String.valueOf(num), 0, new StringBuilder(), 0)) {
                System.out.println(i);
                pn += num;
            }
        }

        return pn;
    }

    //checks if square of number can be partioned such that sum of partitioned is equal to number itself
    private boolean helper(String num, int sum, StringBuilder sb, int ind) {

        // base case
        if (ind >= num.length()) {
            try {
                sum += sb.length() > 0 ? Integer.parseInt(sb.toString()) : 0;
            } catch (Throwable t) {
            } finally {
                return sum == Math.sqrt(Integer.parseInt(num));
            }
        }


        for (int i = ind; i < num.length(); i++) {
            //partition
            int add = 0;
            try {
                add += sb.length() > 0 ? Integer.parseInt(sb.toString()) : 0;
            } catch (Throwable t) {
            }


            if (helper(num, sum + add, sb.length() > 0 ? new StringBuilder().append(num.charAt(i)) : sb.append(num.charAt(i)), i + 1))
                return true;


            // not partition
            if (helper(num, sum, sb.append(num.charAt(i)), i + 1)) return true;
        }

        return false;
    }

    public int minimizedStringLength(String s) {
        Set<Character> set = new HashSet<>();
        for (char c : s.toCharArray()) set.add(c);
        return set.size();
    }

    public int semiOrderedPermutation(int[] nums) {

        int n = nums.length;
        int pos1 = -1, posn = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) pos1 = i;
            else if (nums[i] == n) posn = i;
        }


        int swaps = 0;
        while (true) {
            if (pos1 == 0 && posn == nums.length - 1) return swaps;
            // set 1
            if (pos1 > 0) {
                if (nums[pos1 - 1] == n) posn++;
                pos1--;
                swaps++;
            }
            // set n
            if (posn < nums.length - 1) {
                if (nums[posn + 1] == 1) pos1--;
                posn++;
                swaps++;
            }
        }
    }

    //TLE
    public long matrixSumQueries(int n, int[][] queries) {


        long ans = 0L;
        int[][] fm = new int[n][n];
        Map<List<Integer>, List<Integer>> map = new java.util.HashMap<>();

        int cnt = 0;
        for (int[] query : queries) {
            int type = query[0];
            int ind = query[1];
            int val = query[2];
            //set current row
            if (type == 0) {
                map.put(new ArrayList<>(Arrays.asList(ind, -1)), new ArrayList<>(Arrays.asList(val, cnt++)));
            }
            //set current column
            else map.put(new ArrayList<>(Arrays.asList(-1, ind)), new ArrayList<>(Arrays.asList(val, cnt++)));
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                List<Integer> rq = map.getOrDefault(new ArrayList<>(Arrays.asList(i, -1)), new ArrayList<>());
                List<Integer> cq = map.getOrDefault(new ArrayList<>(Arrays.asList(-1, j)), new ArrayList<>());

                if (rq.size() == 2) {
                    if (cq.size() == 2) {
                        if (rq.get(1) > cq.get(1)) {
                            fm[i][j] = rq.get(0);
                        } else fm[i][j] = cq.get(0);
                    } else fm[i][j] = rq.get(0);
                } else if (cq.size() == 2) {
                    fm[i][j] = cq.get(0);
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ans += fm[i][j];
            }
        }

        return ans;
    }

    public int distanceTraveled(int mainTank, int additionalTank) {

        int distance = 0;
        while (mainTank > 0) {
            if (mainTank >= 5) {
                mainTank -= 5;
                distance += 50;
                if (additionalTank > 0) {
                    additionalTank--;
                    mainTank++;
                }
            } else {
                distance += mainTank * 10;
                mainTank = 0;
                additionalTank = 0;
            }
        }
        return distance;
    }

    public int findValueOfPartition(int[] nums) {
        int min = Integer.MAX_VALUE;
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i++) {
            min = Math.min(Math.abs(nums[i - 1] - nums[i]), min);
        }

        return min;
    }

    public int maximumNumberOfStringPairs(String[] words) {

        int cnt = 0;
        Set<String> seen = new HashSet<>();

        for (String word : words) {
            String rev = new StringBuilder(word).reverse().toString();
            if (seen.contains(rev)) {
                cnt++;
                seen.remove(rev);
            } else {
                seen.add(word);
            }
        }

        return cnt;
    }

    //TBD
    public int longestString(int x, int y, int z) {

        return Math.max(Math.max(ml(new StringBuilder().append("AA"), x - 1, y, z, 2),
                        ml(new StringBuilder().append("BB"), x, y - 1, z, 2)),
                ml(new StringBuilder().append("AB"), x, y, z - 1, 2));
    }

    private int ml(StringBuilder sb, int x, int y, int z, int maxLen) {
        // base case
        if (x <= 0 && y <= 0 && z <= 0) {
            return maxLen;
        }

        // System.out.println(sb + ":" + x + ":" + y + ":" + z);

        while (true) {

            boolean lastStatus = false;
            if (x > 0 && !sb.substring(sb.length() - 2, sb.length()).equals("AA")) {
                lastStatus = true;
                maxLen = Math.max(maxLen, ml(sb.append("AA"), x - 1, y, z, maxLen));
            }

            if (y > 0 && !sb.substring(sb.length() - 1, sb.length()).equals("B")) {
                lastStatus = true;
                maxLen = Math.max(maxLen, ml(sb.append("BB"), x, y - 1, z, maxLen));
            }


            if (z > 0 && !sb.substring(sb.length() - 2, sb.length()).equals("AA")) {
                lastStatus = true;
                maxLen = Math.max(maxLen, ml(sb.append("AB"), x, y, z - 1, maxLen));
            }


            if (!lastStatus) return maxLen;

        }
    }

    public int countBeautifulPairs(int[] nums) {

        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int first = Integer.parseInt(String.valueOf(String.valueOf(nums[i]).charAt(0)));
                int last = Integer.parseInt(String.valueOf(String.valueOf(nums[j]).charAt(String.valueOf(nums[j]).length() - 1)));
                if (gcd(first, last) == 1) cnt++;
            }
        }

        System.out.println(gcd(72, 74));
        return cnt;
    }

    //int version for gcd
    public int gcd(int a, int b) {
        if (b == 0)
            return a;

        return gcd(b, a % b);
    }

    public int makeTheIntegerZero(int num1, int num2) {
        int ans = helper(num1, num2, 0, 0, new HashSet<Integer>());
        return ans == Integer.MAX_VALUE ? -1 : ans;
    }

    /*
        Return min operations to make num1 = 0
     */
    private int helper(int num1, int num2, int op, int ind, Set<Integer> vis) {

        // base case
        if (num1 == 0) return 0;
        int min = Integer.MAX_VALUE;

        if (vis.contains(num1)) return Integer.MAX_VALUE;

        for (int i = ind; i < 60; i++) {
            // take i
            int subract = (int) Math.pow(2, i) + num2;
            vis.add(num1 - subract);
            min = Math.min(helper(num1 - subract, num2, op + 1, 0, vis), min);
            vis.remove(num1 - subract); //backtrack
            // not take i
            min = Math.min(helper(num1, num2, op, i + 1, vis), min);
        }

        return min;
    }

    //----------------------------------------------------------------------------

    /*
    Input: nums = [3,2,5,4], threshold = 5
    Output: 3
    Explanation: In this example, we can select the subarray that starts at l = 1 and ends at r = 3 => [2,5,4]. This subarray satisfies the conditions.
    Hence, the answer is the length of the subarray, 3. We can show that 3 is the maximum possible achievable length.

     */
    public int longestAlternatingSubarray(int[] nums, int threshold) {

        int max = 0;
        for (int i = 0; i < nums.length; i++) {
            int len = 0;
            if (nums[i] % 2 == 0 && nums[i] <= threshold) len++;
            else continue;
            for (int j = i + 1; j < nums.length; j++) {
                if ((nums[j - 1] % 2 != nums[j] % 2) && nums[j] <= threshold) len++;
                else {
                    max = Math.max(max, len);
                    break;
                }
            }
            max = Math.max(max, len);
        }

        return max;
    }

    private void sieveOfEratosthenes(long num, int[] s) {
        // Create a boolean array
        // "prime[0..n]"  and initialize
        // all entries in it as false.
        boolean[] prime = new boolean[(int) (num + 1L)];

        // Initializing smallest
        // factor equal to 2
        // for all the even numbers
        for (int i = 2; i <= num; i += 2)
            s[i] = 2;

        // For odd numbers less
        // then equal to n
        for (int i = 3; i <= num; i += 2) {
            if (!prime[i]) {
                // s(i) for a prime is
                // the number itself
                s[i] = i;

                // For all multiples of
                // current prime number
                for (int j = i; (long) j * i <= num; j += 2) {
                    if (!prime[i * j]) {
                        prime[i * j] = true;

                        // i is the smallest prime
                        // factor for number "i*j".
                        s[i * j] = i;
                    }
                }
            }
        }
    }

    public int sumImbalanceNumbers(int[] nums) {
        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            System.out.println("nums[i]=" + nums[i]);
            TreeMap<Integer, Integer> tm = new TreeMap<>(); // elements along with their count
            int start = 0;

            tm.put(nums[i], tm.getOrDefault(nums[i], 0) + 1);
            for (int j = i + 1; j < nums.length; j++) {
                System.out.println("nums[j]=" + nums[j]);
                tm.put(nums[j], tm.getOrDefault(nums[j], 0) + 1);

                System.out.println("Before=" + tm);

                if (nums[j] < tm.lastKey() && tm.size() > 2) {
                    int f = tm.lowerKey(nums[j]) != null ? tm.lowerKey(nums[j]) : tm.floorKey(nums[j]);
                    int c = tm.higherKey(nums[j]) != null ? tm.higherKey(nums[j]) : tm.ceilingKey(nums[j]);
                    cnt += start;
                    if (c - f > 1) cnt--;
                    if (nums[j] - f > 1) cnt++;
                    if (c - nums[j] > 1) cnt++;
                } else {
                    if (tm.get(nums[j]) > 1 && j > (i + 1) && start >= 1) {

                        if (nums[j] < tm.lastKey() && tm.size() > 2) {
                            int f = tm.lowerKey(nums[j]) != null ? tm.lowerKey(nums[j]) : tm.floorKey(nums[j]);
                            int c = tm.higherKey(nums[j]) != null ? tm.higherKey(nums[j]) : tm.ceilingKey(nums[j]);
                            cnt += start;
                            if (c - f > 1) cnt--;

                            if (nums[j] - f > 1) cnt++;
                            if (c - nums[j] > 1) cnt++;
                        } else cnt++;


                        System.out.println(tm);
                        System.out.println(cnt);
                        continue;
                    }

                    if (tm.lastEntry().getValue() > 1) {

                        if (nums[j] >= tm.lastKey()) cnt += start;
                        else {
                            if (nums[j] < tm.lastKey() && tm.size() > 2) {
                                int f = tm.lowerKey(nums[j]) != null ? tm.lowerKey(nums[j]) : tm.floorKey(nums[j]);
                                int c = tm.higherKey(nums[j]) != null ? tm.higherKey(nums[j]) : tm.ceilingKey(nums[j]);

                                cnt += start;
                                if (c - f > 1) cnt--;
                                if (nums[j] - f > 1) cnt++;
                                if (c - nums[j] > 1) cnt++;
                            } else {
                                int c = tm.higherKey(nums[j]);
                                System.out.println("higherKey=" + c);

                                if (c - nums[j] > 1) cnt++;
                            }

                        }
                    } else {

                        System.out.println("default");
                        System.out.println("start=" + start);

                        int lk = tm.lastKey();
                        Map.Entry<Integer, Integer> last = tm.pollLastEntry();
                        System.out.println("After poll=" + tm);
                        System.out.println("eq=" + (lk - tm.lastKey() > 1 ? 1 : 0));

                        cnt += start + (lk - tm.lastKey() > 1 ? 1 : 0);
                        start += (lk - tm.lastKey() > 1 ? 1 : 0);

                        tm.put(last.getKey(), last.getValue());
                        System.out.println("After=" + tm);
                        System.out.println(tm);
                    }
                    System.out.println(cnt);
                }
            }

        }

        return Math.max(cnt, 0);
    }

    private int lower(int[] arr, int target) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int l = 0;
        int r = arr.length - 1;
        if (target <= arr[0]) {
            return 0;
        }
        if (target > arr[r]) {
            return -1;
        }
        while (l < r) {
            int m = l + (r - l) / 2;

            if (arr[m] >= target) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return r;
    }

    public int sumOfSquares(int[] nums) {
        int ind = 1, n = nums.length, ans = 0;
        for (int num : nums) if (n % ind++ == 0) ans += num * num;
        return ans;
    }

    public boolean isGood(int[] nums) {
        int[] counter = new int[201];
        if (nums.length == 1) return false;
        for (int num : nums) counter[num - 1] += num;
        for (int i = 0; i < nums.length - 1; i++) {
            if (counter[i] == 0) return false;
            if (i == nums.length - 2 && counter[i] != (2 * (nums.length - 1))) return false;
        }
        return true;
    }

    public String sortVowels(String s) {


        StringBuilder sb = new StringBuilder();
        List<Character> list = new ArrayList<>();
        Set<Character> vowels = new HashSet<>();
        vowels.add('a');
        vowels.add('A');
        vowels.add('e');
        vowels.add('E');
        vowels.add('i');
        vowels.add('I');
        vowels.add('o');
        vowels.add('O');
        vowels.add('u');
        vowels.add('U');

        //Extract the vowels
        for (int i = 0; i < s.length(); i++) {
            if (vowels.contains(s.charAt(i))) {
                list.add(s.charAt(i));
            }
        }

        Collections.sort(list);

        int ind = 0;
        for (int i = 0; i < s.length(); i++) {
            if (vowels.contains(s.charAt(i))) {
                sb.append(list.get(ind++));
            } else sb.append(s.charAt(i));
        }

        return sb.toString();
    }


    // 23rd  july

    public List<String> splitWordsBySeparator(List<String> words, char separator) {
        List<String> ans = new ArrayList<>();
        for (String word : words) {
            List<String> arr = new ArrayList<>();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < word.length(); i++) {
                if (word.charAt(i) == separator) {
                    if (sb.length() > 0) {
                        arr.add(sb.toString());
                        sb = new StringBuilder();
                    }
                } else sb.append(word.charAt(i));
            }

            if (sb.length() > 0) arr.add(sb.toString());

            System.out.println(arr);
            ans.addAll(arr);
        }

        return ans;
    }

    public int makePrefSumNonNegative(int[] nums) {
        int cnt = 0;

        while (true) {
            long ps = 0L;
            List<Integer> displaced = new ArrayList<>();
            List<Integer> queue = new ArrayList<>(); // insertion-deletion takes place at both ends

            TreeMap<Integer, Integer> neg = new TreeMap();

            System.out.println(Arrays.toString(nums));
            boolean dis = false;
            for (int num : nums) {

                if (num < 0) neg.put(num, neg.getOrDefault(num, 0) + 1);
                if (num + ps < 0) {
                    int removed = neg.firstKey();
                    ps -= removed;
                    queue.remove(new Integer(removed));
                    displaced.add(removed);
                    dis = true;
                    neg.put(removed, neg.get(removed) - 1);
                    if (neg.get(removed) <= 0) neg.remove(removed);
                    cnt++;
                } else {
                    queue.add(num);
                    ps += num;
                }
                System.out.println("ps=" + ps);
            }
            queue.addAll(displaced);
            if (!dis) break;
            nums = queue.stream().mapToInt(x -> x).toArray();
        }
        return cnt;
    }

    public int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
        int cnt = 0;
        for (int hour : hours) if (hour >= target) cnt++;
        return cnt;
    }

    public int countCompleteSubarrays(int[] nums) {

        int cnt = 0;
        int total = Arrays.stream(nums).distinct().toArray().length;
        for (int i = 0; i < nums.length; i++) {
            Set<Integer> set = new HashSet<>();
            set.add(nums[i]);
            for (int j = i + 1; j < nums.length; j++) {
                set.add(nums[j]);
                if (set.size() == total) cnt++;
            }
        }
        return cnt;
    }

    public String minimumString(String a, String b, String c) {

        List<List<String>> order = new ArrayList<>();

        List<String> pos = new ArrayList<>();
        pos.add(a);
        pos.add(b);
        pos.add(c);

        order.add(pos);

        pos = new ArrayList<>();
        pos.add(a);
        pos.add(c);
        pos.add(b);

        order.add(pos);

        pos = new ArrayList<>();
        pos.add(b);
        pos.add(a);
        pos.add(c);

        order.add(pos);
        pos = new ArrayList<>();
        pos.add(b);
        pos.add(c);
        pos.add(a);

        order.add(pos);

        pos = new ArrayList<>();
        pos.add(c);
        pos.add(a);
        pos.add(b);

        order.add(pos);

        pos = new ArrayList<>();
        pos.add(c);
        pos.add(b);
        pos.add(a);
        order.add(pos);

        List<String> ans = new ArrayList<>();

        for (List<String> seq : order) {
            a = seq.get(0);
            b = seq.get(1);
            c = seq.get(2);
            StringBuilder sb = new StringBuilder();

            for (String s : seq) {
                if (sb.length() == 0) sb.append(s);
                else {
                    if (sb.toString().contains(s)) continue;
                    int ind;
                    int ni = -1;
                    // check if suffix matches with prefix
                    for (ind = 0; ind < s.length(); ind++) {
                        if (sb.toString().endsWith(s.substring(0, ind))) ni = ind;
                    }

                    if (s.contains(sb.toString())) {
                        sb = new StringBuilder();
                        sb.append(s);
                    } else if (s.equals(c) && s.contains(b) && s.contains(a)) {
                        sb = new StringBuilder();
                        sb.append(s);
                    } else {
                        String append = s.substring(ni);
                        sb.append(append);
                    }
                }
            }
            ans.add(sb.toString());
        }

        ans.sort((o1, o2) -> {
            if (o1.length() < o2.length()) return -1;
            if (o2.length() < o1.length()) return 1;
            return o1.compareTo(o2);
        });
        return ans.get(0);
    }

    //TODO: Find better approach
    public int maximumBeauty(int[] nums, int k) {
        Arrays.sort(nums);
        int max = 0;
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);

        System.out.println(Arrays.toString(nums));

        if (k == 0) {
            for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
                max = Math.max(max, entry.getValue());
            }
            return max;
        }
        boolean same = true;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i - 1] != nums[i]) {
                same = false;
                break;
            }
        }

        if (same) return nums.length;

        for (int i = -100000; i <= 100000; i++) {
            int ui = bs(nums, k + i, false);

            while (ui >= 0 && ui < nums.length - 1 && nums[ui + 1] == nums[ui] && nums[ui] <= k + i) ui++;
            while (ui >= 0 && ui < nums.length && nums[ui] > i + k) ui--;
            while (ui >= 0 && ui < nums.length - 1 && nums[ui] < i + k) ui++;

            int li = bs(nums, i - k, true);
            while (li > 0 && li < nums.length && nums[li - 1] == nums[li] && nums[li - 1] >= i - k) li--;
            while (li > 0 && li < nums.length && nums[li] < i - k) li++;

            while (li > 0 && li < nums.length && nums[li] > i - k) li--;

            ui = Math.min(nums.length - 1, ui);
            li = Math.max(0, li);
            // System.out.println(li + ":" + ui + ":" + i);
            if (freq.containsKey(k + i) && freq.containsKey(i - k)
                    && (freq.get(k + i) > 1 || freq.get(i - k) > 1)) {
                max = Math.max(Math.abs(ui - li) + 1, max);
            } else max = Math.max(Math.abs(ui - li), max);
        }
        return Math.min(max, nums.length);
    }

    /*
    Input: nums = [4,9,6,10]
    Output: true
    Explanation: In the first operation: Pick i = 0 and p = 3, and then subtract 3 from nums[0], so that nums becomes [1,9,6,10].
    In the second operation: i = 1, p = 7, subtract 7 from nums[1], so nums becomes equal to [1,2,6,10].
    After the second operation, nums is sorted in strictly increasing order, so the answer is true.
     */

    private int bs(int[] ps, int bus, boolean lower) {
        int l = 0, h = ps.length - 1;
        while (l <= h) {
            int mid = l + (h - l) / 2;

            if (ps[mid] < bus) {
                l = mid + 1;
            } else if (ps[mid] > bus) h = mid;
            else {
                if (lower) return mid;
                return mid + 1;
            }

            if (l == h && l == mid) {
                return l;
            }

        }
        return l;
    }

    public int accountBalanceAfterPurchase(int purchaseAmount) {
        int roundedAmount = purchaseAmount / 10;
        int multiple = purchaseAmount % 10;
        if (multiple >= 5) return Math.abs(100 - 10 * (roundedAmount + 1));
        else return Math.abs(100 - 10 * (roundedAmount));
    }

    public boolean primeSubOperation(int[] nums) {

        primeSieve(1000);
        TreeMap<Integer, Boolean> primeNumbers = new TreeMap<>();

        System.out.println(primeNumbers);
        for (int i = 0; i < nums.length; i++) {
            if (primeNumbers.firstKey() >= nums[i]) continue;
            int prime = primeNumbers.lowerKey(nums[i]);
            if (i == 0) {
                nums[i] -= prime;
                continue;
            }

            while (nums[i - 1] >= (nums[i] - prime)) {
                if (primeNumbers.firstKey() >= prime) break;
                prime = primeNumbers.lowerKey(prime);
            }

            if (nums[i - 1] < (nums[i] - prime)) nums[i] -= prime;
        }


        for (int i = 0; i < nums.length; i++) {
            if (i == 0) continue;
            if (nums[i - 1] >= nums[i]) {
                return false;
            }
        }

        return true;
    }

    //Optimise solution
    public int minimumSeconds(List<Integer> nums) {

        Map<Integer, Integer> freq = new HashMap();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);
        freq = sortByValue((HashMap<Integer, Integer>) freq);
        System.out.println(freq);

        int d = -1;
        if (new HashSet<>(freq.values()).size() == 1) d = freq.keySet().stream().findAny().get();
        int n = nums.size();
        int ans = 0;

        while (freq.size() != 1) {
            List<Integer> nl = new ArrayList<>();

            for (int i = 0; i < n; i++) {

                if (freq.get(nums.get((i - 1 + n) % n)) < freq.get(nums.get((i + 1) % n))) {
                    if (freq.get(nums.get((i + 1) % n)) > freq.get(nums.get(i))) {
                        nl.add(nums.get((i + 1) % n));
                    } else nl.add(nums.get(i));
                } else if (freq.get(nums.get((i - 1 + n) % n)) > freq.get(nums.get((i + 1) % n))) {
                    if (freq.get(nums.get((i - 1 + n) % n)) > freq.get(nums.get(i))) {
                        nl.add(nums.get((i - 1 + n) % n));
                    } else nl.add(nums.get(i));
                } else {
                    if (freq.get(nums.get((i - 1 + n) % n)) > freq.get(nums.get(i))) {
                        nl.add(nums.get((i - 1 + n) % n));
                    } else if (freq.get(nums.get((i - 1 + n) % n)) < freq.get(nums.get(i)) || d == -1)
                        nl.add(nums.get(i));
                    else nl.add(d);
                }

                System.out.println(Arrays.toString(nl.toArray()));

            }

            nums.clear();
            nums.addAll(nl);

            System.out.println(Arrays.toString(nums.toArray()));
            freq = new HashMap();
            for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);
            freq = sortByValue((HashMap<Integer, Integer>) freq);
            ans++;
        }

        return ans;
    }

    public String finalString(String s) {

        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == 'i') sb.reverse();
            else sb.append(c);
        }
        return sb.toString();
    }

    public boolean canSplitArray(List<Integer> nums, int m) {
        int n = nums.size();
        while (true) {

            boolean wassplit = false;
            if (subArrays.isEmpty()) {
                if (split(nums, m, subArrays)) {
                    wassplit = true;
                }
            } else {
                List<List<Integer>> copy = new ArrayList<>();
                copy.addAll(subArrays);
                for (List<Integer> list : subArrays) {
                    if (split(list, m, copy)) {
                        copy.remove(list);
                        wassplit = true;
                    }
                }

                subArrays = copy;
            }

            System.out.println(Arrays.deepToString(subArrays.toArray()));

            if (subArrays.size() == n) return true;
            if (!wassplit) return false;
        }
    }

    private boolean split(List<Integer> nums, int m, List<List<Integer>> subArrays) {
        int ts = Arrays.stream(nums.stream().mapToInt(x -> x).toArray()).sum();
        int ssf = 0;
        for (int i = 0; i < nums.size(); i++) {
            ssf += nums.get(i);
            if ((ssf >= m || i == 0) && ((ts - ssf) >= m || i == nums.size() - 2)) {
                // make a split

                subArrays.add(nums.subList(0, i + 1));
                subArrays.add(nums.subList(i + 1, nums.size()));
                return true;
            }
        }
        return false;
    }

    public int minimumRightShifts(List<Integer> nums) {
        int op = 0;
        List<Integer> nl = new ArrayList<>();
        nl.addAll(nums);
        if (sorted(nl)) return op;
        rs(nl);
        op++;

        while (true) {
            if (sorted(nl)) return op;
            if (nl.equals(nums)) return -1;
            rs(nl);
            op++;
        }
    }

    private void rs(List<Integer> nl) {
        int n = nl.size();
        int l = nl.get(n - 1);
        for (int i = nl.size() - 1; i > 0; i--) nl.set(i, nl.get(i - 1));
        nl.set(0, l);
    }

    private boolean sorted(List<Integer> nl) {
        for (int i = 1; i < nl.size(); i++) if (nl.get(i) < nl.get(i - 1)) return false;
        return true;
    }

    /* A utility function to check if i,j are
       valid indexes for N*N chessboard */
    boolean isSafe(int x, int y, int[][] sol) {
        return (x >= 0 && x < N && y >= 0 && y < N
                && sol[x][y] == -1);
    }

    /* A utility function to print solution
       matrix sol[N][N] */
    void printSolution(int[][] sol) {
        for (int x = 0; x < N; x++) {
            for (int y = 0; y < N; y++)
                System.out.print(sol[x][y] + " ");
            System.out.println();
        }
    }

    /* This function solves the Knight Tour problem
       using Backtracking.  This  function mainly
       uses solveKTUtil() to solve the problem. It
       returns false if no complete tour is possible,
       otherwise return true and prints the tour.
       Please note that there may be more than one
       solutions, this function prints one of the
       feasible solutions.  */
    boolean solveKT() {
        int[][] sol = new int[8][8];

        /* Initialization of solution matrix */
        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++)
                sol[x][y] = -1;

        /* xMove[] and yMove[] define next move of Knight.
           xMove[] is for next value of x coordinate
           yMove[] is for next value of y coordinate */
        int[] xMove = {2, 1, -1, -2, -2, -1, 1, 2};
        int[] yMove = {1, 2, 2, 1, -1, -2, -2, -1};

        // Since the Knight is initially at the first block
        sol[0][0] = 0;

        /* Start from 0,0 and explore all tours using
           solveKTUtil() */
        if (!solveKTUtil(0, 0, 1, sol, xMove, yMove)) {
            System.out.println("Solution does not exist");
            return false;
        } else
            printSolution(sol);

        return true;
    }

    /* A recursive utility function to solve Knight
       Tour problem */
    boolean solveKTUtil(int x, int y, int movei,
                        int[][] sol, int[] xMove,
                        int[] yMove) {
        int k, next_x, next_y;
        if (movei == N * N)
            return true;

        /* Try all next moves from the current coordinate
            x, y */
        for (k = 0; k < 8; k++) {
            next_x = x + xMove[k];
            next_y = y + yMove[k];
            if (isSafe(next_x, next_y, sol)) {
                sol[next_x][next_y] = movei;
                if (solveKTUtil(next_x, next_y, movei + 1,
                        sol, xMove, yMove))
                    return true;
                else
                    sol[next_x][next_y]
                            = -1; // backtracking
            }
        }

        return false;
    }


    public boolean checkValidGrid(int[][] grid) {

        this.N = grid.length;
        // Function Call
        return solveKT();

    }

    public int sumIndicesWithKSetBits(List<Integer> nums, int k) {
        int sum = 0;
        int ind = 0;
        for (int num : nums) {
            if (Integer.bitCount(ind++) == k) sum += num;
        }
        return sum;
    }

    public int countWays(List<Integer> nums) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int num : nums) pq.add(num);
        int cnt = 0;
        int total = 0;
        boolean first = false;
        while (!pq.isEmpty()) {
            int elem = pq.poll();

            if (!first) {
                if (elem != 0) cnt++; // no element is selected
                first = true;
            }

            total++;
            if (elem < total) {
                if (pq.size() == 0) cnt++;
                else if (pq.peek() > total) cnt++;
            }
        }
        return cnt;
    }

    // 24th september

    public String maximumOddBinaryNumber(String s) {
        long one = s.chars().filter(c -> c == '1').count();
        int len = s.length();
        StringBuilder sb = new StringBuilder();
        while (len-- > 1) {
            if (one > 1) {
                sb.append('1');
                one--;
            } else sb.append('0');
        }
        if (one == 1) sb.append('1');
        return sb.toString();
    }

    public long maximumSumOfHeights(List<Integer> maxHeights) {
        long ans = 0;
        for (int idx = 0; idx < maxHeights.size(); idx++)
            ans = Math.max(ans, mr(maxHeights, idx));
        return ans;
    }

    private long mr(List<Integer> maxHeights, int ind) {

        long res = 0;

        res += maxHeights.get(ind);
        // left half
        int li = ind;
        int lv = Integer.MAX_VALUE;
        if (li > 0) lv = Math.min(maxHeights.get(li), maxHeights.get(li - 1));

        while (li-- > 0) {
            lv = Math.min(maxHeights.get(li), lv);
            res += lv;
        }


        // right half
        int ri = ind;
        int rv = Integer.MAX_VALUE;
        if (ri < (maxHeights.size() - 1)) rv = Math.min(maxHeights.get(ri), maxHeights.get(ri + 1));

        while (ri++ < maxHeights.size() - 1) {
            rv = Math.min(maxHeights.get(ri), rv);
            res += rv;
        }
        return res;
    }

    // 30th Sep
    public int minOperations(List<Integer> nums, int k) {
        int op = 0;
        Map<Integer, Boolean> tm = new HashMap<>();
        for (int i = nums.size() - 1; i >= 0; i--) {
            int curr = nums.get(i);
            op++;
            if (curr >= 1 && curr <= k) tm.putIfAbsent(curr, true);
            // check
            if (tm.size() == k) return op;
        }
        return -1;
    }


    public int minOperations(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);
        int op = 0;
        for (int e : freq.values()) {
            if (e == 1) return -1;
            op += (e / 3) + (e % 3 != 0 ? 1 : 0);
        }
        return op;
    }


    //TODO: Complete this
    class Solution {
        public int maxSubarrays(int[] nums) {

            int split = 0;
            int score = -1;

            Set<Integer> set = Arrays.stream(nums).boxed().collect(Collectors.toSet());
            if (set.size() == 1) {
                return nums.length;
            }

            for (int i = 0; i < nums.length; i++) {
                if (i == 0) {
                    score = nums[i];
                    System.out.println(score);
                    continue;
                }
                int prev = score & nums[i];
                int next = i < nums.length - 1 ? nums[i] & nums[i + 1] : Integer.MAX_VALUE;

                if (prev < score || (prev == score && prev != 0)) {
                    score = prev;
                    System.out.println(score);
                    if (score == 0) {
                        score = i < nums.length - 1 ? nums[i + 1] : 0;
                        i++;
                        split++;
                    }
                    continue;
                }
                if (prev - score >= next) {
                    score = nums[i] & nums[i + 1];
                    i++;
                } else {
                    score = prev;
                }
                System.out.println("prev=" + prev + ", score=" + score + ",next=" + next + ",nums[i]=" + nums[i]);
                split++;
            }

            return split == 0 ? 1 : split;
        }
    }

    /*
    Input: nums = [12,6,1,2,7]
    Output: 77
    Explanation: The value of the triplet (0, 2, 4) is (nums[0] - nums[2]) * nums[4] = 77.
    It can be shown that there are no ordered triplets of indices with a value greater than 77.

    Algo:-
        The idea is to calculate maximum from the left, {max, min} to the right and evaluate
        maximum value for the expression (A[i]-A[j])*A[k] where i < j < k for all {i,j,k} E [0,n-1] where n = size of array
     */
    public long maximumTripletValue(int[] nums) {
        long ans = 0L;

        int n = nums.length;
        int[] maxL = new int[n], maxR = new int[n];
        for (int i = 0; i < n; i++) {
            maxL[i] = i == 0 ? nums[i] : Math.max(maxL[i - 1], nums[i]);
            maxR[n - 1 - i] = i == 0 ? nums[n - 1 - i] : Math.max(maxR[n - i], nums[n - 1 - i]);
        }

        for (int i = 1; i < n - 1; i++)
            ans = Math.max(ans, (long) (maxL[i - 1] - nums[i]) * maxR[i + 1]);

        return ans;
    }


    public int differenceOfSums(int n, int m) {
        int ts = n * (n + 1) / 2;
        int ds = 0;
        int i = 1;
        int nm = m;
        while (nm <= n) {
            ds += nm;
            nm = m * ++i;
        }

        return ts - 2 * ds;
    }

    public int minProcessingTime(List<Integer> processorTime, List<Integer> tasks) {

        Collections.sort(processorTime);
        Collections.sort(tasks, Collections.reverseOrder());

        int cores = 0;
        int tt = 0;
        for (int pt : processorTime) {
            tt = Math.max(tt, pt + tasks.get(cores));
            cores += 4;
        }

        return tt;
    }

    // 14th Oct
    public List<Integer> lastVisitedIntegers(List<String> words) {
        List<Integer> ans = new ArrayList<>();
        List<String> digits = new ArrayList<>();
        int k = 0;
        for (String word : words) {
            if (word.equals("prev")) {
                k++;
                ans.add((digits.size() - k < digits.size() && digits.size() - k >= 0) ? Integer.parseInt(digits.get(digits.size() - k)) : -1);
            } else {
                digits.add(word);
                k = 0;
            }
        }
        return ans;
    }

    public List<String> getWordsInLongestSubsequence(int n, String[] words, int[] groups) {

        int zero = -1, one = -1;
        for (int i = 0; i < groups.length; i++) {
            if (groups[i] == 0 && zero == -1) {
                zero = i;
            } else if (one == -1) {
                one = i;
            }
        }

        List<String> z = new ArrayList<>();
        List<String> o = new ArrayList<>();
        if (zero != -1) z.addAll(helper(zero, words, groups));
        if (one != -1) o.addAll(helper(one, words, groups));
        if (z.size() > o.size()) return z;
        return o;
    }

    private List<String> helper(int ind, String[] words, int[] groups) {
        int prev = -1;
        List<String> ans = new ArrayList<>();
        for (int i = ind; i < groups.length; i++) {
            if (prev == -1) {
                ans.add(words[i]);
            } else if (prev != groups[i]) {
                ans.add(words[i]);
            }
            prev = groups[i];
        }
        return ans;
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

    class ModeBST {
        List<Integer> ans;
        int val = Integer.MIN_VALUE, freq = 0, maxFreq = 0;

        // Author: Anand
        public int[] findMode(TreeNode root) {

            ans = new ArrayList<>();
            if (root.left == null && root.right == null) return new int[]{root.val};
            helper(root);

            return ans.stream().mapToInt(x -> x).toArray();
        }

        private void helper(TreeNode root) {
            // base case
            if (root == null) return;
            helper(root.left);

            if (val == root.val) freq++;
            else {
                val = root.val;
                freq = 1;
            }

            if (freq > maxFreq) {
                maxFreq = freq;
                ans = new ArrayList<>(Arrays.asList(root.val));
            } else if (freq == maxFreq) ans.add(val);

            helper(root.right);
        }
    }


    public String shortestBeautifulSubstring(String s, int k) {
        String ans = "";
        for (int i = 0; i < s.length(); i++) {
            int cnt = 0;
            for (int j = i; j < s.length(); j++) {
                if (s.charAt(j) == '1') cnt++;
                if (cnt == k) {
                    String str = s.substring(i, j + 1);
                    if (ans.isEmpty() || str.length() < ans.length()) ans = str;
                    else if (str.length() == ans.length()) {
                        if (str.compareTo(ans) < 0) ans = str;
                    }
                }
            }
        }
        return ans;
    }

    public int[] findIndices(int[] nums, int indexDifference, int valueDifference) {
        TreeMap<Integer, int[]> tm = new TreeMap<>();//{index, new int[]{largI, smallI}}
        for (int i = nums.length - 1; i >= 0; i--) {
            if (i == nums.length - 1) tm.put(i, new int[]{i, i});
            else {
                int[] next = tm.get(i + 1);
                tm.put(i, new int[]{(nums[i] > nums[next[0]] ? i : next[0]), (nums[i] < nums[next[1]] ? i : next[1])});
            }
            // check for valid answer
            int ni = i + indexDifference;
            int[] values = ni < nums.length ? tm.get(ni) : new int[]{};
            if (values.length == 0) continue;
            if (Math.abs(nums[values[0]] - nums[i]) >= valueDifference) return new int[]{i, values[0]};
            if (Math.abs(nums[values[1]] - nums[i]) >= valueDifference) return new int[]{i, values[1]};
        }
        return new int[]{-1, -1};
    }


    public int[][] constructProductMatrix(int[][] grid) {

        int m = grid.length, n = grid[0].length;
        int[] prefixProd = new int[m * n], suffixProd = new int[m * n];
        Arrays.fill(prefixProd, 1);
        Arrays.fill(suffixProd, 1);
        List<Integer> flatten = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                flatten.add(grid[i][j] % 12345);
        }

        int p = 1, s = 1;
        for (int i = 1; i < flatten.size(); i++) {
            p = p * flatten.get(i - 1) % 12345;
            s = s * flatten.get(flatten.size() - i) % 12345;
            prefixProd[i] = p;
            suffixProd[flatten.size() - 1 - i] = s;
        }

        int ind = 0;
        int[][] ans = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ans[i][j] = prefixProd[ind] * suffixProd[ind] % 12345;
                ind++;
            }
        }
        return ans;
    }


    public int minimumSum(int[] nums) {
        int[] left = new int[nums.length], right = new int[nums.length];
        Arrays.fill(left, Integer.MAX_VALUE);
        Arrays.fill(right, Integer.MAX_VALUE);
        for (int i = 1; i < nums.length; i++) {
            left[i] = Math.min(nums[i - 1], left[i - 1]);
            right[nums.length - 1 - i] = Math.min(nums[nums.length - i], right[nums.length - i]);
        }

        int ans = Integer.MAX_VALUE;
        for (int i = 1; i < nums.length - 1; i++) {
            int cs = Integer.MAX_VALUE;
            if (left[i] != Integer.MAX_VALUE && right[i] != Integer.MAX_VALUE
                    && nums[i] > left[i] && right[i] < nums[i]
            ) cs = left[i] + right[i] + nums[i];

            ans = Math.min(ans, cs);
        }
        return ans == Integer.MAX_VALUE ? -1 : ans;
    }
    // 28th oct

    public int sumCounts(List<Integer> nums) {

        int ans = 0;
        for (int i = 0; i < nums.size(); i++) {
            Set<Integer> values = new HashSet<>();
            for (int j = i; j < nums.size(); j++) {
                values.add(nums.get(j));
                ans += values.size() * values.size();
            }
        }

        return ans;
    }


    public int minChanges(String s) {
        int c1 = 0, c0 = 0, ans = 0, len = 0;
        for (int i = 0; i < s.length(); i++) {
            int c = s.charAt(i) - '0';
            if (c1 == 0 && c0 == 0) {
                if (c == 0) c0++;
                else c1++;
                len++;
            } else {
                if (c == 0 && c1 > c0) {
                    //make o->1 
                    ans++;
                    len++;
                    // if even length then make a partition
                    if (len % 2 == 0) {
                        c1 = 0;
                        c0 = 0;
                        len = 0;
                    }
                } else if (c == 1 && c0 > c1) {
                    //make 1->0 and partition
                    ans++;
                    len++;
                    // if even length then make a partition
                    if (len % 2 == 0) {
                        c1 = 0;
                        c0 = 0;
                        len = 0;
                    }
                } else {
                    len++;
                    if (c == 1) c1++;
                    else c0++;
                    if (len % 2 == 0) {
                        c1 = 0;
                        c0 = 0;
                        len = 0;
                    }
                }
            }
        }

        return ans;
    }


    //TLE
    public int sumCounts(int[] nums) {

        int ans = 0;
        final int mod = (int) 1e9 + 7;

        for (int i = 0; i < nums.length; i++) {
            Set<Integer> values = new HashSet<>();
            for (int j = i; j < nums.length; j++) {
                values.add(nums[j]);
                ans = (ans + values.size() * values.size()) % mod;
            }
        }

        return ans;
    }

    // 29th oct
    public int findKOr(int[] nums, int k) {
        int largest = -1;
        for (int num : nums) largest = Math.max(largest, num);
        int cnt = Integer.toBinaryString(largest).length();
        int ans = 0;
        cnt--;

        while (cnt >= 0) {
            int setbit = 0;
            for (int num : nums) {
                // set bit
                if ((num & (1 << cnt)) != 0) setbit++;

                if (setbit == k) {
                    ans += (int) Math.pow(2, cnt);
                    break;
                }
            }
            cnt--;
        }

        return ans;
    }

    public int[] sortByBits(int[] arr) {
        int[] ans = new int[arr.length];

        TreeMap<Integer, List<Integer>> tm = new TreeMap<>(); // no of 1's, numbers list

        int ind = 0;
        for (int a : arr) {
            int cnt = Integer.bitCount(a);
            tm.putIfAbsent(cnt, new ArrayList<>());
            tm.get(cnt).add(a);
        }

        for (List<Integer> list : tm.values()) {
            Collections.sort(list);
            for (int l : list) ans[ind++] = l;
        }

        return ans;
    }

    public long minSum(int[] nums1, int[] nums2) {
        BigInteger bigsum1 = BigInteger.valueOf(0L);
        BigInteger bigsum2 = BigInteger.valueOf(0L);
        for (int num : nums1) bigsum1 = bigsum1.add(new BigInteger(String.valueOf(num)));
        for (int num : nums2) bigsum2 = bigsum2.add(new BigInteger(String.valueOf(num)));
        long sum1 = bigsum1.longValue();
        long sum2 = bigsum2.longValue();
        long cnt01 = 0;
        for (int num : nums1) if (num == 0) cnt01++;
        long cnt02 = 0;
        for (int num : nums2) if (num == 0) cnt02++;
        //sum1>sum2
        if (sum1 > sum2) {
            if (cnt01 == 0) {
                if (cnt02 > 0) {
                    if (sum1 - sum2 >= cnt02) return sum1;
                    else return -1;
                } else return -1;
            } else if (cnt02 > 0) {
                BigInteger bsum2 = BigInteger.valueOf(sum2);
                BigInteger bsum1 = BigInteger.valueOf(sum1);
                BigInteger bcnt01 = BigInteger.valueOf(cnt01);
                BigInteger bcnt02 = BigInteger.valueOf(cnt02);

                return Math.max(bsum1.add(bcnt01).longValue(), bsum2.add(bcnt02).longValue());
            } else return -1;
        }
        //sum2>sum1
        else if (sum2 > sum1) {
            if (cnt02 == 0) {
                if (cnt01 > 0) {
                    if (sum2 - sum1 >= cnt01) return sum2;
                    else return -1;
                } else return -1;
            } else if (cnt01 > 0) {
                BigInteger bsum2 = BigInteger.valueOf(sum2);
                BigInteger bsum1 = BigInteger.valueOf(sum1);
                BigInteger bcnt01 = BigInteger.valueOf(cnt01);
                BigInteger bcnt02 = BigInteger.valueOf(cnt02);

                return Math.max(bsum1.add(bcnt01).longValue(), bsum2.add(bcnt02).longValue());
            } else return -1;
        }
        //sum1==sum2
        else {
            if (cnt01 == cnt02) {
                return sum1 + cnt01;
            } else if (cnt01 != 0 && cnt02 != 0) {
                BigInteger bsum2 = BigInteger.valueOf(sum2);
                BigInteger bsum1 = BigInteger.valueOf(sum1);
                BigInteger bcnt01 = BigInteger.valueOf(cnt01);
                BigInteger bcnt02 = BigInteger.valueOf(cnt02);

                return Math.max(bsum1.add(bcnt01).longValue(), bsum2.add(bcnt02).longValue());
            } else return -1;
        }
    }

    public List<String> buildArray(int[] target, int n) {

        int start = 1;
        List<String> ans = new ArrayList<>();
        for (int t : target) {
            while (start != t && start < n) {
                start++;
                ans.addAll(new ArrayList<>(Arrays.asList("Push", "Pop")));
            }

            start++;
            ans.add("Push");
        }

        return ans;
    }

    public int getLastMoment(int n, int[] left, int[] right) {
        int lm = -1, rm = Integer.MAX_VALUE;
        for (int l : left) lm = Math.max(l, lm);
        for (int r : right) rm = Math.min(r, rm);
        return Math.max(lm, n - rm);
    }

    // 5th November
    public int findChampion(int[][] grid) {
        int ans = -1;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    if (ans == -1) ans = i;
                    else if (j == ans) ans = i;
                }
            }
        }

        return ans;
    }
    //Compute max by traversing wisely

    public int findChampion(int n, int[][] edges) {

        TreeMap<Integer, List<Integer>> graph = new TreeMap<>();
        for (int[] edge : edges) {
            int k = edge[0];
            int v = edge[1];
            graph.putIfAbsent(k, new ArrayList<>());
            graph.get(k).add(v);
        }

        if (edges.length == 0) {
            if (n == 1) return 0;
            return -1;
        }

        // If there is a node in isolation then return -1;
        Set<Integer> nodes = new HashSet<>();
        graph.values().forEach(x -> nodes.addAll(new ArrayList<>(x)));
        nodes.addAll(graph.keySet());
        if (nodes.size() < n) return -1;


        System.out.println(graph);


        List<Integer> ans = new ArrayList<>();
        Set<Integer> values = new HashSet<>();

        //Compute the max node here
        for (Map.Entry<Integer, List<Integer>> entry : graph.entrySet()) {

            if (ans.isEmpty()) {
                ans.add(entry.getKey());
                values.addAll(entry.getValue());
            } else {
                for (int e : entry.getValue()) if (values.contains(e)) values.remove(e);
            }
            values.add(entry.getKey());
        }

        return ans.size() != 1 ? -1 : ans.get(0);
    }

    //Daily LC
    //TODO: Solve in O(N) time
    public int eliminateMaximum(int[] dist, int[] speed) {
        // Sort the monster based on the abiltiies to reach city earliest
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> (double) a.getKey() / a.getValue()));
        for (int i = 0; i < dist.length; i++) pq.add(new Pair<>(dist[i], speed[i]));

        int cnt = 0;
        int time = 0;
        while (!pq.isEmpty()) {
            Pair<Integer, Integer> p = pq.poll();
            //Update distance of nearest monster
            Pair<Integer, Integer> nearestMonster = new Pair<>((p.getKey() - p.getValue() * time), p.getValue());
            // if nearest monster reached the city game over
            if (nearestMonster.getKey() <= 0) return cnt;
            cnt++;
            // take 1 min to charge weopon
            time++;
        }
        // Return count of monster killed
        return cnt;
    }

    public int maximumStrongPairXor(int[] nums) {

        int ans = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i; j < nums.length; j++) {
                if (Math.abs(nums[i] - nums[j]) <= Math.min(nums[i], nums[j])) {
                    ans = Math.max(ans, nums[i] ^ nums[j]);
                }
            }
        }

        return ans;
    }

    public List<String> findHighAccessEmployees(List<List<String>> access_times) {

        List<String> ans = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();// emp, access_times,
        for (List<String> at : access_times) {
            String emp = at.get(0);
            String time = at.get(1);
            map.putIfAbsent(emp, new ArrayList<>());
            map.get(emp).add(time);
        }

        Map<String, List<Integer>> sm = new HashMap<>();// emp, access_times,

        for (Map.Entry<String, List<String>> entry : map.entrySet()) {
            List<Integer> value = entry.getValue().stream()
                    .mapToInt(x -> Integer.parseInt(x.substring(0, 2)) * 60 + Integer.parseInt(x.substring(2, 4)))
                    .boxed()
                    .sorted()
                    .collect(Collectors.toList());
            sm.put(entry.getKey(), value);
        }

        for (Map.Entry<String, List<Integer>> entry : sm.entrySet()) {
            List<Integer> times = new ArrayList<>();
            for (int at : entry.getValue()) {
                // check diff is less than an hour
                if (!times.isEmpty() && Math.abs(times.get(0) - at) >= 60) times.remove(0);
                times.add(at);
                if (times.size() == 3) {
                    ans.add(entry.getKey());
                    break;
                }
            }
        }
        return ans;
    }

    // To compute min maximum sum of a pair in the array
    // we need to maintain count of min,max, element of array
    public int minPairSum(int[] nums) {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;

        for (int num : nums) {
            max = Math.max(max, num);
            min = Math.min(min, num);
        }

        int[] hash = new int[max + 1];
        for (int num : nums) hash[num]++;

        int low = min, high = max;
        int ans = Integer.MIN_VALUE;
        while (low <= high) {
            if (hash[low] == 0) low++;
            else if (hash[high] == 0) high--;
            else {
                ans = Math.max(ans, low + high);
                hash[low]--;
                hash[high]--;
            }
        }
        return ans;
    }

    public int reductionOperations(int[] nums) {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;

        for (int num : nums) {
            max = Math.max(max, num);
            min = Math.min(min, num);
        }

        int[] hash = new int[max + 1];
        for (int num : nums) hash[num]++;

        int low = min, high = max;
        int ans = 0, prev = -1;
        while (low <= high) {
            if (hash[high] != 0) {
                if (prev == -1) prev = hash[high];
                else {
                    ans += prev;
                    hash[high] += prev;
                    prev = hash[high];
                }
            }
            high--;
        }

        return ans;
    }

    //19th Nov
    //Continue to delete character from behind and count the number of operations needed
    public int findMinimumOperations(String s1, String s2, String s3) {
        if (s1.length() == s2.length() && s2.length() == s3.length()) {
            int sz = 0;
            for (int i = 0; i < s1.length(); i++) {
                if (s1.charAt(i) == s2.charAt(i) && s2.charAt(i) == s3.charAt(i)) sz++;
                else break;
            }
            if (sz > 0) return s1.length() * (s1.length() - sz);
            return s1.equals(s2) && s2.equals(s3) ? 0 : -1;
        }

        PriorityQueue<String> pq = new PriorityQueue<>(String::compareTo);
        pq.add(s1);
        pq.add(s2);
        pq.add(s3);
        int op = 0;
        while (!pq.isEmpty()) {
            String curr = pq.poll();
            String next = pq.peek();
            if (next == null) {
                break;
            } else {
                if (!next.startsWith(curr)) return -1;
                op += next.length() - curr.length();
                pq.poll();
                pq.add(curr);
            }
        }

        return op;
    }

    public int maxCoins(int[] nums) {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;

        for (int num : nums) {
            max = Math.max(max, num);
            min = Math.min(min, num);
        }

        int[] hash = new int[max + 1];
        for (int num : nums) hash[num]++;

        int low = min, high = max;
        int ans = 0;
        while (low <= high) {
            if (hash[high] == 0) high--; // haven't got max number continue search
            else if (hash[high] > 0 && hash[low] > 0) { // got max number as well min
                hash[high]--; // pass on max number
                // move to search next maximum
                if (hash[high] == 0 && high > 0) high--;
                // move to search next minimum
                while (hash[low] == 0 && low < nums.length) low++;
                hash[low]--; // pass on  min number
                // if min exists only once it's time to move to next larger number
                if (hash[low] == 0 && low < nums.length) low++;
                // if max exists only once it's time to move to next smaller number
                while (hash[high] == 0 && high > 0) high--;

                // This is second maximum, get it and reduce frequency
                if (hash[high] != 0) {
                    ans += high;
                    hash[high]--;
                }
            }

            // haven't got min number continue search
            if (hash[low] == 0) low++;
        }
        return ans;
    }

    public int[] getSumAbsoluteDifferences(int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n], ans = new int[n];

        for (int i = 0; i < n; i++)
            prefix[i] += nums[i] + (i > 0 ? prefix[i - 1] : 0);

        for (int i = 0; i < n; i++) {
            ans[i] = prefix[n - 1] + nums[i] * (2 * (i + 1) - n) - 2 * prefix[i];
        }
        return ans;
    }


    public int getWinner(int[] arr, int k) {
        Map<Integer, Integer> freq = new LinkedHashMap<>();
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            max = Math.max(arr[i], max);
            freq.put(max, freq.getOrDefault(max, 0) + 1);
        }

        for (Map.Entry<Integer, Integer> entry : freq.entrySet())
            if (entry.getValue() >= k) return entry.getKey();
        return max;
    }


    //TODO: write better code
    public int[] findDiagonalOrder(List<List<Integer>> nums) {
        ArrayList<ArrayList<Integer>> adj = new ArrayList<>();
        Map<Integer, List<Integer>> map = new LinkedHashMap<>();

        Set<Pair<Integer, Integer>> visited = new HashSet<>();
        for (int i = 0; i < nums.size(); i++) {
            List<Integer> row = nums.get(i);
            for (int j = 0; j < row.size(); j += 2) {
                map.putIfAbsent(row.get(j), new ArrayList<>());
                visited.add(new Pair<>(i, j));
                if (i + 1 < nums.size() && nums.get(i + 1).size() > j) {
                    map.get(row.get(j)).add(nums.get(i + 1).get(j));
                    visited.add(new Pair<>(i + 1, j));
                }
                if (j + 1 < row.size()) {
                    map.get(row.get(j)).add(row.get(j + 1));
                    visited.add(new Pair<>(i, j + 1));
                }
            }
        }

        System.out.println(map);
        List<Integer> result = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        int start = new ArrayList<>(map.keySet()).get(0);
        queue.add(start);
        result.add(start);
        while (!queue.isEmpty()) {
            int elem = queue.poll();
            for (int i = 0; i < (map.containsKey(elem) ? map.get(elem).size() : 0); i++) {
                //  Not visited and seats are available
                queue.add(map.get(elem).get(i));
                result.add(map.get(elem).get(i));
            }
        }
        return result.stream().mapToInt(x -> x).toArray();
    }

    public List<Integer> findWordsContaining(String[] words, char x) {
        int ind = 0;
        List<Integer> ans = new ArrayList<>();
        for (String word : words) {
            if (word.contains(String.valueOf(x)))
                ans.add(ind);
            ind++;
        }
        return ans;
    }

    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for (String word : word1) sb1.append(word);
        for (String word : word2) sb2.append(word);
        return sb1.toString().equals(sb2.toString());
    }

    public int countCharacters(String[] words, String chars) {
        Map<Character, Integer> freq = new HashMap<>();
        for (Character c : chars.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);

        int ans = 0;
        for (String word : words) {
            Map<Character, Integer> map = new HashMap<>(freq);
            boolean present = true;
            for (Character c : word.toCharArray()) {
                if (!map.containsKey(c)) {
                    present = false;
                    break;
                }
                map.put(c, map.get(c) - 1);
                if (map.get(c) <= 0) map.remove(c);
            }

            if (present) ans += word.length();
        }
        return ans;
    }
    
    

    public boolean checkPowersOfThree(int n) {
        int max = 0;
        while (Math.pow(3, max) <= n) max++;
        int sum = 0;
        while (max >= 0) {
            if (sum + Math.pow(3, max) == n) return true;
            if (sum + Math.pow(3, max) < n) sum += (int) Math.pow(3, max);
            max--;
        }
        return sum == n;
    }

    public int maximizeSquareHoleArea(int n, int m, int[] hBars, int[] vBars) {
        int length = n + 2;
        int breadth = m + 2;
        Arrays.sort(hBars);
        Arrays.sort(vBars);
        if (m < n) {
            int start = 1;
            Set<Integer> barsRemoved = new HashSet<>();
            for (int hb : hBars)
                barsRemoved.add(hb);
        } else {
            for (int vb : vBars) {
            }
        }
        return 1 + Math.min(m + 2, n + 2) * Math.min(m + 2, n + 2);
    }

    //TODO : Do Memo
    public int minimumCoins(int[] prices) {
        return helper(prices, 0, 0);
    }

    private int helper(int[] prices, int ind, int capacity) {
        // base case
        if (ind >= prices.length) return 0;
        int ans = 0;

        System.out.println("ind=" + ind + ", capacity=" + capacity);

        for (int i = ind; i < prices.length; i++) {
            if (capacity == 0) {
                ans += prices[ind];
                ans += helper(prices, ind + 1, ind + 1);
            } else {
                int b = 0, uc = 0;
                // buy
                b = prices[ind] + helper(prices, ind + 1, ind + 1);

                // use capacity
                uc = helper(prices, ind + 1, capacity - 1);
                ans += (Math.min(b, uc) == 0 ? Math.max(b, uc) : Math.min(b, uc));
            }

            System.out.println("ans=" + ans);
        }

        return ans;
    }


}
/*
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

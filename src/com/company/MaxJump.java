package com.company;

import java.util.HashMap;import java.util.*;

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter obj = new WordFilter(words);
 * int param_1 = obj.f(prefix,suffix);
 */

/*
[6,4,14,6,8,13,9,7,10,6,12]
2
 */

class MaxJump {
    int n;
    int[] dp;

    public int maxJumps(int[] arr, int d) {
        this.n = arr.length;
        if (this.n < 3) return this.n;
        dp = new int[this.n + 1];

        Arrays.fill(dp, -1);
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < this.n; i++) {
            // find max jump from given index
            if (dp[i] == -1) dp[i] = mxj(arr, dp, i, this.n, d);
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    private int mxj(int[] ar, int[] dp, int i, int n, int d) {

        if (dp[i] != -1) return dp[i];

        // check to the left of current position
        int lmax = 0;
        int l = i - 1;
        while (l >= i - d && l >= 0 && ar[l] < ar[i]) {
            lmax = Math.max(lmax, dp[l]);
            l--;
        }


        // check to the right of current position
        int rmax = 0;
        int r = i + 1;

        while (r <= i + d && r < this.n && ar[r] < ar[i]) {
            if (dp[r] == -1) dp[r] = mxj(ar, dp, r, this.n, d); // For every right index we need to check
            rmax = Math.max(rmax, dp[r]);
            r++;
        }

        return Math.max(lmax, rmax) + 1; // +1 is done to add current position index as well
    }


    /*
    String ans[];

    public int[] prevPermOpt1(int[] arr) {
        int n = arr.length;
        if (n == 0) return new int[]{};
        String str = Arrays.stream(arr).boxed().map(Object::toString).collect(Collectors.joining(","));


        int index = 0;
        this.ans[index] = maxPermute(str, "");
        Arrays.sort(this.ans);
        List<Integer> res = this.ans[1].trim().chars().boxed().collect(Collectors.toList());

        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    private String maxPermute(String s, String answer) {

        if (s.length() == 0) {
            System.out.print(answer + "  ");
            return answer;
        }

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            String left_substr = s.substring(0, i);
            String right_substr = s.substring(i + 1);
            String rest = left_substr + right_substr;
            maxPermute(rest, answer + ch);
        }
        return null;
    }

     */

    public int[] prevPermOpt1(int[] arr) {
        int index = arr.length - 2;

        while (index >= 0 && arr[index] <= arr[index + 1]) {
            index--;
        }

        if (index >= 0) {
            int index2 = index + 1;
            for (int i = index; i < arr.length; i++) {
                if (arr[i] > arr[index2] && arr[i] < arr[index]) {
                    index2 = i;
                }
            }

            int temp = arr[index];
            arr[index] = arr[index2];
            arr[index2] = temp;
        }

        return arr;
    }

    public int[] replaceElements(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            int max = -1;
            for (int j = i + 1; j < arr.length; j++) {
                max = Math.max(max, arr[j]);
            }
            arr[i] = max;
        }
        return arr;
    }

    /*
    [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
    TC=O(n), SC=O(1)
     */
    /*
    public int[][] insert(int[][] intervals, int[] newInterval) {

        if (intervals.length == 0) return new int[][]{new int[]{newInterval[0], newInterval[1]}};

        int nl = newInterval[0];
        int nr = newInterval[1];
        int prev = Integer.MIN_VALUE;

        for (int i = 0; i < intervals.length; i++) {
            int vl = intervals[i][0];
            int vr = intervals[i][1];

            if (prev == vl) {
                intervals[i - 1][1] = vr;
            }

            if (vr < nr) {
                intervals[i][1] = nr;
                if (vl > nl) {
                    intervals[i][0] = nl;
                }
                prev = nr;
            }
        }
        return intervals;
    }
     */


    public int[][] insert(int[][] intervals, int[] newInterval) {
        int i = 0;
        int n = intervals.length;
        List<int[]> ans = new ArrayList<>();


        while (i < n && intervals[i][1] < newInterval[0]) ans.add(intervals[i++]);

        int[] temp = newInterval;
        while (i < n && intervals[i][0] < newInterval[1]) {
            temp[0] = Math.min(intervals[i][0], temp[0]);
            temp[1] = Math.max(intervals[i++][1], temp[1]);

        }
        ans.add(temp);

        while (i < n && intervals[i][0] > newInterval[1]) ans.add(intervals[i++]);
        return ans.toArray(new int[ans.size()][2]);

    }

    // [1,3,7,11,12,14,18]
    // [1,2,3,4,5,6,7,8]


    public int lenLongestFibSubseq(int[] A) {
        int n = A.length;

        // Store all array elements in a hash
        // table
        TreeSet<Integer> S = new TreeSet<>();
        for (int t : A) {
            // Add each element into the set
            S.add(t);
        }
        int maxLen = 0, x, y;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {

                x = A[j];
                y = A[i] + A[j];
                int length = 3;

                // check until next fib element is found
                while (S.contains(y)) {
//                    S.tailSet(4);

                    if (y != S.last()) {
                        // next element of fib subseq
                        int z = x + y;
                        x = y;
                        y = z;
                        maxLen = Math.max(maxLen, length);
                        length++;
                    } else {
                        maxLen = Math.max(maxLen, length);
                        break;
                    }
                }
            }
        }
        return maxLen >= 3 ? maxLen : 0;
    }


    public int lenLongestFibSubseqDP(int[] A) {
        int n = A.length;
        int res = 0;
        // Store all array elements in a hashMap
        HashMap<Integer, Integer> map = new HashMap<>();

        // Initialize dp table
        int[][] dp = new int[n][n];

        // Iterate till N
        for (int j = 0; j < n; ++j) {
            map.put(A[j], j);
            for (int i = 0; i < j; ++i) {
                // Check if the current integer
                // forms a fibonacci sequence
                int k = map.get(A[j] - A[i]) != null ?
                        (map.get(A[j] - A[i]).equals(map.get(A[j])) ? -1 : map.get(A[j] - A[i])) : -1;

                // Update the dp table
                dp[i][j] = (A[j] - A[i] < A[i] && k >= 0) ? dp[k][i] + 1 : 2;
                res = Math.max(res, dp[i][j]);
            }
        }
        // Return the answer
        return res > 2 ? res : 0;
    }


    /*
    Eating right now
    Eating
     */
    public boolean areSentencesSimilar(String sentence1, String sentence2) {
        String[] arr1 = sentence1.split(" "), arr2 = sentence2.split(" ");

        int s1 = 0, s2 = 0, e1 = arr1.length - 1, e2 = arr2.length - 1;

        for (; s1 <= e1 && s2 <= e2 && arr1[s1].equals(arr2[s2]); s1++, s2++) ;
        for (; e1 >= 0 && e2 >= 0 && arr1[e1].equals(arr2[e2]); e1--, e2--) ;
        return s1 > e1 || s2 > e2;
        /*
        // sentence2 is smaller in this case
        if (s1.length < s2.length) {
            String prefix = s1[0];
            String suffix = s1[s1.length - 1];
            if (prefix.equals(s2[0]) && suffix.equals(s2[s2.length - 1])
            ) {
                // check for remaining words of sentence
                for (int i = 1; i < s1.length - 1; i++) {
                    if (!s1[i].equals(s2[i])) return false;
                }
            } else if ((prefix.equals(s2[0]) || suffix.equals(s2[s2.length - 1]))
                    && prefix.equals(suffix) && s1.length == 1) return true;
            else return false;
        } else if (s1.length > s2.length) {
            String prefix = s2[0];
            String suffix = s2[s2.length - 1];
            if ((prefix.equals(s1[0]) && suffix.equals(s1[s1.length - 1]))) {
                // check for remaining words of sentence
                for (int i = 1; i < s2.length - 1; i++) {
                    if (!s2[i].equals(s1[i])) return false;
                }
            } else if ((prefix.equals(s1[0]) || suffix.equals(s1[s1.length - 1]))
                    && prefix.equals(suffix) && s2.length == 1) return true;
            else return false;
        } else {
            for (int i = 0; i < s1.length; i++) {
                if (!s1[i].equals(s2[i])) return false;
            }
        }
        return true;

         */
    }

    /*
    private int isFibonacci(int n) {
        if (n == 1 || n == 0) {
            return true;
        }
        int[] dp = new int[n];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i < n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n] == dp[n - 1] + dp[n - 2];
    }
     */

//-----------------------------------------------------------


    //[0,2] -> [1,2] -> [2,2]
    /*
    {{3,6,2},
    {4,9,7},
    {33,8,11}}
    Print spirally

    o/p - [3,6,27,11,8,33,4,9]


    {{3,6,2,11},
    {4,9,7,32},
    {33,8,11,22},
    {34,9,13,29}}

3, 6, 2, 11, 32, 22, 29, 13, 9 , 34, 33, 4 , 9, 7 ,11 , 8

     */
    public static void spiral(int[][] matrix) {
        int n = matrix.length;
        if (n == 0) return;
        for (int i = 0; i < n; i++) {
            int k = 0;
            // start traversing row wise
            for (int j = 0; j < n; i++) {
                System.out.println(matrix[i][j]);
            }

            // start traversing column wise
            for (int j = 0; j < n; i++) {
                System.out.println(matrix[j][i]);
            }

            // start traversing rowise to the left
            if (i == n - 1 && k == n - 1) {
                for (int m = n - 1; m >= 0; m--) {
                    System.out.println(matrix[i][m]);
                }
            }

            // upwards traversal
            if (i == n - 1 && k == 0) {
                for (int m = n - 1; m >= i; m--) {
                    System.out.println(matrix[k][m]);
                }
            }
            k++;
        }
    }

    // TC = ((n^2))
    // sc = O(1)

    public static void main(String[] args) {

        int[][] matrix = new int[][]{{3, 6, 2, 11},
                {4, 9, 7, 32},
                {33, 8, 11, 22},
                {34, 9, 13, 29}};

        spiral(matrix);
    }


    //[0,2] -> [1,2] -> [2,2]
    /*
    {{3,6,2},
    {4,9,7},
    {33,8,11}}
    Print spirally

    o/p - [3,6,27,11,8,33,4,9]


    {{3,6,2,11},
    {4,9,7,32},
    {33,8,11,22},
    {34,9,13,29}}

3, 6, 2, 11, 32, 22, 29, 13, 9 , 34, 33, 4 , 9, 7 ,11 , 8

     */

    // TC = ((n^2))
    // sc = O(1)

    /*

    public List<Integer> spiralOrder(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        if (rows != cols) return spiralRec(matrix);
        else return spiralSquare(matrix);
    }

    private List<Integer> spiralSquare(int[][] matrix) {

        int n = matrix.length;
        if (n == 0) new ArrayList<>();
        List<Integer> ans = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int j, m, p;
            for (j = i; j < n - i; j++) ans.add(matrix[i][j]);
            j--;

            for (m = i + 1; m < n - i; m++) ans.add(matrix[m][j]);

            m -= 2;

            for (p = m; p >= i; p--) ans.add(matrix[j][p]);

            j--;
            p++;

            for (int u = j; u > i; u--) ans.add(matrix[u][p]);
        }
        return ans;
    }
     */
    public List<Integer> spiralOrder(int[][] arr) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        int i, k = 0, l = 0;
        int r = arr.length;
        int c = arr[0].length;
        while (k < r && l < c) {
            for (i = l; i < c; i++) {
                res.add(arr[k][i]);
            }
            k++;

            for (i = k; i < r; i++) {
                res.add(arr[i][c - 1]);
            }
            c--;

            if (k < r) {
                for (i = c - 1; i >= l; i--) {
                    res.add(arr[r - 1][i]);
                }
                r--;
            }

            if (l < c) {
                for (i = r - 1; i >= k; i--) {
                    res.add(arr[i][l]);
                }
                l++;
            }
        }
        return res;
    }

    /*

    public List<Integer> diffWaysToCompute(String expression) {
        return helper(0, expression.length() - 1, expression);

    }

    private List<Integer> helper(int start, int end, String expression) {

        List<Integer> curr = new ArrayList<>();
        // base case
        if (start > end) return curr;
        boolean operatorPresent = false;
        for (int i = start; i < end; i++) {
            if (!Character.isDigit(expression.charAt(i))) {
                operatorPresent = true;
                List<Integer> left = helper(start, i - 1, expression);
                List<Integer> right = helper(i + 1, end, expression);

                for (int leftVal : Objects.requireNonNull(left)) {
                    for (int rightVal : Objects.requireNonNull(right)) {
                        curr.add(calculate(leftVal, rightVal, expression.charAt(i)));
                    }
                }
            }
        }
        if (!operatorPresent) curr.add(Integer.parseInt(expression.substring(start, end + 1)));
        return curr;
    }

    private int calculate(int leftVal, int rightVal, char operator) {

        if (operator == '+') return leftVal + rightVal;
        else if (operator == '-') return leftVal - rightVal;
        else if (operator == '/') return leftVal / rightVal;
        else return leftVal * rightVal;
    }

     */


    // using dp  + memoisation

    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer>[][] dp = new List[expression.length() + 1][expression.length() + 1];
        return find(new StringBuilder(expression), 0, expression.length(), dp);
    }

    private List<Integer> find(StringBuilder expression, int start, int end, List<Integer>[][] dp) {

        List<Integer> ans = new ArrayList<>();
        if (dp[start][end] != null) return dp[start][end];

        for (int i = start; i < end; i++) {
            char c = expression.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                List<Integer> left = find(expression, start, i, dp);
                List<Integer> right = find(expression, i + 1, end, dp);

                for (int leftVal : left) {
                    for (int rightVal : right) {
                        switch (c) {
                            case '+':
                                ans.add(leftVal + rightVal);
                                break;
                            case '-':
                                ans.add(leftVal - rightVal);
                                break;
                            default:
                                ans.add(leftVal * rightVal);
                                break;
                        }
                    }
                }
            }
        }
        // when no operator is found
        if (ans.isEmpty()) {
            ans.add(Integer.parseInt(expression.substring(start, end + 1)));
        }

        dp[start][end] = ans;
        return ans;

//    int n = expression.length();
//        List<Integer> ans = new ArrayList<>();
//        Map<Character, Integer> map = new HashMap<>();
//
//        for (int i = 0; i < n; i++) {
//            if (expression.charAt(i) == 'a') {
//
//            }
//        }

// 2 pointer technique
// using 2 stacks

    }
}

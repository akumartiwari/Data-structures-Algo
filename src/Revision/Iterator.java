package com.company;

import java.util.HashMap;import java.util.*;
import java.util.stream.Collectors;

import static java.util.Arrays.asList;

class Iterator {

    static class MyIter<T extends Comparable> implements java.util.Iterator {

        public MyIter(List<List<T>> lists) {
        }

        class IterState implements Comparable<IterState> {
            private java.util.Iterator<T> iterator;

            /*
             * PROBLEM: Get current value (Helper)
             * Get current value of iterator state.
             *
             * ALGORITHM: Direct field access
             * TC: O(1) | SC: O(1)
             */
            public T getCurrVal() {
                return currVal;
            }

            /*
             * PROBLEM: Check has next (Helper)
             * Check if underlying iterator has more elements.
             *
             * ALGORITHM: Delegate to iterator
             * TC: O(1) | SC: O(1)
             */
            public boolean hasNext() {
                return iterator.hasNext();
            }

            /*
             * PROBLEM: Advance iterator (Helper)
             * Advance to next element, setting currVal to null if exhausted.
             *
             * ALGORITHM: Iterator traversal
             * TC: O(1) | SC: O(1)
             */
            public void next() {
                if (iterator.hasNext()) currVal = iterator.next();
                else currVal = null;
            }

            private T currVal;

            public IterState(java.util.Iterator<T> i) {
                iterator = i;
                next();
            }


            /*
             * PROBLEM: Compare iterator states (Helper)
             * Compare by current value for priority queue ordering.
             *
             * ALGORITHM: Natural ordering via Comparable
             * TC: O(1) | SC: O(1)
             */
            @Override
            public int compareTo(IterState iterState) {
                return currVal.compareTo(iterState.getCurrVal());
            }

            /*
             * PROBLEM: Merge init (Helper)
             * Initialize priority queue with all list iterators.
             *
             * ALGORITHM: Min-heap initialization
             * TC: O(n log n) | SC: O(n)
             */
            public void MyIter(List<List<T>> lists) {
                states = new PriorityQueue<IterState>();
                for (List<T> list : lists) {
                    java.util.Iterator<T> listIter = list.iterator();
                    while (listIter.hasNext()) {
                        states.add(new IterState(listIter));
                    }
                }

            }
        }

        private PriorityQueue<IterState> states;

        /*
         * PROBLEM: Check merged iterator has next (Helper)
         * Check if merged iterator has more elements.
         *
         * ALGORITHM: Priority queue empty check
         * TC: O(1) | SC: O(1)
         */
        @Override
        public boolean hasNext() {
            return !states.isEmpty();
        }

        /*
         * PROBLEM: Merge k Sorted Lists (Helper)
         * Return next smallest element from k sorted lists.
         *
         * ALGORITHM: Min-heap (Priority Queue)
         * TC: O(log k) | SC: O(k)
         */
        @Override
        public T next() {

            IterState n = states.poll();

            T retval = n.getCurrVal();
            // if more than 1 elements exist then paste them
            if (n.hasNext()) {
                n.next();
                states.add(n);
            }

            return null;
        }
    }

    /*
     * PROBLEM: Convert sorted lists to array (Helper)
     * Convert k sorted lists to merged sorted array using the MyIter iterator.
     *
     * ALGORITHM: Merge via iterator
     * TC: O(n log k) | SC: O(n)
     */
    private static <T extends Comparable> ArrayList toArray(List<List<T>> lists) {

        MyIter<T> i = new MyIter<T>(lists);
        List<T> retVal = new ArrayList<>();
        while (i.hasNext()) {
            retVal.add(i.next());
        }
        return (ArrayList) retVal;
    }

    /*
     * PROBLEM: Entry point (Main)
     * Tests the merged iterator with multiple sorted lists.
     *
     * ALGORITHM: N/A
     * TC: O(n log k) | SC: O(n)
     */
    public static void main(String[] args) {

        System.out.println("Hello world!");
        ArrayList arr = toArray(asList(
                asList(1, 3, 5),
                asList(2, 4, 6),
                asList(7, 9),
                asList(8),
                asList(0, 10),
                asList()));

        for (Object i : arr) {
            System.out.println(i);
        }

        arr = toArray(asList());
        for (Object i : arr) {
            System.out.println(i);
        }

    }

    /*
     * PROBLEM: Two Sum (LeetCode 1)
     * Find indices of two numbers in the array that add up to target.
     *
     * ALGORITHM: HashMap
     * TC: O(n) | SC: O(n)
     */
    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int index = 0;
        for (int num : nums) {
            if (map.containsKey(target - num)) {
                index++;
                return new int[]{map.get(target - num), index};
            } else {
                index++;
                return new int[]{map.get(num), index};
            }
        }
        return new int[]{};
    }


    /*
     * PROBLEM: Calculate range sum (Helper)
     * Calculate sum from start to end using arithmetic series formula.
     *
     * ALGORITHM: Arithmetic series formula
     * TC: O(1) | SC: O(1)
     */
    int calSum(int start, int end) {
        int n = end - start;
        return n * (n + 1) / 2;
    }

    /*
     * PROBLEM: Find subarrays less than k (Helper)
     * Find all contiguous subarrays whose elements are all less than k.
     *
     * ALGORITHM: Two-pointer sliding window
     * TC: O(n^2) | SC: O(n^2)
     */
    public List<List<Integer>> findSubarrays(int[] nums, int k) {
        int n = nums.length;
        if (n == 0) return new ArrayList<>();

        int ptr = 0, sum = 0;
        List<List<Integer>> subarrays = new ArrayList<>();

        while (ptr < n) {
            List<Integer> ans = new ArrayList<>();

            if (nums[ptr] < k) {
                int ptrForward = ptr;

                while (ptrForward < n && nums[ptrForward] < k) {
                    ans.add(nums[ptrForward]);
                    ptrForward++;

                }
                subarrays.add(ans);
                ans = new ArrayList<>();
                sum += calSum(ptr, ptrForward);
            }
            ptr++;
        }
        return subarrays;
    }



    /*
         [2,3,1,5,4] and k = 3
         void reverse(int [] arr, int k) -->
         method to sort the array by incorporating reverse method inside
     */

    // O(k/2) ~ O(k)

    // For reverse of an array traverse till n/2 times and follow process
//    public void reverse(int[] arr, int k) {
//        int n = arr.length;
//        if (n == 0) return;
//
//        int totalOper = (int) Math.floor(k / 2);
//        for (int i = 0; i < totalOper; i++) {
//            int newIndex = k - 1 - i;
//            if (arr[i] < arr[newIndex]) continue;
//                // swap
//            else {
//                int temp = arr[newIndex];
//                arr[newIndex] = arr[i];
//                arr[i] = temp;
//            }
//        }
//    }

    /*
     * PROBLEM: Reverse first k elements (Helper)
     * Reverse first k elements of array in-place.
     *
     * ALGORITHM: Collections.reverse
     * TC: O(k) | SC: O(k)
     */
    public Object[] reverse(Object[] arr, int k) {
        Collections.reverse(Collections.singletonList(arr).subList(0, k));
        return arr;
    }

    /*
     * PROBLEM: Sort using reverse (Helper)
     * Sort array using embedded reverse function.
     *
     * ALGORITHM: Custom sort with reverse
     * TC: O(n^2) | SC: O(n)
     */
    public void sort(int[] arr, int k) {
        int n = arr.length;
        if (n == 0) return;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (arr[j] > arr[j + 1]) {
                    Object[] revert = reverse(Collections.singletonList(arr).subList(j, j + 2).toArray(), 2);
                    List<Integer> pre = new ArrayList<>(Arrays.stream(arr).boxed().collect(Collectors.toList()).subList(0, i));
                    // Object array to int array conversion
                    Integer[] post = Arrays.stream(revert)
                            .map(Object::toString)
                            .map(Integer::valueOf)
                            .toArray(Integer[]::new);
                    pre.addAll(Arrays.asList(post));
                    pre.addAll(Arrays.stream(arr).boxed().collect(Collectors.toList()).subList(i, n));
                    arr = pre.stream().mapToInt(x -> x).toArray();
                }
            }
        }
        System.out.println(Arrays.toString(arr));
    }


//    ['a', 'b', 'c'];
    /*
     // Combination (Maths)
Given a length n, count the number of strings of length n that can be made using ‘a’, ‘b’ and ‘c’ with at-most one ‘b’ and two ‘c’s allowed.
     */


    //O(3^n), O(n^3)

    final static int n = 10;
    final static int[][] memo = new int[n][n];

    /*
     * PROBLEM: Count strings with limited b and c (Helper)
     * Count strings of length n using 'a','b','c' with at most bCount 'b's and cCount 'c's.
     *
     * ALGORITHM: Memoized recursion (Combination DP)
     * TC: O(n^3) | SC: O(n^2)
     */
    static int countStr(int n, int bCount, int cCount) {
        // base case
        if (bCount < 0 || cCount < 0) return 0;
        if (bCount == 0 && cCount == 0) return 1;
        int res = memo[bCount][cCount] != 0 ? memo[bCount][cCount] : countStr(n - 1, bCount, cCount);
        res += memo[bCount - 1][cCount] != 0 ? memo[bCount - 1][cCount] : countStr(n - 1, bCount - 1, cCount);
        res += memo[bCount][cCount - 1] != 0 ? memo[bCount][cCount - 1] : countStr(n - 1, bCount, cCount - 1);
        return res;
    }
    // Efficient solution

    static class GFG {
        /*
         * PROBLEM: Entry point (Main)
         * Entry point for GFG test class — tests countStrEff.
         *
         * ALGORITHM: N/A
         * TC: O(n*b*c) | SC: O(n*b*c)
         */
        public static void main(String[] args) {
            int n = 3; // Total number of characters
            int bCount = 1, cCount = 2;
            System.out.println(countStrEff(n, bCount, cCount));
        }
    }

    /*
     * PROBLEM: Count strings with limited b and c — efficient (Helper)
     * Efficient version of countStr using 3D DP table.
     *
     * ALGORITHM: 3D DP (top-down memoization)
     * TC: O(n*b*c) | SC: O(n*b*c)
     */
    private static int countStrEff(int n, int bCount, int cCount) {
        int[][][] dp = new int[n + 1][2][3];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    dp[i][j][k] = -1;
                }
            }
        }
        return countStrEffUtil(dp, n, bCount, cCount);
    }

    /*
     * PROBLEM: DP utility for countStrEff (Helper)
     * Recursive DP utility that computes the count using a prebuilt 3D memo table.
     *
     * ALGORITHM: 3D DP memoized recursion
     * TC: O(n*b*c) | SC: O(1) per call
     */
    private static int countStrEffUtil(int[][][] dp, int n, int bCount, int cCount) {
        // base case
        if (bCount < 0 || cCount < 0) return 0;
        if (n == 0) return 1;
        if (bCount == 0 && cCount == 0) return 1;
        if (dp[n][bCount][cCount] != -1) return dp[n][bCount][cCount];

        int res = countStr(n - 1, bCount, cCount);
        res += countStr(n - 1, bCount - 1, cCount);
        res += countStr(n - 1, bCount, cCount - 1);
        return res;
    }

    /*
     * PROBLEM: Find the Index of the First Occurrence in a String (LeetCode 28)
     * Return the index of the first occurrence of needle in haystack, or -1 if not found.
     *
     * ALGORITHM: Sliding window / brute force
     * TC: O(n*m) | SC: O(1)
     */
    public int strStr(String haystack, String needle) {

        int n = haystack.length();
        if (needle.length() == 0) return 0;

        for (int i = 0; i < n; i++) {
            if (String.valueOf(haystack.charAt(i)).equals(String.valueOf(needle.charAt(0)))) {

                boolean isExist = true;
                int index = 0;
                while (i < n && index < needle.length()) {
                    if (String.valueOf(haystack.charAt(i)).equals(String.valueOf(needle.charAt(index)))) {
                        i++;
                        index++;
                    } else {
                        isExist = false;
                        break;
                    }
                }
                if (isExist && index == needle.length() - 1) return i - index;
            }

        }

        return -1;
    }

    /*
     * PROBLEM: Wildcard Matching (LeetCode 44)
     * Match string s against pattern p containing '?' (any single char) and '*' (any sequence).
     *
     * ALGORITHM: 2D DP (bottom-up)
     * TC: O(m*n) | SC: O(m*n)
     */
    public boolean regexMatch(String s, String p) {
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (i == 0 && j == 0) dp[i][j] = true;
                else if (i == 0) dp[i][j] = false;
                else if (j == 0) {
                    if (p.charAt(i - 1) == '*') dp[i][j] = dp[i - 1][j];
                } else {
                    if (p.charAt(i - 1) == '?' || (p.charAt(i - 1) == s.charAt(j - 1)))
                        dp[i][j] = dp[i - 1][j - 1]; // take prev value as here it will always be true
                    else dp[i][j] = p.charAt(i - 1) == '*' || p.charAt(i - 1) == '.';
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

    /*
     * PROBLEM: Regular Expression Matching (LeetCode 10)
     * Match string s against regex pattern p with '.' (any char) and '*' (zero or more of preceding).
     *
     * ALGORITHM: 2D DP (bottom-up)
     * TC: O(m*n) | SC: O(m*n)
     */
    public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (i == 0 && j == 0) dp[i][j] = true;
                else if (i == 0) dp[i][j] = false;
                else if (j == 0) {
                    if (p.charAt(i - 1) == '*') dp[i][j] = dp[i - 1][j];
                } else {
                    if (p.charAt(i - 1) == '?' || (p.charAt(i - 1) == s.charAt(j - 1)))
                        dp[i][j] = dp[i - 1][j - 1]; // take prev value as here it will always be true
                    else if (p.charAt(i - 1) == '*') dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                    else dp[i][j] = false;
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }


    /*
     * PROBLEM: Longest Palindromic Substring (LeetCode 5)
     * Find the longest palindromic substring in s (attempt using DP array).
     *
     * ALGORITHM: DP (attempt)
     * TC: O(n^2) | SC: O(n)
     */
    public String longestPalindrome(String s) {
        int n = s.length();

        if (n == 0) return "";
        String[] dp = new String[n + 1];

        if (n % 2 != 0) { // string length is odd

            int index = 0;
            String s1 = s.substring(0, n / 2 + 1);
            String s2 = s.substring(n / 2 + 1);

            List<String> post = asList(Arrays.stream(reverse(Collections.singletonList(s2.toCharArray()).toArray(), s2.length()))
                    .map(Object::toString)
                    .toArray(String[]::new));
            s2 = String.join("", post);

            int k = 0;
            while (k < s1.length() && index < s1.length()) {
                String str = "";

                for (int i = 0; i < s1.length(); i++) {
                    if (s1.charAt(i) == s2.charAt(i)) {
                        i++;
                        str += s1.charAt(i);
                    } else {
                        index++;
                    }
                    if (dp[i - 1].length() > str.length()) {
                        dp[i] = dp[i - 1];
                    } else dp[i] = str;

                }
                k++;
            }
        } else {
            for (int i = 0; i < n; i++) {
                String str = "";

                if (s.charAt(i) == s.charAt(n - 1 - i)) {
                    str += s.charAt(i);
                } else break;
                if (dp[i - 1].length() > str.length()) {
                    dp[i] = dp[i - 1];
                } else dp[i] = str;
            }
        }
        return dp[n - 1];
    }

    /*
     * PROBLEM: Longest Palindromic Substring (LeetCode 5)
     * Find the longest palindromic substring by expanding around each center.
     *
     * ALGORITHM: Expand around center
     * TC: O(n^2) | SC: O(1)
     */
    public String longestPalindromSubstring(String s) {
        int n = s.length();
        if (n == 0) return "";
        int maxLength = 0, start = -1;
        for (int i = 0; i < n; i++) {
            int length = Math.max(getLength(i, i, s), getLength(i, i + 1, s));
            if (length > maxLength) maxLength = length;
            start = i - (length - 1) / 2;
        }
        return s.substring(start, start + maxLength);
    }

    /*
     * PROBLEM: Get palindrome length from center (Helper)
     * Get palindrome length by expanding outward from a given center position.
     *
     * ALGORITHM: Two-pointer expansion
     * TC: O(n) | SC: O(1)
     */
    private int getLength(int start, int end, String s) {
        int length = 0;
        while (start > 0 && end < n) {
            if (s.charAt(start) != s.charAt(end)) break;

            length += 2;
            start--;
            end++;
        }

        return length;
    }


    /*
     * PROBLEM: Recursive palindrome length checker (Helper)
     * Recursively compute palindrome length expanding from center, with deletion fallback.
     *
     * ALGORITHM: Recursive expansion with backtracking
     * TC: O(n^2) | SC: O(n)
     */
    private int getLengthRec(int start, int end, String s) {
        if (end >= s.length()) return 0;
        int length = (start == end) ? -1 : 0;
        while (start >= 0 && end < s.length()) {
            if (s.charAt(start) != s.charAt(end)) {
                StringBuilder sb = new StringBuilder(s);
                length = Math.max(Math.max(getLengthRec(start++, end++, sb.deleteCharAt(start).toString()),
                                getLengthRec(start--, end--, sb.deleteCharAt(end).toString())),
                        (start == end) ? getLengthRec(start--, end++, sb.deleteCharAt(start).toString())
                                : getLengthRec(start++, end--, sb.deleteCharAt(start).deleteCharAt(end - 1).toString())
                );
            } else {
                length += 2;
                start--;
                end++;
            }
        }
        return length;
    }


//    int[][] memoTable = new int[n][n];
//        for(int[] val:memoTable) Arrays.fill(val, -1);

    /*
     * PROBLEM: Longest Palindromic Subsequence (LeetCode 516)
     * Compute length of longest palindromic subsequence in s[i..j] using memoization.
     *
     * ALGORITHM: Top-down memoized recursion
     * TC: O(n^2) | SC: O(n^2)
     */
    public int s2(String s, int i, int j, int[][] memoTable) {

        if (i == j) return 1;
        if (i > j) return 0;

        if (memoTable[i][j] != -1) return memoTable[i][j];

        char ch1 = s.charAt(i);
        char ch2 = s.charAt(j);
        int max = 1;
        if (ch1 == ch2) {
            int c = s2(s, i + 1, j - 1, memoTable);
            max = Math.max(max, c + 2);
        } else {
            max = Math.max(s2(s, i + 1, j, memoTable), max);
            max = Math.max(max, s2(s, i, j - 1, memoTable));
        }

        return max;
    }


    boolean[][] memob = new boolean[n][n];

    /*
     * PROBLEM: Palindromic Substrings (LeetCode 647)
     * Count all palindromic substrings in s.
     *
     * ALGORITHM: DP with memoized palindrome check
     * TC: O(n^2) | SC: O(n^2)
     */
    public int countSubstrings(String s) {
        int n = s.length();
        boolean[][] memob = new boolean[n][n];

        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n - i; k++) {
                if (isPalindrome(s, k, k + i, memob)) count += 1;
            }
        }
        return count;
    }

    /*
     * PROBLEM: Longest Palindromic Subsequence (LeetCode 516)
     * Bottom-up DP to compute length of longest palindromic subsequence in s[0..n-1].
     *
     * ALGORITHM: Bottom-up 2D DP
     * TC: O(n^2) | SC: O(n^2)
     */
    public int s3(String s, int n) {

        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;

            for (int j = i + 1; j < n; j++) {
                char ch1 = s.charAt(i);
                char ch2 = s.charAt(j);

                if (ch1 == ch2) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    /*
     * PROBLEM: Count Palindromic Substrings (LeetCode 647)
     * Count all palindromic substrings in s (alternate implementation).
     *
     * ALGORITHM: DP with memoized palindrome check
     * TC: O(n^2) | SC: O(n^2)
     */
    public int countPalindromSubstrings(String s) {
        int n = s.length();
        boolean[][] memob = new boolean[n][n];

        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n - i; k++) {
                if (isPalindrome(s, k, k + i, memob)) count += 1;
            }
        }
        return count;
    }

    /*
     * PROBLEM: Memoized palindrome check (Helper)
     * Check if s[i..j] is a palindrome using a memoization table.
     *
     * ALGORITHM: Memoized recursion
     * TC: O(n^2) | SC: O(n^2)
     */
    private boolean isPalindrome(String s, int i, int j, boolean[][] memob) {
        if (i == j || i > j) return true;
        if (memob[i][j]) return memob[i][j];

        char ch1 = s.charAt(i);
        char ch2 = s.charAt(j);

        if (ch1 == ch2) {
            memob[i][j] = isPalindrome(s, i + 1, j - 1, memob);
        } else return false;
        return memob[i][j];
    }

    /*
    public int countPalindromicSubsequences(String s) {
        int n = s.length();

        // create the palindrom subsequence from given string first
        String[][] dp = new String[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = String.valueOf(s.charAt(i));

            for (int j = i + 1; j < n; j++) {
                char ch1 = s.charAt(i);
                char ch2 = s.charAt(j);

                if (ch1 == ch2) {
                    dp[i][j] = dp[i + 1][j - 1] + ch1 + ch2;
                } else {
                    dp[i][j] = dp[i + 1][j].length() > dp[i][j - 1].length() ? dp[i + 1][j] : dp[i][j - 1];
                }
            }
        }

        String maxPalindrom = dp[0][n - 1].replaceAll("null", "");

        int mn = maxPalindrom.length();
        boolean[][] memob = new boolean[mn][mn];

        HashSet<String> set = new HashSet<>();

        for (int i = 0; i < mn; i++) {
            for (int k = 0; k < mn - i; k++) {
                if (isPalindrome(maxPalindrom, k, k + i, memob)) {
                    System.out.println(maxPalindrom.substring(k, k + i));
                    set.add(maxPalindrom.substring(k, k + i));
                }
            }
        }
        return set.size();
    }

    public int maxLength(String s, boolean[][] memob) {

        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;

            for (int j = i + 1; j < n; j++) {
                char ch1 = s.charAt(i);
                char ch2 = s.charAt(j);

                if (ch1 == ch2) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
}

*/
}

package com.company;

import java.util.*;


class CountPalindromicSubsequencesSolution {
    public int countPalindromicSubsequences(String s) {
        int len = s.length();
        int[][] dp = new int[len][len];

        char[] chs = s.toCharArray();
        for (int i = 0; i < len; i++) {
            dp[i][i] = 1;   // Consider the test case "a", "b" "c"...
        }

        for (int distance = 1; distance < len; distance++) {
            for (int i = 0; i < len - distance; i++) {
                int j = i + distance;
                if (chs[i] == chs[j]) {
                    int low = i + 1;
                    int high = j - 1;

                    /* Variable low and high here are used to get rid of the duplicate*/

                    while (low <= high && chs[low] != chs[j]) {
                        low++;
                    }
                    while (low <= high && chs[high] != chs[j]) {
                        high--;
                    }
                    if (low > high) {
                        // consider the string from i to j is "a...a" "a...a"... where there is no character 'a' inside the leftmost and rightmost 'a'
                       /* eg:  "aba" while i = 0 and j = 2:  dp[1][1] = 1 records the palindrome{"b"},
                         the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"b"},
                         and additional time as {"aba"}. The reason why 2 counted is that we also count {"a", "aa"}.
                         So totally dp[i][j] record the palindrome: {"a", "b", "aa", "aba"}.
                         */

                        dp[i][j] = dp[i + 1][j - 1] * 2 + 2;
                    } else if (low == high) {
                        // consider the string from i to j is "a...a...a" where there is only one character 'a' inside the leftmost and rightmost 'a'
                       /* eg:  "aaa" while i = 0 and j = 2: the dp[i + 1][j - 1] records the palindrome {"a"}.
                         the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"a"},
                         and additional time as {"aaa"}. the reason why 1 counted is that
                         we also count {"aa"} that the first 'a' come from index i and the second come from index j. So totally dp[i][j] records {"a", "aa", "aaa"}
                        */
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 1;
                    } else {
                        // consider the string from i to j is "a...a...a... a" where there are at least two character 'a' close to leftmost and rightmost 'a'
                       /* eg: "aacaa" while i = 0 and j = 4: the dp[i + 1][j - 1] records the palindrome {"a",  "c", "aa", "aca"}.
                          the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"a",  "c", "aa", "aca"},
                          and additional time as {"aaa",  "aca", "aaaa", "aacaa"}.  Now there is duplicate :  {"aca"},
                          which is removed by deduce dp[low + 1][high - 1]. So totally dp[i][j] record {"a",  "c", "aa", "aca", "aaa", "aaaa", "aacaa"}
                          */
                        dp[i][j] = dp[i + 1][j - 1] * 2 - dp[low + 1][high - 1];
                    }
                } else {
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1];  //s.charAt(i) != s.charAt(j)
                }
                dp[i][j] = dp[i][j] < 0 ? dp[i][j] + 1000000007 : dp[i][j] % 1000000007;
            }
        }

        return dp[0][len - 1];
    }


    final int mod = 1_000_000_007;

    public int countPalindromes(String s) {

        int ans = 0;
        int len = s.length();
        int[][] dp = new int[len][len];

        /* compute how many palindromes of length 3 are possible for every 2 characters match */
        for (int i = len - 2; i >= 0; --i) {
            for (int j = i + 2; j < len; ++j) {
                dp[i][j] = (dp[i][j - 1] + (dp[i + 1][j] == dp[i + 1][j - 1] ? 0 : dp[i + 1][j] - dp[i + 1][j - 1])) % mod;
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = (dp[i][j] + j - i - 1) % mod;
                }
            }
        }

        /* re-use the above data to calculate for palindromes of length 5*/
        for (int i = 0; i < len; ++i) {
            for (int j = i + 4; j < len; ++j) {
                if (s.charAt(i) == s.charAt(j)) {
                    ans = (ans + dp[i + 1][j - 1]) % mod;
                }
            }
        }

        // for(int i=0;i<len;++i) System.out.println(Arrays.toString(dp[i]));

        return ans;
    }
    /*
      // 2-d array chacters
  wedfx

  z x f c f g
     */

    public static void main(String[] args) {
        char[][] arr = {{'q', 'w', 'e', 'r', 't', 'y'}, {'a', 's', 'd', 'f', 'g', 'h'}};
        String word = "werty";

        System.out.println(isWordFound(arr, word));
    }

    public static boolean isWordFound(char[][] s, String word) {
        int rows = s.length;
        int cols = s[0].length;
        System.out.println(rows);
        System.out.println(cols);

        // Traverse horizontally
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                String str = "";
                str += String.valueOf(s[i][j]);
                if (str.contains(word)) return true;
            }
        }

        // Traverse vertically
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                String str = "";
                str += String.valueOf(s[j][i]);
                if (str.contains(word)) return true;
            }
        }
        return false;
    }

    /*
    a > $1000

    the amount exceeds $1000, or;
if it occurs within (and including) 60 minutes of another transaction with the same name in a different city.
     */

    // This class is used to store index and corresponding value element of array
    class Triplet<I extends String, I1 extends Number, I2 extends String, I3 extends Number> implements Comparable<Triplet<String, Number, String, Number>> {
        // field member and comparable implementation

        String name;
        int time;
        String city;
        int amount;


        Triplet(String name, int time, String city, int amount) {
            this.name = name;
            this.time = time;
            this.city = city;
            this.amount = amount;

        }

        @Override
        public int compareTo(Triplet<String, Number, String, Number> stringNumberStringTriplet) {
            return this.time - stringNumberStringTriplet.time;
        }
    }


    public int twoCitySchedCost(int[][] costs) {
        int mincost = 0;
        PriorityQueue<Integer> pq1 = new PriorityQueue<>();
        PriorityQueue<Integer> pq2 = new PriorityQueue<>();

        for (int i = 0; i < costs.length; i++) {
            pq1.add(costs[i][0]);
            pq2.add(costs[i][1]);
        }

        for (int i = 0; i < costs.length / 2; i++) {
            int r1 = pq1.poll();
            pq1.remove(r1);
            int r2 = pq2.poll();
            pq2.remove(r2);

            System.out.println(r1);
            System.out.println(r2);

            mincost += r1 + r2;
        }
        return mincost;
    }


    public List<String> invalidTransactions(String[] transactions) {
        int n = transactions.length;
        if (n == 0) return new ArrayList<>();
        List<String> ans = new ArrayList<>();

        HashMap<String, List<Triplet>> map = new HashMap<>();
        for (String t : transactions) {
            String[] arr = t.split(",");
            String name = arr[0];
            String time = arr[1];
            String amount = arr[2];
            String city = arr[3];

            if (Integer.parseInt(amount) > 1000) {
                ans.add(t);
                continue;
            }
            Triplet<String, Integer, String, Integer> triplet = new Triplet(name, Integer.parseInt(time), city, Integer.parseInt(amount));

            List<Triplet> data = new ArrayList<>();
            data.add(triplet);
            if (map.get(name) != null) {
                List<Triplet> existing = map.get(name);
                existing.add(triplet);
                map.put(name, existing);
            } else map.put(name, data);

        }

        for (String key : map.keySet()) {

            List<Triplet> result = map.get(key);

            int lastTime = Integer.MAX_VALUE;
            String lastCity = "";
            int lastAmount = 0;

            for (Triplet res : result) {
                if (lastTime == Integer.MAX_VALUE) {
                    lastTime = res.time;
                    lastCity = res.city;
                    lastAmount = res.amount;
                }
                if (lastTime - res.time < 60 || !res.city.equalsIgnoreCase(lastCity)) {
                    String invalid = res.name + "," + res.time + "," + res.amount + "," + res.city;
                    ans.add(invalid);

                    String lastInValid = res.name + "," + lastTime + "," + lastAmount + "," + lastCity;
                    ans.add(lastInValid);

                    lastTime = res.time;
                    lastCity = res.city;
                    lastAmount = res.amount;
                }
            }
        }
        return new ArrayList<>(new HashSet<>(ans));
    }


    public int lengthOfLongestSubstring(String s) {

        int n = s.length();

        if (n == 0) return 0;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            HashSet<Character> set = new HashSet<Character>();
            int length = 0;
            for (int j = i; j < n; j++) {
                if (set.contains(s.charAt(j))) break;
                else {
                    length++;
                    set.add(s.charAt(j));
                }
            }
            if (length > max) {
                max = length;
            }
        }
        return max;

    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }

        int num1 = 11;
        StringBuilder sb = new StringBuilder(num1);
    }

    public int hammingWeight(int n) {
        StringBuilder number = new StringBuilder(n);
        System.out.println(number);
        int ans = 0;
        for (int i = 0; i < number.toString().length(); i++) {
            if (number.toString().charAt(i) == '1') ans++;
        }
        return ans;
    }

    List<List<Node>> ans = new ArrayList<>();

    public Node connect(Node root) {

        if (root == null)
            return null;

        List<Node> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<Node> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root

        while (!q.isEmpty()) {
            int n = q.size();
            // Dequeue an item from queue
            Node p = q.peek();
            q.remove();

            if (p != null) {
                // next element in queue represents next
                // node at current Level
                p.next = q.peek();

                result.add(p);
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
            } else if (!q.isEmpty())
                q.add(null);
        }
        return root;
    }

}

package CF_Templates;

import java.io.*;
import java.util.HashMap;
import java.util.HashMap;
import java.util.*;

import static java.lang.Math.floor;
import static java.lang.Math.sqrt;

public class topcoder {

    public static class TreeNode {
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

        static int x = -1;
        static int y = -1;

        public static int first_search(TreeNode root, TreeNode main_root) {

            if (root == null) return 0;


            int a = first_search(root.left, main_root);
            int b = first_search(root.right, main_root);

            if (a > main_root.val)
                x = a;

            if (b < main_root.val) y = b;

            return root.val;

        }


        public static void fix(TreeNode root, TreeNode main_root) {

            if (root == null) return;


            fix(root.left, main_root);
            if (root.val > main_root.val) {
                root.val = y;
            }

            fix(root.right, main_root);

            if (root.val < main_root.val) ;
            root.val = x;


        }


        public static int max(int[] nums, int s, int e) {
            int max = Integer.MIN_VALUE;

            for (int i = s; i <= e; i++) {
                max = Math.max(max, nums[i]);
            }

            return max;
        }

        public static TreeNode new_node(int[] nums, int s, int e) {


            int max = max(nums, s, e);
            TreeNode node = new TreeNode(max);

            return node;

        }

        public static TreeNode root;

        public static void res(int[] nums, int s, int e) {
            if (s > e) return;

            if (root == null) {
                root = new_node(nums, s, e);
            }
            root.left = new_node(nums, s, e);
            root.right = new_node(nums, s, e);

            return;
        }


        static int depth(TreeNode root) {

            if (root == null) return 0;
            int a = 1 + depth(root.left);
            int b = 1 + depth(root.right);

            return Math.max(a, b);
        }


        static HashSet<Integer> set = new HashSet<>();

        static void deepestLeaves(TreeNode root, int cur_depth, int depth) {

            if (root == null) return;
            if (cur_depth == depth) set.add(root.val);

            deepestLeaves(root.left, cur_depth + 1, depth);
            deepestLeaves(root.right, cur_depth + 1, depth);


        }

        public static void print(TreeNode root) {

            if (root == null) return;

            System.out.print(root.val + " ");
            System.out.println("er");
            print(root.left);
            print(root.right);
        }

        public static HashSet<Integer> original(TreeNode root) {
            int d = depth(root);
            deepestLeaves(root, 0, d);
            return set;
        }

        static HashSet<Integer> set1 = new HashSet<>();

        static void leaves(TreeNode root) {

            if (root == null) return;

            if (root.left == null && root.right == null) set1.add(root.val);

            leaves(root.left);
            leaves(root.right);


        }

        public static boolean check(HashSet<Integer> s, HashSet<Integer> s1) {

            if (s.size() != s1.size()) return false;

            for (int a : s) {
                if (!s1.contains(a)) return false;
            }

            return true;
        }

        static TreeNode subTree;

        public static void smallest_subTree(TreeNode root) {

            if (root == null) return;

            smallest_subTree(root.left);
            smallest_subTree(root.right);


            set1 = new HashSet<>();
            leaves(root);
            boolean smallest = check(set, set1);

            if (smallest) {
                subTree = root;
                return;

            }
        }

        public static TreeNode answer(TreeNode root) {
            smallest_subTree(root);

            return subTree;
        }
    }

    static class pair {

        long first;
        long second;

        public pair(long first, long second) {
            this.first = first;
            this.second = second;
        }

        public long compareTo(pair p) {
            if (first == p.first) return second - p.second;
            return first - p.first;
        }
    }


    static class Compare {

        static void compare(ArrayList<pair> arr, long n) {

            Collections.sort(arr, new Comparator<pair>() {
                public int compare(pair p1, pair p2) {
                    return (int) (p1.first - p2.first);
                }
            });

        }
    }

    public static HashMap<Integer, Integer> sortByValue(HashMap<Integer, Integer> hm) {


        List<Map.Entry<Integer, Integer>> list = new LinkedList<Map.Entry<Integer, Integer>>(hm.entrySet());

        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
            public int compare(Map.Entry<Integer, Integer> o1,
                               Map.Entry<Integer, Integer> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        HashMap<Integer, Integer> temp = new LinkedHashMap<Integer, Integer>();
        for (Map.Entry<Integer, Integer> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }

        return temp;
    }


    static class pairr implements Comparable<pairr> {
        static Long value;
        Long index;

        public pairr(Long value, Long index) {
            this.value = value;
            this.index = index;
        }

        public int compareTo(pairr o) {
            return (int) (value - o.value);
        }
    }


    static class Key<K1, K2> {
        public K1 key1;
        public K2 key2;

        public Key(K1 key1, K2 key2) {
            this.key1 = key1;
            this.key2 = key2;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }

            if (o == null || getClass() != o.getClass()) {
                return false;
            }

            Key key = (Key) o;
            if (key1 != null ? !key1.equals(key.key1) : key.key1 != null) {
                return false;
            }

            if (key2 != null ? !key2.equals(key.key2) : key.key2 != null) {
                return false;
            }

            return true;
        }

        @Override
        public int hashCode() {
            int result = key1 != null ? key1.hashCode() : 0;
            result = 31 * result + (key2 != null ? key2.hashCode() : 0);
            return result;
        }

        @Override
        public String toString() {
            return "[" + key1 + ", " + key2 + "]";
        }
    }


    public static int sumOfDigits(long n) {

        int sum = 0;

        while (n > 0) {
            sum += n % 10;
            n /= 10;
        }

        return sum;

    }


    public static long binary_search(int s, int e, long num, long[] ar) {

        if (s > e) {
            return -1;
        }


        int mid = (s + e) / 2;

        if (s == e && ar[s] >= num) {
            return ar[s];
        } else if (s == e && ar[s] < num) {
            return -1;
        } else if (ar[mid] < num) {
            return binary_search(mid + 1, e, num, ar);
        } else if (ar[mid] >= num) {
            return binary_search(s, mid, num, ar);
        }

        return -1;

    }


    public static int index_search(int s, int e, long num, long[] ar) {

        if (s > e) {
            return -1;
        }


        int mid = (s + e) / 2;

        if (s == e && ar[s] >= num) {
            return s;
        } else if (s == e && ar[s] < num) {
            return -1;
        } else if (ar[mid] < num) {
            return index_search(mid + 1, e, num, ar);
        } else if (ar[mid] >= num) {
            return index_search(s, mid, num, ar);
        }

        return -1;

    }

    public static void swap(int[] ar, int i, int j) {


        for (int k = j; k >= i; k--) {
            int temp = ar[k];
            ar[k] = ar[k + 1];
            ar[k + 1] = temp;
        }
    }

    public static boolean digit_exists(long n) {

        while (n > 0) {
            if (n % 10 == 9)
                return true;
            n = n / 10;
        }

        return false;
    }

    public static int log(int n) {

        int c = 0;
        while (n > 0) {
            c++;
            n /= 2;
        }

        return c;
    }

    public static int findOr(int[] bits) {
        int or = 0;
        for (int i = 0; i < 32; i++) {
            or = or << 1;
            if (bits[i] > 0)
                or = or + 1;
        }
        return or;
    }

    static void simpleSieve(int limit, Vector<Integer> prime) {
        // Create a boolean array "mark[0..n-1]" and initialize
        // all entries of it as true. A value in mark[p] will
        // finally be false if 'p' is Not a prime, else true.
        boolean mark[] = new boolean[limit + 1];

        for (int i = 0; i < mark.length; i++)
            mark[i] = true;

        for (int p = 2; p * p < limit; p++) {
            // If p is not changed, then it is a prime
            if (mark[p] == true) {
                // Update all multiples of p
                for (int i = p * p; i < limit; i += p)
                    mark[i] = false;
            }
        }

        // Print all prime numbers and store them in prime
        for (int p = 2; p < limit; p++) {
            if (mark[p] == true) {
                prime.add(p);
            }
        }
    }

    // Prints all prime numbers smaller than 'n'
    public static void segmentedSieve(int n, ArrayList<Integer> l) {
        // Compute all primes smaller than or equal
        // to square root of n using simple sieve
        int limit = (int) (floor(sqrt(n)) + 1);
        Vector<Integer> prime = new Vector<>();

        simpleSieve(limit, prime);

        // Divide the range [0..n-1] in different segments
        // We have chosen segment size as sqrt(n).
        int low = limit;
        int high = 2 * limit;

        // While all segments of range [0..n-1] are not processed,
        // process one segment at a time
        while (low < n) {
            if (high >= n)
                high = n;

            // To mark primes in current range. A value in mark[i]
            // will finally be false if 'i-low' is Not a prime,
            // else true.
            boolean mark[] = new boolean[limit + 1];

            for (int i = 0; i < mark.length; i++)
                mark[i] = true;

            // Use the found primes by simpleSieve() to find
            // primes in current range
            for (int i = 0; i < prime.size(); i++) {
                // Find the minimum number in [low..high] that is
                // a multiple of prime.get(i) (divisible by prime.get(i))
                // For example, if low is 31 and prime.get(i) is 3,
                // we start with 33.
                int loLim = (int) (floor(low / prime.get(i)) * prime.get(i));
                if (loLim < low)
                    loLim += prime.get(i);

			                /*  Mark multiples of prime.get(i) in [low..high]:
			                    We are marking j - low for j, i.e. each number
			                    in range [low, high] is mapped to [0, high-low]
			                    so if range is [50, 100]  marking 50 corresponds
			                    to marking 0, marking 51 corresponds to 1 and
			                    so on. In this way we need to allocate space only
			                    for range  */
                for (int j = loLim; j < high; j += prime.get(i))
                    mark[j - low] = false;
            }

            // Numbers which are not marked as false are prime
            for (int i = low; i < high; i++)
                if (mark[i - low] == true)
                    l.add(i);

            // Update low and high for next segment
            low = low + limit;
            high = high + limit;
        }
    }

    public static int find_indexNum(long k) {

        long k1 = k;
        int power = 0;

        while (k > 0) {
            power++;
            k /= 2;
        }

        long check = (long) Math.pow(2, power - 1);
        if (k1 == check) {
            return power;
        }
        //   System.out.println(power);
        long f = (long) Math.pow(2, power - 1);
        long rem = k1 - f;
        return find_indexNum(rem);
    }

    public static void sortPair(ArrayList<pair> l, int n) {
        n = l.size();

        Compare obj = new Compare();
        obj.compare(l, n);
    }


    public static void shuffle(int[] array, int num, int t_index, boolean[] vis, int m) {


        for (int i = 0; i < m; i++) {
            if (vis[i] == false) {


                int temp = array[i];
                if (i < t_index) {
                    vis[i] = true;
                }
                array[i] = num;
                array[t_index] = temp;
                //	System.out.println(array[t_index]+" "+array[i]);
                break;
            }
        }
    }

    public static void rotate(int[] arr, int j, int times, int m) {


        if (j == 0) {
            int temp1 = arr[0];
            arr[0] = arr[times];
            arr[times] = temp1;

        } else {
            int temp = arr[j];
            int z = arr[0];

            arr[0] = arr[times];

            arr[j] = z;
            arr[times] = temp;


        }

    }

    public static void recur(int i, int A, int B, int[] dp, int[] metal, int n, boolean took, int ind) {

        if (i - A <= 0 && i - B <= 0) return;
        int count = 0;

        for (int j = 1; j <= n; j++) {
            if (dp[j] >= metal[j]) {
                count++;
            }
        }

        if (count == n) return;
        if (i - A >= 0 && i - B >= 0 && dp[i] > 0 && dp[i] > metal[i]) {
            dp[i]--;

            dp[i - A]++;

            dp[i - B]++;
        }


        if (ind == 6) {
            //	System.out.println(Arrays.toString(dp));
        }
        recur(i - A, A, B, dp, metal, n, took, ind);
        recur(i - B, A, B, dp, metal, n, took, ind);
    }


    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (int i = 5; i * i <= n; i = i + 6)
            if (n % i == 0 || n % (i + 2) == 0)
                return false;
        return true;
    }

    public static boolean[] getSieve(int n) {
        boolean[] isPrime = new boolean[n + 1];
        for (int i = 2; i <= n; i++) isPrime[i] = true;
        for (int i = 2; i * i <= n; i++)
            if (isPrime[i])
                for (int j = i; i * j <= n; j++) isPrime[i * j] = false;
        return isPrime;
    }


    public static long gcd(long a, long b) {
        if (a == 0)
            return b;

        return gcd(b % a, a);
    }


    public static void dfs(LinkedList<Integer>[] list, HashMap<Integer, Integer> map, int parent, int n) {

        Stack<Integer> st = new Stack<>();


    }

    public static boolean pos(int n) {


        int i = 1;
        boolean pos = false;

        while (i * i <= n) {
            if (i * i * 2 == n || i * i * 4 == n) {
                pos = true;
                break;
            }
            i++;
        }
        if (pos) return true;
        return false;

    }

    static long count = 0;

    public static void pairs(int[] ar, int s, int e) {

        if (e <= s) return;
        //   System.out.println(ar[s]+" "+ar[e]+" "+s+" "+e);
        if (ar[e] - ar[s] == e - s) {
            count++;
            //System.out.println("sdf");
        }

        pairs(ar, s + 1, e);
        pairs(ar, s, e - 1);

    }


    public static class Pair1 implements Comparable<Pair1> {
        int value;
        int index;

        public Pair1(int value, int index) {
            this.value = value;
            this.index = index;

        }

        public int compareTo(Pair1 o) {
            return o.value - value;
        }


    }

    public static long ways(long n) {

        return (n * (n - 1)) / 2;
    }


    static boolean isPrime(long n) {

        if (n <= 1) return false;
        if (n <= 3) return true;

        if (n % 2 == 0 || n % 3 == 0) return false;


        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0)
                return false;
        }

        return true;
    }

    static long nextPrime(long n) {

        boolean found = false;
        long prime = n;

        while (!found) {
            prime++;
            if (isPrime(prime))
                found = true;
        }

        return prime;
    }

    public static boolean isValid(int h, int m, int hour, int minute) {
        int a = flip(hour / 10);
        if (a == -1) return false;
        int b = flip(hour % 10);
        if (b == -1) return false;
        int c = flip(minute / 10);
        if (c == -1) return false;
        int d = flip(minute % 10);
        if (d == -1) return false;
        if (10 * d + c >= h) return false;
        if (10 * b + a >= m) return false;
        return true;
    }

    public static int flip(int x) {
        if (x == 0) return 0;
        if (x == 1) return 1;
        if (x == 2) return 5;
        if (x == 5) return 2;
        if (x == 8) return 8;
        return -1;
    }

    static long maximum(long a, long b, long c, long d) {

        long m = Math.max(a, b);
        long m1 = Math.max(c, d);
        return Math.max(m1, m1);
    }

    static long minimum(long a, long b, long c, long d) {

        long m = Math.min(a, b);
        long m1 = Math.min(c, d);
        return Math.min(m, m1);
    }

    static long ans = 0;

    public static void solve1(boolean[][] vis, long[][] mat, int r, int c, int r2, int c2, int r1, int c1, int r3, int c3) {

        if (r > r1 || c > c1 || r > r2 || c > c2 || r1 > r3 || c1 > c3 || r3 < r2 || c3 < c2 ||
                vis[r][c] || vis[r1][c1] || vis[r2][c2] || vis[r3][c3])
            return;

        vis[r][c] = true;
        vis[r1][c1] = true;
        vis[r2][c2] = true;
        vis[r3][c3] = true;
        long max = maximum(mat[r][c], mat[r1][c1], mat[r2][c2], mat[r3][c3]);
        long min = minimum(mat[r][c], mat[r1][c1], mat[r2][c2], mat[r3][c3]);

        long a = mat[r][c];
        long b = mat[r1][c1];
        long c4 = mat[r2][c2];
        long d = mat[r3][c3];
        long[] p = {a, b, c4, d};
        Arrays.sort(p);
        long temp = (p[2] + p[3] - p[0] - p[1]);
        if (r == r1 && r == r2 && r2 == r3 && r1 == r3)
            temp /= 2;
        System.out.println(Arrays.toString(p));
        ans += temp;


        solve1(vis, mat, r + 1, c, r2 + 1, c2, r1 - 1, c1, r3 - 1, c3);
        solve1(vis, mat, r, c + 1, r2, c2 - 1, r1, c1 + 1, r3, c3 - 1);
        solve1(vis, mat, r + 1, c + 1, r2 + 1, c2 - 1, r1 - 1, c1 + 1, r3 - 1, c3 - 1);

    }

    public static int dfs(int parent, LinkedList<Integer>[] list) {


        for (int i : list[parent]) {
            if (list[parent].size() == 0) {
                return 0;
            } else {
                return 1 + dfs(i, list);
            }
        }

        return 0;

    }


    public static long answer = Integer.MAX_VALUE;


    public static void min_Time(int[][] dp, int i, HashSet<Integer> set, int min, int r, int c) {

        if (i > r) {
            answer = Math.min(answer, min);
            return;
        }
        if (min > answer) return;

        for (int j = i; j <= c; j++) {
            if (!set.contains(j)) {
                set.add(j);
                min += dp[i][j];
                min_Time(dp, i + 1, set, min, r, c);
                min -= dp[i][j];
                set.remove(j);
            }
        }
    }


    public static void dp(int[][] dp, int r, int c, int o, int z, long sum) {

        if (r > o) {
            answer = Math.min(answer, sum);
        }
        if (r > o || c > z) {
            return;
        }
        if (sum > answer) return;
        sum += dp[r][c];

        dp(dp, r + 1, c + 1, o, z, sum);
        sum -= dp[r][c];
        dp(dp, r, c + 1, o, z, sum);

    }

    static HashSet<ArrayList<Integer>> l = new HashSet<>();


    public static void fourSum(Deque<Integer> ll, int i, int target, int[] ar, int n) {

        if (ll.size() == 4) {
            int sum = 0;
            ArrayList<Integer> list = new ArrayList<>();

            for (int a : ll) {
                sum += a;
                list.add(a);
            }

            if (sum == target) {

                Collections.sort(list);


                l.add(list);
                //	System.out.println(ll);
            }

            return;
        }

        for (int j = i; j < n; j++) {

            ll.add(ar[j]);

            fourSum(ll, j + 1, target, ar, n);
            ll.removeLast();
        }
    }


    static int max_bottles(int cur, int exchange, int n) {
        if (cur == exchange) {
            cur = 0;
            n++;

        }
        if (n == 0) return 0;


        return 1 + max_bottles(cur + 1, exchange, n - 1);
    }


    public static void fill(int[][] mat, List<Integer> ans, int row_start, int row_end, int col_start, int col_end) {


        for (int i = col_start; i <= col_end; i++) {
            ans.add(mat[row_start][i]);
        }

        for (int i = row_start + 1; i <= row_end; i++) {
            ans.add(mat[i][col_end]);
        }
        if (col_start == col_end) return;
        if (row_start == row_end) return;
        for (int i = col_end - 1; i >= col_start; i--) {
            ans.add(mat[row_end][i]);
        }

        for (int i = row_end - 1; i >= row_start + 1; i--) {
            ans.add(mat[i][col_start]);
        }
    }


    public static void create(int[][] mat, int j, int i, int k) {

        if (i < 1 || j >= mat.length) return;

        mat[j][i] = k;
        create(mat, j + 1, i - 1, k + 1);


    }

    public static long sum(int[][] mat, int x1, int y1, int x2, int y2) {


        long sum = 0;
        while (x1 <= x2) {
            sum += mat[x1][y1];
            //   System.out.println(mat[x1][y1]);
            x1++;
        }
        y1++;
        while (y1 <= y2) {
            sum += mat[x2][y1];
            y1++;
        }

        return sum;
    }

    public static boolean allneg(int[] ar, int n) {

        for (int i = 0; i < n; i++) {
            if (ar[i] >= 0) return false;
        }

        return true;
    }

    public static boolean allpos(int[] ar, int n) {

        for (int i = 0; i < n; i++) {
            if (ar[i] <= 0) return false;
        }

        return true;
    }

    public static int max_pos(int[] ar, int n) {

        int min = Integer.MAX_VALUE;

        for (int i = 1; i < n; i++) {
            if (ar[i] > 0) {
                break;
            }
            int a = Math.abs(ar[i] - ar[i - 1]);
            min = Math.min(min, a);
        }
        int c = 0;
        boolean zero = false;
        TreeSet<Integer> set = new TreeSet<>();
        int neg = 0;
        for (int i = 0; i < n; i++) {
            if (ar[i] <= 0) {
                neg++;
                if (ar[i] == 0) zero = true;
                continue;
            }
            if (ar[i] <= min) {
                c = 1;
            }
        }

        neg += c;
        return neg;


    }


    static final int MAX = 10000000;

    // prefix[i] is going to store count
    // of primes till i (including i).
    static int prefix[] = new int[MAX + 1];

    static void buildPrefix() {

        // Create a boolean array "prime[0..n]". A
        // value in prime[i] will finally be false
        // if i is Not a prime, else true.
        boolean prime[] = new boolean[MAX + 1];
        Arrays.fill(prime, true);

        for (int p = 2; p * p <= MAX; p++) {

            // If prime[p] is not changed, then
            // it is a prime
            if (prime[p] == true) {

                // Update all multiples of p
                for (int i = p * 2; i <= MAX; i += p)
                    prime[i] = false;
            }
        }

        // Build prefix array
        prefix[0] = prefix[1] = 0;
        for (int p = 2; p <= MAX; p++) {
            prefix[p] = prefix[p - 1];
            if (prime[p])
                prefix[p]++;
        }
    }

    static int query(int L, int R) {
        return prefix[R] - prefix[L - 1];
    }

    static void alter(int n) {

        int ans = 0;


        boolean[] vis = new boolean[n + 1];


        for (int i = 2; i <= n; i++) {
            boolean p = false;
            if (vis[i] == false) {
                for (int j = i; j <= n; j += i) {
                    if (vis[j] == true) {
                        p = true;
                    } else {
                        vis[j] = true;
                    }
                }
                if (!p) ans++;
            }
        }

        System.out.println(ans);
    }

    public static void solveDK(int[] dp, int i, int D, int K) {

        int d = D / K;
        int ans = -1;
        int ind = d + 1;

        while (ind < i) {
            int temp = i / ind;

            temp--;
            if (dp[temp * ind] == temp) {
                ans = dp[temp * ind] + 1;
                dp[i] = ans;
                break;
            }
            ind = ind * 2;
        }
        if (ans == -1)
            dp[i] = 1;


    }


    public static void solveKD(int[] dp, int i, int D, int K) {

        int d = K / D;
        int ans = -1;
        int ind = d + 1;

        while (ind < i) {
            int temp = i / ind;

            temp--;
            if (dp[temp * ind] == temp) {
                ans = dp[temp * ind] + 1;
                dp[i] = ans;
                break;
            }
            ind = ind * 2;
        }
        if (ans == -1)
            dp[i] = 1;


    }

    static int countGreater(int arr[], int n, int k) {
        int l = 0;
        int r = n - 1;

        // Stores the index of the left most element
        // from the array which is greater than k
        int leftGreater = n;

        // Finds number of elements greater than k
        while (l <= r) {
            int m = l + (r - l) / 2;

            // If mid element is greater than
            // k update leftGreater and r
            if (arr[m] > k) {
                leftGreater = m;
                r = m - 1;
            }

            // If mid element is less than
            // or equal to k update l
            else
                l = m + 1;
        }

        // Return the count of elements greater than k
        return (n - leftGreater);
    }

    static ArrayList<Integer> printDivisors(int n) {
        // Note that this loop runs till square root

        ArrayList<Integer> list = new ArrayList<>();

        for (int i = 1; i <= Math.sqrt(n); i++) {
            if (n % i == 0) {
                // If divisors are equal, print only one
                if (n / i == i)
                    list.add(i);

                else // Otherwise print both
                    list.add(i);
                list.add(n / i);
            }
        }

        return list;
    }


    static boolean isPossible(String s, String str, int i, int j) {
        //	 System.out.println(i+" "+j);
        int x = i;
        int y = j;
        while (i >= 0 && j < str.length()) {
            if (s.charAt(i) != str.charAt(j)) {
                break;
            }
            i--;
            j++;
        }

        if (j == str.length()) {
            System.out.println(x + " " + y);
            return true;
        }

        return false;
    }


    static void leftRotate(int l, int r, int arr[], int d) {
        for (int i = 0; i < d; i++)
            leftRotatebyOne(l, r, arr);
    }

    static void leftRotatebyOne(int l, int r, int arr[]) {
        int i, temp;
        temp = arr[l];
        for (i = l; i < r; i++)
            arr[i] = arr[i + 1];
        arr[r] = temp;
    }

    static class Pair {
        int x;
        int y;

        // Constructor
        public Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    static class Compare1 {

        static void compare(Pair arr[], int n) {
            // Comparator to sort the pair according to second element
            Arrays.sort(arr, new Comparator<Pair>() {
                @Override
                public int compare(Pair p1, Pair p2) {
                    return p2.x - p1.x;
                }
            });


        }
    }


    static long modInverse(long a, long m) {
        long m0 = m;
        long y = 0, x = 1;

        if (m == 1)
            return 0;

        while (a > 1) {
            // q is quotient
            long q = a / m;

            long t = m;

            // m is remainder now, process
            // same as Euclid's algo
            m = a % m;
            a = t;
            t = y;

            // Update x and y
            y = x - q * y;
            x = t;
        }

        // Make x positive
        if (x < 0)
            x += m0;

        return x;
    }

    static long power(long x, long y, long p) {
        long res = 1; // Initialize result

        x = x % p; // Update x if it is more than or
        // equal to p

        if (x == 0)
            return 0; // In case x is divisible by p;

        while (y > 0) {

            // If y is odd, multiply x with result
            if ((y & 1) != 0)
                res = (res * x) % p;

            // y must be even now
            y = y >> 1; // y = y/2
            x = (x * x) % p;
        }
        return res;
    }

    static long solvetheproblem(long n, long k) {

        long mod = 1000000007;

        long ansss = 0;
        while (n > 0) {

            // Nearest power of 2<=N
            long p = (long) (Math.log(n) / Math.log(2));
            ;

            // Now insert k^p in the answer
            long temp = (long) power(k, p, mod);
            ansss += temp;
            ansss %= mod;


            // update n
            n %= (long) Math.pow(2, p);
        }

        // Print the ans in sorted order


        return ansss % mod
                ;
    }

    static boolean pos(int[][] mat, int r, int c, int n, boolean[][] vis) {
        if (r <= 0 || c <= 0 || r > 2 || c > n) return false;
        if (r == 2 && c == n) return true;

        if (vis[r][c]) return false;
        vis[r][c] = true;
        if (mat[r][c] == 1) return false;

        boolean a = pos(mat, r + 1, c, n, vis);
        boolean b = pos(mat, r, c + 1, n, vis);
        boolean d = pos(mat, r + 1, c + 1, n, vis);
        boolean e = pos(mat, r - 1, c + 1, n, vis);

        return a || b || d || e;
    }


    static long sameremdiv(long x, long y) {
        if (x <= y) {
            y = x - 1;
        }

        long sq = (long) Math.sqrt(x);


        if (y <= sq) {
            y--;
            long ans = (y * (y + 1)) / 2;
            return ans;
        } else {
            long ans = 0;
            long dif = y - sq;
            sq -= 2;
            if (sq > 1)
                ans += (sq * (sq + 1)) / 2;
            long d = x / y;
            sq += 2;
            for (int i = (int) sq; i <= y; i++) {
                if (i > 1) {
                    long temp = x / i;
                    if (x % i < temp) temp--;
                    ans += temp;
                }
            }

            return ans;
        }
    }

    static int binary(long[] ar, long element, int s, int e) {

        int mid = (s + e) / 2;
        //  System.out.println(mid);
        if (s > e) return mid;
        if (ar[mid] == element) {
            return mid;
        } else if (ar[mid] > element) {
            return binary(ar, element, s, mid - 1);
        } else {
            return binary(ar, element, mid + 1, e);
        }


    }

    static boolean isGibbrish(HashSet<String> set, String str, int j) {


        StringBuilder sb = new StringBuilder();
        if (j >= str.length()) {
            return true;
        }
        for (int i = j; i < str.length(); i++) {
            sb.append(str.charAt(i));
            String temp = sb.toString();
            if (set.contains(temp)) {
                boolean test = isGibbrish(set, str, i + 1);
                if (test) return true;
            }
        }

        return false;


    }


    static TreeNode buildTree(TreeNode root, int[] ar, int l, int r) {

        if (l > r) return null;
        int len = l + r;
        if (len % 2 != 0) len++;
        int mid = (len) / 2;
        int v = ar[mid];


        TreeNode temp = new TreeNode(v);
        root = temp;
        root.left = buildTree(root.left, ar, l, mid - 1);
        root.right = buildTree(root.right, ar, mid + 1, r);

        return root;

    }

    public static int getClosest(int val1, int val2,
                                 int target) {
        if (target - val1 >= val2 - target)
            return val2;
        else
            return val1;
    }

    public static int findClosest(int start, int end, int arr[], int target) {
        int n = arr.length;

        // Corner cases
        if (target <= arr[start])
            return arr[start];
        if (target >= arr[end])
            return arr[end];

        // Doing binary search
        int i = start, j = end + 1, mid = 0;
        while (i < j) {
            mid = (i + j) / 2;

            if (arr[mid] == target)
                return arr[mid];

                      /* If target is less than array element,
                         then search in left */
            if (target < arr[mid]) {

                // If target is greater than previous
                // to mid, return closest of two
                if (mid > 0 && target > arr[mid - 1])
                    return getClosest(arr[mid - 1],
                            arr[mid], target);

                /* Repeat for left half */
                j = mid;
            }

            // If target is greater than mid
            else {
                if (mid < n - 1 && target < arr[mid + 1])
                    return getClosest(arr[mid],
                            arr[mid + 1], target);
                i = mid + 1; // update i
            }
        }

        // Only single element left after search
        return arr[mid];
    }

    static int lis(int arr[], int n) {
        int lis[] = new int[n];
        int i, j, max = 0;

        /* Initialize LIS values for all indexes */
        for (i = 0; i < n; i++)
            lis[i] = 1;

                  /* Compute optimized LIS values in
                     bottom up manner */
        for (i = 1; i < n; i++)
            for (j = 0; j < i; j++)
                if (arr[i] >= arr[j] && lis[i] < lis[j] + 1)
                    lis[i] = lis[j] + 1;

        /* Pick maximum of all LIS values */
        for (i = 0; i < n; i++)
            if (max < lis[i])
                max = lis[i];

        return max;
    }

    static List<List<Integer>> res = new ArrayList<>();

    public static void lists(List<Integer> list, List<Integer> temp, int target, int j) {

        int sum = 0;
        for (int i = 0; i < temp.size(); i++) {
            sum += temp.get(i);
        }
        if (sum > target) {
            return;
        } else if (sum == target) {
            Collections.sort(temp);
            boolean exist = false;
            for (List<Integer> l : res) {
                if (l.size() == temp.size()) {
                    int c = 0;
                    for (int i = 0; i < l.size(); i++) {
                        if (l.get(i) == temp.get(i)) c++;
                    }
                    if (c == l.size()) {
                        exist = true;
                        break;
                    }
                }
            }
            if (!exist) {
                res.add(new ArrayList<>(temp));
            }


        } else if (j == list.size()) return;
        for (int i = j; i < list.size(); i++) {
            temp.add(list.get(i));
            lists(list, temp, target, i + 1);
            temp.remove(temp.size() - 1);

        }
        return;

    }


    static ArrayList<Integer> dfs(int p, int c, ArrayList<Integer> l, HashMap<Integer, Integer> map,
                                  LinkedList<Integer>[] list, int v) {
        l.add(map.get(c));
        System.out.println(c);
        if (c == v) {
            //System.out.println(c);
            return l;
        }
        for (int j : list[c]) {
            if (c == 1) {
                //   System.out.println("Yes"+" "+j);
                //	 System.out.println("yes"+" "+list[c]);
            }
            if (j != p) {

                dfs(c, j, l, map, list, v);

            }
        }


        return new ArrayList<>();


    }


    static boolean pos(char[] a, int j) {
        char[] ar = new char[a.length];
        for (int i = 0; i < a.length; i++) {
            ar[i] = a[i];
        }
        ar[j] = 'c';
        ar[j - 1] = 'a';
        ar[j - 2] = 'b';
        ar[j - 3] = 'a';
        ar[j + 1] = 'a';
        ar[j + 2] = 'b';
        ar[j + 3] = 'a';
        int count = 0;

        for (int i = 3; i < ar.length - 3; i++) {
            if (ar[i] == 'c' && ar[i - 1] == 'a' && ar[i - 2] == 'b' && ar[i - 3] == 'a'
                    && ar[i + 1] == 'a' && ar[i + 2] == 'b' && ar[i + 3] == 'a') {
                count++;
            }
        }

        if (count == 1) return true;

        return false;

    }

    static void bruteforce(String s) {

        String ans = s.charAt(0) + "";
        ans += ans;
        StringBuilder sb = new StringBuilder();
        sb.append(s.charAt(0));

        for (int i = 1; i < s.length(); i++) {
            String d = sb.toString();
            sb.reverse();
            d += sb.toString();
            boolean con = true;
            System.out.println(d + " " + s);
            for (int j = 0; j < Math.min(d.length(), s.length()); j++) {
                if (s.charAt(j) > d.charAt(j)) {
                    con = false;
                    break;

                }
            }
            sb.reverse();
            sb.append(s.charAt(i));

            if (con) {
                ans = d;

                break;
            }
        }
        System.out.println(ans + " " + "yes");


    }

    static void permute(String s, String answer) {
        if (s.length() == 0) {
            System.out.print(answer + "  ");
            return;
        }

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            String left_substr = s.substring(0, i);
            String right_substr = s.substring(i + 1);
            String rest = left_substr + right_substr;
            permute(rest, answer + ch);
        }
    }


    static List<List<Integer>> result = new ArrayList<>();

    public static void comb(List<Integer> list, int n, int[] ar, HashSet<Integer> set) {


        if (list.size() == n) {

            boolean exist = false;

            for (List<Integer> l : result) {
                int c = 0;
                for (int i = 0; i < l.size(); i++) {
                    if (l.get(i) == list.get(i)) c++;
                }
                if (c == n) {
                    exist = true;
                    break;
                }
            }
            if (!exist) result.add(new ArrayList<>(list));
        }


        for (int j = 0; j < n; j++) {
            if (!set.contains(j)) {
                list.add(ar[j]);
                set.add(j);
                comb(list, n, ar, set);
                set.remove(j);
                list.remove(list.size() - 1);

            }

        }
        return;

    }

    static void pinkSeat(int[][] mat, int i, int j, int n, int m, boolean vis[][], int[][] res) {
        if (i <= 0 || j <= 0 || i > n || j > m || vis[i][j]) return;

        int a = Math.abs(i - 1);
        int b = Math.abs(j - 1);
        int c = Math.abs(i - 1);
        int d = Math.abs(j - m);
        int e = Math.abs(i - n);
        int f = Math.abs(j - m);
        int x = Math.abs(i - n);
        int y = Math.abs(j - 1);
        vis[i][j] = true;
        int max = Math.max(a + b, c + d);
        max = Math.max(max, e + f);
        max = Math.max(max, x + y);

        res[i][j] = max;
        pinkSeat(mat, i - 1, j - 1, n, m, vis, res);
        pinkSeat(mat, i + 1, j - 1, n, m, vis, res);
        pinkSeat(mat, i - 1, j + 1, n, m, vis, res);
        pinkSeat(mat, i + 1, j + 1, n, m, vis, res);
        pinkSeat(mat, i, j - 1, n, m, vis, res);
        pinkSeat(mat, i, j + 1, n, m, vis, res);
        pinkSeat(mat, i - 1, j, n, m, vis, res);
        pinkSeat(mat, i + 1, j, n, m, vis, res);

    }

    public static void main(String args[]) throws IOException {


        //	 System.setIn(new FileInputStream("Case.txt"));
        BufferedReader ob = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));


        int t = Integer.parseInt(ob.readLine());


        while (t-- > 0) {

	 	            	/*
	 	            	StringTokenizer st = new StringTokenizer(ob.readLine());
	 	            	int n = Integer.parseInt(st.nextToken());
	 	            	int n = Integer.parseInt(ob.readLine());
	 	            	int []ar = new int[n];

	 	            	 */

            //  int n = Integer.parseInt(ob.readLine());

            String s = ob.readLine();
            int f = -1;
            int sec = -1;
            int ind = -1;
            int ind2 = -1;
            for (int i = s.length() - 1; i >= 1; i--) {
                int a = s.charAt(i) - '0';
                int b = s.charAt(i - 1) - '0';
                if (a + b >= 10) {
                    int v = a + b;
                    f = 1;
                    sec = v % 10;
                    ind = i - 1;
                    ind2 = i;
                    break;
                }
            }

            int a = 2;
            char v = (char) (a + '0');

            if (f != -1) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < ind; i++) {
                    sb.append(s.charAt(i));
                }
                sb.append('1');
                char v1 = (char) (sec + '0');
                sb.append(v1);
                for (int i = ind2 + 1; i < s.length(); i++) {
                    sb.append(s.charAt(i));
                }
                String ans = sb.toString();
                System.out.println(ans);
            } else {
                int x = s.charAt(0) - '0';
                int y = s.charAt(1) - '0';
                char h = (char) (x + y + '0');
                StringBuilder sb = new StringBuilder();
                sb.append(h);
                for (int i = 2; i < s.length(); i++) {
                    sb.append(s.charAt(i));
                }

                String ans = sb.toString();
                System.out.println(ans);
            }
        }
    }
}

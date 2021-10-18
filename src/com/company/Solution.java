package com.company;

import java.awt.*;
import java.math.BigInteger;
import java.util.List;
import java.util.*;
import java.util.Map.Entry;

class Solution {


    public static void usingPointsAsKeys() {
        Point points[] =
                {
                        new Point(1, 2),
                        new Point(1, 1),
                        new Point(5, 7)
                };
        Map<Point, Integer> map = new HashMap<Point, Integer>();
        for (int i = 0; i < points.length; i++) {
            map.put(points[i], 1);
        }
        for (Entry<Point, Integer> entry : map.entrySet()) {
            Point p = entry.getKey();
            String s = "{" + p.x + ", " + p.y + "}";
            System.out.println(s + " " + entry.getValue());
        }

        //=====================================================================
        // Important: This shows that it WILL work as expected!
        Point somePoint = new Point(1, 2);
        Integer value = map.get(somePoint);
        System.out.println("Value for " + somePoint + " is " + value);

    }


    public static void usingArraysAsKeys() {
        int[][] points =
                {
                        {1, 2},
                        {1, 1},
                        {5, 7}
                };
        HashMap<int[], Integer> map = new HashMap<int[], Integer>();
        for (int i = 0; i < points.length; i++) {
            map.put(points[i], 1);
        }
        for (Entry<int[], Integer> entry : map.entrySet()) {
            // This would print the arrays as "[I@139a55"
            //System.out.println(entry.getKey() + " " + entry.getValue());

            // This will print the arrays as [1, 2]:
            System.out.println(
                    Arrays.toString(entry.getKey()) + " " + entry.getValue());
        }

        //=====================================================================
        // Important: This shows that it will NOT work as expected!
        int somePoint[] = {1, 2};
        Integer value = map.get(somePoint);
        System.out.println(
                "Value for " + Arrays.toString(somePoint) + " is " + value);

    }


    public int nearestValidPoint(int x, int y, int[][] points) {

        Map<Point, Integer> sanityMap = new HashMap<Point, Integer>();

        for (int i = 0; i < points.length; i++) {
            if (sanityMap.containsKey(new Point(points[i][0], points[i][1]))) return 0;
            sanityMap.put(new Point(points[i][0], points[i][1]),
                    sanityMap.getOrDefault(new Point(points[i][0], points[i][1]), 0) + 1);
        }

        Map<Point, Double> distMap = new HashMap<Point, Double>();

        for (int i = 0; i < points.length; i++) {
            if ((points[i][0] == x) || (points[i][1] == y)) { // points lies on same plane
                double dist = Math.abs(points[i][0] - x) + Math.abs(points[i][1] - y);
                distMap.put(new Point(points[i][0], points[i][1]), dist);
            }
        }


        if (distMap.keySet().isEmpty()) return -1;


        // Create a list from elements of HashMap
        List<Entry<Point, Double>> list = new LinkedList<>(distMap.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Entry<Point, Double>>() {
            public int compare(Entry<Point, Double> o1,
                               Entry<Point, Double> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        Double smallestDistance = list.get(0).getValue();
        int smallestIndex = Integer.MAX_VALUE;

        for (Entry<Point, Double> sortedMap : list) {


            if ((sortedMap.getKey().x) == x && (sortedMap.getKey().y) == y) return 0;
            if (smallestDistance == sortedMap.getValue()) {
                if (smallestIndex > sortedMap.getKey().x) {
                    smallestIndex = sortedMap.getKey().x;
                } else if (smallestIndex == sortedMap.getKey().x) {
                }

            } else break;
        }
        return smallestIndex;

    }
}

class SolutionMaxSliding {
    // O(n2*logn)
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int index = 0;
        int[] arr = new int[n - k + 1];

        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());

        for (int i = 0; i < n - k + 1; i++) {
            for (int j = i; j < i + k; j++) {
                pq.add(nums[j]);
            }
            arr[index] = pq.poll();
            pq.clear();
            index++;
        }
        return arr;
    }
}


class Solution1 {
    public int nearestValidPoint(int x, int y, int[][] points) {

        Map<Point, Double> distMap = new HashMap<Point, Double>();

        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());


        double min = Double.MAX_VALUE;
        Point minPoint = new Point();

        HashMap<Character, Integer> freq = new HashMap<>();

        // Create a list from elements of HashMap
        List<Entry<Character, Integer>> list =
                new LinkedList<Entry<Character, Integer>>(freq.entrySet());

        Collections.sort(list, new Comparator<Entry<Character, Integer>>() {
            @Override
            public int compare(Entry<Character, Integer> characterIntegerEntry, Entry<Character, Integer> t1) {
                return characterIntegerEntry.getValue() - t1.getValue();
            }
        });

        for (int i = 0; i < points.length; i++) {
            if ((points[i][0] == x) && (points[i][1] == y)) return 0;
            if (points[i][0] == points[i][1]) return 0;

            if ((points[i][0] == x) || (points[i][1] == y)) { // points lies on same plane
                double dist = Math.abs(points[i][0] - x) + Math.abs(points[i][1] - y);
                if (dist < min) {
                    minPoint.x = points[i][0];
                    minPoint.y = points[i][1];
                    min = dist;
                }
            }
        }

        if (min == Integer.MAX_VALUE) return -1;
        if (minPoint.x == x) return Math.abs(minPoint.y - y);
        if (minPoint.y == y) return minPoint.x;
        return -1;
    }

    /*
input
- [1,3,-1,-3,5,3,6,7]
- 3
output
[3,3,5,5,6,7]
*/
    public String frequencySort(String s) {

        HashMap<Character, Integer> freq = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            freq.put(s.charAt(i), freq.getOrDefault(s.charAt(i), 0) + 1);
        }

        // Create a list from elements of HashMap
        List<Entry<Character, Integer>> list =
                new LinkedList<Entry<Character, Integer>>(freq.entrySet());

        // Sort the list
        list.sort(new Comparator<Entry<Character, Integer>>() {
            public int compare(Entry<Character, Integer> o1,
                               Entry<Character, Integer> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        StringBuilder sb = new StringBuilder("");
        for (Entry<Character, Integer> aa : list) {
            for (int i = 0; i < aa.getValue(); i++) {
                sb.append(aa.getKey());
            }
        }
        return sb.toString();
    }
}

// nums = [1,3,-1,-3,5,3,6,7], k = 3
class SlidingWindowMax {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] result = new int[nums.length - k + 1];
        PriorityQueue<Pair> pq = new PriorityQueue<>();
        int i = 0, j = 0;

        // loop for all values of array
        while (j < nums.length) {
            Pair pair = new Pair(j, nums[j]);
            pq.add(pair);
            // check of  pointer distance is beyond k
            // if yes then  update j else do the below logic
            if (j - i + 1 < k) {
                j++;
            } else {
                result[i] = pq.peek() != null ? pq.peek().val : 0;

                while (!pq.isEmpty() && pq.peek().key < i + 1) {
                    pq.remove();
                }
                i++;
                j++;
            }
        }
        return result;
    }


    /*
// Dry run
    i,j=0+1+1;
   result[0] = 3;
   2 < 1
   i = 1, j = 3
    result[1] = 3
    i=2, j = 4;
    result[2]=5
    i=3, j=5
    result[3]=5
    i=4, j=6
    result[4]=6
    i=5, j=7
    result[5]=7
    i =6, j=8
     */

//result[]  = [3,3,5,5,6,7]


    // This class is used to store index and corresponding value element of array
    static class Pair implements Comparable<Pair> {
        // field member and comparable implementation

        int key;
        int val;

        Pair(int key, int val) {
            this.key = key;
            this.val = val;
        }


        @Override
        public int compareTo(Pair o) {
            return o.val - this.key;
        }
    }

    class SolutionIsSumEqual {
        public boolean isSumEqual(String firstWord, String secondWord, String targetWord) {

            StringBuilder firstSum = new StringBuilder("");
            StringBuilder secondSum = new StringBuilder("");
            StringBuilder targetSum = new StringBuilder("");

            int init = (int) 'a';

            for (int i = 0; i < firstWord.length(); i++) {
                firstSum.append((int) firstWord.charAt(i) - init);
            }

            for (int i = 0; i < secondWord.length(); i++) {
                secondSum.append((int) secondWord.charAt(i) - init);
            }

            for (int i = 0; i < targetWord.length(); i++) {
                targetSum.append((int) targetWord.charAt(i) - init);
            }

            return (Integer.parseInt(firstSum.toString()) + Integer.parseInt(secondSum.toString())) == Integer.parseInt(targetSum.toString());
        }
    }


    class Solution {
        public String maxValue(String n, int x) {

            // max pq
            PriorityQueue<BigInteger> pq = new PriorityQueue<>(Collections.reverseOrder());
            String sign = "";

            BigInteger abs = new BigInteger(n);

            if (abs.compareTo(BigInteger.valueOf(0L)) < 0) {
                sign = "-";
                n = n.replace("-", "");
            }

            for (int i = 0; i < n.length(); i++) {
                BigInteger number = new BigInteger(sign + n.substring(0, i) + x + n.substring(i, n.length()));
                pq.add(number);
            }

            pq.add(new BigInteger(sign + n + x));

            assert pq.peek() != null;
            return pq.peek().toString();
        }


        /*
        jth task is assigned at time `t` then
         it will be fried again  at time t + tasks[j]
         Input: servers = [3,3,2], tasks = [1,2,3,2,1,2]
         Output: [2,2,0,2,1,2]

         */

        public PriorityQueue<Pair> insertPQ(int[] servers) {
            PriorityQueue<Pair> pq = new PriorityQueue<>(); // min-heap
            for (int i = 0; i < servers.length; i++) {
                pq.add(new Pair(i, servers[i]));
            }
            return pq;
        }

        public int[] assignTasks(int[] servers, int[] tasks) {
            int index = 0;
            int[] ans = new int[tasks.length];
            PriorityQueue<Pair> pq = insertPQ(servers);

            HashMap<Integer, Integer> map = new HashMap<>(); // ith index server blocked for `t`th seconds

            // traverse all tasks and assign to relevant servers
            for (int i = 0; i < tasks.length; i++) {
                int minIndex = pq.peek().key;

                boolean refill = false;
                while (map.get(minIndex) != null && map.get(minIndex) > 0) { // server is blocked
                    pq.remove();
                    minIndex = pq.peek().key;
                    refill = true;
                }
                if (refill) insertPQ(servers);

                map.put(minIndex, map.getOrDefault(minIndex, tasks[i]) - 1);
                ans[index] = minIndex;
                index++;

                List<Integer> triplet = new ArrayList<Integer>();
                triplet.addAll(Arrays.asList(1, 2, 3));
//                ans.add(triplet);

            }
            return ans;
        }

        // This class is used to store index and corresponding value element of array
        class Pair implements Comparable<Pair> {
            // field member and comparable implementation

            int key;
            int val;

            Pair(int key, int val) {
                this.key = key;
                this.val = val;
            }

            @Override
            public int compareTo(Pair pair) {
                return pair.val - this.val;
            }
        }
        // Dry-run
        /*



         */
    }


    class SolutionTriplet {
        public List<List<Integer>> threeSum(int[] nums) {
            int n = nums.length;
            int l, r;
            /* Sort the elements */
            quickSort(nums, 0, n - 1);
            List<List<Integer>> ans = new ArrayList<>();
            for (int i = 0; i < n - 2; i++) {
                l = i + 1;
                r = n - 1;

                while (l < r) {
                    if (nums[i] + nums[l] + nums[r] == 0) {
                        List<Integer> triplet = new ArrayList<Integer>();
                        triplet.addAll(Arrays.asList(nums[l], nums[i], nums[r]));
                        ans.add(triplet);
                    } else if (nums[i] + nums[l] + nums[r] < 0) l++;
                    else r--;
                }
            }
            return ans;
        }

        int partition(int A[], int si, int ei) {
            int x = A[ei];
            int i = (si - 1);
            int j;

            for (j = si; j <= ei - 1; j++) {
                if (A[j] <= x) {
                    i++;
                    int temp = A[i];
                    A[i] = A[j];
                    A[j] = temp;
                }
            }
            int temp = A[i + 1];
            A[i + 1] = A[ei];
            A[ei] = temp;
            return (i + 1);
        }

        /* Implementation of Quick Sort
   A[] --> Array to be sorted
   si  --> Starting index
   ei  --> Ending index
    */
        void quickSort(int A[], int si, int ei) {
            int pi;

            /* Partitioning index */
            if (si < ei) {
                pi = partition(A, si, ei);
                quickSort(A, si, pi - 1);
                quickSort(A, pi + 1, ei);
            }
        }

        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

        public boolean isScramble(String s1, String s2) {
            if (s1.length() == 1 || s2.length() == 0) return false;
            int randomNum = new Random().nextInt((s1.length()) + 1);
            while (map.get(randomNum) != null) randomNum = new Random().nextInt((s1.length()) + 1);

            String part1 = s1.substring(randomNum, s1.length());
            String part2 = s1.substring(0, randomNum);
            map.put(randomNum, 1);
            if (s2.equals(part1.concat(part2)) || s2.equals(part2.concat(part1))) return true;
            else return isScramble(part1, s2) || isScramble(part2, s2);
        }


        /*
        Input: n = 12, primes = [2,7,13,19]
Output: 32

         */
        public PriorityQueue<Integer> insertPq(int[] primes) {
            PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());// max-heap
            pq.add(1);

            for (int i = 0; i < primes.length; i++) {
                pq.add(primes[i]);
            }
            return pq;
        }

        //Function to check and return prime numbers
        boolean checkPrime(long n) {
            // Converting long to BigInteger
            BigInteger b = new BigInteger(String.valueOf(n));

            return b.isProbablePrime(1);
        }

        public int countPrimes(int n) {
            int count = 0;
            for (int i = 2; i <= n; i++) {
                if (checkPrime((long) i)) {
                    count++;
                }
            }
            return count;
        }

        public int nthSuperUglyNumber1(int n, int[] primes) {


            PriorityQueue<Long> pq = new PriorityQueue<>();// max-heap
            pq.add(1L);
            for (int prime : primes) {
                pq.add(prime * 1L);
            }

            long ans = -1L;

            while (n > 0) {
                long curr = pq.poll();
                if (curr != n) {
                    for (int prime : primes) {
                        pq.add(prime * curr);
                    }
                }
                ans = curr;
            }

            return (int) ans;
        }

        public int nthSuperUglyNumber(int n, int[] primes) {

            PriorityQueue<Integer> ans = new PriorityQueue<>(Collections.reverseOrder());// max-heap
            ans.add(1);
            PriorityQueue<Integer> pq = insertPq(primes);
            // traverse for all positive integers
            for (int i = 1; i <= n; i++) {
                boolean factor = false;
                // check if no. exists and of factor of given prime numbers
                while (i > 1) {
                    int element = pq.peek();
                    if (!pq.isEmpty()) {
                        if (i % element == 0) {
                            i = i / 10;
                            factor = true;
                        } else {
                            pq.poll();
                            factor = false;
                            break;
                        }
                    } else {
                        pq = insertPq(primes);
                        break;
                    }
                    if (factor) ans.add(i);
                }
            }

            return ans.peek();
        }

        // This class is used to store index and corresponding value element of array
        class Pair<I extends Number, I1 extends Number> implements Comparable<Pair<Number, Number>> {
            // field member and comparable implementation

            int key;
            int val;

            Pair(int key, int val) {
                this.key = key;
                this.val = val;
            }

            @Override
            public int compareTo(Pair<Number, Number> pair) {
                return pair.val - this.val;
            }
        }
    }

    class SolutionPrime {


        HashMap<Integer, Boolean> map = new HashMap<>();

        //Function to check and return prime numbers
        boolean checkPrime(int n) {

            // Check if number is less than
            // equal to 1
            if (n <= 1)
                return false;

                // Check if number is 2
            else if (n == 2)
                return true;

                // Check if n is a multiple of 2
            else if (n % 2 == 0)
                return false;

            // If not, then just check the odds
            for (int i = 3; i <= Math.sqrt(n); i += 2) {
                if (n % i == 0)
                    return false;
            }
            return true;
        }


        public int countPrimes(int n) {
            for (int i = 2; i < n; i++) {
                int multiple = 1;
                if (!map.containsKey(i)) {
                    while (multiple * i <= n) { // all multiple till must not be prime
                        map.put(multiple * i, false);
                        multiple++;
                    }
                }
            }
            System.out.println(map.size());
            int count = 0;
            for (int i = 2; i < n; i++) {
                //if a factor exists in map then it must not be prime
                if (map.containsKey(n)) continue;
                if (checkPrime(i)) {
                    count++;
                }
            }
            return count;
        }

        public int maxCoins(int[] nums) {
            int n = nums.length;
            if (n == 0) return 0;
            if (n == 1) return nums[0];
            if (n == 2) return nums[0] * nums[1] + nums[1];
            List<Integer> arrayList = new ArrayList<>();
            int sum = 0;
            for (int i = 0; i < n; i++) {
                arrayList.add(nums[i]);
            }

            while (arrayList.size() > 2) {
                arrayList.forEach(System.out::println);
                sum += (arrayList.get(0) * arrayList.get(1) * arrayList.get(2));
                arrayList.remove(1);
            }

            arrayList.forEach(System.out::println);

            System.out.println(sum);
            if (arrayList.size() == 2) sum += arrayList.get(0) * arrayList.get(1);
            if (arrayList.size() == 1) sum += arrayList.get(0);

            return sum;
        }

        int numSquaresDp(int n) {
            //O(n*sqrt(n))

            // dp[i] = min no of squares needed to make sum i.
            int[] dp = new int[n + 1];

            dp[0] = 0; //(Note this)

            for (int i = 1; i <= n; i++) {
                for (int j = 1; j * j <= i; j++)
                    dp[i] = Math.min(dp[i], dp[i - j * j] + 1); // +1 b/c of j*j is only one square.
            }


            return dp[n];
        }


        public int numSquares(int n) {
            PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
            int min = 0;

            for (int i = 2; i * i < n; i++) {
                pq.add(i * i);
            }

            int sum = 0;
            int count = 0;

            while (pq.iterator().hasNext()) {
                if (pq.peek() < n) {
                    sum += pq.poll();
                    count++;
                }
            }

            while (pq.peek() < n) {
                int number = pq.poll();
                min++;
            }
            return min;
        }

        // {l,r,u,d,ld,rd,lu,ru}
        // directions matrix
        int[] dx8 = {-1, 0, 1, 0, 1, 1, -1, -1};
        int[] dy8 = {0, -1, 0, 1, -1, 1, -1, 1};

        public int shortestPathBinaryMatrix(int[][] grid) {
            return bfs(grid);
        }

        private boolean isSafe(int[][] grid, int i, int j) {
            int n = grid.length;

            return i >= 0 && i < n && j >= 0 && j < n && grid[i][j] == 0;
        }

        //  bfs of matrix to count the number of islands
        private int bfs(int[][] grid) {
            int n = grid.length, cnt = 0;

            // to store nearest points based on distance
            PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[a.length - 1]));
            boolean[][] vis = new boolean[n][n];

            if (grid[0][0] == 1) return -1;
            cnt++;
            vis[0][0] = true; // mark as visited
            queue.offer(new int[]{0, 0, cnt});

            while (!queue.isEmpty()) {
                int[] elem = queue.poll();
                if (elem[0] == n - 1 && elem[1] == n - 1) return elem[2];

                for (int val = 0; val < 8; val++) {
                    int x = dx8[val] + elem[0];
                    int y = dy8[val] + elem[1];
                    // apply the check for boundary condition
                    if (isSafe(grid, x, y) && !vis[x][y]) {
                        queue.offer(new int[]{x, y, elem[2] + 1});
                        vis[x][y] = true;
                    }
                }
            }
            return -1;
        }
    }

    class BFSBallPLAYER {

        // {l,r,u,d,ld,rd,lu,ru}
        // directions matrix
        int[] dx = {-1, 0, 1, 0};
        int[] dy = {0, -1, 0, 1};


        public int minPushBox(char[][] grid) {
            int m = grid.length;
            int n = grid[0].length;

            Point ball = new Point();
            Point player = new Point();
            Point target = new Point();

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (grid[i][j] == 'B') {
                        ball.x = i;
                        ball.y = j;
                    } else if (grid[i][j] == 'T') {
                        target.x = i;
                        target.y = j;
                    } else if (grid[i][j] == 'S') {
                        player.x = i;
                        player.y = j;
                    }
                }
            }


            return bfs(grid, player, ball, target);
        }

        private boolean isSafe(char[][] grid, int i, int j) {
            int m = grid.length;
            int n = grid[0].length;

            return i >= 0 && i < m && j >= 0 && j < n && (grid[i][j] != '#');
        }

        //  bfs of matrix to count the number of islands
        private int bfs(char[][] grid, Point player, Point ball, Point target) {
            int m = grid.length;
            int n = grid[0].length;
            // to store nearest points based on distance
            boolean[][] vis = new boolean[m][n];

            // to store nearest points based on distance
            PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[a.length - 1]));

            queue.offer(new int[]{ball.x, ball.y, 0});
            vis[ball.x][ball.y] = true; // mark as visited

            while (!queue.isEmpty()) {
                int[] elem = queue.poll();

                if (elem[0] == target.x && elem[1] == target.y) return elem[2];

                for (int val = 0; val < 4; val++) {
                    int x = dx[val] + elem[0];
                    int y = dy[val] + elem[1];
                    // apply the check for boundary condition
                    if (isSafe(grid, x, y) && !vis[x][y]) {
                        if (isReachable(grid, new Point(elem[0] - dx[val], elem[1] - dy[val]), player)) {
                            // System.out.println("new point:" + new Point(x, y));
                            queue.offer(new int[]{x, y, elem[2] + 1});
                            vis[x][y] = true;
                            player = new Point(elem[0], elem[1]);
                            // System.out.println("player location:" + player);
                        }
                    }
                }
            }
            return -1;
        }


        private boolean isReachable(char[][] grid, Point ball, Point player) {
            int m = grid.length;
            int n = grid[0].length;

            // to store nearest points based on distance
            Queue<Point> queue = new LinkedList<>();
            boolean[][] vis = new boolean[m][n];
            queue.offer(player);
            vis[player.x][player.y] = true; // mark as visited

            while (!queue.isEmpty()) {
                Point elem = queue.poll();
                if (elem.x == ball.x && elem.y == ball.y) return true;

                for (int val = 0; val < 4; val++) {
                    int x = dx[val] + elem.x;
                    int y = dy[val] + elem.y;
                    // apply the check for boundary condition
                    if (isSafe(grid, x, y) && !vis[x][y]) {
                        queue.offer(new Point(x, y));
                        vis[x][y] = true;
                    }
                }
            }
            return false;

        }
    }

    // Quick sort algorithm to find out Kth largest element from the array 
	public int findKthLargest(int[] nums, int k) {
		int n = nums.length;
		k = n - k;
		return quickSelect(nums, 0, n - 1, k);
	}
	// Quick select function to find the partition index 
	private int quickSelect(int[] nums, int l, int h, int k) {
		if (l == h) return nums[k];
		int pIndex = partition(nums, l, h);
		if (pIndex<k) {
			return quickSelect(nums, pIndex + 1, h, k);
		} else {
			return quickSelect(nums, l, pIndex, k);
		}
	}

	// Find pIndex based on input provided 
	private int partition(int[] nums, int l, int h) {
		int i = l - 1, j = h + 1;
		int pIndex = (l + h) / 2;
		int pivot = nums[pIndex];
		while (true) {
			while (nums[++i]<pivot);
			while (nums[--j] > pivot);
			if (i >= j) return j;
			swap(nums, i, j);
		}
	}

	// swap indexes in array
	private void swap(int[] nums, int i, int j) {
		int temp = nums[i];
		nums[i] = nums[j];
		nums[j] = temp;
	}
    
     public int minMovesToSeat(int[] seats, int[] students) {
        int moves = 0;
        Arrays.sort(seats);
        Arrays.sort(students);
        for (int i=0;i<students.length;i++){
             moves += Math.abs(students[i]-seats[i]);
        }
        return moves;
    }
    
    static long INF = (long) 1e10;
	public long kthSmallestProduct(int[] nums1, int[] nums2, long k) {
		int m = nums1.length, n = nums2.length;
		long l = -INF - 1, h = INF + 1;

		while (l<h) {
			long mid = l + ((h - l) >> 1), cnt = 0;

			// binary search for cnt lesser than expected element
			for (int num: nums1) {
				if (num >= 0) {
					int i = 0, j = n - 1, p = 0;
					while (i<= j) {
						int c = i + ((j - i) >> 1);
						long mul = num * (long) nums2[c];
						if (mul<= mid) {
							p = c + 1;
							i = c + 1;
						} else j = c - 1;
					}
					cnt += p;
				} else {
					int i = 0, j = n - 1, p = 0;
					while (i<= j) {
						int c = i + ((j - i) >> 1);
						long mul = num * (long) nums2[c];
						if (mul<= mid) {
							p = n - c;
							j = c - 1;
						} else i = c + 1;
					}
					cnt += p;
				}
			}
			if (cnt >= k) h = mid;
			else l = mid + 1L;
		}
		return l;
	}
    	// The element with greater than n/2 occurrence will have count at least 1 for the its existence vs non-existence
	public int majorityElement(int[] nums) {
		int candidate = Integer.MIN_VALUE;
		int n = nums.length;
		int cnt = 0;
		for (int num: nums) {
			if (cnt == 0) {
			   candidate = num;
			}
			cnt += (candidate == num) ? 1 : -1;
		}
		return candidate;
	}
	
	
	/*
class Solution {
	// For all numbers compute the max bitwiuse OR then recursively find all possible subsets
	public int countMaxOrSubsets(int[] nums) {
		int n = nums.length;
		int a = 0;
		for (int num: nums) a |= num;
		return subset(nums, n - 1, a, 0);
	}
	private int subset(int[] nums, int len, int a, int b) {
		// Base case 
		if (len<0) return 0;
		int ans = 0;
		if (a == (b | nums[len])) ans = 1;
		return ans + subset(nums, len - 1, a, b) // Not taken
			+
			subset(nums, len - 1, a, b | nums[len]); // Taken
	}
}
*/
    int cnt = 0, maxOR=0;
    public int countMaxOrSubsets(int[] nums) {
        int n = nums.length;
        subset(nums, 0, 0);
        return cnt;
    }
    
    private void subset(int [] nums, int ind, int OR){
        // base case 
        int n = nums.length;
        if (ind == n){
            if (OR > maxOR){
                maxOR = OR;
                cnt = 1;
            } else if (OR == maxOR) cnt ++;
            return;
        }
        
        
        // include 
        subset(nums, ind+1, OR|nums[ind]);
        // exclude
        subset(nums, ind+1, OR);
    }
}

class Bank {
	long[] balance;
	int n;
	public Bank(long[] balance) {
		// Intialise objects
		this.balance = balance;
		this.n = this.balance.length;
	}

	public boolean transfer(int account1, int account2, long money) {
		// validation of accounts
		if (!(account1 >= 1 && account1<= n) || (!(account2 >= 1 && account2<= n))) return false;

		// check transfer conditions
		if (balance[account1 - 1] >= money) {
			balance[account1 - 1] -= money;
			balance[account2 - 1] += money;
			return true;
		}
		return false;
	}

	public boolean deposit(int account, long money) {
		//  validation of accounts
		if (!(account >= 1 && account<= n)) return false;
		// Deposit amount in account
		balance[account - 1] += money;
		return true;
	}

	public boolean withdraw(int account, long money) {
		// validation of account
		if (!(account >= 1 && account<= n)) return false;
		// withdraw amount
		if (balance[account - 1] >= money) {
			balance[account - 1] -= money;
			return true;
		}
		return false;
	}
}

/**
 * Your Bank object will be instantiated and called as such:
 * Bank obj = new Bank(balance);
 * boolean param_1 = obj.transfer(account1,account2,money);
 * boolean param_2 = obj.deposit(account,money);
 * boolean param_3 = obj.withdraw(account,money);
 */



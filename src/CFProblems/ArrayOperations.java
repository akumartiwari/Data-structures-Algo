package CFProblems;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.*;

public class ArrayOperations {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        //	 System.setIn(new FileInputStream("Case.txt"));
        BufferedReader ob = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        while (t-- > 0) {
            String number = sc.next();
            int n = Integer.parseInt(number.split(" ")[0]);
            int k = Integer.parseInt(number.split(" ")[1]);
            Integer[] array = Arrays.stream(sc.next().split(" ")).toArray(Integer[]::new);

            operations(n, k, array);
        }
    }

    /*

    [1 1 1 2 1 3 1]
    [ 1 1 1 1 1 2 3]
    |1/3| + |1/2| + |1| = 1 + 1 = 2

     DP  based solution of states
     */
    static int score = 0;

    private static void operations(int n, int k, Integer[] array) {
        System.out.println(score(Arrays.asList(array), n, k, 0, 0, new HashSet<>()));
    }

    private static int score(List<Integer> array, int n, int k, int i, int j, Set<Integer> occuiped) {
        if (i >= array.size() || j >= array.size() || k <= 0) return 0;

        int min = Integer.MAX_VALUE;
        for (int ind = 0; ind < array.size(); ind++) {
            if (!occuiped.contains(ind)) {
                occuiped.add(ind);
            }

            int ind1;
            for (ind1 = 0; ind1 < array.size(); ind1++) {
                if (!occuiped.contains(ind1) && ind1 == ind) {
                    occuiped.add(ind);
                    break;
                }
            }
            min = (int) Math.min(min, Math.floor(ind / ind1));
        }
        return 0;
    }

    public int minimumCost(int[] cost) {
        int n = cost.length;
        if (n == 1) return cost[0];
        int ans = 0;
        int ind = n - 1;
        Arrays.sort(cost);
        while (ind > 0) {
            ans += cost[ind] + cost[ind - 1];
            ind -= 3;
        }
        if (ind == 0) ans += cost[0];
        return ans;
    }


    public int numberOfArrays(int[] differences, int lower, int upper) {
        long x = 0;
        long min = 0, max = 0;

        for (int d : differences) {
            x += d;
            min = Math.min(min, x);
            max = Math.max(max, x);
        }

        return (int) Math.max(0, (upper - lower + 1) - (max - min));
    }

    // Author: Anand
    // TC = O(R*C*log(RC))
    private static final int[] d = {0, 1, 0, -1, 0};

    public List<List<Integer>> highestRankedKItems(int[][] grid, int[] pricing, int[] start, int k) {
        int R = grid.length, C = grid[0].length;
        int x = start[0], y = start[1], low = pricing[0], high = pricing[1];
        Set<String> seen = new HashSet<>(); // Set used to prune duplicates.
        seen.add(x + "," + y);
        List<List<Integer>> ans = new ArrayList<>();
        // PriorityQueue sorted by (distance, price, row, col).
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] == b[0] ? a[1] == b[1] ? a[2] == b[2] ? a[3] - b[3] : a[2] - b[2] : a[1] - b[1] : a[0] - b[0]);
        pq.offer(new int[]{0, grid[x][y], x, y}); // BFS starting point.
        while (!pq.isEmpty() && ans.size() < k) {
            int[] cur = pq.poll();
            int distance = cur[0], price = cur[1], r = cur[2], c = cur[3]; // distance, price, row & column.
            if (low <= price && price <= high && ans.size() < k) { // price in range and size less than k?
                ans.add(Arrays.asList(r, c));
            }
            for (int m = 0; m < 4; ++m) { // traverse 4 neighbors.
                int i = r + d[m], j = c + d[m + 1];
                // in boundary, not wall, and not visited yet?
                if (0 <= i && i < R && 0 <= j && j < C && grid[i][j] > 0 && seen.add(i + "," + j)) {
                    pq.offer(new int[]{distance + 1, grid[i][j], i, j});
                }
            }
        }
        return ans;
    }

    // Author: Anand
    public int countElements(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int smallest = nums[0];
        int largest = nums[n - 1];
        int cnt = 0;
        for (int i = 1; i < n; i++) {
            if (nums[i] > smallest && nums[i] < largest) cnt++;
        }
        return cnt;
    }

    //Author: Anand
    public int[] rearrangeArray(int[] nums) {
        List<Integer> pos = new ArrayList<>();
        List<Integer> neg = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] >= 0) pos.add(nums[i]);
            else neg.add(nums[i]);
        }

        int[] array = new int[nums.length];
        int ind = 0;
        for (int i = 0; i < pos.size(); i++) {
            array[ind++] = pos.get(i);
            array[ind++] = neg.get(i);
        }
        return array;
    }

    // Author: Anand
    public List<Integer> findLonely(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> lonely = new ArrayList<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        for (int i = 0; i < nums.length; i++) {
            if (map.get(nums[i]) == 1) {
                if (map.containsKey(nums[i] + 1) || map.containsKey(nums[i] - 1)) continue;
                else lonely.add(nums[i]);
            }
        }
        return lonely;
    }

    // Author: Anand
    /*
    Input: statements = [[2,1,2],[1,2,2],[2,0,2]]
    Output: 2
    list =  []

    Approach:-
    consider all possible cases and maximise total no. of persons in a group
    // TC = O(n2)
     */
    int max = 0;

    public int maximumGood(int[][] statements) {
        int n = statements.length;
        // arr stores status of people ie. either good=1, or bad=0 or noStatement=2
        int[] arr = new int[n];
        Arrays.fill(arr, 2);
        dfs(0, arr, statements, n);
        return max;
    }

    private void dfs(int index, int[] arr, int[][] statements, int n) {
        //base case
        if (index >= n) {
            int res = 0;
            for (int i = 0; i < n; i++) {
                if (arr[i] == 1) res++;
            }
            max = Math.max(max, res);
            return;
        }

        // 3- cases:-
        int[] temp = arr.clone();
        if (temp[index] == 0) { // statement doesn't case ie. bad person
            dfs(index + 1, temp, statements, n);
        } else if (temp[index] == 1) { // good person
            int[] st = statements[index];
            // bfs all possiblities
            for (int i = 0; i < n; i++) {
                if ((st[i] == 0 && temp[i] == 1) || (st[i] == 1 && temp[i] == 0)) return;
                if (st[i] != 2) temp[i] = st[i];
            }
            dfs(index + 1, temp, statements, n);
        } else {
            // ie. no statement
            //assume person says lie
            temp[index] = 0;
            dfs(index + 1, temp, statements, n);
            temp[index] = 1;
            // person says truth
            int[] st = statements[index];
            for (int i = 0; i < n; i++) {
                if ((st[i] == 0 && temp[i] == 1) || (st[i] == 1 && temp[i] == 0)) return;
                if (st[i] != 2) temp[i] = st[i];
            }
            dfs(index + 1, temp, statements, n);
        }
    }

}


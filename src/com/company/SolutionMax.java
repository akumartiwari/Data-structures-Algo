package com.company;

import java.util.HashMap;import java.util.*;
import java.util.stream.Collectors;

class SolutionMax {
    public void nextPermutation(int[] nums) {
        int elem = Integer.MIN_VALUE;
        int index = 0;
        int n = nums.length;
        for (int i = n - 1; i > 0; i--) {
            index++;
            if (nums[i] >= elem) {
                elem = nums[i];
                continue;
            } else {
                break;
            }
        }

        if (index > n - 1) {
            for (int i = 0; i < nums.length / 2; i++) {
                int temp = nums[i];
                nums[i] = nums[nums.length - 1 - i];
                nums[nums.length - 1 - i] = temp;
            }
        } else {
            int temp = elem;
            nums[index] = nums[n - index];
            nums[n - index] = temp;

            // One by one reverse first
            // and last elements of a[0..k-1]
            for (int i = index + 1; i < n; i++) {
                int tempswap = nums[i];
                nums[i] = nums[n - i];
                nums[n - i] = tempswap;
            }
        }
        for (int num : nums) {
            System.out.println(num);
        }
    }


    public void swap(int[] nums, int i, int j) {
        int temp;
        temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    /*
    public List<List<Integer>> permute(int[] nums) {
        int n = nums.length;
        if (n == 0) return new ArrayList<>();
        // brute force solution

        List<List<Integer>> permute = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nums = swap(nums, i+1, j);
                List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
                permute.add(list);
                System.out.println(list);
                nums = swap(nums, i+1, j);
                System.out.println(list);
            }
        }
     */

    public List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> result = new ArrayList<>();
        permute(nums, 0, nums.length - 1, result);
        return result;
    }

    private void permute(int[] nums, int l, int r, List<List<Integer>> result) {
        if (l == r) {
            List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
            result.add(list);
            return;
        }
        for (int i = l; i <= r; i++) {
            swap(nums, l, i);
            permute(nums, l + 1, r, result);
            swap(nums, l, i);
        }
    }


    public int numSquarefulPerms(int[] nums) {

        if (nums.length == 0) return 0;
        List<List<Integer>> result = new ArrayList<>();
        permute(nums, 0, nums.length - 1, result);
        result = result.stream().distinct().collect(Collectors.toList());
        // getting the unique elements
        if (result.size() > 0) {
            int arraySize = result.get(0).size();

            while (arraySize-- > 0) {
                int elem = Integer.MIN_VALUE;
                List<List<Integer>> remove = new ArrayList<>();

                for (List<Integer> r : result) {
                    if (elem == r.get(arraySize)) {
                        remove.add(r);
                    }
                    elem = r.get(arraySize);
                }

                result.removeAll(remove);
            }

            System.out.println(result);

//             get the perfect squares elements
            while (result.listIterator().hasNext()) {
                List<Integer> eligibleElem = result.listIterator().next();
                int value = Integer.MIN_VALUE;
                for (int i = 0; i < eligibleElem.size() - 1; i++) {
                    if (i == 0) {
                        value = eligibleElem.get(0);
                    } else {
                        value += eligibleElem.get(i);
                        int sr = (int) Math.sqrt(value);
                        if (sr * sr == value) value = eligibleElem.get(i);
                        else {
                            result.remove(eligibleElem); // not a perfect square
                            break;
                        }
                    }
                }
            }
            System.out.println(result);
            result.remove(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        }
        return result.size();
    }

    public int minStickers(String[] stickers, String target) {
        int min = 0;
        int n = target.length();
        int[] dp = new int[n];
        dp[0] = 0;

        for (int i = 0; i < n; i++) {
            // get possible sustring from a given string
            target.substring(0, i).concat(target.substring(i));

        }


        return min;
    }

    public String getPermutation(int n, int k) {
        int[] nums = new int[n];
        for (int i = 1; i <= n; i++) {
            nums[i - 1] = i;
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        permuteNum(nums, 0, nums.length - 1, pq);

        if (pq.peek() != null) {
            return pq.peek().toString();
        }

        return null;
    }

    private void permuteNum(int[] nums, int l, int r, PriorityQueue<Integer> result) {
        if (l == r) {
            StringBuilder number = new StringBuilder();
            for (int num : Arrays.stream(nums).boxed().collect(Collectors.toList())) {
                number.append(num);
            }
            List<Integer> result1 = new ArrayList<>();
            result.add(Integer.parseInt(number.toString()));
            return;
        }
        for (int i = l; i <= r; i++) {
            swapNum(nums, l, i);
            permuteNum(nums, l + 1, r, result);
            swapNum(nums, l, i);
        }
    }

    public void swapNum(int[] nums, int i, int j) {
        int temp;
        temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

//
//    public List<List<String>> partition(String s) {
//
//        StringBuilder builder = new StringBuilder();
//        for (int i = 1; i <= n; i++) {
//            builder.append(i);
//        }
////        return getSequence(new StringBuilder(), builder, k - 1);
//
//
//    }
//
//    public List<List<Integer>> combine(int n, int k) {
//
//        StringBuilder builder = new StringBuilder();
//        for (int i = 1; i <= n; i++) {
//            builder.append(i);
//        }
//        getSequence(new StringBuilder(), builder, k - 1);
//    }

    /*
    4
2
1234
gs(sb, "1234", 1)
     */
    public List<List<Integer>> getSequence(StringBuilder prefix, StringBuilder builder, int k) {
        int n = builder.toString().length();
        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                List<Integer> result = new ArrayList<>();
                result.add(Integer.parseInt(String.valueOf(builder.charAt(i))));
                while (k-- > 0) {
                    result.add(Integer.parseInt(String.valueOf(builder.charAt(j + k))));
                }
                ans.add(result);
            }
        }
        return ans;
    }


    /*
    Input: nums = [5,1,3]
Output: 3

[135]
[133]
[113]
[111]
     */
    public int reductionOperations(int[] nums) {
        Arrays.sort(nums);

        int first = Integer.MIN_VALUE;
        int second = Integer.MIN_VALUE;


        for (int i = nums.length - 1; i >= 0; i--) {
            first = nums[i];
            second = nums[i - 1];
            if (first != second) nums[i] = second;
            else continue;

        }


//            List<Integer> list = new ArrayList<>();
//            for (int num : nums) list.add(num);
//            int[] large = largest(nums);
//
//            int operation = 0;
//            int[] nextLargest = nextLargest(nums);
//            Integer max = Collections.max(list);
//            nums[large[0]] = nextLargest[1];
//            operation = 3;
//            return operation;
        return 0;
    }

    public int[] nextLargest(int[] nums) {
        int[] ans = new int[2];
        for (int i = nums.length - 1; i > 0; i--) {
            if (nums[i] != nums[i - 1]) {
                ans[1] = nums[i];
                ans[0] = i;
                break;
            }
        }
        return ans;
    }

    public int[] largest(int[] nums) {
        int[] ans = new int[2];
        ans[1] = Integer.MIN_VALUE;

        for (int i = nums.length - 1; i > 0; i--) {
            ans[1] = nums[i];
            ans[0] = i;
            if (nums[i] != nums[i - 1]) break;
        }
        return ans;
    }

    /*
    [[50,50]]
1
50

[[1,2],[3,4],[5,6]]
     */

    public boolean isCovered(int[][] ranges, int left, int right) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int n = ranges.length;
        for (int i = 0; i < n; i++) {
            int ls = ranges[i][0];
            int rs = ranges[i][1];
            for (int k = ls; k <= rs; k++) {
                map.put(k, 1); // mark existence of all values
            }
        }

        for (int i = left; i <= right; i++) {
            if (map.get(i) == null) return false;
        }
        return true;
    }

    /*
    Input: chalk = [3,4,1,2], k = 25

25-c[0] = 25-3 = 22

[5,1,5]
22

22-5-1-5 = 11-5= 6-1 = 5-5 = 0
index =3
index = 0
     */

    public int chalkReplacer(int[] chalk, int k) {
        int n = chalk.length;
        if (n == 0) return 0;
        int index = 0;

        int sum = 0;
        for (int c : chalk) sum += c;
        int loops = k / sum;
        int rem = k % sum;
        if (loops <= 1) {
            while (k >= 0) {
                if (index == n) index = 0;
                k -= chalk[index];
                if (k >= 0) index++;
            }
        } else {
            k = rem;
            while (k >= 0) {
                if (index == n) index = 0;
                k -= chalk[index];
                if (k >= 0) index++;
            }
        }
        return index;
    }


    //    Definition for a binary tree node.
    public class TreeNode {
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
        public int compareTo(Pair<Number, Number> numberNumberPair) {
            return numberNumberPair.val - this.val;
        }
    }


    // This class is used to store index and corresponding value element of array
    class Triplet<I extends String, I1 extends Integer, I2 extends Number> implements Comparable<Triplet<String, Integer, Number>> {
        // field member and comparable implementation

        public Triplet() {
        }

        public Triplet(String key, int val, int number) {
            this.key = key;
            this.val = val;
            this.number = number;
        }

        String key;
        int val;
        int number;

        @Override
        public int compareTo(Triplet<String, Integer, Number> stringIntegerNumberTriplet) {
            return stringIntegerNumberTriplet.val - stringIntegerNumberTriplet.number;
        }
    }

    /*
    Input: groupSizes = [3,3,3,3,3,1,3]
    Output: [[5],[0,1,2],[3,4,6]]
     TC = O(n + n-logn) ~ O(nlogn)

     */

    /*
    Input: nums = [1,-3,2,3,-4]
Output: 5
     */


    /*
    dp[0]=1
    prev= -3+1 = -2
    dp[1] = -2
    [1,-3,2,3,-4]
    a = -2
    b = -3
     */
    public int maxAbsoluteSum(int[] nums) {

//        int n = nums.length;
//        int[] dp = new int[n];
//        dp[0] = nums[0];
//        int max = Integer.MIN_VALUE;
//        for (int i = 1; i < n; i++) {
//            int a = nums[i] + dp[i-1];
//            int b  = nums[i];
//            if (a < 0 || b < 0){
//                dp[i] = -Math.max(Math.abs(a), Math.abs(b));
//            }
//            else {
//                dp[i] = Math.max(Math.abs(a), Math.abs(b));
//            }
//            if (Math.abs(dp[i]) > max) max = dp[i];
//        }
//        return Math.abs(max);

        /*
            Input: nums = [1,-3,2,3,-4]
Output: 5

         */
        // kaden's algorithm solution
        int sMax = nums[0], gMax = nums[0], sMin = nums[0], gMin = nums[0];
        for (int i = 1; i < nums.length; i++) {
            sMax = Math.max(sMax + nums[i], nums[i]);
            gMax = Math.max(gMax, sMax);

            sMin = Math.min(sMin + nums[i], nums[i]);
            gMin = Math.min(gMin, sMin);
        }
        return Math.max(gMax, -gMin);
    }

    public int[] processQueries(int[] queries, int m) {
        int n = queries.length;
        int[] ans = new int[n];
        int[] p = new int[m];
        for (int i = 0; i < m; i++) {
            p[i] = i + 1;
        }

        for (int i = 0; i < n; i++) {
            // get the index of element
            for (int k = 0; k < p.length; k++) {
                if (p[k] == queries[i]) ans[i] = k;
            }
            reshuffle(p, ans[i]);
        }
        return ans;
    }

    private void reshuffle(int[] perm, int pos) {
        int[] copy = Arrays.copyOfRange(perm, 0, perm.length);
        perm[0] = perm[pos];
        for (int i = 1; i <= pos; i++) {
            perm[i] = copy[i - 1];
        }
    }


    public List<List<Integer>> groupThePeople(int[] groupSizes) {
        int n = groupSizes.length;
        HashMap<Integer, List<Integer>> map = new HashMap<>(); // map to store persion i with size of group
        List<List<Integer>> ans = new ArrayList<>(); // answer arrays list
        List<Integer> list = new ArrayList<>();
        list.add(0);
        map.put(groupSizes[0], list);
        for (int i = 1; i < n; i++) {
            // if group element is found
            if (map.get(groupSizes[i]) != null) {
                List<Integer> old = map.get(groupSizes[i]);
                old.add(i);
                map.put(groupSizes[i], old);
            } else {
                //if new element is found with next group
                List<Integer> indices = new ArrayList<>();
                indices.add(i);
                map.put(groupSizes[i], indices);
            }
        }

        for (Integer key : map.keySet()) {
            List<Integer> indices = map.get(key);

            // g
            int groupSize = key;
            while (indices.size() > 0) {
                List<Integer> ansList = new ArrayList<>();
                // Fill all elements of the group in 1st list  till all indices is covered
                while (groupSize-- > 0) {
                    ansList.add(indices.remove(0));
                }
                groupSize = key;
                ans.add(ansList);
            }
        }

        return ans;
    }


    public int[] getDepthAndSum(TreeNode node) {
        int[] ans = new int[2];
        int[] left_ans;
        int[] right_ans;

        if (node == null) return ans;
        if (node.left == null && node.right == null) {
            ans[0] = 1;
            ans[1] = node.val;
            return ans;
        }

        Pair<Integer, Integer> pair = new Pair(10, 12);
        Triplet<String, Integer, Integer> triplet = new Triplet<>();

        System.out.println(pair);
        System.out.println(triplet);
//        Quartet<String, String, String, String> quartet = Quartet.fromCollection(listOf4Names);

        left_ans = getDepthAndSum(node.left);
        right_ans = getDepthAndSum(node.right);

        if (left_ans[0] == right_ans[0]) {
            ans[0] = 1 + left_ans[0];
            ans[1] = left_ans[1] + right_ans[1];
        } else if (left_ans[0] > right_ans[0]) {
            ans[0] = 1 + left_ans[0];
            ans[1] = left_ans[1];
        } else {
            ans[0] = 1 + right_ans[0];
            ans[1] = right_ans[1];
        }
        return ans;
    }

    public int deepestLeavesSum(TreeNode root) {
        return getDepthAndSum(root)[1];
    }


    //dp based
    public int minOperationsToFlip(String expression) {
        return 0;
    }


    public int largestMagicSquare(int[][] grid) {
        int max = 0;
        int n = grid.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][i] = grid[i][i]; // d
                dp[i][j] = grid[i][j]; // c
                dp[j][i] = grid[j][i]; // r
                if (ms(dp) != 0) {
                    max = Math.max(ms(dp), max);
                }
            }
        }
        return max;
    }

    public int ms(int[][] magicSquare) {
        int N = magicSquare.length;
        // sumd1 and sumd2 are the sum of the two diagonals
        int sumd1 = 0, sumd2 = 0;
        for (int i = 0; i < N; i++) {
            sumd1 += magicSquare[i][i];
            sumd2 += magicSquare[i][N - 1 - i];
        }
        // if the two diagonal sums are unequal then it is not a magic square
        if (sumd1 != sumd2)
            return 0;
        return magicSquare.length - 1;
    }

}

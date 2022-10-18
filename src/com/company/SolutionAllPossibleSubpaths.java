package com.company;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

class SolutionAllPossibleSubpaths {
    int paths;
    int target;

    public int pathSum(TreeNode root, int targetSum) {
        this.target = targetSum;
        this.paths = 0;
        if (root == null) return 0;
        countAllPossiblePath(root, new ArrayList<>());
        return this.paths;
    }

    private void countAllPossiblePath(TreeNode node, ArrayList<TreeNode> nodelist) {
        if (node != null) {
            nodelist.add(node);
        }

        // count paths where sum is equla to target
        if (nodelist.stream().mapToInt(e -> e.val).sum() == target) this.paths++;

        if (node.left != null) {
            countAllPossiblePath(node.left, nodelist);
        }

        if (node.right != null) {
            countAllPossiblePath(node.right, nodelist);
        } else if (node.left == null) {
            List<Integer> list = new ArrayList<>();
            for (TreeNode treeNode : nodelist) {
                list.add(treeNode.val);
            }
            if (list.stream().mapToInt(e -> e).sum() == target) this.paths++;
        }
        nodelist.remove(node);

    }

    int count = 0;
    int k;
    HashMap<Integer, Integer> h = new HashMap<>();

    public void preorder(TreeNode node, int currSum) {
        if (node == null)
            return;

        // current prefix sum
        currSum += node.val;

        // here is the sum we're looking for
        if (currSum == k)
            count++;

        // number of times the curr_sum − k has occured already,
        // determines the number of times a path with sum k
        // has occured upto the current node
        count += h.getOrDefault(currSum - k, 0);

        // add the current sum into hashmap
        // to use it during the child nodes processing
        h.put(currSum, h.getOrDefault(currSum, 0) + 1);

        // process left subtree
        preorder(node.left, currSum);
        // process right subtree
        preorder(node.right, currSum);

        // remove the current sum from the hashmap
        // in order not to use it during
        // the parallel subtree processing
        h.put(currSum, h.get(currSum) - 1);
    }

    public int allPathSum(TreeNode root, int sum) {
        k = sum;
        preorder(root, 0);
        return count;
    }

    int max = Integer.MIN_VALUE;

    public int longestUnivaluePath(TreeNode root) {

        // base case
        if (root == null) return 0;
        preorder(root);
        return max;
    }

    private int preorder(TreeNode root) {
        if (root == null) {
            return 0;
        }

        // recursively traverse left subtree
        int left = preorder(root.left);

        // recursively traverse right subtree
        int right = preorder(root.right);

        int towardsLeft = 0, towardsRight = 0;
        if (root.left != null && root.left.val == root.val) towardsLeft += left + 1;

        if (root.right != null && root.right.val == root.val) towardsRight += right + 1;


        max = Math.max(towardsLeft + towardsRight, max);
        return Math.max(towardsLeft, towardsRight);
    }

    String ans = "~";
    static String dir = "DRUL";
    static int[] dirx = {1, 0, -1, 0};
    static int[] diry = {0, 1, 0, -1};

    public String smallestFromLeaf(TreeNode root) {
        dfs(root, new StringBuilder());
        return ans;
    }

    private void dfs(TreeNode root, StringBuilder sb) {

        if (root == null) return;
        sb.append((char) ('a' + root.val));
        // if node is child
        if (root.left == null && root.right == null) {
            sb.reverse();
            String S = sb.toString();
            sb.reverse();
            if (S.compareTo(ans) < 0) ans = S;
        }

        dfs(root.left, sb);
        dfs(root.right, sb);
        sb.deleteCharAt(sb.length() - 1);

    }

    // [1 2 3] , target = 4  , you are allowed to take any element as many times you want

    private int count(int[] arr, int n, int sum, int index, String arrStr) {
        // base case
        if (index == n) {
            if (sum == 0) {
                System.out.println(arrStr);
                return 1;
            }
            return 0;
        }
        int left = 0;
        int right = 0;


        // when element is included
        if (arr[index] <= sum) {
            // element included
            sum -= arr[index];
            left = count(arr, n, sum, index, arrStr + arr[index]);
            //  restore sum
            sum += arr[index];
        }

        //  when element is not taken
        right = count(arr, n, sum, index + 1, arrStr);

        // removed the last character
        arrStr = arrStr.length() > 0 ? arrStr.substring(0, arrStr.length() - 1) : "";

        return left + right;
    }

    static int rows, cols;

    private static void findPaths(int i, int j, String s) {
        // base case
        if (i == rows - 1 && j == cols - 1) {
            System.out.println(s);
            return;
        }

        int[][] vis = new int[rows][cols];

        Arrays.fill(vis, 0);
        // move to right
        s += "R";
        // recursive call to right
        findPaths(i, j + 1, s);
        // backtrack
        s = s.substring(0, s.length() - 1);

        // move to down
        s += "D";
        findPaths(i + 1, j, s);
        s = s.substring(0, s.length() - 1);
    }


    private static void ratInAMaze(int i, int j, String s, int[][] vis) {

        // boundary conditions
        if (i >= rows || j >= cols || i < 0 || j < 0 || vis[i][j] == 1) return;

        // base case
        if (i == rows - 1 && j == cols - 1) {
            System.out.println(s);
            return;
        }

        // mark visited as true
        vis[i][j] = 1;

        // Downward direction
        s += "D";
        ratInAMaze(i + 1, j, s, vis);
        s = s.substring(0, s.length() - 1);


        // Right direction
        s += "R";
        ratInAMaze(i, j + 1, s, vis);
        s = s.substring(0, s.length() - 1);

        s += "U";
        ratInAMaze(i - 1, j, s, vis);
        ;
        s = s.substring(0, s.length() - 1);

        s += "L";
        ratInAMaze(i, j - 1, s, vis);
        s = s.substring(0, s.length() - 1);
        // backtrack
        vis[i][j] = 0;
    }


    private static void ratInAMazeV1(int i, int j, String s, int[][] vis) {

        // boundary conditions
        if (i >= rows || j >= cols || i < 0 || j < 0 || vis[i][j] == 1) return;

        // base case
        if (i == rows - 1 && j == cols - 1) {
            System.out.println(s);
            return;
        }

        // mark visited as true
        vis[i][j] = 1;

        for (int x = 0; x < 4; x++) {
            s += dir.charAt(x);
            ratInAMazeV1(i + dirx[x], j + diry[x], s, vis);
            s = s.substring(0, s.length() - 1);
        }
        // backtrack
        vis[i][j] = 0;

    }

    /**
     * Function to count number of valid paths
     *
     * @param maze
     * @param i
     * @param j
     * @param vis
     * @return
     */
    private static int countPaths(int[][] maze, int i, int j, int[][] vis) {

        // boundary conditions
        if (i >= rows || j >= cols || i < 0 || j < 0 || vis[i][j] == 1 || maze[i][j] == 1) return 0;

        // base case
        if (i == rows - 1 && j == cols - 1) {
            return 1;
        }

        // mark visited as true
        vis[i][j] = 1;

        int cnt = 0;
        for (int x = 0; x < 4; x++) {
            cnt += countPaths(maze, i + dirx[x], j + diry[x], vis);
        }
        // backtrack
        vis[i][j] = 0;

        return cnt;

    }

    /*
       TC ~= O(2^n * n), SC = O(n+n) where n for stack and remaining n for ds

     */
    private static void func(int ind, List<Integer> ds, int[] arr, int n) {
        if (ind == n) {
            for (int e : ds) System.out.print(e + " ");
            System.out.println();
            return;
        }

        // curr index is taken
        ds.add(arr[ind]);
        func(ind + 1, ds, arr, n);
        ds.remove(arr[ind]);

        // step for backtracking
        // curr index is not taken
        func(ind + 1, ds, arr, n);
    }

    /*
     Print subsequences whose sum is divisible by k
     IO :-
      arr = [ 4 3 2]
      k = 3

      O:-
      {[3], [4,2]}

     */

    /***
     * Fun prints subsequences whose sum is divisible by k
     * @param ind
     * @param ds
     * @param arr
     * @param n
     * @param sum
     * @param k
     */
    private static void subsetDivByK(int ind, List<Integer> ds, int[] arr, int n, int sum, int k) {
        // base case
        /*
           1.  index has reached to extremities
           return
           2 sum is divisible by k
           print result and return
         */
        if (ind == n) {
            if (sum % k == 0) {
                for (int e : ds) System.out.print(e + " ");
                System.out.println();
                return;
            }
            return;
        }

        // curr index is taken
        ds.add(arr[ind]);
        sum += arr[ind];
        subsetDivByK(ind + 1, ds, arr, n, sum, k);
        sum -= arr[ind];
        ds.remove(arr[ind]);

        // step for backtracking
        // curr index is not taken
        subsetDivByK(ind + 1, ds, arr, n, sum, k);
    }

    /**
     * Count of subsequences whose sum is divisible by k
     *
     * @param ind
     * @param arr
     * @param n
     * @param sum
     * @param k
     * @return
     */
    private static int subsequenceDivByKCount(int ind, int[] arr, int n, int sum, int k) {
        if (ind == n) {
            if (sum % k == 0) {
                return 1;
            }
            return 0;
        }


        int left = 0, right = 0;

        sum += arr[ind];
        left += subsequenceDivByKCount(ind + 1, arr, n, sum, k);
        sum -= arr[ind];

        // step for backtracking
        // curr index is not taken
        right += subsequenceDivByKCount(ind + 1, arr, n, sum, k);
        return left + right;
    }


    private static int subsetDivByKCount(int ind, int[] arr, int n, int sum) {
        if (ind == n) {
            if (sum == 0) {
                return 1;
            }
            return 0;
        }


        int left = 0, right = 0;

        while (arr[ind] <= sum) {
            sum += arr[ind];
            left += subsetDivByKCount(ind, arr, n, sum);
            sum -= arr[ind];
        }

        // step for backtracking
        // curr index is not taken
        right += subsetDivByKCount(ind + 1, arr, n, sum);
        return left + right;
    }


/*

[[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]]
source = [0,4]
destination = [4,4]

Dry-run

 */

    private boolean isValidDestination(int[][] maze, int[] destination) {
        int s = destination[0];
        int e = destination[1];
        if (maze[s][e] == 1) return false;
        // this means  ball can go downwards

        if (s + 1 < rows && maze[s + 1][e] == 1) {
            return false;
        }
        return true;
    }


    /*

    [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]]
[0,4]
[3,2]

*/
// N QUEEN PROBLEM- NP HARD

    int n; // no. of columns
    int[] rowhash = new int[n];
    int[] thirdhash = new int[2 * n - 1];
    int[] firsthash = new int[2 * n - 1];

    // TC  = O(N * N^N)
    //
    private boolean NQueen(int col, int[][] mat) {

        if (col == n) {
            // print the paths
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    System.out.println(mat[i][j] + " ");
                }
                System.out.println();
            }
            return true;
        }


        for (int row = 0; row < n; row++) {
            if (isSafe(row, col, mat)) {
                rowhash[row] = 1;
                firsthash[n - 1 + row - col] = 1;
                mat[row][col] = 1;
                thirdhash[row + col] = 1;
                if (NQueen(col + 1, mat)) return true;
                mat[row][col] = 0;
                rowhash[row] = 0;
                thirdhash[row + col] = 0;
                firsthash[n - 1 + row - col] = 0;
            }
        }

        return false;
    }


    private void NQueenAllPaths(int col, int[][] mat) {
        if (col == n) {
            // print the paths
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    System.out.println(mat[i][j] + " ");
                }
                System.out.println();
            }
        }

        for (int row = 0; row < n; row++) {
            if (isSafe2(row, col, mat)) {
                rowhash[row] = 1;
                firsthash[n - 1 + row - col] = 1;
                mat[row][col] = 1;
                thirdhash[row + col] = 1;
                NQueen(col + 1, mat);
                mat[row][col] = 0;
                rowhash[row] = 0;
                thirdhash[row + col] = 0;
                firsthash[n - 1 + row - col] = 0;
            }
        }
    }

    /***
     * This func tell us whether it is a safe place ot place queen
     * Dir to be checked as follows:-
     * left --> {i,j--}
     * left-up --> {i --, j --}
     * left-down --> {i ++, j --}
     * @param row
     * @param col
     * @param mat
     * @return
     */
    //. TC  = O(N)
    // TC = O(1)
    private boolean isSafe(int row, int col, int[][] mat) {

        // left
        for (int i = row, j = col; j >= 0; j--) {
            if (mat[i][j] == 1) return false;
        }

        // left-up
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (mat[i][j] == 1) return false;
        }


        // left-down
        for (int i = row, j = col; j >= 0 && i < n; i++, j--) {
            if (mat[i][j] == 1) return false;
        }
        return false;
    }

    /***
     * This func tell us whether it is a safe place ot place queen
     * Dir to be checked as follows:-
     * left --> {i,j--}
     * left-up --> {i --, j --}
     * left-down --> {i ++, j --}
     * @param row
     * @param col
     * @param mat
     * @return
     */
    // TC = O(1)
    private boolean isSafe2(int row, int col, int[][] mat) {
        if (rowhash[col] == 1 || firsthash[n - 1 + row - col] == 1 || thirdhash[row + col] == 1) return false;
        return true;
    }
}

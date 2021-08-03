package com.company;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class EmployeeSolution {
    String name;
    Integer salary;

    EmployeeSolution(String n, Integer s) {
        this.name = n;
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


    String ans = "~";

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
}


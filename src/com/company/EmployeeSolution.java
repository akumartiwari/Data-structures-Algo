package com.company;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
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

    Set<String> ans = new HashSet<>();

    public List<String> addOperators(String num, int target) {
        calculate(num, 0, target, "");
        return new ArrayList<>(ans);
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
                ans.add(res);
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
            while (exp[0].length() > i && isNumber((int) exp[0].charAt(i))) {
                i++;
            }
            res = Double.parseDouble(exp[0].substring(0, i));
            exp[0] = exp[0].substring(i);
        }
        return res;
    }


    private boolean isNumber(int c) {
        int zero = (int) '0';
        int nine = (int) '9';
        return (c >= zero && c <= nine) || c == '.';
    }
        
}

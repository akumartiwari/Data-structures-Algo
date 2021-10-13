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

    
    // Greedy approach
	// The approach to traverse through the array and check if we get a n 'X'  character then move 3 steps ahead 
	//  else move only 1 step (normal pace)

	public int minimumMoves(String s) {
		int n = s.length(), cnt = 0;
		for (int i = 0; i<n;) {
			if (s.charAt(i) == 'X') {
				i += 3;
				cnt++;
			} else i++;
		}
		return cnt;
	}
    
}


/*
    private static final int[][] DIRS = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    public int minPushBox(char[][] grid) {
        int R = grid.length, C = grid[0].length;
        int[] box = new int[2], player = new int[2];
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (grid[i][j] == 'B') {
                    box[0] = i;
                    box[1] = j;
                } else if (grid[i][j] == 'S') {
                    player[0] = i;
                    player[1] = j;
                }
            }
        }
        Queue<Pair<int[], int[]>> queue = new LinkedList();
        queue.add(new Pair(box, player));
        boolean[][] visited = new boolean[R * C][R * C];
        // in some cases, player needs to push the box further in order to change its direction; hence, tracking the box itself isn't enough,
        // we need to track both box and player locations. for example,
        // . # T # .
        // . . . B S
        // . . . # .
        // `B` needs to land on location(1,2) twice
        visited[box[0] * C + box[1]][player[0] * C + player[1]] = true;
        int step = 0;
        while (!queue.isEmpty()) {
            step++;
            for (int i = queue.size() - 1; i >= 0; i--) {
                Pair<int[], int[]> state = queue.poll();
                int[] b = state.getKey(), p = state.getValue();
                for (int j = 0; j < DIRS.length; j++) {
                    int[] nb = new int[]{b[0] + DIRS[j][0], b[1] + DIRS[j][1]};
                    if (nb[0] >= 0 && nb[0] < R && nb[1] >= 0 && nb[1] < C && grid[nb[0]][nb[1]] != '#') {
                        // check where was it pushed from. basically, the opposite direction where the box moves to.
                        int[] np = new int[]{b[0] - DIRS[j][0], b[1] - DIRS[j][1]};
                        if (np[0] >= 0 && np[0] < R && np[1] >= 0 && np[1] < C
                                && grid[np[0]][np[1]] != '#'
                                && !visited[nb[0] * C + nb[1]][np[0] * C + np[1]]) {
                            // can the player reach to the box-pushing location
                            if (isReachable(grid, R, C, b, p, np)) {
                                if (grid[nb[0]][nb[1]] == 'T') {
                                    return step;
                                }
                                visited[nb[0] * C + nb[1]][np[0] * C + np[1]] = true;
                                queue.add(new Pair(nb, b));
                            }
                        }
                    }
                }
            }
        }
        return -1;
    }

    private boolean isReachable(char[][] grid, int R, int C, int[] box, int[] from, int[] to) {
        Queue<int[]> queue = new LinkedList();
        queue.add(from);
        boolean[][] visited = new boolean[R][C];
        visited[from[0]][from[1]] = true;
        while (!queue.isEmpty()) {
            int[] loc = queue.poll();
            if (loc[0] == to[0] && loc[1] == to[1]) {
                return true;
            }
            for (int j = 0; j < DIRS.length; j++) {
                int nr = loc[0] + DIRS[j][0], nc = loc[1] + DIRS[j][1];
                if (nr >= 0 && nr < R && nc >= 0 && nc < C && !visited[nr][nc]
                        && grid[nr][nc] != '#'
                        && (nr != box[0] || nc != box[1])) {
                    visited[nr][nc] = true;
                    queue.add(new int[]{nr, nc});
                }
            }
        }
        return false;
    }
 */


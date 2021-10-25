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
	
	
	// KADAN's ALGO
   public int maxSubArray(int[] nums) {
 	int n = nums.length;
 	int max_sum_so_far = Integer.MIN_VALUE, max_ending_here = 0;

 	for (int i = 0; i<n; i++) {
 		max_ending_here += nums[i];
 		max_sum_so_far = Math.max(max_ending_here, max_sum_so_far);
 		if (max_ending_here<0) max_ending_here = 0;
 	}
 	return max_sum_so_far;
 }

      // O(n+m), O(1)
	/*
     This problem is called Merge Sorted Array
     Example:- 
     Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

    Algo :- We start filling the array from right end till all elements of nums1 is consumed 
        After that remaining element of nums2 is utitlised 

        */

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int i = m - 1, j = n - 1, k = m + n - 1;
		while (j > -1) {
			if (i > -1) {
				nums1[k--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
			} else {
				nums1[k--] = nums2[j--];
			}
		}
	}
	
		// Thoughts:- 

	/* 

	   TC = O(2^n), Sc = O(n)

	   Algorithm:-
	  - The idea is to split array in two parts such that 
	     avg(A) = avg(B)
	  - Iterate thrugh array elements and for each elem 
	     check if we can split it in two parts with equals avg 

	  -  We have choice of take or dont take in first part 
	     ie. if arr(i) is taken in part1 sumA+arr(i)
	     else sumB + arr(i)

	  - Do above step recursilvely and backtrack
	  - check if sumA == sumB && (index == n-1) { that means all elements have been segregated into two parts successfuly
	  } 
	     - if true return true 
	      else return false and recurse further 
	  - Add Memoization to improve exponential time complexity        

	  total  = sumA + sumB
	  sumB = total - sumA

	  A+B=n
	  B=n-A

	  sumA/A = sumB/B

	  sumA/A = total-sumA/B
	  sumA/A = total-sumA/n-A
	  n*sumA/A  = total
	  sumA = total * lenA / n

	  problem boils down to finding a subsequence of length len1
	  with sum equals sumA
	*/

	public boolean splitArraySameAverage(int[] nums) {
		int n = nums.length, total = 0;
		for (int i = 0; i<n; i++) total += nums[i];

		HashMap<String, Boolean> map = new HashMap<>();
		for (int cnt = 1; cnt<n; cnt++) {
			if ((total * cnt) % n == 0) {
				if (isPossible(nums, 0, cnt, (total * cnt) / n, map)) return true;
			}
		}
		return false;
	}

	private boolean isPossible(int[] nums, int ind, int len, int sum, HashMap<String, Boolean> map) {
		int n = nums.length;
		// base case 
		if (sum == 0 && len == 0) {
			return true;
		}

		if (ind >= n || len == 0) return false;
		String key = len + "-" + sum + "-" + ind;
		if (map.containsKey(key)) return map.get(key);
		// if number can be taken
		if (sum - nums[ind] >= 0) {
			// taken 
			boolean case1 = isPossible(nums, ind + 1, len - 1, sum - nums[ind], map);

			// not taken 
			boolean case2 = isPossible(nums, ind + 1, len, sum, map);

			map.put(key, (case1 || case2));
			return case1 || case2;
		}

		// Can't be taken 
		boolean case2 = isPossible(nums, ind + 1, len, sum, map);

		map.put(key, case2);
		return case2;
	}
	
    // Min. no. of steps to  make both strings equal
    // Input: word1 = "sea", word2 = "eat"
    // Output: 2
    // TC = O(n1*n2), SC =  O(n1*n2)
	public int minDistance(String word1, String word2) {
		int n1 = word1.length();
		int n2 = word2.length();
		int[][] dp = new int[n1 + 1][n2 + 1]; // To get no of steps after removing a character from either strings 

		for (int i = 0; i<= n1; i++) {
			for (int j = 0; j<= n2; j++) {
				// if no character is removed from word1 then steps = no. of characters of word2
				if (i == 0) dp[i][j] = j;
				// if no character is removed from word2 then steps = no. of character of word1
				else if (j == 0) dp[i][j] = i;
				else {
					// if prev chartacters are same then steps = prev steps
					if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
						dp[i][j] = dp[i - 1][j - 1];
					}

					// Else take min of both cases 
					else {
						dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1]);
					}
				}
			}
		}
		return dp[n1][n2];

	}
	
    // TC = O(V+E), SC = O(V+E)
    long maxScore;
    int count;
    public int countHighestScoreNodes(int[] parents) {
        int n = parents.length;
        // create an adjacancy list of edges of each vertex
        List<Integer> list[] = new ArrayList[n];
        
        for (int i=0;i<n;i++){
              list[i] = new ArrayList<>();
        }
        
        for (int i=1;i<n;i++){
            list[parents[i]].add(i);
        }
     
        maxScore = 0L;
        count = 0;
        dfs(0, list, n); // dfs to count the  number of nodes in tree with root 0 
        
        return count;
    }
    
    
    // This function calculates the number of node in the subtree of root u 
    private long dfs(int u, List<Integer> list[], int n){

        int total=0;
        long prod=1L, rem, val;
        
        for (Integer v : list[u]){
            val = dfs(v, list, n);
            total += val;
            prod *= val;
        }
        
        // if nodes  remaning beyond subtree with root u the  it will be taken into consideration
        rem = (long)(n-total-1);
        if (rem > 0) prod *= rem;
        
        // Logic for maxScore
        if (prod > maxScore){
            maxScore = prod;
            count = 1;
        } else if (prod == maxScore){
            count ++;
        }
        return total+1;
    }
	
    // Input: n = 1000
    // Output: 1333
    // TC = O(n)
    // SC = O(1)
    public int nextBeautifulNumber(int n) {
	int number = n;
	while (true) {
		number++;
		String s = Integer.toString(number);

		int len = s.length();
		HashMap<Integer, Integer> freq = new HashMap<>();
		// calculate the frequency of each digit
		for (int i = 0; i<len; i++) {
			Integer digit = Integer.parseInt(String.valueOf(s.charAt(i)));
			freq.put(digit, freq.getOrDefault(digit, 0) + 1);
			if (digit<freq.get(digit)) break;
		}

		boolean isValid = true;
		// For every digit verify its freq
		for (Integer key: freq.keySet()) {
			if (freq.get(key) != key) {
				isValid = false;
				break;
			}
		}

		if (isValid) return number;
	}
   }

  // TOPOLOGICAL SORT	
  // Use toplogical sort for indegree and pq to minisnmise the time taken to complete the course 
  // TC = O(V+E) // As Simple DFS, SC = O(V) {Stack space}

  public int minimumTime(int n, int[][] relations, int[] time) {
  	// create adjancy list of graph
  	List<List<Integer>> graph = new ArrayList<>();

  	for (int i = 0; i<n; i++) {
  		graph.add(new ArrayList<>());
  	}
  	// create an indegree array for each node
  	int[] indegree = new int[n];
  	for (int[] e: relations) {
  		// create the graph
  		graph.get(e[0] - 1).add(e[1] - 1);
  		indegree[e[1] - 1] += 1;
  	}

  	int maxTime = 0;
  	// create a MIN-PQ to minimise time taken to complete all courses 

  	PriorityQueue<int[] > pq = new PriorityQueue<>(10, (t1, t2) -> t1[1] - t2[1]);

  	//Insert all dead end nodes ie. nodes with 0 indegree
  	for (int i = 0; i<n; i++) {
  		if (indegree[i] == 0) pq.offer(new int[] {
  			i, time[i]
  		});
  	}

  	while (!pq.isEmpty()) {
  		int[] curr = pq.poll();
  		int currCourse = curr[0];
  		int currTime = curr[1];

  		maxTime = Math.max(maxTime, currTime);

  		// Visit all adjance vertex of curr node and update the time taken to complete the courses 
  		for (int next: graph.get(currCourse)) {
  			// reduce indegreee by 1 as node as visited
  			indegree[next] -= 1;

  			// if indegree=0 means all its adajancent node has been visisted , Now move on to next  parent node
  			if (indegree[next] == 0) {
  				pq.offer(new int[] {
  					next, currTime + time[next]
  				});
  			}
  		}
  	}

  	// return minTime Taken to  complete all courses 
  	return maxTime;
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



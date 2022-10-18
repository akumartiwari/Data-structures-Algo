package com.company;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode() {}
 * ListNode(int val) { this.val = val; }
 * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class SolutionClosetElements {

    /**
     * @param head The linked list's head.
     * Note that the head is guaranteed to be not null, so it contains at least one node.
     */
    int size;
    ListNode curr;


    public SolutionClosetElements(ListNode head) {
        this.size = 0;
        this.curr = new ListNode();
        while (head != null) {
            head = head.next;
            this.size++;
        }
    }


    // To generate ramdom vaklue within a range
    /*

         min + (int ) Math.random() * (max - 1)
     */

    /**
     * Returns a random node's value.
     */
    public int getRandom() {
        return 1 + (int) (Math.random() * ((this.size - 1) + 1));
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * Solution obj = new Solution(head);
     * int param_1 = obj.getRandom();
     */


/*
    private List<Integer> expand(int[] arr, List<Integer> ans, int i, int j, int k) {
        int n = arr.length;
        while (k > 0) {
            if (i >= 0) {
                ans.add(arr[i]);
                i--;
            }
            if (j < n) {
                ans.add(arr[j]);
                j++;
            }
            k--;
        }
        Collections.sort(ans);
        return ans;
    }

    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int n = arr.length;
        List<Integer> ans = new ArrayList<>();
        int[] copy = Arrays.copyOfRange(arr, 0, n);
        Arrays.sort(copy);
        int index = Arrays.binarySearch(copy, x);
        if (index == -1) {
            int i = index - 1, j = index + 1;
            return expand(arr, ans, i, j, k);
        } else {
            Arrays.sort(copy);
            int diff = Integer.MAX_VALUE;
            int diffi = Integer.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                if (Math.abs(arr[i] - x) < diff) {
                    diff = Math.abs(arr[i] - x);
                    diffi = i;
                }
            }

            // expand towrds the centre
            return expand(arr, ans, diffi - 1, diffi + 1, k);
        }
    }

 */
    public List<Integer> findClosestElements(int[] nums, int k, int x) {

        int len = nums.length, count = 0;
        PriorityQueue<int[]> heap = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] ints, int[] t1) {
                return t1[0] - ints[0];
            }
        });

        for (int i = 0; i < len; i++) {
            if (count < k) {
                count++;
                heap.offer(new int[]{Math.abs(nums[i] - x), nums[i]});
            } else {
                if (Math.abs(nums[i] - x) >= heap.peek()[0])
                    continue;
                heap.remove();
                heap.offer(new int[]{Math.abs(nums[i] - x), nums[i]});
            }
        }

        List<Integer> ans = new ArrayList<>();
        while (!heap.isEmpty())
            ans.add(heap.remove()[1]);

        Collections.sort(ans);
        return ans;
    }

    public List<Integer> findClosestElementsBinary(int[] nums, int k, int x) {
        List<Integer> ans = new ArrayList<>();
        int len = nums.length, lo = 0, hi = len - 1, left = -1, right = len;

        // binary search to get closet number
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;

            // element is found
            if (nums[mid] == x) {
                left = mid;
                right = mid + 1;
                break;
            } else if (nums[mid] < x) {
                right = mid;
                hi = mid - 1;
            } else {
                left = mid;
                lo = mid + 1;
            }
        }

        // As of now we have range of target elements
        while (left >= 0 && right < left && k >= 1) {
            if (Math.abs(nums[left] - x) <= Math.abs(nums[right] - x)) left++;
            else right--;
            k--;
        }

        // This is done to handle case when any 1 ont them have reached at  extremeties
        while (left >= 0 && k >= 1) {
            left--;
            k--;
        }

        while (right < len && k >= 1) {
            right++;
            k--;
        }

        // addd the range in ans object
        for (int i = left + 1; i < right; i++) ans.add(nums[i]);

        Collections.sort(ans);
        return ans;

    }

    List<ListNode> arrlist1 = new ArrayList<>();
    List<ListNode> arrlist2 = new ArrayList<>();
    List<ListNode> ans = new ArrayList<>();

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // insert ListNodes to respective arrLists
        while (true) {
            if (l1 != null && l2 != null) {
                arrlist1.add(l1);
                arrlist2.add(l2);
                l1 = l1.next;
                l2 = l2.next;
            } else if (l1 != null) {
                arrlist1.add(l1);
                l1 = l1.next;
            } else if (l2 != null) {
                arrlist2.add(l2);
                l2 = l2.next;
            } else break;
        }

        int carry = 0, sum;
        for (int i = arrlist1.size() - 1, j = arrlist2.size() - 1; i >= 0 || j >= 0;
             i--, j--) {

            if (i >= 0 && j >= 0) {
                sum = carry + arrlist1.get(i).val + arrlist2.get(j).val;
                carry = sum / 10;
                sum %= 10;
                ans.add(new ListNode(sum));
            } else if (i >= 0) {
                sum = carry + arrlist1.get(i).val;
                carry = sum / 10;
                sum %= 10;
                ans.add(new ListNode(sum));
            } else {
                sum = carry + arrlist2.get(j).val;
                carry = sum / 10;
                sum %= 10;
                ans.add(new ListNode(sum));
            }
        }

        // if any carry is pending then add it as well
        if (carry != 0) ans.add(new ListNode(carry));

        // Iterate them and start addition of numbers
        ListNode dummyNode = ans.get(ans.size() - 1);
        ListNode curr = dummyNode;

        // traverse ans arraylist in reverse order and build final linked list
        for (int i = ans.size() - 2; i >= 0; i--) {
            curr.next = ans.get(i);
            curr = curr.next;
        }

        return dummyNode;
    }

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode() {}
     * ListNode(int val) { this.val = val; }
     * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
  /*
  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
       ListNode rl1 = reverse(l1);
       ListNode rl2 = reverse(l2);

       int carry = 0, sum;
       ListNode dummyNode = new ListNode();
       ListNode curr = dummyNode;

       while (rl1 != null && rl2 != null){
           sum = rl1.val + rl2.val + carry;
           carry = sum/10;
           sum %= 10;

           // How to point node to dummy node
           curr.next = new ListNode(sum);
           curr = curr.next;
       }

      return reverse(dummyNode);
  }

  private ListNode reverse(ListNode  node){
      if (node == null || node.next == null) return node;
      ListNode p = reverse(node.next);
      node.next.next = node;
      node.next = null ;
      return p;
  }
  */

    // [1,3,4,2,2]
    /*
"pwwkew"

{w=5}
l = 2
i=1
ml=2
l=3
i=2
i=2
ml= 3
l=1

{b=}
bbbbbbb
l=1
i=0
ml=1
l=1

"dvdf"  

     */
    // Floyd cycle algorithm solve the problem
    public int findDuplicate(int[] nums) {
        int n = nums.length;
        int slow = 0, fast = 0;
        boolean hascycle = false;
        while (slow < n) {
            if (fast > n - 1) fast -= n;
            if (slow != 0 && nums[slow] == nums[fast]) {
                hascycle = true;
                return nums[slow];
            }
            slow++;
            fast += 2;
        }

        // if (hascycle) {
        //     slow = 0;
        //     while (slow != fast) {
        //         slow++;
        //         fast++;
        //     }
        // }

        return hascycle ? nums[slow] : 0;
    }

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        if (n == 0) return 0;

        // To store character along with its index
        HashMap<Character, Integer> map = new HashMap<>();

        int maxLength = Integer.MIN_VALUE;
        int length = 0;
        for (int i = 0; i < n; i++) {
            if (map.containsKey(s.charAt(i))) {
                maxLength = Math.max(maxLength, length);
                int prev = map.get(s.charAt(i));
                length = i - prev;
                map = new HashMap<>();
            } else length++;

            map.put(s.charAt(i), i);
        }

        return Math.max(maxLength, length);

    }

    //"abcabcbb"
    /*
        abcabcbb

     */
    public int lengthOfLongestSubstrings(String s) {
        int n = s.length();
        if (n == 0) return 0;

        HashSet<Character> set = new HashSet<>();
        int i = 0, j = 0, maxCount = Integer.MIN_VALUE;

        while (j < n) {
            if (!set.contains(s.charAt(j))) {
                set.add(s.charAt(j));
                j++;
                maxCount = Math.max(set.size(), maxCount);
            } else {
                set.remove(s.charAt(i));
                i++;
            }
        }
        return maxCount;
    }

    public int maximumUniqueSubarray(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;

        HashSet<Integer> set = new HashSet<>();
        int i = 0, j = 0, maxSum = Integer.MIN_VALUE;
        int setSum = 0;

        while (j < n) {
            if (!set.contains(nums[j])) {
                set.add(nums[j]);
                setSum += nums[j];
                j++;
                maxSum = Math.max(setSum, maxSum);
            } else {
                setSum -= nums[i];
                set.remove(nums[i]);
                i++;
            }
        }
        return maxSum;
    }

    public int maxAbsValExpr(int[] A, int[] B) {
        int i, max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        int max3 = Integer.MIN_VALUE, max4 = Integer.MIN_VALUE, min3 = Integer.MAX_VALUE, min4 = Integer.MAX_VALUE;
        int a, b, c;
        //when you open the modulus and solve, you get 4 similar cases mentined below

        for (i = 0; i < A.length; i++) {
            //for case type 1 --> (A[i]+B[i]+i)-(A[j]+B[j]+j)
            max1 = Math.max(max1, A[i] + B[i] + i);
            min1 = Math.min(min1, A[i] + B[i] + i);

            //for case type 2 --> (A[i]+B[i]-i)-(A[j]+B[j]-j)
            max2 = Math.max(max2, A[i] + B[i] - i);
            min2 = Math.min(min2, A[i] + B[i] - i);

            //for case type 3 --> (A[i]-B[i]+i)-(A[j]-B[j]+j)
            max3 = Math.max(max3, A[i] - B[i] + i);
            min3 = Math.min(min3, A[i] - B[i] + i);

            //for case type 4 --> (A[i]-B[i]-i)-(A[j]-B[j]-j)
            max4 = Math.max(max4, A[i] - B[i] - i);
            min4 = Math.min(min4, A[i] - B[i] - i);
        }

        a = Math.max(max1 - min1, max2 - min2);
        b = Math.max(max3 - min3, max4 - min4);
        c = Math.max(a, b);

        return c;
    }

    // TC = O(nlogn), SC= O(n)

    // [1,3]

    public int minDifference(int[] nums) {
        int n = nums.length;
        if (n <= 4) return 0;
        Arrays.sort(nums);
        int res = Integer.MAX_VALUE;
        int interval = n - 4;

        for (int i = 0; i + interval < n; i++) res = Math.min(nums[i + interval] - nums[i], res);
        return res;
    }

            /* - use a stack to store char/by char
           - validate top of stack on each input and if it matches the config pair remove top
           - if not matches store in stack
           - check stack size once traverse all char, if empty it is valid otw not
        O(n) space and O(n) time complexity
        */


    // Input: s = "([)]"

    public boolean isValid(String s) {
        int n = s.length();
        // configure the map
        HashMap<String, String> map = new HashMap<>();
        map.put("(", ")");
        Stack<String> stk = new Stack<>();

        for (int i = 0; i < n; i++) {
            String value = String.valueOf(s.charAt(i));
            if (!stk.empty() && map.get(stk.peek()) != null && (map.get(stk.peek()).equalsIgnoreCase(value))) stk.pop();
            else stk.push(value);
        }
        return stk.isEmpty();
    }

    public List<String> generateParenthesis(int n) {
        Set<String> validParantheses = new HashSet<>();

        StringBuilder s = new StringBuilder();
        while (n > 0) {
            s.append("()");
            n--;
        }

        List<List<Character>> result = new ArrayList<>();
        char[] nums = s.toString().toCharArray();
        permute(nums, 0, nums.length - 1, result);

        for (List<Character> value : result) {
            String str = value.stream().map(Object::toString).collect(Collectors.joining());
            if (isValid(str)) validParantheses.add(str);
        }
        return new ArrayList<>(validParantheses);
    }

    private void permute(char[] nums, int l, int r, List<List<Character>> result) {
        if (l == r) {

            List<Character> list = new ArrayList<Character>();
            for (char c : nums) {
                list.add(c);
            }

            result.add(list);
            return;
        }
        for (int i = l; i <= r; i++) {
            swap(nums, l, i);
            permute(nums, l + 1, r, result);
            swap(nums, l, i);
        }
    }

    public void swap(char[] nums, int i, int j) {
        char temp;
        temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    public List<String> generateParenthesisBackTrack(int n) {

        final List<String> result = new ArrayList<>();
        final char[] chars = new char[n * 2];
        backtrack(result, 0, 0, n, 0, chars);
        return result;
    }

    private void backtrack(List<String> result, int open, int close, int n, int index, char[] arr) {

        if (open < n) {
            arr[index] = '(';
            backtrack(result, open + 1, close, n, index + 1, arr);
        }
        if (close < open) {
            arr[index] = ')';
            backtrack(result, open, close + 1, n, index + 1, arr);
        }
        if (open == n && close == n) {
            result.add(new String(arr));
        }
    }
    /*
       Set<String> paranAns = new HashSet<>();
    public List<String> generateParenthesisOne(int n) {
        generate(n, 0, 0, "", "L");
        return new ArrayList<>(paranAns);
    }

    private void generate(int n, int l, int r, String curr, String lorr) {

        if (l == r && l == n) this.paranAns.add(curr);

        if (lorr.equalsIgnoreCase("L") && l < n) {
            generate(n, l + 1, r, curr.concat("("), "L");
            generate(n, l + 1, r, curr.concat("("), "R");
        } else if (lorr.equalsIgnoreCase("R") && r < n) {
            if (r < l) {
                generate(n, l, r + 1, curr.concat(")"), "L");
                generate(n, l, r + 1, curr.concat(")"), "R");
            }
        }
    }

    public boolean canJump(int[] nums) {
        int n = nums.length;
        if (n == 0) return false;

        return bfsArray(nums);
    }

    private boolean bfsArray(int[] nums) {
        //min Pq to store elements along with index
        Queue<Pair<Integer, Integer>> queue = new LinkedList<>();
        queue.add(new Pair<>(nums[0], 0));

        while (!queue.isEmpty()) {
            Pair<Integer, Integer> element = queue.peek();
            if (element.second() == nums.length - 1) return true;

            int allPair = 0;
            while (allPair < element.first()) {
                queue.add(new Pair<>(nums[element.first() - allPair], element.second() - allPair));
                allPair++;
            }
        }

        return false;
    }

     */

    static class Pair implements Comparable<Pair> {
        int first;
        int second;

        Pair(int key, int val) {
            this.first = key;
            this.second = val;
        }

        @Override
        public int compareTo(Pair pair) {
            return 0;
        }
    }

    public boolean canJump(int[] nums) {
        int n = nums.length;
        if (n == 0) return false;

        return bfsArray(nums);
    }

    private boolean bfsArray(int[] nums) {
        /*
        int n = nums.length;
        //min Pq to store elements along with index
        Queue<Pair> queue = new LinkedList<>();
        int depth = 0;
        boolean[] visited = new boolean[n];
        queue.add(new Pair(nums[0], 0));
        visited[nums[0]] = true;
        while (!queue.isEmpty()) {
            Pair element = queue.poll();

            if (element.second == nums.length - 1) return true;

            int allPair = 0;
            while (allPair < element.first) {
                if (!visited[nums[Math.abs(element.second - allPair)]]) {
                    queue.add(new Pair(nums[Math.abs(element.second - allPair)], Math.abs(element.second - allPair)));
                    visited[nums[Math.abs(element.second - allPair)]] = true;
                }
                allPair++;
            }
            depth++;
        }
        return false;

         */
        int n = nums.length;
        //min Pq to store index
        Queue<Integer> queue = new LinkedList<>();
        int depth = 0;
        // To store status of index of array ie. either visited or not
        boolean[] visited = new boolean[n];

        queue.add(0);
        visited[0] = true;

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int index = queue.remove();
                if (index == n - 1) return true;
                for (int j = 1; j <= nums[index]; j++) {
                    int neighbourIndex = j + index;
                    if (neighbourIndex > n - 1) break;// overflow
                    if (visited[neighbourIndex]) continue; // do nothing go ahead
                    queue.add(neighbourIndex);
                    visited[neighbourIndex] = true;
                }

            }
            depth++;
        }
        return false;
    }

    // [1,2,3,4,5,6,7,8]

    /*
    public boolean splitArraySameAverage(int[] nums) {
        int n = nums.length;
        if (n < 2) return false;
        if (n == 2) return nums[0] == nums[1];
        double avg = Arrays.stream(nums).average().getAsDouble();

        List<Double> listP = new ArrayList<>();
        List<Double> listM = new ArrayList<>();

        // Put the average difference total avg with numbers
        for (int num : nums) {
            double a = num - avg;
            if (a == 0) return true;
            if (a > 0) {
                if (listM.contains(a)) return true;
                listP.add(a);
            } else {
                if (listP.contains(Math.abs(a))) return true;
                listM.add(Math.abs(a));
            }
        }

        Optional<Double> mSum = listM.stream().reduce(Double::sum);
        Set<Double> mSumSet = new HashSet<>();

        for (double m : listM) {
            Set<Double> mSumSetTemp = new HashSet<>();

            for (double mm : mSumSet) {
                mSumSetTemp.add(mm + m);
            }

            mSumSet.addAll(mSumSetTemp);
            mSumSet.add(m);
        }
        mSumSet.remove(mSum.get());

        Set<Double> pSumSet = new HashSet<>();

        for (double p : listP) {
            Set<Double> pSumSetTemp = new HashSet<>();

            for (double pp : pSumSet) {
                if (mSumSet.contains(p + pp)) return true;
                pSumSetTemp.add(pp + p);
            }

            pSumSet.addAll(pSumSetTemp);
            if (mSumSet.contains(p)) return true;
            pSumSet.add(p);
        }


        for (int i = 0, j = 0; i < mSumSet.size() && j < pSumSet.size(); i++, j++) {
            if (new ArrayList<>(mSumSet).get(i) - new ArrayList<>(pSumSet).get(i) < 0.00000001) return true;
        }
        return false;
    }
     */


   /*
       sum1 / len1  = sum2 /len2 --->1
       sum2   = total-sum1 ---->2
       eqn 2 to :-
       sum1 / len1 = total-sum1/len2
       sum1/len1 + sum1/len2 = total/len2
       sum1(len1+len2)/len1*len2= total/len2

       sum1 = total * len1 / n where n is size of array
       Find all possible subsequences of length 1 n-2 such that avgs are equal

    */

    // To store count * index * sum as key present in map --> To avoid duplicate calcuations (memoization)
    private Map<String, Boolean> map;

    public boolean splitArraySameAverage(int[] nums) {
        int n = nums.length;
        if (n < 2) return false;
        if (n == 2) return nums[0] == nums[1];

        int sum = Arrays.stream(nums).sum();
        map = new HashMap<>();

        for (int count = 1; count < n - 1; count++) {
            if ((count * sum) % n == 0) {
                return isPossible(nums, 0, count, (count * sum) / n);
            }
        }

        return false;
    }

    //This method tell us is it possible to segrgate array into parts having size = count such avgs equal
    private boolean isPossible(int[] nums, int index, int count, int sum) {

        // base cases
        if (sum == 0 && count == 0) return true;
        if (count == 0 || index == nums.length) return false;

        String key = index + "*" + count + "*" + sum;
        if (map.containsKey(key)) return map.get(key);

        if (sum - nums[index] > 0) {
            boolean case1 = isPossible(nums, index + 1, count - 1, sum - nums[index]);
            boolean case2 = isPossible(nums, index + 1, count, sum);

            map.put(key, case1 || case2);
            return case1 || case2;
        }

        boolean case2 = isPossible(nums, index + 1, count, sum);
        map.put(key, case2);
        return case2;
    }
}

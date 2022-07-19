package com.company;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.Arrays.asList;

//import for Scanner and other utility classes
public class DynamicArray {
    private int[] array; // original array
    int size; // max size of array
    int index; // count number of elements of array

    public DynamicArray(int[] array, int size, int count) {
        this.array = array;
        this.size = size;
        this.index = count;
    }

    public void add(int data) {
        // check if size of array overflows
        if (array.length == size) {
            growSize(array);
        }
        array[index] = data;
        index++;
    }

    private void growSize(int[] array) {
        int[] temp = null;
        if (index == size) {
            temp = new int[size * 2]; // initialise with size double with given size
            // copy all existing elements to new temp array
            for (int i = 0; i < size; i++) {
                temp[i] = array[i];
            }
        }
        this.array = temp;
        size *= 2;
    }


    public void insertAt(int data, int index) {

        // check if size of array overflows
        if (array.length == size) {
            growSize(array);
        }

        for (int i = array.length - 1; i > index; i--) {
            array[i + 1] = array[i];
        }

        array[index] = data;
        this.index++;
    }

    // This function will shrink the array to its actual size
    public void shrinkSize() {
        int[] temp = null;

        if (index > 0) {
            temp = new int[index];

            for (int i = 0; i < array.length; i++) {
                temp[i] = array[i];
            }
            array = temp;
            size = index;
        }
    }

    // function remove last element or put
    // zero at last index
    public void remove() {
        if (index > 0) {
            array[index - 1] = 0;
            index--;
        }
    }

    // function shift all element of right
    // side from given index in left
    public void removeAt(int index) throws Exception {
        // check if size of array overflows
        if (array.length <= index) {
            throw new Exception("Invalid index!");
        }

        for (int i = array.length - 1; i > index; i--) {
            array[i - 1] = array[i];
        }

//        for (int i = index; i < count - 1; i++) {
//            array[i] = array[i + 1];
//        }
        array[this.index - 1] = 0;
        this.index--;
    }

    /*
      public int findKthNumber(int m, int n, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder()); //max-heap

        for (int rows = 1; rows <= m; rows++) {
            for (int cols = 1; cols <= n; cols++) {
                if (pq.size() <= k) {
                    pq.add(rows * cols);
                } else break;
            }
            break;
        }
        int ans = pq.peek();
        pq.clear();
        return ans;
    }

     */
    // 2 pointer approach to solve problem
    /*
       Algorithm :-
       * create a min-heap named  pq and an integer counter .
       * Start with ptrs i,j : i = 0 , j = arr.length-1
       * arr[i+1]/arr[j] <  arr[i]/arr[j-1] and push smaller one into pq (mean-heap)
       * repeat above step till counter reaches to k
       * Reaturn pq.peek()
       * Done
     */

    /*
    Input: arr = [1,2,3,5], k = 3
    Output: [2,5]

    l =0 , h = 3;
    l=1 , h=3 counter = 2
     */

    private int[] getMapValueAt(LinkedHashMap<Integer, Integer> hashMap, int index) {
        Map.Entry<Integer, Integer> entry = (Map.Entry<Integer, Integer>) hashMap.entrySet().toArray()[index];
        return new int[]{entry.getKey(), entry.getValue()};
    }

    public int[] kthSmallestPrimeFraction(int[] arr, int k) {
        int n = arr.length;
        int l = 0;
        int h = n - 1;
        int[] ans = new int[2];
        if (k == 0) return new int[]{};
        if (k == 1) return new int[]{arr[0], arr[n - 1]};
        LinkedHashMap<Integer, Integer> map = new LinkedHashMap<>();
        map.put(arr[l], arr[h]);

        while (l < h) {

            double rd = (float) arr[l] / arr[h - 1];
            double li = (float) arr[l + 1] / arr[h];

            if (li <= rd) {
                map.put(arr[l + 1], map.getOrDefault(arr[l + 1], arr[h]) + 1);
                map.put(arr[l], map.getOrDefault(arr[l], arr[h - 1]) + 1);
                l++;
            } else {
                h--;
                map.put(arr[l], map.getOrDefault(arr[l], arr[h - 1]) + 1);
                map.put(arr[l + 1], map.getOrDefault(arr[l + 1], arr[h]) + 1);
            }
            if (map.size() >= k) {
                ans = getMapValueAt(map, k);
                return ans;
            }
        }
        return ans;
    }


    public int findKthNumber(int m, int n, int k) {
        int index = 0;
        int[] ans = new int[m * n];
        for (int rows = 1; rows <= m; rows++) {
            for (int cols = 1; cols <= n; cols++) {
                ans[index] = rows * cols;
                index++;
            }
        }
        Arrays.sort(ans);
        return ans[k - 1];
    }


    public int kthSmallest(int[][] matrix, int k) {

        int index = 0;
        int n = matrix.length;
        int[] ans = new int[n * n];
        for (int rows = 0; rows < n; rows++) {
            for (int cols = 0; cols < matrix[rows].length; cols++) {
                ans[index] = matrix[rows][cols];
                index++;
            }
        }
        Arrays.sort(ans);
        return ans[k - 1];
    }

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) throws Exception {
        List<List<Integer>> ans = new ArrayList<>();
        HashMap<Integer, Pair> map = new HashMap<>();
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        if (nums1.length <= nums2.length) {
            for (int i = 0; i < nums1.length; i++) {
                int sum = nums1[i] + nums2[i];
                map.put(sum, map.getOrDefault(sum, new Pair(nums1[i], nums2[i])));
            }
        } else new ArrayList<>();

        // TreeMap to store values of HashMap
        TreeMap<Integer, Pair> sorted = new TreeMap<>();

        // Copy all data from hashMap into TreeMap
        sorted.putAll(map);

        sorted.keySet().stream().map(data -> {
            List<Integer> list = new ArrayList<>();
            list.add(sorted.get(data).key);
            list.add(sorted.get(data).val);
            ans.add(list);
            return index;
        });

        return ans;
    }


    static class Pair implements Comparable<Pair> {
        int key;
        int val;

        Pair(int key, int val) {
            this.key = key;
            this.val = val;
        }

        @Override
        public int compareTo(Pair pair) {
            return this.val - pair.val;
        }
    }


    /*
    Input: s = "abc", t = "ahbgdc"
    Output: true

    Input: s = "abcde", words = ["a","bb","acd","ace"]
    Output: 3

    "abcde"
    ["a","bb","acd","ace"]

         */


    //O(words*s*log(t))
    // O(words*s))

    HashMap<Character, Integer> map = new HashMap<>();

    public int numMatchingSubseq(String s, String[] words) {
        int ans = 0;
        for (String word : words) {
            if (isSubsequence(word, s)) ans++;
        }
        return ans;
    }


    /*
    abcde

    bb
    map ()

    "kguhsugfxvwxakdcovjeczhqvbevkhlixsrhumxykbkihjdfxxxwragzcbhngbzgasxysxdtwntvbpdihtvkffacmxhbxxqniyqm"
    ["ykbkihjdfxxxwragzcbhngbzgasxysxdtwn","wxakdcovjeczhqvbevkhlixsrhumxykbkihj","diht","covjeczhqvbevkhlixsrhumxykbkihjdfxxxwragzcbhngbz","ovjeczhqvbevkhlixsrhumxykbkihjdfxxxwragzcbhng","qhzucvqxalfrtlrdjverseuldzymzbunhugekoyyghmkpkfqmd","eydmbsorvjnnifqxsyuypsrijzrosukrhtbneprpyyoawbvoki","uanfzlfmmtvhzzebrazyuslbapsfzwtlvqbhqxsfmqagwxetro","fffaawedchlcyqvzzxbzczbwyfjkllpsjhoozyresqemmawban","astrknwzefcmuswdxalooatmiduspjuofthtomoqilgdojwhon"]

     */
    public boolean isSubsequence(String s, String t) {
        if (s.length() > t.length()) {
            return false;
        }
        if (s.length() == 0) return true;

        int index = 0;
        int[][] table = new int[s.length()][t.length()];

        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < t.length(); j++) {
                table[i][j] = 0;
            }
        }

        HashMap<Integer, Integer> storeMap = new HashMap<>(); // To Store indexes of character from both strings {s,t}

        for (int i = 0; i < s.length(); i++) {

            // if character is found then index  will get started from next element
            if (map.get(s.charAt(i)) != null && storeMap.get(map.get(s.charAt(i))) == null) {
                index = map.get(s.charAt(i)) + 1;
                storeMap.put(map.get(s.charAt(i)), i);
                continue;
            }
            boolean isCharFound = false;
            for (int j = index; j < t.length(); j++) {
                if (s.charAt(i) == t.charAt(j)) {
                    index++;
                    isCharFound = true;
                    map.put(t.charAt(j), j); // put in  map with character and latest index
                    storeMap.put(j, i); // to store indexes from desired string and original
                    break;
                }
                index++;
            }
            if (!isCharFound) {
                return false;
            }
        }
        return true;
    }

    /*
    Input: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
Output: 3
E
     */

    //  [[2000,3200], [200,1300], [1000,1250], [100,200]]
    //courses[i] = [durationi, lastDayi]
    public int scheduleCourse(int[][] courses) {
        // sort arrays based on lastday to complete a course in desc
        Arrays.sort(courses, (a, b) -> a[1] - b[1]);
        // To store timetaken to complete a course
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);

        int time = 0; // current time of course in progress
        for (int[] c : courses) {
            // if time taken to complete current course is lesser than last day to pick then good to go
            if (time + c[0] <= c[1]) {
                pq.offer(c[0]);
                time += c[0];
            } else if (!pq.isEmpty() && pq.peek() > c[0]) { // else compare it with latest course taken wrt duration and pick the one least duration
                time += c[0] - pq.poll();
                pq.offer(c[0]);
            }
        }
        return pq.size();
    }


    /*
    [[0,1],[1,0]]
    [[1,0],[0,1]]
    */

    public boolean findRotation(int[][] mat, int[][] target) {


        int[][] current = mat;
        if (Arrays.deepEquals(current, target)) return true;

        // At max current mat can be rotated 4 times
        for (int i = 0; i < 4; i++) {
            if (Arrays.deepEquals(current, target)) return true;
            current = rotate(mat);
        }

        return false;
    }

    /*
    public void rotate(int[] nums, int k) {

        int n = nums.length;

        for (int j = 0; j < k; j++) {
            int sp = nums[n - 1];
            for (int i = n - 1; i > 0; i--) {
                nums[i] = nums[i - 1];
            }
            nums[0] = sp;
        }
    }

     */

    public void rotate(int[] nums, int k) {
        int n = nums.length;
        String str = "";
        for (int j = 0; j < n; j++) {
            str += nums[j];
        }

        int index = 0;
        for (char c : str.substring(n - k).concat(str.substring(0, k + 1)).toCharArray()) {
            if (String.valueOf(c) == "-") {
                nums[index] = -Integer.parseInt(String.valueOf(++c));
            }
            index++;
        }
    }


    public int[][] rotate(int[][] matrix) {
        int n = matrix.length;
        int[][] nmatrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nmatrix[i][j] = matrix[n - 1 - j][i];
                System.out.println(nmatrix[i][j]);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = nmatrix[i][j];
            }
        }
        return matrix;

    }


    public int reductionOperations(int[] nums) {


        return 0;
    }

    public boolean findRotation1(int[][] mat, int[][] target) {
        int countzerom = 0;
        int countonem = 0;
        int countzerot = 0;
        int countonet = 0;

        if (mat.length == 0 && target.length == 0) return true;
        if (mat.length == 0 || target.length == 0) return false;
        if (mat.length != target.length) return false;
        System.out.println("sizes not equal");

        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if (mat[i][j] == 0) countzerom++;
                else countonem++;
            }
        }

        for (int i = 0; i < target.length; i++) {
            for (int j = 0; j < target[i].length; j++) {
                if (target[i][j] == 0) countzerot++;
                else countonet++;
            }
        }

        if (countonem != countzerot || countonem != countonet) {
            System.out.println("count not equal");
            return false;
        }

        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if (mat[i][mat.length - 1 - i] != target[mat.length - 1 - i][i]) {
                    System.out.println("diagonal not equal");
                    return false;
                }
            }
        }
        return true;
    }

    //"111000"

    public int minFlips(String s) {
        char last = ' ';
        int res = 0;

        for (int i = 0; i < s.length(); i++) {
            if (i != 0 && last == s.charAt(i)) {
                res++;
            }
            last = s.charAt(i);
        }

        // To return min flips
        return res - 1;
    }



    /*
       index =1
       index =2
       index=3
     */


    // [1,2,3,5]
    // [1,5,4 2,5,4, 3,5,4 5,5,4]
    public int[] kthArraySmallestPrimeFraction(int[] arr, int k) {
        //min-heap with increasing order of fraction magnitude
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>((a, b) -> a[0] * b[1] - a[1] * b[0]);

        int[] res = new int[2];
        if (arr.length == 0 || k == 0) return res;

        int i = 0, j = arr.length - 1;
        while (i < j) {
            pq.offer(new int[]{arr[i], arr[j], j});
            i++;
        }

        while (k-- > 0 && !pq.isEmpty()) {
            int[] curr = pq.poll();
            System.out.println(curr[0] + " " + curr[1] + " " + curr[2]);
            if (k == 0) {
                res[0] = curr[0];
                res[1] = curr[1];
            }
            if (curr[2] == 0) continue;
            pq.offer(new int[]{curr[0], arr[curr[2] - 1], curr[2] - 1});
        }


        return res;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        if (k == 0) return head;
        ListNode fast = head;
        int l = 1;
        while (fast.next != null) {
            fast = fast.next;
            l++;
        }
        int shifts = k % l;

        if (l == k || shifts == 0) return head;

        ListNode current = head;
        int curr = 1;
        while (curr < l - shifts) {
            current = current.next;
            curr++;
        }
        ListNode temp = current.next;
        current.next = null;
        fast.next = head;
        return temp;
    }

    // O(n)
// [1,2,3,4]
//5
    /*
    Dry run:-
    l = 4
    r = 4;
    q = 1
  k = 5
     */

    public int size(ListNode root) {
        ListNode curr = root;
        int size = 0;
        while (curr != null) {
            curr = curr.next;
            size++;
        }
        return size;
    }

    public ListNode[] splitListToPartsO(ListNode root, int k) {
        ListNode[] res = new ListNode[k];
        int size = size(root);  // First you find out the size of linked list.
        ListNode curr = root;
        ListNode prev = root;
        int i = 0;
        int div = size / k;  // Get how many part it can really divided.
        int mod = size % k; // Get how many additional one node need to be added from start.
        while (curr != null) {
            res[i++] = curr;
            int temp = div;
            while (temp-- != 0) {
                prev = curr;
                curr = curr.next;
            }
            if (mod != 0) {    // Take one extra node till mod > 0 i.e. ensures that one extra node is added from starting
                prev = curr;
                curr = curr.next;
                mod--;
            }

            prev.next = null;
        }
        return res;
    }


    public ListNode[] splitListToParts(ListNode root, int k) {
        ListNode[] list = new ListNode[k];

        if (root == null) return list;
        if (k == 0) {
            ListNode[] nodes = new ListNode[1];
            nodes[0] = root;
            return nodes;
        }
        ListNode fast = root;
        ListNode head = root;
        System.out.println(head.val);

        int l = 1;
        while (fast.next != null) {
            fast = fast.next;
            l++;
        }

        int r = l % k;
        int q = l / k;
        if (k < l) {
            boolean isFirstPart = true;

            while (l / k-- > 0) {
                while (k-- != 0) {
                    ListNode node = new ListNode();
                    int curr = 0;

                    if (isFirstPart) {
                        while (curr < r + q) {
                            root = root.next;
                            curr++;
                        }
                        node = root.next;
                        isFirstPart = false;
                    } else {
                        while (curr < q) {
                            root = root.next;
                            curr++;
                        }
                        node = root.next;
                    }
                    list[l - l / k] = node;
                }
            }
        } else if (k == l) list[0] = root;
        else {
            int count = 1;
            while (count == l) {
                list[count - 1] = new ListNode(root.val, null);
                root = root.next;
                count++;
            }
        }
        return list;
    }


    public ListNode oddEvenList(ListNode head) {

        int counter = 1;
        ListNode odd = head;
        ListNode even = head;

        while (head.next != null) {
            if (counter % 2 == 0) {
                even.next = head.next;
                System.out.println("Even= " + even.val);
            } else {
                odd.next = head.next;
                System.out.println("Even= " + odd.val);
            }
            head = head.next;
            counter++;
        }

        even = odd.next;
        head = odd.next;
        return head;
    }


    /*
"abcd"
"cdef"
3
s = "abcd", t = "acde", maxCost = 0


------------------------------------------------
s = "abcd", t = "bcdf", maxCost = 3



// start = 0, end = 0;

"krrgw"
"zjxss"
19
     */

    /*
    find the increasing sequence from the last having x breaks the sequence
find next greater element than x from last i.e z
swap z with x
reverse subarray from x + 1 to end

[1,2,3]
[1, 3,2]
index = 2, elem = 3;
temp = 3;
nums[2] = nums[3-1-2]
nums[2] = nums[0]
nums[2] = 1;
nums[0]= 3;
3,2,1,

tempswap = 1;
nums[2]=2;
nums[1]=1;

     3,1,2
     [3,2,1]




         */


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

        int temp = elem;
        nums[index] = nums[n - index];
        nums[n - index] = temp;


        if (index >= n - 1) {
            Collections.reverse(asList(nums));
            System.out.println(asList(nums));
        } else {
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


    public int equalSubstringOptimal(String s, String t, int maxCost) {
        int start = 0, end = 0, currcost = 0, maxlength = 0;
        int n = s.length();
        while (end < n) {
            currcost += Math.abs(s.charAt(end) - t.charAt(end));
            if (currcost > maxCost) {
                currcost -= Math.abs(s.charAt(start) - t.charAt(start));
                start++;
            }

            maxlength = Math.max(maxlength, end - start + 1);
            end++;
        }

        return maxlength;
    }


    public int equalSubstring(String s, String t, int maxCost) {
        if (s.length() == 0 || t.length() == 0) return 0;
        if (s.length() != t.length()) return 0;
        if (maxCost == 0) return 1;

        int[] dp = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            int curr = Math.abs((int) (s.charAt(i) - t.charAt(i)));
            int cost;
            if (i == 0) {
                cost = curr;
            } else {
                cost = Math.min(dp[i - 1], curr);
            }
            dp[i] = cost;
            System.out.println(dp[i]);
        }


        int index = 0;
        int maxLength = 0;
        int oriCost = 0;
        while (maxLength < maxCost && oriCost + dp[s.length() - 1 - index] <= maxCost) {
            maxLength += 1;
            oriCost += dp[s.length() - 1 - index];
            System.out.println(oriCost);
            index++;
        }
        return dp[s.length() - 1] > maxCost ? 1 : maxLength;
    }

}

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

class House {
    public class TreeNode {
        int val;
        House.TreeNode left;
        House.TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, House.TreeNode left, House.TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // This [paise class stores if value/node is taken or not as first and its actual value and second param
    // {0,23} -> not taken  with value=23
    static class Pair<I extends Number, I1 extends Number> implements Comparable<Pair<Number, Number>> {
        int first;
        int second;

        Pair(int key, int val) {
            this.first = key;
            this.second = val;
        }

        @Override
        public int compareTo(Pair<Number, Number> pair) {
            return this.first - pair.second;
        }
    }


    public Pair<Number, Number> dfs1(House.TreeNode root) {
        if (root == null) return new House.Pair<>(0, 0);
        Pair<Number, Number> left = dfs1(root.left);
        Pair<Number, Number> right = dfs1(root.right);

        return new House.Pair<Number, Number>((root.val + left.second + right.second), (Math.max(left.first, left.second) + Math.max(right.first, right.second)));
    }

    public int rob(House.TreeNode root) {
        Pair<Number, Number> ans = dfs1(root);
        return Math.max(ans.first, ans.second);
    }

    /*
     public static int longestCommonSubstrLengthRec(String s1, String s2, int i1, int i2, int count) {
    if (i1 == s1.length() || i2 == s2.length())
      return count;

    if (s1.charAt(i1) == s2.charAt(i2))
      count = longestCommonSubstrLengthRec(s1, s2, i1 + 1, i2 + 1, count + 1);

    int c1 = longestCommonSubstrLengthRec(s1, s2, i1, i2 + 1, 0);
    int c2 = longestCommonSubstrLengthRec(s1, s2, i1 + 1, i2, 0);

    return Math.max(count, Math.max(c1, c2));
  }

  public static int longestCommonSubstrLength(String s1, String s2)
  {
    return longestCommonSubstrLengthRec(s1, s2, 0, 0, 0);
  }

     */

    /*
    [1,3,1,3,100]
    robRec(nums, 2, 1, 1);
    O(n) O(1)
    2 pointer approach
     */
    public int rob(int[] nums) {
        if (nums.length == 1) return nums[0];
        if (nums.length == 0) return 0;
        return Math.max(robRec(nums, 0, nums.length - 2), robRec(nums, 1, nums.length - 1));
    }

    public int robRec(int[] nums, int l, int r) {
        int prev = 0, prepre = 0, curr = 0;
        for (int i = l; i <= r; i++) {
            curr = Math.max(nums[i] + prepre, prev);
            prepre = prev;
            prev = curr;
        }
        return curr;
    }


    class Solution {
        // Recurison Implementation
       /*
        int[] nums;

        public int rob(int[] nums) {
            if (nums.length == 0) return 0;
            if (nums.length == 1) return nums[0];


            // setting global int[] with current nums
            this.nums = nums;
            return Math.max(recurse(0, nums.length - 1), recurse(1, nums.length));
        }

        public int recurse(int curr, int end) {
            if (curr >= end) return 0;
            // 2 cases":-
            //   - Excluding the current num value ==> curr +=1
            //   - Including the current num ==>curr +=2


            return Math.max(recurse(curr + 1, end), nums[curr] + recurse(curr + 2, end));



//   2. Top-down DP ----> O(n), O(n)

        int[] nums, memo;

        public int rob(int[] nums) {

            int n = nums.length;
            this.nums = nums;
            // base cases
            if (n == 0) return 0;
            if (n == 1) return nums[0];
            memo = new int[n];

            // Reset memo table to get exact values
            Arrays.fill(memo, -1);
            int left = recurse(0, n - 1);

            // Reset again for next step
            Arrays.fill(memo, -1);
            int right = recurse(1, n);

            return Math.max(left, right);
        }

        public int recurse(int curr, int end) {

            if (curr >= end) return 0;
            else if (memo[curr] != -1) return memo[curr];

            Arrays.fill(memo, -1);

            memo[curr] = Math.max(recurse(curr + 1, end), nums[curr] + recurse(curr + 2, end));
            return memo[curr];

        }


    }


        // 3. Bottom-up DP O(n), O(n)
        int[] nums;
        int n;

        public int rob(int[] nums) {
            this.nums = nums;
            n = nums.length;
            // base cases
            if (n == 0) return 0;
            else if (n == 1) return nums[0];
            return Math.max(recurse(0, n - 1), recurse(1, n));
        }

        public int recurse(int start, int end) {
            int[] dp = new int[end];
            dp[start] = nums[start];
            if (end > start + 1) dp[start] = Math.max(dp[start], dp[start + 1]);

            for (int i = start + 2; i < n; i++) {
                // by including current num and by excluding it
                dp[i] = Math.max(nums[i] + dp[i - 1], dp[i - 2]);
            }

            return dp[end - 1];
        }
        */


        // 4. Bottom-up DP with constant space

        int[] nums;
        int n;

        public int rob(int[] nums) {
            this.nums = nums;
            n = nums.length;
            // base cases
            if (n == 0) return 0;
            else if (n == 1) return nums[0];
            return Math.max(recurse(0, n - 1), recurse(1, n));
        }

        public int recurse(int start, int end) {
            int prev = 0, prepre = 0;
            if (end > start + 1) {
                prev = Math.max(nums[start], nums[start + 1]);
                prepre = nums[start];
            } else prev = nums[start];


            for (int i = start + 2; i < end; i++) {
                // by including current num and by excluding it
                int curr = Math.max(nums[i] + prepre, prev);
                prepre = prev;
                prev = curr;
            }
            return prev;
        }
        /*
        / 0 < x < n and n % x == 0.
// n = n-x
Input: n = 3
Output: true
dpa[0]=0
dpb[0]=0
dpa[1]=1
dpv[1]=1

n == 2
         */

        public boolean divisorGame(int n) {
            if (n == 0) return false;

            int[] dpa = new int[n];
            int[] dpb = new int[n];

            Arrays.fill(dpa, -1);
            Arrays.fill(dpb, -1);
            dpa[0] = 0;
            dpb[0] = 0;

            int k = 0;
            while (n > 1) {
                dpa[k] = 1;
                n -= 1;
                if (n > 1) {
                    dpb[k] = 1;
                    n -= 1;
                }
                k++;
            }

            int a = 0;
            int b = 0;
            for (int i = dpa.length - 1; i > 0; i--) {
                if (dpa[i] != -1) {
                    a = i;
                    break;
                }
            }
            for (int i = dpb.length - 1; i > 0; i--) {
                if (dpb[i] != -1) {
                    b = i;
                    break;
                }
            }

            return a > b;
        }


        public int[][] matrixBlockSum(int[][] mat, int k) {
            int rows = mat.length;
            int cols = mat[0].length;
            int[][] ans = new int[rows][cols];

            if (n == 0) return new int[rows][cols];

            //Logic /crux of code
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {


                }
            }


            return ans;
        }

        int[] arr;

        // Complete the maxSubsetSum function below.
        int maxSubsetSum(int[] arr) {
            this.arr = arr;
            n = arr.length;
            // base cases
            if (n == 0) return 0;
            else if (n == 1) return nums[0];
            return Math.max(maxSum(0, n - 1), maxSum(1, n));

        }

        public int maxSum(int start, int end) {
            int[] dp = new int[end];
            dp[start] = nums[start];
            if (end > start + 1) dp[start] = Math.max(dp[start], dp[start + 1]);

            for (int i = start + 2; i < n; i++) {
                // by including current num and by excluding it
                dp[i] = Math.max(nums[i] + dp[i - 1], dp[i - 2]);
            }

            return dp[end - 1];
        }


        public int maxSubsetSum1(int[] arr) {
            if (arr.length == 0) return 0;
            arr[0] = Math.max(0, arr[0]);
            if (arr.length == 1) return arr[0];
            arr[1] = Math.max(arr[0], arr[1]);
            for (int i = 2; i < arr.length; i++)
                arr[i] = Math.max(arr[i - 1], arr[i] + arr[i - 2]);
            return arr[arr.length - 1];
        }


        /*
        8
2
4
3
5
2
6
4
5
Sample Output 2

12
[22344556] ---> [21231212]
10


10

2
4
2
6
1
7
8
9
2
1
[1212]
6
3 5 7 11 5 8
5 7 11 10 5 8


3 5 5 7 8 11
5 5 7 8 10 11

         */


        public int beautifulPairs(List<Integer> A, List<Integer> B) {
            int ans = 0;
            boolean isSwitched = false;
            Collections.sort(A);
            Collections.sort(B);
            for (int i = 0; i < A.size(); i++) {
                int index = Arrays.binarySearch(B.toArray(), A.get(i));
                if (index == -1) {
                    if (!isSwitched) {
                        if (B.get(i) != null) {
                            B.remove(i);
                        }
                        B.add(A.get(i));
                        Collections.sort(B);
                        isSwitched = true;
                        ans++;
                    }
                    continue;
                }
                ans++;
            }
            return ans;
        }


        public long candies(int n, List<Integer> arr) {
            int descending_seq = 0;
            long sum = 0;
            int prev_c = 0;
            int prev_num_of_candies = 0;
            for (int c : arr) {
                if (c >= prev_c) {
                    if (descending_seq > 0) {
                        // agjust local max value if descending sequence
                        // was longer than ascending
                        if (descending_seq >= prev_num_of_candies) {
                            sum += 1 + descending_seq - prev_num_of_candies;
                        }
                        // last of descending = local minimum
                        prev_num_of_candies = 1;
                        descending_seq = 0;
                    }
                    if (c > prev_c) {
                        ++prev_num_of_candies;
                    } else {
                        // optimal if previous value is the same
                        prev_num_of_candies = 1;
                    }
                    sum += prev_num_of_candies;
                } else {
                    ++descending_seq;
                    // For 3 descending numbers in a row this summing strategy
                    // will increment like sum+=1+2+3 which is the same as
                    // more usual and expected sum+=3+2+1
                    sum += descending_seq;
                }
                prev_c = c;
            }
            // If we finished on descending order, update last local max
            if (descending_seq >= prev_num_of_candies) {
                sum += 1 + descending_seq - prev_num_of_candies;
            }
            return sum;
        }
    }


    public static int vowelsubstring(String s) {
        int n = s.length();
        if (n < 5) return 0;

        int ans = 0;
        String[] dp = new String[n * (n + 1) / 2]; // to store lengths of all valid substrings

        for (int i = 0; i < n; i++)
            for (int j = i + 1; j <= n; j++)
                dp[i] = (s.substring(i, j));


        for (int m = 0; m < dp.length; m++) {

            String str = dp[m];
            // Hash Array of size 5
            // such that the index 0, 1, 2, 3 and 4
            // represent the vowels a, e, i, o and u
            int[] hash = new int[5];

            // Loop the String to mark the vowels
            // which are present
            for (int i = 0; i < str.length(); i++) {

                if (str.charAt(i) == 'A' || str.charAt(i) == 'a')
                    hash[0] = 1;

                else if (str.charAt(i) == 'E' || str.charAt(i) == 'e')
                    hash[1] = 1;

                else if (str.charAt(i) == 'I' || str.charAt(i) == 'i')
                    hash[2] = 1;

                else if (str.charAt(i) == 'O' || str.charAt(i) == 'o')
                    hash[3] = 1;

                else if (str.charAt(i) == 'U' || str.charAt(i) == 'u')
                    hash[4] = 1;
            }

            // Loop to check if there is any vowel
            // which is not present in the String
            for (int i = 0; i < 5; i++) {
                if (hash[i] == 0) {
                    ans++;
                }
            }
        }
        return ans;
    }


    public static int selectStock(int saving, List<Integer> currentValue, List<Integer> futureValue) {
        int n = currentValue.size();
        int maxProfit = 0;
        for (int k = 0; k < n; k++) {
            int prev = 0, prepre = 0;

            for (int i = k; i < n; i++) {
                if (currentValue.get(i) < saving) {
                    int profit = futureValue.get(i) - currentValue.get(i);
                    int curr = Math.max(prev, prepre + profit);
                    prepre = prev;
                    prev = curr;
                    saving -= currentValue.get(i);
                }
            }
            if (prev > maxProfit) maxProfit = prev;
        }
        return maxProfit;
    }


    public static String reachTheEnd(List<String> grid, int maxTime) {
        // Write your code here

        int n = grid.size();
        if (n == 0) return "No";

        // matrix is ready
        int[][] matrix = new int[n][n];
        for (int k = 0; k < grid.size(); k++) {
            for (int i = 0; i < grid.get(k).length(); i++) {
                matrix[i][k] = grid.get(k).charAt(i);
            }
        }

        // get min time for btoph cases --> to the right and bottom
        int minTime = dfs(matrix, n, n, 0, 0, new int[n * n], 0, Integer.MAX_VALUE);

        return minTime > maxTime ? "No" : "Yes";
    }


    private static int dfs(int mat[][], int m, int n,
                           int i, int j, int path[], int idx, int min) {
        path[idx] = mat[i][j];

        // Reached the bottom of the matrix so we are left with
        // only option to move right
        if (i == m - 1) {
            for (int k = j + 1; k < n; k++) {
                if (mat[i][k] != '#') path[idx + k - j] = mat[i][k];
                else return 0;
            }
            int pathArrSize = Math.toIntExact(Arrays.stream(path).filter(x -> x != -1).count());

            if (min > pathArrSize) min = pathArrSize;
            return min;
        }

        // Reached the right corner of the matrix we are left with
        // only the downward movement.
        if (j == n - 1) {
            for (int k = i + 1; k < m; k++) {
                if (mat[i][k] != '#') path[idx + k - i] = mat[k][j];
                else return 0;
            }
            int pathArrSize = Math.toIntExact(Arrays.stream(path).filter(x -> x != -1).count());
            if (min > pathArrSize) min = pathArrSize;
            return min;
        }
        // Find all the paths that are possible after moving down
        dfs(mat, m, n, i + 1, j, path, idx + 1, min);

        // Find all the paths that are possible after moving right
        dfs(mat, m, n, i, j + 1, path, idx + 1, min);
        return min;
    }

    public static int dfs(int[][] matrix, int row, int col) {

//        int n = matrix.length;
//        // dp array to store minimum via paths
//        int[] dp = new int[n  *  n];
//        Arrays.fill(dp, -1);
//
//        dp[0] = 0;
//
//        if (col == 1) {
//          // matrix traversal
//            for (int r = 0; r < n; r++) {
//                for (int c = 1; c < n; c++) {
//                    // check for boundary conditions
//                     if (c < n-1  && r < n-1 ){
//                         // if all directions are blocked
//                         if (matrix[r+1][c] == '#' && matrix[r][c+1] == '#' && matrix[r-1][c] == '#' && matrix[r][c-1] == '#')
//                             break;
//                     }
//                    dp[r][c] = Math.min()
//                }
//            }
//
//        }
//
        return 0;
    }


    int[] nums;

    public int[] minDifference(int[] nums, int[][] queries) {
        this.nums = nums;
        int[] ans = new int[queries.length];
        if (nums.length == 0) return ans;
        if (queries.length == 0) return ans;

        int k = 0;
        for (int[] q : queries) {
            ans[k] = processQuery(q);
            k++;
        }
        return ans;
    }

    protected int processQuery(int[] query) {
        int min = Integer.MAX_VALUE;
        int start = query[0];
        int end = query[1];
        int[] array = new int[end - start + 1];

        int index = 0;
        for (int i = start; i <= end; i++) {
            array[index] = this.nums[i];
            index++;
        }
        Arrays.sort(array);
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i] + " ");
        }
        System.out.println();
        int k = 0;
        for (int i = 0; i < array.length - 1; i++) {
            if (Math.abs(array[k + 1] - array[k]) == 0) min = -1;
            else if ((min == -1 || min > Math.abs(array[k + 1] - array[k]) && Math.abs(array[k + 1] - array[k]) != 0)) {
                min = Math.abs(array[k + 1] - array[k]);
            }
            k++;
        }

        System.out.println(min);
        return min;
    }

    public int countSubIslands(int[][] grid1, int[][] grid2) {

        int[] dp1 = new int[grid1.length];
        int[] dp2 = new int[grid2.length];


        return 0;

    }

    public String largestOddNumber(String num) {
        String largest = "";
        if (num.length() == 0) return "";

        for (int i = 0; i < num.length(); i++) {
            if (Integer.parseInt(String.valueOf(num.charAt(i))) % 2 != 0)
                largest = (num.substring(0, i + 1));
        }

        return largest;
    }

    public int numberOfRounds(String startTime, String finishTime) {
        int before = 0, after = 0;
        if (Integer.parseInt(startTime.split(":")[1]) % 15 != 0) { // Time can't be taken

            if (Integer.parseInt(startTime.split(":")[1]) <= 45) {
                startTime = startTime.split(":")[0] + ":" + String.valueOf((Integer.parseInt(startTime.split(":")[1]) / 15 + 1) * 15);
            } else {
                startTime = Integer.parseInt(startTime.split(":")[0] + 1) + ":00";
                System.out.println(startTime);
            }
        }
        if (Integer.parseInt(startTime.split(":")[0]) > Integer.parseInt(finishTime.split(":")[0])) {
            before = (24 - Integer.parseInt(startTime.split(":")[0])) * 60 + Integer.parseInt(startTime.split(":")[1]);
            after = Integer.parseInt(finishTime.split(":")[0]) * 60 + Integer.parseInt(finishTime.split(":")[1]);
        } else if (Integer.parseInt(startTime.split(":")[0]) == 0 && Integer.parseInt(finishTime.split(":")[0]) == 0) {
            after = 24 * 60 - (Integer.parseInt(finishTime.split(":")[1]) - Integer.parseInt(startTime.split(":")[1]));

        } else {
            after = (Integer.parseInt(finishTime.split(":")[0]) - Integer.parseInt(startTime.split(":")[0])) * 60 +
                    (Integer.parseInt(finishTime.split(":")[1]) - Integer.parseInt(startTime.split(":")[1]));
        }

        return (before + after) / 15;


    }

    PriorityQueue<Integer> pq = new PriorityQueue<Integer>();

    public int kthSmallest(TreeNode root, int k) {
        dfs(root);
        int prev = pq.peek() != null ? pq.peek() : 0;

        while (pq.peek() != null && k > 0) {
            prev = pq.poll();
            if (pq.peek() != null && pq.peek() == prev) continue;
            else k--;
        }

        return prev;
    }

    public void dfs(TreeNode root) {

        if (root == null)
            return;

        /* first print data of root */
        System.out.print(root.val + " ");

        pq.add(root.val);
        /* then recur on left sutree */
        dfs(root.left);


        pq.add(root.val);
        /* now recur on right subtree */
        dfs(root.right);

    }


    // left root  right
    List<Integer> ans = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {

        inorder(root);
        return this.ans;
    }


    public void inorder(TreeNode root) {

        if (root == null)
            return;

        /* then recur on left subtree */
        dfs(root.left);

        ans.add(root.val);
        /* first print data of root */
        System.out.print(root.val + " ");

        /* now recur on right subtree */
        dfs(root.right);

    }

//    List<List<Integer>> ans = new ArrayList<>();

    /*
    public List<List<Integer>> levelOrder(Node root) {

        if (root == null)
            return new ArrayList<>();

        List<Integer> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<Node> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();

            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                Node p = q.peek();
                q.remove();
                result.add(p.val);
                // Enqueue all children of
                // the dequeued item
                for (int i = 0; i < p.children.size(); i++)
                    q.add(p.children.get(i));
                n--;
            }
            ans.add(result);
            result = new ArrayList<>();
        }
        Collections.reverse(ans);
        return ans;

        return ans;
    }

     */

    List<Double> answer = new ArrayList<>();

    public List<Double> averageOfLevels(TreeNode root) {

        if (root == null)
            return new ArrayList<>();

        List<Integer> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<TreeNode> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();

            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                TreeNode p = q.peek();
                q.remove();
                result.add(p.val);
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
                n--;
            }

            double value = result.stream().mapToDouble(a -> a).sum() / result.size();
            answer.add(value);
            result = new ArrayList<>();
        }
        return answer;
    }

    public TreeNode convertBST(TreeNode root) {
        convertBSTRec(root, 0);
        return root;
    }

    private int convertBSTRec(TreeNode root, int parentVal) {
        if (root == null) return 0;
        int rightVal = convertBSTRec(root.right, parentVal);
        root.val = rightVal + parentVal + root.val;
        int leftVal = convertBSTRec(root.left, parentVal);
        return root.val - parentVal + leftVal;
    }


}

class GFG {

    // Pair class
    static class Pair {

        int first;
        int second;

        Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }

    static void SelectActivities(int s[], int f[]) {

        // Vector to store results.
        ArrayList<Pair> ans = new ArrayList<>();

        // Minimum Priority Queue to sort activities in
        // ascending order of finishing time (f[i]).
        PriorityQueue<Pair> p = new PriorityQueue<>(Comparator.comparingInt(p2 -> p2.first));

        for (int i = 0; i < s.length; i++) {
            // Pushing elements in priority queue where the
            // key is f[i]
            p.add(new Pair(f[i], s[i]));
        }

        Pair it = p.poll();
        int start = Objects.requireNonNull(it).second;
        int end = it.first;
        ans.add(new Pair(start, end));

        while (!p.isEmpty()) {
            Pair itr = p.poll();
            if (itr.second >= end) {
                start = itr.second;
                end = itr.first;
                ans.add(new Pair(start, end));
            }
        }
        System.out.println("Following Activities should be selected. \n");

        for (Pair itr : ans) {
            System.out.println("Activity started at: " + itr.first + " and ends at  " + itr.second);
        }
    }

    // Driver Code
    public static void main(String[] args) {

        int s[] = {1, 3, 0, 5, 8, 5};
        int f[] = {2, 4, 6, 7, 9, 9};

        // Function call
        SelectActivities(s, f);
    }

    /*
    Input: nums = [2,3,0,1,4]
Output: 2

   Greedy approach to solve problem
     TC = O(n), SC = O(1)
     njmi= 2
     cj = 1
     cjmi=2
     nj=2
     njmx=4
     */

    public int jump(int[] nums) {

        int n = nums.length;
        if (n == 0) return 0;

        int currjumps = 0, currjumpmaxindex = 0, nextjumps = 1, nextjumpmaxindex = 0;// variables to store jumps indexes

        for (int i = 0; i < n; i++) {
            if (i > currjumpmaxindex) { // case when you have consumed all indexes of currjumps
                currjumps = nextjumps;
                currjumpmaxindex = nextjumpmaxindex;
                nextjumps = currjumps + 1;
                nextjumpmaxindex = 0;
            }
            nextjumpmaxindex = Math.max(nums[i] + i, nextjumpmaxindex);
        }

        return currjumps;
    }

    public void format() throws IOException {
        //Scanner
        Scanner s = new Scanner(System.in);
        String variables = s.nextLine();                 // Reading input from STDIN
        int n = variables.split(",").length;
        HashMap<String, String> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map.put(variables.split(",")[i], "");
        }

        for (int i = 0; i < n; i++) {
            String expression = s.nextLine();
            String[] kv = expression.split("=");
            String k = kv[0].trim();
            String v = kv[1].trim();
            map.put(map.getOrDefault(k, v), "");
        }
        // Traverse map and put ans
        StringBuilder ans = new StringBuilder();
        int index = 0;
        String lastValue = "";

        for (Map.Entry<String, String> key : map.entrySet()) {
            String expression = map.get(key);
            if (index == 0) {
                ans.append("1").append(key);
                lastValue = expression;
                index++;
            } else {
                // case 2 handled
                if (lastValue.replaceAll("[0-9]|\\s", "").equalsIgnoreCase(expression.replaceAll("[0-9]|\\s", ""))) {
                    int prev = Integer.parseInt(lastValue.replaceAll("[a-z|A-Z|\\s]", ""));
                    int curr = Integer.parseInt(expression.replaceAll("[a-z|A-Z|\\s]", ""));
                    if (key.equals(lastValue.replaceAll("[0-9]|\\s", ""))) ans.append(" = ").append(prev).append(key);
                    else ans.append(" = ").append(prev / curr).append(key);
                }
            }
        }
        System.out.println(ans);
    }


    /*
    input = [(1,4), (2,3)]
return 3

input = [(4,6), (1,2)]

{[1,2],[4,6]}

return 3

[[1,4],[4,6],[6,8],[10,15]]
3+2+2+5=12
{{1,3}, {2,4}, {5,7}, {6,8}}.

s = 1 ,e=4

r[0][0]=1
r[0][1]=4
[1,4][5,8]
     */
    public static int mergeSegments(int[][] segments) {
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int result = 0;
        int last = 0;
        for (int[] seg : segments) {
            result += Math.max(seg[1] - Math.max(last, seg[0]), 0);
            last = Math.max(last, seg[1]);
        }
        return result;
    }


    public static int[][] mergeOverlapingSegments(int[][] segments) {
        // O(n), O(n*n)
        /*
        int n = segments.length;
        if (n == 0) return new int[n][n];
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int[][] result = new int[n][n];
        int index = 0;

        int start = Integer.MAX_VALUE;
        int end = Integer.MAX_VALUE;
        for (int[] seg : segments) {
            if (seg[0] < end) { // the value of start will not change
                end = seg[1];
                if (start == Integer.MAX_VALUE) start = seg[0];
            } else {
                result[index][0] = start;
                result[index][1] = end;
                index++;
                start = seg[0];
                end = seg[1];
            }
        }
        result[index][0] = start;
        result[index][1] = end;
        return result;

         */
        // Using stack
        int n = segments.length;
        if (n == 0) return new int[n][n];
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int[][] result = new int[n][n];


        return result;
    }


    class pair {
        int l, r, index;

        pair(int l, int r, int index) {
            this.l = l;
            this.r = r;
            this.index = index;
        }
    }

    void findOverlapSegement(int N, int[] a, int[] b) {

        ArrayList<pair> tuple = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            int x, y;
            x = a[i];
            y = b[i];
            tuple.add(new pair(x, y, i));
        }

        // sorted the tuple base on l values -> (leftmost value of tuple)
        Collections.sort(tuple, (aa, bb) -> (aa.l != bb.l) ? aa.l - bb.l : aa.r - bb.r);
        // store r-value 0f current
        int curr = tuple.get(0).r;
        // store index of current
        int curpos = tuple.get(0).index;

        for (int i = 1; i < N; i++) {
            pair currpair = new pair(tuple.get(i).l, tuple.get(i).r, tuple.get(i).index);

            // get L-value of prev
            int L = tuple.get(i - 1).l;
            int R = currpair.r;
            if (L == R) {
                if (tuple.get(i - 1).index < currpair.index)
                    System.out.println(tuple.get(i).index + " " + currpair.index);
                else System.out.println(currpair.index + " " + tuple.get(i).index);
                return;
            }

            if (currpair.r < curr) {
                System.out.print(tuple.get(i).index + " " + curpos);
                return;
            }
            // update the pos and the index
            else {
                curpos = currpair.index;
                curr = currpair.r;
            }

            // If such intervals found
            System.out.print("-1 -1");

        }

    }
}

class BinarySystem {

    public static void main(String[] args) {
        String s1 = "110";// "0000100";// "110";// "1011"; // "101";
        String s2 = "1000";// "101";// "1000"; // "100";

        String res = "";
        res = add(s1.toCharArray(), s2.toCharArray(), s1.length(), s2.length(), 0, res);
        System.out.println(s1 + " + " + s2 + " = " + res);

    }

    private static String add(char[] A, char[] B, int i, int j, int carry, String result) {

        if (i < 0 && j < 0) {
            return (carry > 0 ? carry : "") + result;
        }

        int a = (i >= 0 ? A[i] - '0' : 0);
        int b = (j >= 0 ? B[j] - '0' : 0);
        int sum = a + b + carry;
        result = sum % 2 + result;

        return add(A, B, i - 1, j - 1, sum / 2, result);
    }


    public static String add(String s1, String s2) {
        int len = Math.max(s1.length(), s2.length());

        final StringBuilder result = new StringBuilder(len);
        int carryOver = 0;

        for (int s1Iter = s1.length() - 1, s2Iter = s2.length() - 1; s1Iter >= 0 || s2Iter >= 0; s1Iter--, s2Iter--) {
            final int s1Val = (s1Iter >= 0 ? s1.charAt(s1Iter) - '0' : 0), s2Val = s2Iter >= 0 ? s2.charAt(s2Iter) - '0' : 0;
            final int subsum = s1Val + s2Val + carryOver;
            result.append(subsum % 2);
            carryOver = subsum / 2;
            if (carryOver == 1) result.append(carryOver);
        }

        return result.reverse().toString();
    }

}


class SubArrays {
    public static void main(String[] args) {
        int[] input = {3, 5, 2, 7, 8, 9, 11, 2, 5, 8, 3};
        SubArrays subArrays = new SubArrays();
        System.out.println(subArrays.count(input, 9).stream().collect(Collectors.toSet()));

    }


    // To create tuples in  java we have top create their oops class
    static class pair {
        int index;
        int key;
        String value;

        pair(int index) {
            this.index = index;
            this.key = 0;
            this.value = null;
        }

    }

    static class sort implements Comparable<pair> {
        pair p;

        @Override
        public int compareTo(pair pair) {
            return p.key - pair.key;
        }
    }

    private Map<Integer, pair> map(int[] input, int k) {
        int n = input.length;
        if (n == 0) return new HashMap<>();
        Map<Integer, pair> map = new HashMap<>();
        pair p = new pair(n);
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>(Comparator.comparingInt(a -> a));
        pq.offer(0);
        for (int e : input) {
            map.put(e, map.getOrDefault(e, new pair(e % n)));
            n /= e;
        }
        return map;
    }

    private List<Integer> count(int[] input, int k) {
        int count = 0;
        int n = input.length;
        if (n == 0) return new ArrayList<>();
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (input[i] < k) {
                while (i < n && input[i] < k) {
                    ans.add(input[i]);
                    count++;
                    i++;
                }
            }
        }
        return ans;
    }

    //  Total sum of subarray must be less than k
    private List<Integer> subarrays(int[] input, int k) {
        int n = input.length;
        if (n == 0) return new ArrayList<>();
        int sum = 0;
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (input[i] < k) {
                while (i < n && input[i] < k) {
                    sum += input[i];
                    i++;
                    ans.add(input[i]);
                }
            }
            sum = 0;
        }
        return ans;
    }

// all  possible total sum of subarray must be less than k
//    private

    private List<Integer> allPossibleSubarrays(int[] input, int k) {
        int n = input.length;
        if (n == 0) return new ArrayList<>();
        int sum = 0;
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (input[i] < k) {
                while (input[i] < k) {
                    sum += input[i];
                    i++;
                    ans.add(input[i]);
                }

            }
            sum = 0;
        }
        return ans;
    }


    int calc(int left, int right) {
        int n = right - left;
        return n * (n + 1) / 2;
    }

    int findSubArrays(int[] arr, int k) {
        int ans = 0, sum = 0, ptr = 0;
        while (ptr < arr.length) {
            if (arr[ptr] < k) {
                int ptrForward = ptr;
                while (arr[ptrForward] < k)
                    ptrForward++;
                sum += calc(ptr, ptrForward);
            } else {
                ptr++;
            }
        }
        return ans;
    }
}

class Iterator {

    static class MyIter<T extends Comparable> implements java.util.Iterator {

        public MyIter(List<List<T>> lists) {
        }

        class IterState implements Comparable<IterState> {
            private java.util.Iterator<T> iterator;

            public T getCurrVal() {
                return currVal;
            }

            public boolean hasNext() {
                return iterator.hasNext();
            }

            public void next() {
                if (iterator.hasNext()) currVal = iterator.next();
                else currVal = null;
            }

            private T currVal;

            public IterState(java.util.Iterator<T> i) {
                iterator = i;
                next();
            }


            @Override
            public int compareTo(IterState iterState) {
                return currVal.compareTo(iterState.getCurrVal());
            }

            public void MyIter(List<List<T>> lists) {
                states = new PriorityQueue<IterState>();
                for (List<T> list : lists) {
                    java.util.Iterator<T> listIter = list.iterator();
                    while (listIter.hasNext()) {
                        states.add(new IterState(listIter));
                    }
                }

            }
        }

        private PriorityQueue<IterState> states;

        @Override
        public boolean hasNext() {
            return !states.isEmpty();
        }

        @Override
        public T next() {

            IterState n = states.poll();

            T retval = n.getCurrVal();
            // if more than 1 elements exist then paste them
            if (n.hasNext()) {
                n.next();
                states.add(n);
            }

            return null;
        }
    }

    private static <T extends Comparable> ArrayList toArray(List<List<T>> lists) {

        MyIter<T> i = new MyIter<T>(lists);
        List<T> retVal = new ArrayList<>();
        while (i.hasNext()) {
            retVal.add(i.next());
        }
        return (ArrayList) retVal;
    }

    public static void main(String[] args) {

        System.out.println("Hello world!");
        ArrayList arr = toArray(asList(
                asList(1, 3, 5),
                asList(2, 4, 6),
                asList(7, 9),
                asList(8),
                asList(0, 10),
                asList()));

        for (Object i : arr) {
            System.out.println(i);
        }

        arr = toArray(asList());
        for (Object i : arr) {
            System.out.println(i);
        }

    }

    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int index = 0;
        for (int num : nums) {
            if (map.containsKey(target - num)) {
                index++;
                return new int[]{map.get(target - num), index};
            } else {
                index++;
                return new int[]{map.get(num), index};
            }
        }
        return new int[]{};
    }


    int calSum(int start, int end) {
        int n = end - start;
        return n * (n + 1) / 2;
    }

    public List<List<Integer>> findSubarrays(int[] nums, int k) {
        int n = nums.length;
        if (n == 0) return new ArrayList<>();

        int ptr = 0, sum = 0;
        List<List<Integer>> subarrays = new ArrayList<>();

        while (ptr < n) {
            List<Integer> ans = new ArrayList<>();

            if (nums[ptr] < k) {
                int ptrForward = ptr;

                while (ptrForward < n && nums[ptrForward] < k) {
                    ans.add(nums[ptrForward]);
                    ptrForward++;

                }
                subarrays.add(ans);
                ans = new ArrayList<>();
                sum += calSum(ptr, ptrForward);
            }
            ptr++;
        }
        return subarrays;
    }



    /*
         [2,3,1,5,4] and k = 3
         void reverse(int [] arr, int k) -->
         method to sort the array by incorporating reverse method inside
     */

    // O(k/2) ~ O(k)

    // For reverse of an array traverse till n/2 times and follow process
//    public void reverse(int[] arr, int k) {
//        int n = arr.length;
//        if (n == 0) return;
//
//        int totalOper = (int) Math.floor(k / 2);
//        for (int i = 0; i < totalOper; i++) {
//            int newIndex = k - 1 - i;
//            if (arr[i] < arr[newIndex]) continue;
//                // swap
//            else {
//                int temp = arr[newIndex];
//                arr[newIndex] = arr[i];
//                arr[i] = temp;
//            }
//        }
//    }

    public Object[] reverse(Object[] arr, int k) {
        Collections.reverse(Collections.singletonList(arr).subList(0, k));
        return arr;
    }

    // sorting of function using reverse function embedded
    // O(n*n)
    public void sort(int[] arr, int k) {
        int n = arr.length;
        if (n == 0) return;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (arr[j] > arr[j + 1]) {
                    Object[] revert = reverse(Collections.singletonList(arr).subList(j, j + 2).toArray(), 2);
                    List<Integer> pre = new ArrayList<>(Arrays.stream(arr).boxed().collect(Collectors.toList()).subList(0, i));
                    // Object array to int array conversion
                    Integer[] post = Arrays.stream(revert)
                            .map(Object::toString)
                            .map(Integer::valueOf)
                            .toArray(Integer[]::new);
                    pre.addAll(Arrays.asList(post));
                    pre.addAll(Arrays.stream(arr).boxed().collect(Collectors.toList()).subList(i, n));
                    arr = pre.stream().mapToInt(x -> x).toArray();
                }
            }
        }
        System.out.println(Arrays.toString(arr));
    }


//    ['a', 'b', 'c'];
    /*
     // Combination (Maths)
Given a length n, count the number of strings of length n that can be made using ‘a’, ‘b’ and ‘c’ with at-most one ‘b’ and two ‘c’s allowed.
     */


    //O(3^n), O(n^3)

    final static int n = 10;
    final static int[][] memo = new int[n][n];

    static int countStr(int n, int bCount, int cCount) {
        // base case
        if (bCount < 0 || cCount < 0) return 0;
        if (bCount == 0 && cCount == 0) return 1;
        int res = memo[bCount][cCount] != 0 ? memo[bCount][cCount] : countStr(n - 1, bCount, cCount);
        res += memo[bCount - 1][cCount] != 0 ? memo[bCount - 1][cCount] : countStr(n - 1, bCount - 1, cCount);
        res += memo[bCount][cCount - 1] != 0 ? memo[bCount][cCount - 1] : countStr(n - 1, bCount, cCount - 1);
        return res;
    }
    // Efficient solution

    static class GFG {
        public static void main(String[] args) {
            int n = 3; // Total number of characters
            int bCount = 1, cCount = 2;
            System.out.println(countStrEff(n, bCount, cCount));
        }
    }

    private static int countStrEff(int n, int bCount, int cCount) {
        int[][][] dp = new int[n + 1][2][3];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    dp[i][j][k] = -1;
                }
            }
        }
        return countStrEffUtil(dp, n, bCount, cCount);
    }

    private static int countStrEffUtil(int[][][] dp, int n, int bCount, int cCount) {
        // base case
        if (bCount < 0 || cCount < 0) return 0;
        if (n == 0) return 1;
        if (bCount == 0 && cCount == 0) return 1;
        if (dp[n][bCount][cCount] != -1) return dp[n][bCount][cCount];

        int res = countStr(n - 1, bCount, cCount);
        res += countStr(n - 1, bCount - 1, cCount);
        res += countStr(n - 1, bCount, cCount - 1);
        return res;
    }

    /*
  Input:  s = "aab", p = "c*a*b"
Output: true
"aaa"
"aaaa"
     */

    public int strStr(String haystack, String needle) {

        int n = haystack.length();
        if (needle.length() == 0) return 0;

        for (int i = 0; i < n; i++) {
            if (String.valueOf(haystack.charAt(i)).equals(String.valueOf(needle.charAt(0)))) {

                boolean isExist = true;
                int index = 0;
                while (i < n && index < needle.length()) {
                    if (String.valueOf(haystack.charAt(i)).equals(String.valueOf(needle.charAt(index)))) {
                        i++;
                        index++;
                    } else {
                        isExist = false;
                        break;
                    }
                }
                if (isExist && index == needle.length() - 1) return i - index;
            }

        }

        return -1;
    }

    public boolean regexMatch(String s, String p) {
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (i == 0 && j == 0) dp[i][j] = true;
                else if (i == 0) dp[i][j] = false;
                else if (j == 0) {
                    if (p.charAt(i - 1) == '*') dp[i][j] = dp[i - 1][j];
                } else {
                    if (p.charAt(i - 1) == '?' || (p.charAt(i - 1) == s.charAt(j - 1)))
                        dp[i][j] = dp[i - 1][j - 1]; // take prev value as here it will always be true
                    else dp[i][j] = p.charAt(i - 1) == '*' || p.charAt(i - 1) == '.';
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

    public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[p.length() + 1][s.length() + 1];
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (i == 0 && j == 0) dp[i][j] = true;
                else if (i == 0) dp[i][j] = false;
                else if (j == 0) {
                    if (p.charAt(i - 1) == '*') dp[i][j] = dp[i - 1][j];
                } else {
                    if (p.charAt(i - 1) == '?' || (p.charAt(i - 1) == s.charAt(j - 1)))
                        dp[i][j] = dp[i - 1][j - 1]; // take prev value as here it will always be true
                    else if (p.charAt(i - 1) == '*') dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                    else dp[i][j] = false;
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }


    public String longestPalindrome(String s) {
        int n = s.length();

        if (n == 0) return "";
        String[] dp = new String[n + 1];

        if (n % 2 != 0) { // string length is odd

            int index = 0;
            String s1 = s.substring(0, n / 2 + 1);
            String s2 = s.substring(n / 2 + 1);

            List<String> post = asList(Arrays.stream(reverse(Collections.singletonList(s2.toCharArray()).toArray(), s2.length()))
                    .map(Object::toString)
                    .toArray(String[]::new));
            s2 = String.join("", post);

            int k = 0;
            while (k < s1.length() && index < s1.length()) {
                String str = "";

                for (int i = 0; i < s1.length(); i++) {
                    if (s1.charAt(i) == s2.charAt(i)) {
                        i++;
                        str += s1.charAt(i);
                    } else {
                        index++;
                    }
                    if (dp[i - 1].length() > str.length()) {
                        dp[i] = dp[i - 1];
                    } else dp[i] = str;

                }
                k++;
            }
        } else {
            for (int i = 0; i < n; i++) {
                String str = "";

                if (s.charAt(i) == s.charAt(n - 1 - i)) {
                    str += s.charAt(i);
                } else break;
                if (dp[i - 1].length() > str.length()) {
                    dp[i] = dp[i - 1];
                } else dp[i] = str;
            }
        }
        return dp[n - 1];
    }

    public String longestPalindromSubstring(String s) {
        int n = s.length();
        if (n == 0) return "";
        int maxLength = 0, start = -1;
        for (int i = 0; i < n; i++) {
            int length = Math.max(getLength(i, i, s), getLength(i, i + 1, s));
            if (length > maxLength) maxLength = length;
            start = i - (length - 1) / 2;
        }
        return s.substring(start, start + maxLength);
    }

    private int getLength(int start, int end, String s) {
        int length = 0;
        while (start > 0 && end < n) {
            if (s.charAt(start) != s.charAt(end)) break;

            length += 2;
            start--;
            end++;
        }

        return length;
    }


    private int getLengthRec(int start, int end, String s) {
        if (end >= s.length()) return 0;
        int length = (start == end) ? -1 : 0;
        while (start >= 0 && end < s.length()) {
            if (s.charAt(start) != s.charAt(end)) {
                StringBuilder sb = new StringBuilder(s);
                length = Math.max(Math.max(getLengthRec(start++, end++, sb.deleteCharAt(start).toString()),
                                getLengthRec(start--, end--, sb.deleteCharAt(end).toString())),
                        (start == end) ? getLengthRec(start--, end++, sb.deleteCharAt(start).toString())
                                : getLengthRec(start++, end--, sb.deleteCharAt(start).deleteCharAt(end - 1).toString())
                );
            } else {
                length += 2;
                start--;
                end++;
            }
        }
        return length;
    }


//    int[][] memoTable = new int[n][n];
//        for(int[] val:memoTable) Arrays.fill(val, -1);

    public int s2(String s, int i, int j, int[][] memoTable) {

        if (i == j) return 1;
        if (i > j) return 0;

        if (memoTable[i][j] != -1) return memoTable[i][j];

        char ch1 = s.charAt(i);
        char ch2 = s.charAt(j);
        int max = 1;
        if (ch1 == ch2) {
            int c = s2(s, i + 1, j - 1, memoTable);
            max = Math.max(max, c + 2);
        } else {
            max = Math.max(s2(s, i + 1, j, memoTable), max);
            max = Math.max(max, s2(s, i, j - 1, memoTable));
        }

        return max;
    }


    boolean[][] memob = new boolean[n][n];

    /*
    Input: s = "abc"
Output: 3

  int max_sum = Integer.MIN_VALUE;

        // Consider all blocks starting with i.
        for (int i = 0; i < n - k + 1; i++) {
            int current_sum = 0;
            for (int j = 0; j < k; j++)
                current_sum = current_sum + arr[i + j];

            // Update result if required.
            max_sum = Math.max(current_sum, max_sum);
        }

        return max_sum;


   aaa
     */
    public int countSubstrings(String s) {
        int n = s.length();
        boolean[][] memob = new boolean[n][n];

        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n - i; k++) {
                if (isPalindrome(s, k, k + i, memob)) count += 1;
            }
        }
        return count;
    }

    public int s3(String s, int n) {

        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;

            for (int j = i + 1; j < n; j++) {
                char ch1 = s.charAt(i);
                char ch2 = s.charAt(j);

                if (ch1 == ch2) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    /*
    Input: s = "aaa"
Output: 6

     */
    public int countPalindromSubstrings(String s) {
        int n = s.length();
        boolean[][] memob = new boolean[n][n];

        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n - i; k++) {
                if (isPalindrome(s, k, k + i, memob)) count += 1;
            }
        }
        return count;
    }

    private boolean isPalindrome(String s, int i, int j, boolean[][] memob) {
        if (i == j || i > j) return true;
        if (memob[i][j]) return memob[i][j];

        char ch1 = s.charAt(i);
        char ch2 = s.charAt(j);

        if (ch1 == ch2) {
            memob[i][j] = isPalindrome(s, i + 1, j - 1, memob);
        } else return false;
        return memob[i][j];
    }

    /*
    public int countPalindromicSubsequences(String s) {
        int n = s.length();

        // create the palindrom subsequence from given string first
        String[][] dp = new String[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = String.valueOf(s.charAt(i));

            for (int j = i + 1; j < n; j++) {
                char ch1 = s.charAt(i);
                char ch2 = s.charAt(j);

                if (ch1 == ch2) {
                    dp[i][j] = dp[i + 1][j - 1] + ch1 + ch2;
                } else {
                    dp[i][j] = dp[i + 1][j].length() > dp[i][j - 1].length() ? dp[i + 1][j] : dp[i][j - 1];
                }
            }
        }

        String maxPalindrom = dp[0][n - 1].replaceAll("null", "");

        int mn = maxPalindrom.length();
        boolean[][] memob = new boolean[mn][mn];

        HashSet<String> set = new HashSet<>();

        for (int i = 0; i < mn; i++) {
            for (int k = 0; k < mn - i; k++) {
                if (isPalindrome(maxPalindrom, k, k + i, memob)) {
                    System.out.println(maxPalindrom.substring(k, k + i));
                    set.add(maxPalindrom.substring(k, k + i));
                }
            }
        }
        return set.size();
    }

    public int maxLength(String s, boolean[][] memob) {

        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;

            for (int j = i + 1; j < n; j++) {
                char ch1 = s.charAt(i);
                char ch2 = s.charAt(j);

                if (ch1 == ch2) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
}

*/
}

class CountPalindromicSubsequencesSolution {
    public int countPalindromicSubsequences(String s) {
        int len = s.length();
        int[][] dp = new int[len][len];

        char[] chs = s.toCharArray();
        for (int i = 0; i < len; i++) {
            dp[i][i] = 1;   // Consider the test case "a", "b" "c"...
        }

        for (int distance = 1; distance < len; distance++) {
            for (int i = 0; i < len - distance; i++) {
                int j = i + distance;
                if (chs[i] == chs[j]) {
                    int low = i + 1;
                    int high = j - 1;

                    /* Variable low and high here are used to get rid of the duplicate*/

                    while (low <= high && chs[low] != chs[j]) {
                        low++;
                    }
                    while (low <= high && chs[high] != chs[j]) {
                        high--;
                    }
                    if (low > high) {
                        // consider the string from i to j is "a...a" "a...a"... where there is no character 'a' inside the leftmost and rightmost 'a'
                       /* eg:  "aba" while i = 0 and j = 2:  dp[1][1] = 1 records the palindrome{"b"},
                         the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"b"},
                         and additional time as {"aba"}. The reason why 2 counted is that we also count {"a", "aa"}.
                         So totally dp[i][j] record the palindrome: {"a", "b", "aa", "aba"}.
                         */

                        dp[i][j] = dp[i + 1][j - 1] * 2 + 2;
                    } else if (low == high) {
                        // consider the string from i to j is "a...a...a" where there is only one character 'a' inside the leftmost and rightmost 'a'
                       /* eg:  "aaa" while i = 0 and j = 2: the dp[i + 1][j - 1] records the palindrome {"a"}.
                         the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"a"},
                         and additional time as {"aaa"}. the reason why 1 counted is that
                         we also count {"aa"} that the first 'a' come from index i and the second come from index j. So totally dp[i][j] records {"a", "aa", "aaa"}
                        */
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 1;
                    } else {
                        // consider the string from i to j is "a...a...a... a" where there are at least two character 'a' close to leftmost and rightmost 'a'
                       /* eg: "aacaa" while i = 0 and j = 4: the dp[i + 1][j - 1] records the palindrome {"a",  "c", "aa", "aca"}.
                          the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"a",  "c", "aa", "aca"},
                          and additional time as {"aaa",  "aca", "aaaa", "aacaa"}.  Now there is duplicate :  {"aca"},
                          which is removed by deduce dp[low + 1][high - 1]. So totally dp[i][j] record {"a",  "c", "aa", "aca", "aaa", "aaaa", "aacaa"}
                          */
                        dp[i][j] = dp[i + 1][j - 1] * 2 - dp[low + 1][high - 1];
                    }
                } else {
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1];  //s.charAt(i) != s.charAt(j)
                }
                dp[i][j] = dp[i][j] < 0 ? dp[i][j] + 1000000007 : dp[i][j] % 1000000007;
            }
        }

        return dp[0][len - 1];
    }


    /*
      // 2-d array chacters
  wedfx

  z x f c f g
     */

    public static void main(String[] args) {
        char[][] arr = {{'q', 'w', 'e', 'r', 't', 'y'}, {'a', 's', 'd', 'f', 'g', 'h'}};
        String word = "werty";

        System.out.println(isWordFound(arr, word));
    }

    public static boolean isWordFound(char[][] s, String word) {
        int rows = s.length;
        int cols = s[0].length;
        System.out.println(rows);
        System.out.println(cols);

        // Traverse horizontally
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                String str = "";
                str += String.valueOf(s[i][j]);
                if (str.contains(word)) return true;
            }
        }

        // Traverse vertically
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                String str = "";
                str += String.valueOf(s[j][i]);
                if (str.contains(word)) return true;
            }
        }
        return false;
    }

    /*
    a > $1000

    the amount exceeds $1000, or;
if it occurs within (and including) 60 minutes of another transaction with the same name in a different city.
     */

    // This class is used to store index and corresponding value element of array
    class Triplet<I extends String, I1 extends Number, I2 extends String, I3 extends Number> implements Comparable<Triplet<String, Number, String, Number>> {
        // field member and comparable implementation

        String name;
        int time;
        String city;
        int amount;


        Triplet(String name, int time, String city, int amount) {
            this.name = name;
            this.time = time;
            this.city = city;
            this.amount = amount;

        }

        @Override
        public int compareTo(Triplet<String, Number, String, Number> stringNumberStringTriplet) {
            return this.time - stringNumberStringTriplet.time;
        }
    }


    public int twoCitySchedCost(int[][] costs) {
        int mincost = 0;
        PriorityQueue<Integer> pq1 = new PriorityQueue<>();
        PriorityQueue<Integer> pq2 = new PriorityQueue<>();

        for (int i = 0; i < costs.length; i++) {
            pq1.add(costs[i][0]);
            pq2.add(costs[i][1]);
        }

        for (int i = 0; i < costs.length / 2; i++) {
            int r1 = pq1.poll();
            pq1.remove(r1);
            int r2 = pq2.poll();
            pq2.remove(r2);

            System.out.println(r1);
            System.out.println(r2);

            mincost += r1 + r2;
        }
        return mincost;
    }


    public List<String> invalidTransactions(String[] transactions) {
        int n = transactions.length;
        if (n == 0) return new ArrayList<>();
        List<String> ans = new ArrayList<>();

        HashMap<String, List<Triplet>> map = new HashMap<>();
        for (String t : transactions) {
            String[] arr = t.split(",");
            String name = arr[0];
            String time = arr[1];
            String amount = arr[2];
            String city = arr[3];

            if (Integer.parseInt(amount) > 1000) {
                ans.add(t);
                continue;
            }
            Triplet<String, Integer, String, Integer> triplet = new Triplet(name, Integer.parseInt(time), city, Integer.parseInt(amount));

            List<Triplet> data = new ArrayList<>();
            data.add(triplet);
            if (map.get(name) != null) {
                List<Triplet> existing = map.get(name);
                existing.add(triplet);
                map.put(name, existing);
            } else map.put(name, data);

        }

        for (String key : map.keySet()) {

            List<Triplet> result = map.get(key);

            int lastTime = Integer.MAX_VALUE;
            String lastCity = "";
            int lastAmount = 0;

            for (Triplet res : result) {
                if (lastTime == Integer.MAX_VALUE) {
                    lastTime = res.time;
                    lastCity = res.city;
                    lastAmount = res.amount;
                }
                if (lastTime - res.time < 60 || !res.city.equalsIgnoreCase(lastCity)) {
                    String invalid = res.name + "," + res.time + "," + res.amount + "," + res.city;
                    ans.add(invalid);

                    String lastInValid = res.name + "," + lastTime + "," + lastAmount + "," + lastCity;
                    ans.add(lastInValid);

                    lastTime = res.time;
                    lastCity = res.city;
                    lastAmount = res.amount;
                }
            }
        }
        return new ArrayList<>(new HashSet<>(ans));
    }


    public int lengthOfLongestSubstring(String s) {

        int n = s.length();

        if (n == 0) return 0;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            HashSet<Character> set = new HashSet<Character>();
            int length = 0;
            for (int j = i; j < n; j++) {
                if (set.contains(s.charAt(j))) break;
                else {
                    length++;
                    set.add(s.charAt(j));
                }
            }
            if (length > max) {
                max = length;
            }
        }
        return max;

    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }

        int num1 = 11;
        StringBuilder sb = new StringBuilder(num1);
    }

    public int hammingWeight(int n) {
        StringBuilder number = new StringBuilder(n);
        System.out.println(number.toString());
        int ans = 0;
        for (int i = 0; i < number.toString().length(); i++) {
            if (number.toString().charAt(i) == '1') ans++;
        }
        return ans;
    }

    List<List<Node>> ans = new ArrayList<>();

    public Node connect(Node root) {

        if (root == null)
            return null;

        List<Node> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<Node> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root

        while (!q.isEmpty()) {
            int n = q.size();
            // Dequeue an item from queue
            Node p = q.peek();
            q.remove();

            if (p != null) {
                // next element in queue represents next
                // node at current Level
                p.next = q.peek();

                result.add(p);
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
            } else if (!q.isEmpty())
                q.add(null);
        }
        return root;
    }

}

/*
Input: arr = [1,2,3,4]
Output: "23:41"

 */

class SolutionMaxTime {

    // Function to return the updated frequency map
    // for the array passed as argument
    static HashMap<Integer, Integer> getFrequencyMap(int arr[]) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (hashMap.containsKey(arr[i])) {
                hashMap.put(arr[i], hashMap.get(arr[i]) + 1);
            } else {
                hashMap.put(arr[i], 1);
            }
        }
        return hashMap;
    }

    // Function that returns true if the passed digit is present
    // in the map after decrementing it's frequency by 1
    static boolean hasDigit(HashMap<Integer, Integer> hashMap, int digit) {

        // If map contains the digit
        if (hashMap.containsKey(digit) && hashMap.get(digit) > 0) {

            // Decrement the frequency of the digit by 1
            hashMap.put(digit, hashMap.get(digit) - 1);

            // True here indicates that the digit was found in the map
            return true;
        }

        // Digit not found
        return false;
    }

    // Function to return the maximum possible time in 24-Hours format
    public String largestTimeFromDigits(int[] arr) {
        boolean iszero = false;
        for (int a : arr) {
            if (a == 0) iszero = true;
        }

        HashMap<Integer, Integer> hashMap = getFrequencyMap(arr);
        int i;
        boolean flag;
        String time = "";

        flag = false;

        // First digit of hours can be from the range [0, 2]
        for (i = 2; i >= 0; i--) {
            if (hasDigit(hashMap, i)) {
                flag = true;
                time += i;
                break;
            }
        }

        // If no valid digit found
        if (!flag) {
            return "";
        }

        flag = false;

        // If first digit of hours was chosen as 2 then
        // the second digit of hours can be
        // from the range [0, 3]
        if (time.charAt(0) == '2') {
            for (i = 3; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }
        }

        // Else it can be from the range [0, 9]
        else {
            for (i = 9; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    if (hasDigit(hashMap, 0)) time = "0" + time;
                    break;
                }
            }
        }
        if (!flag) {
            return "";
        }

        // Hours and minutes separator
        time += ":";

        flag = false;

        // First digit of minutes can be from the range [0, 5]
        for (i = 5; i >= 0; i--) {
            if (hasDigit(hashMap, i)) {
                flag = true;
                time += i;
                break;
            }
        }
        if (!flag) {
            return "";
        }

        flag = false;

        // Second digit of minutes can be from the range [0, 9]
        for (i = 9; i >= 0; i--) {
            if (hasDigit(hashMap, i)) {
                flag = true;
                time += i;
                break;
            }
        }
        if (!flag) {
            return "";
        }

        // Return the maximum possible time
        return time;
    }

    public static String getLargestTime(int[] input) {
        String largestTime = "00:00";
        String str = input[0] + "" + input[1] + "" + input[2] + "" + input[3];
        List<String> times = new ArrayList<>();
        permutation(str, times);
        Collections.sort(times, Collections.reverseOrder());
        for (String t : times) {
            int hours = Integer.parseInt(t) / 100;
            int minutes = Integer.parseInt(t) % 100;
            if (hours < 24 && minutes < 60) {
                if (hours < 10 && minutes < 10) {
                    largestTime = "0" + hours + ":0" + minutes;
                } else if (hours < 10) {
                    largestTime = "0" + hours + ":" + minutes;
                } else if (minutes < 10) {
                    largestTime = hours + ":0" + minutes;
                } else {
                    largestTime = hours + ":" + minutes;
                }
            }
        }
        return largestTime;
    }

    public static void permutation(String str, List<String> list) {
        permutation("", str, list);
    }

    private static void permutation(String prefix, String str, List<String> list) {
        int n = str.length();
        if (n == 0) list.add(prefix);
        else {
            for (int i = 0; i < n; i++)
                permutation(prefix + str.charAt(i), str.substring(0, i) + str.substring(i + 1, n), list);
        }
    }


}


class SolutionStr {
    // Function to return the updated frequency map
    // for the array passed as argument
    static HashMap<Integer, Integer> getFrequencyMap(int arr[]) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (hashMap.containsKey(arr[i])) {
                hashMap.put(arr[i], hashMap.get(arr[i]) + 1);
            } else {
                hashMap.put(arr[i], 1);
            }
        }
        return hashMap;
    }

    // Function that returns true if the passed digit is present
    // in the map after decrementing it's frequency by 1
    static boolean hasDigit(HashMap<Integer, Integer> hashMap, int digit) {

        // If map contains the digit
        if (hashMap.containsKey(digit) && hashMap.get(digit) > 0) {

            // Decrement the frequency of the digit by 1
            hashMap.put(digit, hashMap.get(digit) - 1);

            // True here indicates that the digit was found in the map
            return true;
        }

        // Digit not found
        return false;
    }

    public String largestTimeFromDigits(int[] input) {
        int iszero = 0;
        for (int a : input) {
            if (a == 0) iszero++;
        }

        if (iszero != 1) {

            HashMap<Integer, Integer> hashMap = getFrequencyMap(input);
            int i;
            boolean flag;
            String time = "";

            flag = false;

            // First digit of hours can be from the range [0, 2]
            for (i = 2; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }

            // If no valid digit found
            if (!flag) {
                return "";
            }

            flag = false;

            // If first digit of hours was chosen as 2 then
            // the second digit of hours can be
            // from the range [0, 3]
            if (time.charAt(0) == '2') {
                for (i = 3; i >= 0; i--) {
                    if (hasDigit(hashMap, i)) {
                        flag = true;
                        time += i;
                        break;
                    }
                }
            }

            // Else it can be from the range [0, 9]
            else {
                for (i = 9; i >= 0; i--) {
                    if (hasDigit(hashMap, i)) {
                        flag = true;
                        time += i;
                        if (hasDigit(hashMap, 0)) time = "0" + time;
                        break;
                    }
                }
            }
            if (!flag) {
                return "";
            }

            // Hours and minutes separator
            time += ":";

            flag = false;

            // First digit of minutes can be from the range [0, 5]
            for (i = 5; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }
            if (!flag) {
                return "";
            }

            flag = false;

            // Second digit of minutes can be from the range [0, 9]
            for (i = 9; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }
            if (!flag) {
                return "";
            }

            // Return the maximum possible time
            return time;
        } else {
            String largestTime = "00:00";
            String str = input[0] + "" + input[1] + "" + input[2] + "" + input[3];
            List<String> times = new ArrayList<>();
            permutation(str, times);
            Collections.sort(times, Collections.reverseOrder());
            for (String t : times) {
                int hours = Integer.parseInt(t) / 100;
                int minutes = Integer.parseInt(t) % 100;
                if (hours < 24 && minutes < 60) {
                    if (hours < 10 && minutes < 10) {
                        largestTime = "0" + hours + ":0" + minutes;
                    } else if (hours < 10) {
                        largestTime = "0" + hours + ":" + minutes;
                    } else if (minutes < 10) {
                        largestTime = hours + ":0" + minutes;
                    } else {
                        largestTime = hours + ":" + minutes;
                    }
                }
            }
            return largestTime;
        }

    }

    public static void permutation(String str, List<String> list) {
        permutation("", str, list);
    }

    private static void permutation(String prefix, String str, List<String> list) {
        int n = str.length();
        if (n == 0) list.add(prefix);
        else {
            for (int i = 0; i < n; i++)
                permutation(prefix + str.charAt(i), str.substring(0, i) + str.substring(i + 1, n), list);
        }
    }

    int[] nums;

    int[] prefix = new int[nums.length];

    public void NumArray(int[] nums) {
        this.nums = nums;
        int index = 0;
        for (int n : nums) {
            prefix[index] = n;
            index++;
        }

        return;
    }

    public void update(int index, int val) {
        this.nums[index] = val;
        prefix[index] -= this.nums[index];
        prefix[index] += val;
        return;
    }

    public int sumRange(int left, int right) {
        return prefix[right] - prefix[left];
    }
    /**
     * Your NumArray object will be instantiated and called as such:
     * NumArray obj = new NumArray(nums);
     * obj.update(index,val);
     * int param_2 = obj.sumRange(left,right);
     */
}

class SolutionTrie {

    // Alphabet size (# of symbols)
    static final int ALPHABET_SIZE = 26;

    static class TrieNode {
        TrieNode[] children = new TrieNode[ALPHABET_SIZE];

        // isEndOfWord is true if the node represents
        // end of a word
        boolean isEndOfWord;

        TrieNode() {
            isEndOfWord = false;
            for (int i = 0; i < ALPHABET_SIZE; i++)
                children[i] = null;
        }
    }

    static TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public SolutionTrie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String key) {
        int level;
        int length = key.length();
        int index;

        TrieNode pCrawl = root;

        for (level = 0; level < length; level++) {
            index = key.charAt(level) - 'a';
            if (pCrawl.children[index] == null)
                pCrawl.children[index] = new TrieNode();

            pCrawl = pCrawl.children[index];
        }

        // mark last node as leaf
        pCrawl.isEndOfWord = true;
    }


    /**
     * Returns if the word is in the trie.
     */
    public String search(String key) {
        int level;
        int length = key.length();
        int index;
        StringBuilder ans = new StringBuilder();
        TrieNode pCrawl = root;

        char last = '0';
        for (level = 0; level < length; level++) {

            if (last != key.charAt(level)) {
                index = key.charAt(level) - 'a';
                if (pCrawl.children[index] == null) {
                    if (pCrawl.isEndOfWord) return ans.toString();
                    else return "";
                }


                ans.append(key.charAt(level));
                pCrawl = pCrawl.children[index];
            } else return String.valueOf(last);
        }

        return ans.toString();
    }


    public String replaceWords(List<String> dictionary, String sentence) {
        if (dictionary.size() == 0) return "";
        String[] words = sentence.split(" ");

        for (String dic : dictionary) {
            insert(dic);
        }

        StringBuilder finalAns = new StringBuilder("");
        for (String word : words) {
            String root = search(word);
            if (root.length() > 0) {
                finalAns.append(root).append(" ");
            } else finalAns.append(word).append(" ");
        }

        return finalAns.toString().trim();
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String key) {
        int level;
        int length = key.length();
        int index;
        TrieNode pCrawl = root;

        for (level = 0; level < length; level++) {
            index = key.charAt(level) - 'a';

            if (pCrawl.children[index] == null)
                return false;

            pCrawl = pCrawl.children[index];
        }

        return true;
    }

    /**
     * Your Trie object will be instantiated and called as such:
     * Trie obj = new Trie();
     * obj.insert(word);
     * boolean param_2 = obj.search(word);
     * boolean param_3 = obj.startsWith(prefix);
     */


    static class WordDictionary {
        // Alphabet size (# of symbols)
        static final int ALPHABET_SIZE = 26;
        static TrieNode root;

        static class TrieNode {
            TrieNode[] children = new TrieNode[ALPHABET_SIZE];

            // isEndOfWord is true if the node represents
            // end of a word
            boolean isEndOfWord;

            TrieNode() {
                isEndOfWord = false;
                for (int i = 0; i < ALPHABET_SIZE; i++)
                    children[i] = null;
            }
        }

        /**
         * Initialize your data structure here.
         */
        public WordDictionary() {
            root = new TrieNode();
        }

        public void addWord(String word) {
            int level;
            int length = word.length();
            int index;

            TrieNode pCrawl = root;

            for (level = 0; level < length; level++) {
                if (word.charAt(level) == '.') index = pCrawl.children.length - 1;
                else index = word.charAt(level) - 'a';
                if (pCrawl.children[index] == null)
                    pCrawl.children[index] = new TrieNode();

                pCrawl = pCrawl.children[index];
            }

            // mark last node as leaf
            pCrawl.isEndOfWord = true;
        }

        public boolean searchUtil(String word, int indx, TrieNode ptr) {
            if (indx == word.length()) {
                return ptr.isEndOfWord;
            }
            if (word.charAt(indx) != '.') {
                if (ptr.children[word.charAt(indx) - 'a'] == null) {
                    return false;
                }
                if (searchUtil(word, indx + 1, ptr.children[word.charAt(indx) - 'a'])) {
                    return true;
                }
            } else {
                for (int i = 0; i < ALPHABET_SIZE; i++) {
                    if (ptr.children[i] != null) {
                        if (searchUtil(word, indx + 1, ptr.children[i])) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        public boolean search(String word) {
            int len = word.length();
            TrieNode ptr = root;
            return searchUtil(word, 0, ptr);
        }
    }

    /**
     * Your WordDictionary object will be instantiated and called as such:
     * WordDictionary obj = new WordDictionary();
     * obj.addWord(word);
     * boolean param_2 = obj.search(word);
     */


/*
class WordFilter {

    public WordFilter(String[] words) {

        int n = words.length;
        if (n == 0) return;
        for (String word : words) {




        }
    }

    public int f(String prefix, String suffix) {

    }
}
 */

        /*
        Input: dist = [1,1,2,3], speed = [1,1,1,1]
Output: 1
Input: dist = [3,2,4], speed = [5,3,2]
Output: 1

         */

    class Solution1 {
        private TrieNode root;

        public Solution1() {
            root = new TrieNode();
        }

        public String replaceWords(List<String> dictionary, String sentence) {
            for (String root : dictionary) {
                addRoot(root);
            }
            String[] words = sentence.split(" ");
            String[] result = new String[words.length];
            for (int i = 0; i < words.length; i++) {
                char[] chars = words[i].toCharArray();
                TrieNode node = root;
                StringBuilder rootWordBuilder = new StringBuilder();
                for (char c : chars) {
                    if (!node.containsKey(c) || node.isEnd()) {
                        break;
                    }
                    rootWordBuilder.append(c);
                    node = node.get(c);
                }
                result[i] = rootWordBuilder.length() <= 0 || !node.isEnd() ? words[i] : rootWordBuilder.toString();
            }
            return String.join(" ", result);
        }

        public void addRoot(String rootWord) {
            TrieNode node = root;
            char[] chars = rootWord.toCharArray();
            for (char c : chars) {
                if (!node.containsKey(c)) {
                    node.add(c);
                }
                node = node.get(c);
            }
            node.setEnd();
        }

        class TrieNode {
            private TrieNode[] children;
            private boolean isEnd;

            public TrieNode() {
                children = new TrieNode[26];
            }

            public void add(char c) {
                children[c - 'a'] = new TrieNode();
            }

            public boolean containsKey(char c) {
                return children[c - 'a'] != null;
            }

            public TrieNode get(char c) {
                return children[c - 'a'];
            }

            public boolean isEnd() {
                return this.isEnd;
            }

            public void setEnd() {
                this.isEnd = true;
            }
        }
    }

    public int eliminateMaximum(int[] dist, int[] speed) {
        int ans = 0;
        List<Integer> list = Arrays.stream(dist).boxed().sorted().collect(Collectors.toList());

        for (int i = 0; i < speed.length; i++) {
            int s = speed[i];
            int elem = list.get(0);
            if (Arrays.binarySearch(list.stream().mapToInt(Integer::intValue).toArray(), 0) == -1) {
                if (s >= elem) {
                    list.remove(0);
                    ans++;
                }
                Collections.sort(list);
                list.forEach(x -> x = x - 1);
            } else break;
        }
        return ans;
    }


   /*
    public boolean canReach(String s, int minJump, int maxJump) {
        int n = s.length();
        if (n == 0) return false;

        Set<Integer> landingPos = new HashSet<>(); // to store valid landing positions
        landingPos.add(0);

        while (landingPos.size() > 0) {
            Optional<Integer> pos = landingPos.stream().findFirst();
            for (int j = minJump; j <= Math.min(pos.get() + maxJump, n - 1) && s.charAt(j) == '0'; j++) {
                landingPos.add(j);
            }
            if (landingPos.contains(n - 1)) return true;
        }
        return false;
    }

    */

    public boolean canReach(String s, int minJump, int maxJump) {
       /*
        int n = s.length();
        if (n == 0) return false;

        Set<Integer> landingPos = new HashSet<>(); // to store valid landing positions
        landingPos.add(0);

        while (landingPos.size() > 0) {
            Optional<Integer> pos = landingPos.stream().findFirst();

            for (int j = (pos.get() + minJump); j <= (pos.get() + Math.min(pos.get() + maxJump, n - 1)) && s.charAt(j) == '0'; j++) {
                landingPos.add(j);
            }
            if (landingPos.contains(n - 1)) return true;
            landingPos.remove(pos.get());
        }
        return false;
        */

        int n = s.length();
        int[] near = new int[n + 1];
        Arrays.fill(near, 0);
        int index = 0, walker = 0, curPos = 0;

        for (int i = 1; i < n; i++) {
            // whops getting out of range, start jumping
            // at next nearest possible index
            if (i > curPos + maxJump) {
                curPos = near[++index];
                // can't make anymore jumps
                if (curPos == 0) return false;
            }
            if (s.charAt(i) == '0' && i >= curPos + minJump) {
                near[++walker] = i;
            }
        }
        return near[walker] == n - 1;

    }


}


class WordFilter {

    /*
    static class Pair implements Comparable<WordFilter.Pair> {
        int key;
        String val;

        Pair(int key, String val) {
            this.key = key;
            this.val = val;
        }

        @Override
        public int compareTo(Pair pair) {
            return this.key - pair.key;
        }
    }

    private WordFilter.TrieNode root;

    PriorityQueue<Pair> pq = new PriorityQueue<Pair>(); // max PQ
    int index = 0;

    public void addRoot(String rootWord) {
        root = new WordFilter.TrieNode();
        WordFilter.TrieNode node = root;
        char[] chars = rootWord.toCharArray();
        for (char c : chars) {
            if (!node.containsKey(c)) {
                node.add(c);
            }
            node = node.get(c);
        }
        node.setEnd();
        pq.add(new Pair(index, rootWord));
        index++;
    }


    class TrieNode {
        private WordFilter.TrieNode[] children;
        private boolean isEnd;

        public TrieNode() {
            children = new WordFilter.TrieNode[26];
        }

        public void add(char c) {
            children[c - 'a'] = new WordFilter.TrieNode();
        }

        public boolean containsKey(char c) {
            return children[c - 'a'] != null;
        }

        public WordFilter.TrieNode get(char c) {
            return children[c - 'a'];
        }

        public boolean isEnd() {
            return this.isEnd;
        }

        public void setEnd() {
            this.isEnd = true;
        }
    }

    public WordFilter(String[] words) {
        for (String word : words) {
            addRoot(word);
        }
    }

    public int f(String prefix, String suffix) {

        Optional<Pair> linkedHashSet = pq.stream().filter(x -> x.val.startsWith(prefix)
                && x.val.endsWith(suffix)).max((x, y) -> x.key - y.key);

        if (linkedHashSet.isPresent()) return linkedHashSet.get().key;
        else return 0;
    }
    */
    TrieNode trie;

    public WordFilter(String[] words) {
        trie = new TrieNode();
        for (int weight = 0; weight < words.length; ++weight) {
            String word = words[weight] + "{";
            for (int i = 0; i < word.length(); ++i) {
                TrieNode cur = trie;
                cur.weight = weight;
                for (int j = i; j < 2 * word.length() - 1; ++j) {
                    int k = word.charAt(j % word.length()) - 'a';
                    if (cur.children[k] == null)
                        cur.children[k] = new TrieNode();
                    cur = cur.children[k];
                    cur.weight = weight;
                }
            }
        }
    }

    public int f(String prefix, String suffix) {
        TrieNode cur = trie;
        for (char letter : (suffix + '{' + prefix).toCharArray()) {
            if (cur.children[letter - 'a'] == null) return -1;
            cur = cur.children[letter - 'a'];
        }
        return cur.weight;
    }

    class TrieNode {
        TrieNode[] children;
        int weight;

        public TrieNode() {
            children = new TrieNode[27];
            weight = 0;
        }
    }
}

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter obj = new WordFilter(words);
 * int param_1 = obj.f(prefix,suffix);
 */

/*
[6,4,14,6,8,13,9,7,10,6,12]
2
 */

class MaxJump {
    int n;
    int[] dp;

    public int maxJumps(int[] arr, int d) {
        this.n = arr.length;
        if (this.n < 3) return this.n;
        dp = new int[this.n + 1];

        Arrays.fill(dp, -1);
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < this.n; i++) {
            // find max jump from given index
            if (dp[i] == -1) dp[i] = mxj(arr, dp, i, this.n, d);
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    private int mxj(int[] ar, int[] dp, int i, int n, int d) {

        if (dp[i] != -1) return dp[i];

        // check to the left of current position
        int lmax = 0;
        int l = i - 1;
        while (l >= i - d && l >= 0 && ar[l] < ar[i]) {
            lmax = Math.max(lmax, dp[l]);
            l--;
        }


        // check to the right of current position
        int rmax = 0;
        int r = i + 1;

        while (r <= i + d && r < this.n && ar[r] < ar[i]) {
            if (dp[r] == -1) dp[r] = mxj(ar, dp, r, this.n, d); // For every right index we need to check
            rmax = Math.max(rmax, dp[r]);
            r++;
        }

        return Math.max(lmax, rmax) + 1; // +1 is done to add current position index as well
    }


    /*
    String ans[];

    public int[] prevPermOpt1(int[] arr) {
        int n = arr.length;
        if (n == 0) return new int[]{};
        String str = Arrays.stream(arr).boxed().map(Object::toString).collect(Collectors.joining(","));


        int index = 0;
        this.ans[index] = maxPermute(str, "");
        Arrays.sort(this.ans);
        List<Integer> res = this.ans[1].trim().chars().boxed().collect(Collectors.toList());

        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    private String maxPermute(String s, String answer) {

        if (s.length() == 0) {
            System.out.print(answer + "  ");
            return answer;
        }

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            String left_substr = s.substring(0, i);
            String right_substr = s.substring(i + 1);
            String rest = left_substr + right_substr;
            maxPermute(rest, answer + ch);
        }
        return null;
    }

     */

    public int[] prevPermOpt1(int[] arr) {
        int index = arr.length - 2;

        while (index >= 0 && arr[index] <= arr[index + 1]) {
            index--;
        }

        if (index >= 0) {
            int index2 = index + 1;
            for (int i = index; i < arr.length; i++) {
                if (arr[i] > arr[index2] && arr[i] < arr[index]) {
                    index2 = i;
                }
            }

            int temp = arr[index];
            arr[index] = arr[index2];
            arr[index2] = temp;
        }

        return arr;
    }

    public int[] replaceElements(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            int max = -1;
            for (int j = i + 1; j < arr.length; j++) {
                max = Math.max(max, arr[j]);
            }
            arr[i] = max;
        }
        return arr;
    }

    /*
    [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
    TC=O(n), SC=O(1)
     */
    /*
    public int[][] insert(int[][] intervals, int[] newInterval) {

        if (intervals.length == 0) return new int[][]{new int[]{newInterval[0], newInterval[1]}};

        int nl = newInterval[0];
        int nr = newInterval[1];
        int prev = Integer.MIN_VALUE;

        for (int i = 0; i < intervals.length; i++) {
            int vl = intervals[i][0];
            int vr = intervals[i][1];

            if (prev == vl) {
                intervals[i - 1][1] = vr;
            }

            if (vr < nr) {
                intervals[i][1] = nr;
                if (vl > nl) {
                    intervals[i][0] = nl;
                }
                prev = nr;
            }
        }
        return intervals;
    }
     */


    public int[][] insert(int[][] intervals, int[] newInterval) {
        int i = 0;
        int n = intervals.length;
        List<int[]> ans = new ArrayList<>();


        while (i < n && intervals[i][1] < newInterval[0]) ans.add(intervals[i++]);

        int[] temp = newInterval;
        while (i < n && intervals[i][0] < newInterval[1]) {
            temp[0] = Math.min(intervals[i][0], temp[0]);
            temp[1] = Math.max(intervals[i++][1], temp[1]);

        }
        ans.add(temp);

        while (i < n && intervals[i][0] > newInterval[1]) ans.add(intervals[i++]);
        return ans.toArray(new int[ans.size()][2]);

    }

    // [1,3,7,11,12,14,18]
    // [1,2,3,4,5,6,7,8]


    public int lenLongestFibSubseq(int[] A) {
        int n = A.length;

        // Store all array elements in a hash
        // table
        TreeSet<Integer> S = new TreeSet<>();
        for (int t : A) {
            // Add each element into the set
            S.add(t);
        }
        int maxLen = 0, x, y;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {

                x = A[j];
                y = A[i] + A[j];
                int length = 3;

                // check until next fib element is found
                while (S.contains(y)) {
//                    S.tailSet(4);

                    if (y != S.last()) {
                        // next element of fib subseq
                        int z = x + y;
                        x = y;
                        y = z;
                        maxLen = Math.max(maxLen, length);
                        length++;
                    } else {
                        maxLen = Math.max(maxLen, length);
                        break;
                    }
                }
            }
        }
        return maxLen >= 3 ? maxLen : 0;
    }


    public int lenLongestFibSubseqDP(int[] A) {
        int n = A.length;
        int res = 0;
        // Store all array elements in a hashMap
        HashMap<Integer, Integer> map = new HashMap<>();

        // Initialize dp table
        int[][] dp = new int[n][n];

        // Iterate till N
        for (int j = 0; j < n; ++j) {
            map.put(A[j], j);
            for (int i = 0; i < j; ++i) {
                // Check if the current integer
                // forms a fibonacci sequence
                int k = map.get(A[j] - A[i]) != null ?
                        (map.get(A[j] - A[i]).equals(map.get(A[j])) ? -1 : map.get(A[j] - A[i])) : -1;

                // Update the dp table
                dp[i][j] = (A[j] - A[i] < A[i] && k >= 0) ? dp[k][i] + 1 : 2;
                res = Math.max(res, dp[i][j]);
            }
        }
        // Return the answer
        return res > 2 ? res : 0;
    }


    /*
    Eating right now
    Eating
     */
    public boolean areSentencesSimilar(String sentence1, String sentence2) {
        String[] arr1 = sentence1.split(" "), arr2 = sentence2.split(" ");

        int s1 = 0, s2 = 0, e1 = arr1.length - 1, e2 = arr2.length - 1;

        for (; s1 <= e1 && s2 <= e2 && arr1[s1].equals(arr2[s2]); s1++, s2++) ;
        for (; e1 >= 0 && e2 >= 0 && arr1[e1].equals(arr2[e2]); e1--, e2--) ;
        return s1 > e1 || s2 > e2;
        /*
        // sentence2 is smaller in this case
        if (s1.length < s2.length) {
            String prefix = s1[0];
            String suffix = s1[s1.length - 1];
            if (prefix.equals(s2[0]) && suffix.equals(s2[s2.length - 1])
            ) {
                // check for remaining words of sentence
                for (int i = 1; i < s1.length - 1; i++) {
                    if (!s1[i].equals(s2[i])) return false;
                }
            } else if ((prefix.equals(s2[0]) || suffix.equals(s2[s2.length - 1]))
                    && prefix.equals(suffix) && s1.length == 1) return true;
            else return false;
        } else if (s1.length > s2.length) {
            String prefix = s2[0];
            String suffix = s2[s2.length - 1];
            if ((prefix.equals(s1[0]) && suffix.equals(s1[s1.length - 1]))) {
                // check for remaining words of sentence
                for (int i = 1; i < s2.length - 1; i++) {
                    if (!s2[i].equals(s1[i])) return false;
                }
            } else if ((prefix.equals(s1[0]) || suffix.equals(s1[s1.length - 1]))
                    && prefix.equals(suffix) && s2.length == 1) return true;
            else return false;
        } else {
            for (int i = 0; i < s1.length; i++) {
                if (!s1[i].equals(s2[i])) return false;
            }
        }
        return true;

         */
    }

    /*
    private int isFibonacci(int n) {
        if (n == 1 || n == 0) {
            return true;
        }
        int[] dp = new int[n];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i < n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n] == dp[n - 1] + dp[n - 2];
    }
     */

//-----------------------------------------------------------


    //[0,2] -> [1,2] -> [2,2]
    /*
    {{3,6,2},
    {4,9,7},
    {33,8,11}}
    Print spirally

    o/p - [3,6,27,11,8,33,4,9]


    {{3,6,2,11},
    {4,9,7,32},
    {33,8,11,22},
    {34,9,13,29}}

3, 6, 2, 11, 32, 22, 29, 13, 9 , 34, 33, 4 , 9, 7 ,11 , 8

     */
    public static void spiral(int[][] matrix) {
        int n = matrix.length;
        if (n == 0) return;
        for (int i = 0; i < n; i++) {
            int k = 0;
            // start traversing row wise
            for (int j = 0; j < n; i++) {
                System.out.println(matrix[i][j]);
            }

            // start traversing column wise
            for (int j = 0; j < n; i++) {
                System.out.println(matrix[j][i]);
            }

            // start traversing rowise to the left
            if (i == n - 1 && k == n - 1) {
                for (int m = n - 1; m >= 0; m--) {
                    System.out.println(matrix[i][m]);
                }
            }

            // upwards traversal
            if (i == n - 1 && k == 0) {
                for (int m = n - 1; m >= i; m--) {
                    System.out.println(matrix[k][m]);
                }
            }
            k++;
        }
    }

    // TC = ((n^2))
    // sc = O(1)

    public static void main(String[] args) {

        int[][] matrix = new int[][]{{3, 6, 2, 11},
                {4, 9, 7, 32},
                {33, 8, 11, 22},
                {34, 9, 13, 29}};

        spiral(matrix);
    }


    //[0,2] -> [1,2] -> [2,2]
    /*
    {{3,6,2},
    {4,9,7},
    {33,8,11}}
    Print spirally

    o/p - [3,6,27,11,8,33,4,9]


    {{3,6,2,11},
    {4,9,7,32},
    {33,8,11,22},
    {34,9,13,29}}

3, 6, 2, 11, 32, 22, 29, 13, 9 , 34, 33, 4 , 9, 7 ,11 , 8

     */

    // TC = ((n^2))
    // sc = O(1)

    /*

    public List<Integer> spiralOrder(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        if (rows != cols) return spiralRec(matrix);
        else return spiralSquare(matrix);
    }

    private List<Integer> spiralSquare(int[][] matrix) {

        int n = matrix.length;
        if (n == 0) new ArrayList<>();
        List<Integer> ans = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int j, m, p;
            for (j = i; j < n - i; j++) ans.add(matrix[i][j]);
            j--;

            for (m = i + 1; m < n - i; m++) ans.add(matrix[m][j]);

            m -= 2;

            for (p = m; p >= i; p--) ans.add(matrix[j][p]);

            j--;
            p++;

            for (int u = j; u > i; u--) ans.add(matrix[u][p]);
        }
        return ans;
    }
     */
    public List<Integer> spiralOrder(int[][] arr) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        int i, k = 0, l = 0;
        int r = arr.length;
        int c = arr[0].length;
        while (k < r && l < c) {
            for (i = l; i < c; i++) {
                res.add(arr[k][i]);
            }
            k++;

            for (i = k; i < r; i++) {
                res.add(arr[i][c - 1]);
            }
            c--;

            if (k < r) {
                for (i = c - 1; i >= l; i--) {
                    res.add(arr[r - 1][i]);
                }
                r--;
            }

            if (l < c) {
                for (i = r - 1; i >= k; i--) {
                    res.add(arr[i][l]);
                }
                l++;
            }
        }
        return res;
    }

    /*

    public List<Integer> diffWaysToCompute(String expression) {
        return helper(0, expression.length() - 1, expression);

    }

    private List<Integer> helper(int start, int end, String expression) {

        List<Integer> curr = new ArrayList<>();
        // base case
        if (start > end) return curr;
        boolean operatorPresent = false;
        for (int i = start; i < end; i++) {
            if (!Character.isDigit(expression.charAt(i))) {
                operatorPresent = true;
                List<Integer> left = helper(start, i - 1, expression);
                List<Integer> right = helper(i + 1, end, expression);

                for (int leftVal : Objects.requireNonNull(left)) {
                    for (int rightVal : Objects.requireNonNull(right)) {
                        curr.add(calculate(leftVal, rightVal, expression.charAt(i)));
                    }
                }
            }
        }
        if (!operatorPresent) curr.add(Integer.parseInt(expression.substring(start, end + 1)));
        return curr;
    }

    private int calculate(int leftVal, int rightVal, char operator) {

        if (operator == '+') return leftVal + rightVal;
        else if (operator == '-') return leftVal - rightVal;
        else if (operator == '/') return leftVal / rightVal;
        else return leftVal * rightVal;
    }

     */


    // using dp  + memoisation

    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer>[][] dp = new List[expression.length() + 1][expression.length() + 1];
        return find(new StringBuilder(expression), 0, expression.length(), dp);
    }

    private List<Integer> find(StringBuilder expression, int start, int end, List<Integer>[][] dp) {

        List<Integer> ans = new ArrayList<>();
        if (dp[start][end] != null) return dp[start][end];

        for (int i = start; i < end; i++) {
            char c = expression.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                List<Integer> left = find(expression, start, i, dp);
                List<Integer> right = find(expression, i + 1, end, dp);

                for (int leftVal : left) {
                    for (int rightVal : right) {
                        switch (c) {
                            case '+':
                                ans.add(leftVal + rightVal);
                                break;
                            case '-':
                                ans.add(leftVal - rightVal);
                                break;
                            default:
                                ans.add(leftVal * rightVal);
                                break;
                        }
                    }
                }
            }
        }
        // when no operator is found
        if (ans.isEmpty()) {
            ans.add(Integer.parseInt(expression.substring(start, end + 1)));
        }

        dp[start][end] = ans;
        return ans;

//    int n = expression.length();
//        List<Integer> ans = new ArrayList<>();
//        Map<Character, Integer> map = new HashMap<>();
//
//        for (int i = 0; i < n; i++) {
//            if (expression.charAt(i) == 'a') {
//
//            }
//        }

// 2 pointer technique
// using 2 stacks

    }
}


//      Definition for a binary tree node.
class TreeNode {
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


    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> ans = new ArrayList<>();
        return generateTree(1, n, ans);
    }


    private List<TreeNode> generateTree(int start, int end, List<TreeNode> ans) {

        if (start > end) {
            ans.add(null);
            return ans;
        }
        if (start == end) {
            ans.add(new TreeNode(start));
            return ans;
        }

        List<TreeNode> left, right;

        for (int i = start; i < end; i++) {
            left = generateTree(start, i - 1, ans);
            right = generateTree(i + 1, end, ans);

            for (TreeNode leftVal : left) {
                for (TreeNode rightVal : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftVal;
                    root.right = rightVal;
                    ans.add(root);
                }
            }
        }
        return ans;
    }

    List<TreeNode> recursion(int start, int end, int[][] dp) {
        List<TreeNode> list = new ArrayList<>();
        if (dp[start][end] != -1) {
            list.add(new TreeNode(dp[start][end]));
            return list;
        }
        if (start > end) {
            list.add(null);
            return list;
        }
        if (start == end) {
            dp[start][end] = start;
            list.add(new TreeNode(start));
            return list;
        }
        List<TreeNode> left, right;
        for (int i = start; i <= end; i++) {
            left = recursion(start, i - 1, dp);
            right = recursion(i + 1, end, dp);
            for (TreeNode lst : left) {
                for (TreeNode rst : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = lst;
                    root.right = rst;
                    list.add(root);
                }
            }
        }
        return list;
    }

    public int numTrees(int n) {
        int[][] dp = new int[n + 1][n + 1];
        Arrays.fill(dp, -1);
        return recursion(1, n, dp).size();
    }

    // left --> right --> root
    public void flatten(TreeNode root) {
        TreeNode curr = root;
        while (curr != null) {
            TreeNode left = curr.left;
            if (left != null) {
                TreeNode rightMost = getRightMost(left);
                rightMost.right = curr.right;
                curr.right = left;
                curr.left = null;
            }
            curr = curr.right;
        }
        List<Integer> ans = new ArrayList<>();

        ans.stream().forEach(System.out::println);
    }

    public TreeNode getRightMost(TreeNode node) {
        while (node.right != null) node = node.right;
        return node;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        // base cases
           /*
            int n = temperatures.length;

            int[] ans = new int[n];
            // iterate throught all element
            for (int i = 0; i < n; i++) {
                int count = 0;
                for (int j = i + 1; j < n; j++) {
                    count++;
                    if (temperatures[j] > temperatures[i]) ans[i] = count;
                }
            }

            return ans;
        }
        */


        int n = temperatures.length;
        int[] nextWarmerday = new int[n];
        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack

        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday[idx] = idx - i;
            }
            stk.push(i);
        }
        return nextWarmerday;
    }


            /*
            Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
E
            */

        /*
        public int[] nextGreaterElement(int[] nums1, int[] nums2) {
            int n = nums1.length;
            int[] ans = new int[n];
            Arrays.fill(ans, -1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < nums2.length; j++) {
                    if (nums2[j] == nums1[i]) {
                        while (j < nums2.length && nums2[j] <= nums1[i]) {
                            j++;
                        }
                        if (j != nums2.length) ans[i] = nums2[j];
                        break;
                    }
                }
            }
            return ans;
        }

         */

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        int[] ans = new int[n1];
        Arrays.fill(ans, -1);
        HashMap<Integer, Integer> map = new HashMap<>();
        // store nums2 in map
        for (int num = 0; num < n2; num++) map.put(nums2[num], num);

        for (int i = 0; i < n1; i++) {
            Integer index = map.get(nums1[i]);
            if (map.get(nums1[i]) != -1) {
                int j = index;
                while (j < nums2.length && nums2[j] <= nums1[i]) j++;
                if (j != n2) ans[i] = nums2[j];
            }
        }
        return ans;
    }

    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);

        for (int i = 0; i < n; i++) {
            int counter = 0;
            int j = i + 1;
            while (counter < n && j < n && nums[j] <= nums[i]) {
                counter++;
                if (j == n - 1) {
                    j = 0;
                    continue;
                }
                j++;
            }
            if (counter != n) ans[i] = nums[j];
        }
        return ans;
    }


    public int countTriples(int n) {
        if (n == 0 || n == 1) return 0;

        HashMap<Integer, Integer> hashMap = new HashMap<>();

        // hashing
        for (int i = 1; i <= n; i++) {
            hashMap.put(i * i, i);
        }

        // check for conidtion :- a2 + b2 = c2
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                int elem = i * i - j * j;
                if (hashMap.get(elem) != null) ans++;
            }
        }
        return ans;
    }

    public boolean sumGame(String num) {
        int n = num.length();
        if (n == 0) return false;

        int a = 0;
        int b = 0;
        boolean ans = false;

        int first = 0;
        int second = 0;

        for (int i = 0; i < n; i++) {
            if (num.charAt(i) == '?') {
                // cases:-
                    /*
                        case 1  --> next place is an '?' ==> fill it with  9 (max digit)
                        case 2 --> next place is not an '?' ==> then follow the algo below
                     */
                // if total character is in odd place
                if (i % 2 == 0) {
                    for (int j = 0; j < i; j++) {
                        if (j < i / 2) first += Integer.parseInt(String.valueOf(num.charAt(j)));
                        else second += Integer.parseInt(String.valueOf(num.charAt(j)));
                    }
                    second += 9; // to replace '?' with greatest value by alice
                    return first == second;
                }
            }
        }
        return false;
    }


    static final int MAX = 100;
    static final int MAX_CHAR = 26;

    // Precompute the prefix and suffix array.
    static void precompute(String s, int n, int l[][],
                           int r[][]) {
        l[s.charAt(0) - 'a'][0] = 1;

        // Precompute the prefix 2D array
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < MAX_CHAR; j++)
                l[j][i] += l[j][i - 1];

            l[s.charAt(i) - 'a'][i]++;
        }

        r[s.charAt(n - 1) - 'a'][n - 1] = 1;

        // Precompute the Suffix 2D array.
        for (int i = n - 2; i >= 0; i--) {
            for (int j = 0; j < MAX_CHAR; j++)
                r[j][i] += r[j][i + 1];

            r[s.charAt(i) - 'a'][i]++;
        }
    }

    // Find the number of palindromic subsequence of
// length k
    static int countPalindromes(int k, int n, int l[][],
                                int r[][]) {
        int ans = 0;

        // For k greater than 2. Adding all the products
        // of value of prefix and suffix array.
        for (int i = 1; i < n - 1; i++)
            for (int j = 0; j < MAX_CHAR; j++)
                ans += l[j][i - 1] * r[j][i + 1];

        return ans;
    }

    // Driver code
    public int countPalindromicSubsequence(String s) {
        int k = 3;
        int n = s.length();
        int l[][] = new int[MAX_CHAR][MAX];
        int r[][] = new int[MAX_CHAR][MAX];
        precompute(s, n, l, r);
        return countPalindromes(k, n, l, r);
    }

    public TreeNode canMerge(List<TreeNode> trees) {
        TreeNode node = null;
        for (int i = 0; i < trees.size(); i++) {
            ArrayList<Integer> list1;
            ArrayList<Integer> list2;
            ArrayList<Integer> list3;
            if (i == 0 && trees.size() > 1) {
                //Stores Inorder of tree1 to list1
                list1 = storeInorder(trees.get(i));

                //Stores Inorder of tree2 to list2
                list2 = storeInorder(trees.get(i + 1));

                // Merges both list1 and list2 into list3
                list3 = merge(list1, list2, list1.size(), list2.size());

                //Eventually converts the merged list into resultant BST
                node = ALtoBST(list3, 0, list3.size() - 1);
                i++;
            } else {

                //Stores Inorder of tree1 to list1
                list1 = storeInorder(node);

                //Stores Inorder of tree2 to list2
                list2 = storeInorder(trees.get(i));

                // Merges both list1 and list2 into list3
                list3 = merge(list1, list2, list1.size(), list2.size());

                //Eventually converts the merged list into resultant BST
                node = ALtoBST(list3, 0, list3.size() - 1);
            }
        }
        return node;
    }

    // Method that converts an ArrayList to a BST
    TreeNode ALtoBST(ArrayList<Integer> list, int start, int end) {
        // Base case
        if (start > end)
            return null;

        // Get the middle element and make it root
        int mid = (start + end) / 2;
        TreeNode node = new TreeNode(list.get(mid));

        /* Recursively construct the left subtree and make it
        left child of root */
        node.left = ALtoBST(list, start, mid - 1);

        /* Recursively construct the right subtree and make it
        right child of root */
        node.right = ALtoBST(list, mid + 1, end);

        return node;
    }

    // Method that merges two ArrayLists into one.
    ArrayList<Integer> merge(ArrayList<Integer> list1, ArrayList<Integer> list2, int m, int n) {
        // list3 will contain the merge of list1 and list2
        ArrayList<Integer> list3 = new ArrayList<>();
        int i = 0;
        int j = 0;

        //Traversing through both ArrayLists
        while (i < m && j < n) {
            // Smaller one goes into list3
            if (list1.get(i) < list2.get(j)) {
                list3.add(list1.get(i));
                i++;
            } else {
                list3.add(list2.get(j));
                j++;
            }
        }

        // Adds the remaining elements of list1 into list3
        while (i < m) {
            list3.add(list1.get(i));
            i++;
        }
        // Adds the remaining elements of list2 into list3
        while (j < n) {
            list3.add(list2.get(j));
            j++;
        }
        return list3;
    }

    // Method that stores inorder traversal of a tree
    ArrayList<Integer> storeInorder(TreeNode node) {
        ArrayList<Integer> list1 = new ArrayList<>();
        ArrayList<Integer> list2 = storeInorderUtil(node, list1);
        return list2;
    }

    // A Utility Method that stores inorder traversal of a tree
    public ArrayList<Integer> storeInorderUtil(TreeNode node, ArrayList<Integer> list) {
        if (node == null)
            return list;

        //recur on the left child
        storeInorderUtil(node.left, list);

        // Adds data to the list
        list.add(node.val);

        //recur on the right child
        storeInorderUtil(node.right, list);

        return list;
    }

    public int colorTheGrid(int m, int n) {
        return (int) (Math.pow(Math.pow(m, n), 3) % 1000000007);
    }

    public TreeNode constructMaximumBinaryTree(int[] nums) {
        int n = nums.length;
        if (n == 0) return null;
        TreeNode root = new TreeNode();
        form(root, 0, n - 1, nums);
        return root;
    }

    private void form(TreeNode root, int start, int end, int[] nums) {
        // base cases
        if (start > end) return;

        int rootIndex = maxIndex(start, end, nums);
        root.val = nums[rootIndex];

        if (rootIndex > start) {
            root.left = new TreeNode();
            // call recursilvely for left half
            form(root.left, start, rootIndex - 1, nums);
        }
        if (rootIndex < end) {
            root.right = new TreeNode();

            // call recursilvely for right half
            form(root.right, rootIndex + 1, end, nums);
        }

    }

    private int maxIndex(int start, int end, int[] nums) {
        int index = Integer.MIN_VALUE;
        int max = Integer.MIN_VALUE;

        for (int i = start; i <= end; i++) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        return index;
    }


    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     * int val;
     * TreeNode left;
     * TreeNode right;
     * TreeNode() {}
     * TreeNode(int val) { this.val = val; }
     * TreeNode(int val, TreeNode left, TreeNode right) {
     * this.val = val;
     * this.left = left;
     * this.right = right;
     * }
     * }
     */
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode node = new TreeNode(val);
        // base case
        if (root == null) return node;

        // when val is greater than root node itself
        if (root.val < val) {
            node.left = root;
            node.right = null;
            return node;
        }

        TreeNode current = root;
        while (current.right != null && current.right.val > val) {
            current = current.right;
        }
        // if we have reached at last right node

        if (current.right == null) current.right = node;
            // we have found a node with value greater than val
        else {
            node.left = current.right;
            node.right = null;
            current.right = node;
        }
        return root;
    }

    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {

        /*
        List<TreeNode> ans = new ArrayList<>();
        if (root == null) return ans;
        duplicate(root, ans);
        return ans;
    }

    private void duplicate(TreeNode root, List<TreeNode> ans) {

        if (root == null) return;
        List<TreeNode> left = findDuplicateSubtrees(root.left);
        List<TreeNode> right = findDuplicateSubtrees(root.right);


        // check if both halfs are equal
        int index = 0;
        for (TreeNode node : right) {
            if (node != left.get(index)) break;
            index++;
        }
        if (left.size() == right.size() && (index == left.size() - 1)) ans.addAll(left);
        return;
    }

       */

        List<TreeNode> ans = new ArrayList<>();
        getAllSubTrees(root, new HashMap<>(), ans);
        return ans;

    }

    private String getAllSubTrees(TreeNode root, HashMap<String, Integer> map, List<TreeNode> ans) {

        if (root == null) return " ";
        //inorder recursion call stack
        // check if tree is non empty then add to ans

        String curr = "^" + getAllSubTrees(root.left, map, ans) + root.val + getAllSubTrees(root.right, map, ans);
        int val = map.getOrDefault(curr, 0);
        // if curr value already exists in map  ie. duplicate
        // check if tree is non empty then add to ans
        if (val == 1) ans.add(root);
        map.put(curr, val + 1);
        return curr;

    }

// Write a function to return a prime or not

    // 4 , 7, 11
    public boolean isPrime(int n) {
        for (int i = 3; i < Math.sqrt(n); i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    /*

       number = 7
       counter = 4

     */

    public int nthPrimeNumber(int n) {
        int counter = 0;
        int number = 2;
        while (true) {
            if (isPrime(number)) {
                counter++;
            }
            number++;
            if (counter == n) break;
        }
        return number;
    }


    /*
    // Algo :-
     * Iterate through all array elements
     * For every elem split array to m equal halfs
     * Find largest amomg them
     * Update global min value from them
     * return min


Input: nums = [7,2,5,10,8], m = 2
Output: 18


First half till ith position and remaining half can be found from left elements of array

Dry-run :-


[1,4,4]
3
     */
    public int splitArrayOpt(int[] nums, int m) {
        int low = IntStream.of(nums).max().orElse(0);
        int high = IntStream.of(nums).sum();
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (split(nums, mid) > m) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    private int split(int[] nums, int sum) {
        int ret = 1;
        int currentSum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (currentSum + nums[i] > sum) {
                ret++;
                currentSum = 0;
            }
            currentSum += nums[i];
        }
        return ret;
    }


    /*
    Input: nums = [7,2,5,10,8], m = 2
Output: 18
     */
    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int min = Integer.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            List<List<Integer>> ans = new ArrayList<>();
            // Insert elements from first half
            int indexF = 0;
            while (indexF / m <= i) {
                List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(indexF, indexF + m);
                ans.add(list);
                indexF += m;
            }

            // Insert elements from halfs left till now
            int index = indexF;
            while (index / m < n) {
                List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(index, index + m);
                ans.add(list);
                index += m;
            }
            // for unequal halfs add remaining elements
            ans.add(Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(index, n));

            Optional<Integer> largestSum = ans.stream().map(x -> x.stream().mapToInt(Integer::intValue).sum())
                    .collect(Collectors.toList()).stream().max(Comparator.comparingInt(x -> x));

            if (largestSum.isPresent()) min = Math.min(min, largestSum.get());
        }
        return min;
    }


    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] dp = new int[n];
        int min = prices[0];
        dp[0] = 0;

        // keep track of max profit at every stage
        for (int i = 1; i < n; i++) {
            min = Math.min(min, prices[i]);
            dp[i] = Math.max(dp[i - 1], prices[i] - min);
        }
        return dp[n - 1];
    }


    public ListNode reverseBetween(ListNode head, int left, int right) {

        // base case
        if (head == null) return head;

        ListNode curr = head;
        ListNode prev = null;

        while (left > 1) {
            ListNode next = curr.next;
            // move pointer ahead
            prev = curr;
            curr = next;
            left--;
            right--;
        }

        ListNode connection = prev;
        ListNode tail = curr;

        // reverse the actual list nodes
        while (right > 0) {
            // store the next of current on next node
            ListNode next = curr.next;
            // reverse the linked list
            curr.next = prev;
            // move pointer ahead
            prev = curr;
            curr = next;
            right--;
        }

        if (connection != null) {
            connection.next = prev;
        } else {
            head = prev;
        }
        tail.next = curr;
        return head;
    }
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

/*
 class Solution {

        // using Recursion
        public int[] sortArray(int[] nums) {
            int n = nums.length;
            //base condition
            // hypothesis
            // induction
            if (n == 1) return nums;

            List<Integer> sorted = sort(Arrays.stream(nums).boxed().collect(Collectors.toList()));
            return sorted.stream().mapToInt(x -> x).toArray();
        }

        private List<Integer> sort(List<Integer> nums) {
            int n = nums.size();
            if (n == 0 || n == 1) return nums;

            // Hypothesis
            int temp = nums.get(n - 1);
            nums.remove(new Integer(temp));
            sort(nums);
            return insert(nums, temp);
        }


        private List<Integer> insert(List<Integer> sorted, int num) {
            // base case
            int n = sorted.size();
            if (n == 0 || sorted.get(n - 1) <= num) {
                sorted.add(num);
                return sorted;
            }

            // Hypothesis
            int val = sorted.get(n - 1); // last value is stored in a variable to place num at correct position
            sorted.remove(new Integer(val));
            insert(sorted, num);

            // induction
            sorted.add(val);
            return sorted;
        }
 }

 */

class SolutionRecursion {

    Map<String, List<Integer>> map;

    // using Recursion
    public int[] sortArray(int[] nums) {
        int n = nums.length;
        map = new HashMap<>();
        if (n == 1) return nums;

        List<Integer> sorted = sort(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        return sorted.stream().mapToInt(x -> x).toArray();
    }

    private List<Integer> sort(List<Integer> nums) {
        int n = nums.size();
        if (n == 0 || n == 1) return nums;

        // Hypothesis
        int temp = nums.get(n - 1);
        nums.remove(new Integer(temp));
        sort(nums);
        return insert(nums, temp);
    }

    private List<Integer> insert(List<Integer> sorted, int num) {
        // base case
        int n = sorted.size();
        if (n == 0 || sorted.get(n - 1) <= num) {
            System.out.println(num);
            sorted.add(num);
            return sorted;
        }

        String key = (n - 1) + "*" + sorted.get(n - 1);
        int val = sorted.get(n - 1); // last value is stored in a variable to place num at correct position
        sorted.remove(new Integer(val));
        List<Integer> finalSorted;

        if (!map.containsKey(key)) {
            finalSorted = insert(sorted, num);
        } else finalSorted = map.get(key);

        finalSorted.add(val);
        map.put(key, finalSorted);
        return map.get(key);
    }


    private static List<Integer> countcolderDays(int[] arr) {
        Stack<Integer> stk = new Stack<>();
        // used to store elements greater than the current element
        List<Integer> ans = new ArrayList<>();
        int n = arr.length;
        if (n == 0) return ans;
        for (int i = 0; i < n; i++) {
            if (arr[i] > stk.peek()) {
                // push into stack
                stk.push(arr[i]);
            }

            int count = Math.abs(stk.size() - i); // count of elements lesser than current one
            ans.add(count);
        }

        return ans;
    }
    //         int[] arr = {100,10,89,40,1,80,97};

    private static List<Integer> countcolderDays1(int[] temperatures) {
        int n = temperatures.length;
        List<Integer> nextWarmerday = new ArrayList<>();

        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack
        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday.add(i - idx);
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

    public int maxSum(int[] nums1, int[] nums2) {
        return maxSum(nums1, nums2, 0, 0, 0);
    }

    private int maxSum(int[] nums1, int[] nums2, int num1, int num2, int max) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        if (n1 == 0 && n2 == 0) return 0;
        // base case
        if (num1 == n1 - 1 || num2 == n2 - 1) return 0;

        // hypothesis
        // starting from nums1
        for (int i = 0; i < n1; i++) {
            int index = Arrays.binarySearch(nums2, nums1[i]);
            if (index != -1) {
                int in = maxSum(nums1, nums2, i, index, max);
                int out = maxSum(nums1, nums2, i, num2, max);
                max += Math.max(in, out);
                return max;
            }
        }

        for (int i = 0; i < n2; i++) {
            int index = Arrays.binarySearch(nums1, nums2[i]);
            if (index != -1) {
                int in = maxSum(nums1, nums2, index, i, max);
                int out = maxSum(nums1, nums2, num1, i, max);
                max += Math.max(in, out);
                return max;
            }
        }

        // Induction
        return max;
    }

    public List<TreeNode> allPossibleFBT(int n) {
        List<TreeNode> ans = new ArrayList<>();
        // base cases
        if (n == 1) {
            ans.add(new TreeNode());
            return ans;
        }
        for (int i = 1; i <= n - 2; i++) {
            List<TreeNode> left = allPossibleFBT(i);
            List<TreeNode> right = allPossibleFBT(n - 1 - i);

            for (TreeNode l : left) {
                for (TreeNode r : right) {
                    TreeNode tree = new TreeNode();
                    tree.left = l;
                    tree.right = r;
                    ans.add(tree);
                }
            }
        }
        return ans;
    }


    public boolean areOccurrencesEqual(String s) {
        int n = s.length();
        HashMap<Character, Integer> freq = new HashMap<>();
        for (int i = 0; i < n; i++) {
            freq.put(s.charAt(i), freq.getOrDefault(s.charAt(i), 1) + 1);
        }

        int count = Integer.MIN_VALUE;
        for (Character c : freq.keySet()) {
            if (count == Integer.MIN_VALUE) count = freq.get(c);
            else if (count != freq.get(c)) {
                return false;
            }
        }
        return true;
    }

    static class Pair<I extends Number, I1 extends Number> implements Comparable<Pair<Number, Number>> {
        int key;
        int val;

        Pair(int key, int val) {
            this.key = key;
            this.val = val;
        }

        @Override
        public int compareTo(Pair<Number, Number> pair) {
            return 0;
        }
    }

    public int smallestChair(int[][] times, int targetFriend) {
        int row = times.length;
        // arrival and leave time
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < row; i++) {
            map.put(times[i][0], times[i][1]);
        }

        // TreeMap to store values of HashMap

        // Copy all data from hashMap into TreeMap
        TreeMap<Integer, Integer> sorted = new TreeMap<>(map);

        // Copy all data from hashMap into TreeMap
        TreeMap<Integer, Pair<Integer, Integer>> pairIndex = new TreeMap<>();
        // to store key, pair(index,value)

        Map<Integer, Integer> valueSorted =
                map.entrySet().stream()
                        .sorted(Map.Entry.comparingByValue())
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                                (e1, e2) -> e1, LinkedHashMap::new));


        int[] unoccupied = new int[row];
        Arrays.fill(unoccupied, 0);
        int index = 0;

        // Display the TreeMap which is naturally sorted
        for (Map.Entry<Integer, Integer> entry : sorted.entrySet()) {
            pairIndex.put(entry.getKey(), new Pair(index, entry.getValue()));

            // check if any chair has been unoccupied
            for (Map.Entry<Integer, Integer> ent : valueSorted.entrySet()) {
                if (entry.getKey() > (ent.getValue())) {
                    // get index of chair to be unoccupied
                    unoccupied[pairIndex.get(ent.getKey()).key] = 0;
                    break;
                }
            }

            if (entry.getKey().equals(targetFriend)) {
                // return  first zero chair
                return getChair(unoccupied);
            }
            unoccupied[index] = 1;
            index++;
        }
        return 0;
    }

    private int getChair(int[] unoccupied) {
        int chair = 0;
        for (int c : unoccupied) {
            if (c == 0) return chair;
            chair++;
        }
        return 0;
    }

    public List<List<Long>> splitPainting(int[][] segments) {
        int row = segments.length;
        int col = segments[0].length;

        List<List<Long>> ans = new ArrayList<>();
        // arrival and leave time
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < row; i++) {
            map.put(segments[i][0], segments[i][1]);
        }

        return ans;
    }

    // TLE

    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] ans = new int[n];

        for (int i = 0; i < n; i++) {
            int count = 0;
            int prev_max = Integer.MIN_VALUE;
            for (int j = i + 1; j < n; j++) {
                int ht = Math.min(heights[i], heights[j]);

                if (prev_max == Integer.MIN_VALUE) {
                    int max = Integer.MIN_VALUE;
                    int index = i + 1;
                    while (index < j) {
                        max = Math.max(heights[index], max);
                        index++;
                    }
                    if (max != Integer.MIN_VALUE) {
                        prev_max = max;
                    }
                } else {
                    prev_max = Math.max(heights[j - 1], prev_max);
                }


                if (prev_max != Integer.MIN_VALUE && ht > prev_max) count++;
            }
            ans[i] = count;
        }
        return ans;
    }

    /*
    Input: heights = [10,6,8,5,11,9]
    Output: [3,1,2,1,1,0]

     */

    public int[] canSeePersonsCountOpt(int[] heights) {
        int n = heights.length;
        int[] ans = new int[n];


//        List<Integer> list = Arrays.stream(heights).boxed().collect(Collectors.toList());
//        Collections.reverse(list);

        for (int i = n - 1; i >= 0; i--) {
            int count = 0;
            int max = Integer.MIN_VALUE;

            for (int j = i + 1; j < n; j++) {
                int ht = Math.min(heights[i], heights[j]);

                max = Math.max(heights[j - 1], max);
                if (ht > max) count++;
            }
            ans[i] = count;
        }
        return ans;
    }


    // TC = O(N), SC = O(N)
    // FAB PROBLEM
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Stack<Integer> stk = new Stack<>(); // to store next greater elements in stack
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[i % n] >= nums[stk.peek()]) stk.pop();

            ans[i % n] = stk.isEmpty() ? -1 : nums[stk.peek()];
            stk.push(i % n);
        }
        return ans;
    }

    class Solution {
        PriorityQueue<Integer> res;

        public int nextGreaterElement(int n) {
            String str = Integer.toString(n);
            res = new PriorityQueue<>();
            permutation(str.toCharArray(), 0, 0);

            while (!res.isEmpty() && res.peek() <= n) res.poll();
            return res.peek() != null ? res.poll() : -1;
        }

        private void permutation(char[] str, int l, int r) {
            if (l == r) {
                StringBuilder list = new StringBuilder();
                for (char c : str) list.append(String.valueOf(c));
                res.add(Integer.parseInt(list.toString()));
                return;
            }

            for (int i = l; i <= r; i++) {
                swap(str, l, i); // generate all permute
                permutation(str, l + 1, r);// fix position
                swap(str, l, i); // backtrack
            }
        }

        private void swap(char[] str, int i, int j) {
            char temp = str[i];
            str[i] = str[j];
            str[j] = temp;
        }
    }

    public static void reverse(int start, int end, List<Integer> nums) {
        while (start < end) {
            int temp = nums.get(start);
            nums.set(start, nums.get(end));
            nums.set(end, temp);
            start++;
            end--;
        }
    }

    public int nextGreaterElement(int n) {
        List<Integer> nums = new ArrayList<>();
        while (n != 0) {
            int rem = n % 10;
            nums.add(rem);
            n = n / 10;
        }
        Collections.reverse(nums);
        int k = nums.size() - 2;
        while (k >= 0 && nums.get(k) >= nums.get(k + 1))
            k--;
        if (k == -1) {
            return -1;
        }
        reverse(k + 1, nums.size() - 1, nums);
        for (int i = k + 1; i < nums.size(); i++) {
            if (nums.get(i) > nums.get(k)) {
                int temp = nums.get(i);
                nums.set(i, nums.get(k));
                nums.set(k, temp);
                break;
            }
        }
        long num = 0;
        int i = 0;
        while (i < nums.size()) {
            num = num * 10 + nums.get(i++);
        }
        return (num <= Integer.MAX_VALUE) ? (int) num : -1;
    }

    public int findUnsortedSubarray(int[] nums) {
        Stack<Integer> s = new Stack<>(); // To store index of elements greater than current one in stack
        int n = nums.length;
        int left = n - 1;
        int right = 0;
        for (int i = 0; i <= n - 1; i++) {
            while (!s.isEmpty() && nums[i] < nums[s.peek()]) {
                left = Math.min(s.pop(), left);
            }
            s.add(i);
        }

        s = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!s.isEmpty() && nums[i] > nums[s.peek()]) {
                right = Math.max(s.pop(), right);
            }

            s.add(i);
        }

        return right - left > 0 ? right - left + 1 : 0;
    }

    public int[] smallestRange(List<List<Integer>> nums) {
        int n = nums.size();

        new ArrayList<>(nums);


        Stack<Integer> s = new Stack<>(); // To store index of elements greater than current one in stack
        int left = n - 1;
        int right = 0;
        for (int i = 0; i <= n - 1; i++) {
            while (!s.isEmpty() && nums.get(i).get(i) < nums.get(i).get(s.peek())) {
                left = Math.min(s.pop(), left);
            }
            s.add(i);
        }

        s = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!s.isEmpty() && nums.get(i).get(i) > nums.get(i).get(s.peek())) {
                right = Math.max(s.pop(), right);
            }

            s.add(i);
        }

        return right - left > 0 ? new int[]{right - left + 1} : new int[]{0};
    }
    //count smaller on right using AVL
    // TC = O(nlogn) SC = O(n)

    public class HelloWorld {

        protected class TreeNode {
            int key;
            int height;
            int size;
            TreeNode left;
            TreeNode right;
            TreeNode parent;

            public TreeNode(final int key) {
                this.key = key;
                this.size = 1;
                this.height = 1;
                this.left = null;
                this.right = null;
            }
        }

        public int size(final TreeNode node) {
            return node == null ? 0 : node.size;
        }

        public int height(final TreeNode node) {
            return node == null ? 0 : node.height;
        }

        public TreeNode rotateLeft(final TreeNode root) {
            final TreeNode newRoot = root.right;
            final TreeNode leftSubTree = newRoot.left;

            newRoot.left = root;
            root.right = leftSubTree;

            root.height = max(height(root.left), height(root.right)) + 1;
            newRoot.height = max(height(newRoot.left), height(newRoot.right)) + 1;

            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

            return newRoot;
        }

        public TreeNode rotateRight(final TreeNode root) {
            final TreeNode newRoot = root.left;
            final TreeNode rightSubTree = newRoot.right;

            newRoot.right = root;
            root.left = rightSubTree;

            root.height = max(height(root.left), height(root.right)) + 1;
            newRoot.height = max(height(newRoot.left), height(newRoot.right)) + 1;

            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;
            newRoot.size = size(newRoot.left) + size(newRoot.right) + 1;

            return newRoot;
        }

        public int max(final int a, final int b) {
            return a >= b ? a : b;
        }

        public TreeNode insertIntoAVL(final TreeNode node, final int key, final int count[], final int index) {
            if (node == null) {
                return new TreeNode(key);
            }

            if (node.key > key) {
                node.left = insertIntoAVL(node.left, key, count, index);
            } else {
                node.right = insertIntoAVL(node.right, key, count, index);

                // update smaller elements count
                count[index] = count[index] + size(node.left) + 1;
            }

            // update the size and height
            node.height = max(height(node.left), height(node.right)) + 1;
            node.size = size(node.left) + size(node.right) + 1;

            // balance the tree
            final int balance = height(node.left) - height(node.right);
            // left-left
            if (balance > 1 && node.key > key) {
                return rotateRight(node);
            }
            // right-right
            if (balance < -1 && node.key > key) {
                return rotateLeft(node);
            }
            // left-right
            if (balance > 1 && node.key < key) {
                node.left = rotateLeft(node.left);
                return rotateRight(node);
            }
            // right-left
            if (balance > 1 && node.key < key) {
                node.right = rotateRight(node.right);
                return rotateLeft(node);
            }

            return node;
        }

        public int[] countSmallerOnRight(final int[] in) {
            final int[] smaller = new int[in.length];

            TreeNode root = null;
            for (int i = in.length - 1; i >= 0; i--) {
                root = insertIntoAVL(root, in[i], smaller, i);
            }

            return smaller;
        }


        public void main(String[] args) {
            System.out.println("Hello World");

            int[] res = countSmallerOnRight(new int[]{100, 80, 70, 95, 10, 97});

            for (int r : res) System.out.println(r);
        }
    }


    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     * int val;
     * TreeNode left;
     * TreeNode right;
     * TreeNode() {}
     * TreeNode(int val) { this.val = val; }
     * TreeNode(int val, TreeNode left, TreeNode right) {
     * this.val = val;
     * this.left = left;
     * this.right = right;
     * }
     * }
     */

// Algo :-
/*
  - get the height of tree
  - create an array that store no.of nodes at each level
  - get number of nodes at each and store into array recursilvely
  - return max number of nodes at a level

*/


    // TC = O(n^2), SC = O(n)
    // Fn to get max width of tree
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) return 0;
        int h = height(root);
        int level = 0;
        int[] arr = new int[h];

        getMaxWidthRecursively(root, arr, level);
        // preorder traversal of tree is needed
        return getMax(arr);
    }

    // O(n)
    private int getMax(int[] arr) {
        int max = Integer.MIN_VALUE;
        for (int n : arr) {
            max = Math.max(max, n);
        }
        return max;
    }

    // O(n^2) --> as for every pass we are getting 1 more pass for all nodes on that level
    // Preorder tree traversal to get all nodes on a level
    private void getMaxWidthRecursively(TreeNode root, int[] arr, int level) {

        if (root != null) {
            arr[level]++;
            getMaxWidthRecursively(root.left, arr, level + 1);
            getMaxWidthRecursively(root.right, arr, level + 1);
        }
    }

    // Fn to get height of tree
    private int height(TreeNode root) {
        if (root == null) return 0;
        int lh = height(root.left);
        int rh = height(root.right);
        return 1 + Math.max(lh, rh);
    }

    private int maxWidthBfs(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        root.val = 0;
        q.add(root);

        int ans = 1;
        while (!q.isEmpty()) {
            int size = q.size();
            TreeNode start = q.peek();

            while (size-- > 0) {
                TreeNode temp = q.remove();

                if (temp.left != null) {
                    temp.left.val = temp.left.val * 2 + 1;
                    q.add(temp.left);
                }

                if (temp.right != null) {
                    temp.right.val = temp.right.val * 2 + 2;
                    q.add(temp.right);
                }

                if (size == 0) {
                    ans = Math.max(ans, temp.val - start.val + 1);
                }
            }
        }
        return ans;
    }


    public boolean isSubPath(ListNode head, TreeNode root) {

        if (head == null) return true;
        if (root == null) return false;

        return isPath(head, root) && isSubPath(head, root.left) && isSubPath(head, root.right);
    }

    private boolean isPath(ListNode head, TreeNode root) {
        if (head == null) return true;
        if (root == null) return false;

        return head.val == root.val && (isPath(head.next, root.left) || isPath(head.next, root.right));
    }


    // recursive soln
    public int treeDiameter(int[][] edges) {
        int n = edges.length;
        List<Set<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < n + 1; i++) graph.add(new HashSet<>());

        for (int[] e : edges) {
            int u = e[0], v = e[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }

        int[] distance = bfs(graph, 0); // used to find extremities of nodes in graph

        distance = bfs(graph, distance[0]); // distance b/w them
        return distance[1];
    }

    private int[] bfs(List<Set<Integer>> graph, int start) {

        boolean[] visited = new boolean[graph.size()];

        Arrays.fill(visited, false);
        visited[start] = true;
        LinkedList<Integer> queue = new LinkedList<>();
        queue.addLast(start);

        // bfs algo
        int lastNode = start, distance = -1;
        while (!queue.isEmpty()) {

            int size = queue.size();
            while (size-- > 0) {
                int nextNode = queue.removeFirst();
                for (Integer nbr : graph.get(nextNode)) {
                    if (!visited[nbr]) {
                        visited[nbr] = true;
                        queue.addLast(nbr);
                        lastNode = nbr;
                    }
                }
            }

            // after level is traversed
            distance += 1;
        }

        return new int[]{lastNode, distance};
    }

    public int factorial(int n) {
        // base case
        if (n == 0) return 1;

        int smallerOutput = factorial(n - 1);
        int answer = n * smallerOutput;
        return answer;
    }


    /*
        In = 512
        Ot = 1 + d(512/10) = 1+ d(51) = 1 + 1 + d(5) = 2 + 1 + d(0) = 3 + 0 = 3

        In = 102123
        Ot = 1 + d(10212) = 2 + d(1021) = 3 + d(102) = 4 + d(10) = 4 + d(10) = 5 + d(1) = 6 + d(0) = 6

     */
    public int digits(int n) {
        if (n <= 0) return 0;
        return 1 + digits(n / 10);
    }


    /*
       n = 10
       10 9 8  7 ....1
     */
    public void printNumbersDec(int n) {
        // base case
        if (n == 1) return;
        System.out.println(n);

        printNumbersDec(n - 1);
    }

    /*
       n = 10
       1 2  3 4 5 6 ... 10
     */
    public void printNumbersAsc(int n) {
        // base case
        if (n == 1) return;
        printNumbersDec(n - 1);
        System.out.println(n);
    }

    private void towerOfHanoi(int n, char source, char destination, char helper) {

        // base case
        if (n == 1) {
            System.out.println("Move 1st" + " disk from " + source + " to " + destination);
            return;
        }

        towerOfHanoi(n - 1, source, helper, destination);

        System.out.println("Move " + n + " disk from " + source + " to " + destination);


        towerOfHanoi(n - 1, helper, destination, source);
    }

    private static String getOptions(int n) {
        switch (n) {
            case 1:
                return "";
            case 2:
                return "abc";
            case 3:
                return "def";
            case 4:
                return "ghi";
            case 5:
                return "jkl";
            case 6:
                return "mno";
            case 7:
                return "pqrs";
            case 8:
                return "tuy";
            case 9:
                return "wxyz";
            default:
                return "";
        }
    }

    private static String[] keypad(int n) {
        // base-case
        if (n == 0) {
            String[] output = new String[1];
            output[0] = "";
            return output;
        }

        int lastDigit = n % 10;
        int remainingNumber = n / 10;

        String[] output = keypad(remainingNumber);
        String lastOptions = getOptions(lastDigit);


        String[] result = new String[output.length * lastOptions.length()];
        int index = 0;
        for (int i = 0; i < lastOptions.length(); i++) {
            for (int j = 0; j < output.length; j++) {
                result[index] = lastOptions.charAt(i) + output[j];
                index++;
            }
        }

        return result;
    }

    private static String gender(int n, int k, String rootGender) {
        // base-case
        if (n == 1) return rootGender;

        int c = (int) Math.pow(2, n - 1);
        // check in left subtree
        if (k <= c / 2) {
            return gender(n - 1, k, rootGender);
        } else {

            String child = "m";
            if (rootGender.equalsIgnoreCase("m")) child = "f";

            // check in right subtree
            return gender(n - 1, (k - c / 2), child);
        }
    }


    /*
     1->2<-3->4->5

    Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
reverseKnodes(2, 2, 3) -->
     */

    public ListNode reverseKGroup(ListNode head, int k) {
        // base case
        if (head == null) return null;
        ListNode curr = reverseKnodes(head, k);
        return reverseKGroup(curr, k);
    }

    // Fn. to revert first k nodes of list
    private ListNode reverseKnodes(ListNode head, int k) {
        // base case
        if (head == null) return null;
        if (k == 0) return head;

        // hypothsesis
        head.next.next = head.next;
        return reverseKnodes(head.next, k - 1);
    }

    /*
    Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
     Recursion:-


     */

    public ListNode reverseList(ListNode head) {
        // base  case
        if (head == null || head.next == null) return head;

        // hypothesise
        ListNode p = reverseList(head.next);

        // induction
        head.next.next = head; // actual reverse step done
        head.next = null;
        return p;
    }

    public TreeNode correctBinaryTree(TreeNode root) {

        if (root == null) return root;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean found = false;
        Map<TreeNode, TreeNode> parent = new HashMap<>();

        while (!queue.isEmpty()) {
            if (found) break;
            int size = queue.size();
            Map<TreeNode, TreeNode> map = new HashMap<>();
            for (int i = 0; i < size; i++) {
                TreeNode curr = queue.poll();
                if (map.containsKey(curr)) {
                    map.get(curr).right = null;
                    TreeNode pt = parent.get(map.get(curr));
                    if (pt.left == map.get(curr)) {
                        pt.left = null;
                    } else {
                        pt.right = null;
                    }
                    found = true;
                    break;
                } else {
                    if (curr.left != null) {
                        queue.offer(curr.left);
                        parent.put(curr.left, curr);
                    }
                    if (curr.right != null) {
                        queue.offer(curr.right);
                        map.put(curr.right, curr);
                        parent.put(curr.right, curr);
                    }
                }
            }
        }

        return root;
    }

    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<Integer>[] list = new ArrayList[n];
        for (int i = 0; i < n; i++) list[i] = new ArrayList<>();

        int src = 0;
        for (int i = 0; i < manager.length; i++) {
            if (manager[i] == -1) {
                src = i;
            } else list[manager[i]].add(i);
        }

        return helper(src, list, informTime);
    }

    private int helper(int src, List<Integer>[] list, int[] informTime) {
        int max = 0;
        for (int e : list[src]) {
            max = Math.max(helper(e, list, informTime), max);
        }
        return max + informTime[src];
    }

    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return max;
    }

    private int maxGain(TreeNode root) {
        if (root == null) return 0;

        int left = Math.max(maxGain(root.left), 0);
        int right = Math.max(maxGain(root.right), 0);

        int priceNewPath = root.val + left + right;
        max = Math.max(max, priceNewPath);

        return root.val + Math.max(left, right);
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        return getTargetSum(root, new ArrayList<>(), targetSum);
    }

    private boolean getTargetSum(TreeNode node, ArrayList<TreeNode> nodelist, int targetSum) {
        if (node != null) {
            nodelist.add(node);
        }
        if (node.left != null) {
            getTargetSum(node.left, nodelist, targetSum);
        }

        if (node.right != null) {
            getTargetSum(node.right, nodelist, targetSum);
        } else if (node.left == null) {
            StringBuilder path = new StringBuilder();
            for (TreeNode treeNode : nodelist) {
                path.append(treeNode.val);
            }
            if (targetSum == Integer.parseInt(path.toString())) return true;
        }
        nodelist.remove(node);
        return false;
    }


    List<String> paths = new ArrayList<>();

    public int sumNumbers(TreeNode root) {
        printAllPossiblePath(root, new ArrayList<TreeNode>());
        int sum = 0;
        for (String num : paths) {
            System.out.println(num);
            sum += Integer.parseInt(num);
        }
        return sum;
    }


    private void printAllPossiblePath(TreeNode node, ArrayList<TreeNode> nodelist) {
        if (node != null) {
            nodelist.add(node);
        }

        if (node.left != null) {
            printAllPossiblePath(node.left, nodelist);
        }

        if (node.right != null) {
            printAllPossiblePath(node.right, nodelist);
        } else if (node.left == null) {
            StringBuilder path = new StringBuilder();
            for (TreeNode treeNode : nodelist) {
                path.append(treeNode.val);
            }
            paths.add(path.toString());
        }
        nodelist.remove(node);

    }
}

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

class SolutionHasPath {
    int rows;
    int cols;
    int[][] vis;
    String dir = "URLD"; // lexico order
    int[] dirx = {1, 0, -1, 0};
    int[] diry = {0, 1, 0, -1};


    public static String OrderOfFrequencies(String S) {

        //this is default OUTPUT. You can change it.
        StringBuilder result = new StringBuilder(" ");

        HashMap<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        HashMap<Integer, Character> reversedMap = new HashMap<>();

        map.keySet().forEach(elem -> reversedMap.put(map.get(elem), elem));

        Map<Integer, Character> treeMap = new TreeMap<>(reversedMap);

        for (Integer c : treeMap.keySet()) {
            result.append(treeMap.get(c));
        }

        return result.toString();
    }

    private boolean validateDest(int[][] maze, int[] destination) {
        int s = destination[0];
        int e = destination[1];

        boolean isOpen = false;
        // destination is at valid position
        if (maze[s][e] == 1) return false;

        if (s + 1 < rows && maze[s + 1][e] == 0)
            isOpen = true;

        if (s - 1 >= 0 && maze[s - 1][e] == 0) {
            if (isOpen) return false;
            isOpen = true;
        }
        if (e + 1 < cols && maze[s][e + 1] == 0) {
            if (isOpen) return false;
            isOpen = true;
        }
        if (e - 1 >= 0 && maze[s][e - 1] == 0) {
            if (isOpen) return false;
        }
        return true;
    }

    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        rows = maze.length;
        cols = maze[0].length;
        vis = new int[rows][cols];
        for (int[] elem : vis) Arrays.fill(elem, 0);

//            // validate destination
//            if (!validateDest(maze, destination)) return false;

        return hasPathRecursive(maze, start[0], start[1], destination, vis);
    }

    // Recursive solution of maze
    private boolean hasPathRecursive(int[][] maze, int i, int j, int[] destination, int[][] vis) {

        // boundary conditions
        if (i >= rows || j >= cols || i < 0 || j < 0 || vis[i][j] == 1 || maze[i][j] == 1) return false;

        // base case
        // apply case for destination  has 2 adjacent  empty spaces
        if (i == destination[0] && j == destination[1]) {
            return true;
        }

        // mark visited as true
        vis[i][j] = 1;

        for (int x = 0; x < 4; x++) {
            boolean pathFound = hasPathRecursive(maze, i + dirx[x], j + diry[x], destination, vis);
            if (pathFound) return true;
        }
        // backtrack
        vis[i][j] = 0;
        return false;
    }


    int max = Integer.MIN_VALUE;

    public int maxSumBST(TreeNode root) {
        if (root == null) return 0;
        subTree(root);
        return Math.max(max, 0);
    }

    private SubTree subTree(TreeNode root) {

        if (root == null) return null;

        SubTree currTree = new SubTree(root.val);
        SubTree leftSubTree = subTree(root.left);
        SubTree rightSubTree = subTree(root.right);

        if (leftSubTree != null) {
            currTree.bst &= leftSubTree.bst && root.val > leftSubTree.max;
            currTree.min = leftSubTree.min;
            currTree.sum += leftSubTree.sum;
        }

        if (rightSubTree != null) {
            currTree.bst &= rightSubTree.bst && root.val < rightSubTree.min;
            currTree.max = rightSubTree.max;
            currTree.sum += rightSubTree.sum;
        }

        if (currTree.bst) max = Math.max(max, currTree.sum);
        return currTree;
    }

    class SubTree {
        int max;
        int min;
        int sum;
        boolean bst;

        SubTree(int val) {
            this.max = val;
            this.min = val;
            this.sum = val;
            this.bst = true;
        }
    }


    /*

      [5,2,1,3,6]
      val  = 5
      idx = 1
      lo = check(-INF, 5, nums)
      ro = check(5,INF, nums)

     */
    int idx = 0;

    public boolean verifyPreorder(int[] preorder) {
        return check(Integer.MIN_VALUE, Integer.MAX_VALUE, preorder);
    }

    private boolean check(int lb, int up, int[] nums) {
        // base case
        if (idx == nums.length) return true;


        int val = nums[idx];
        if (val < lb || val > up) return false;

        idx++;
        boolean leftOrder = check(lb, val, nums);
        boolean rightOrder = check(val, up, nums);

        return leftOrder || rightOrder;
    }

    public int numIdenticalPairs(int[] nums) {

        int n = nums.length;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int j = i + 1; j < n; j++) {
                if (nums[j] == nums[i]) cnt++;
            }
            ans += cnt;
        }
        return ans;
    }

    List<Integer> ans = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) return ans;
        helper(root, 0);
        return ans;
    }


    private void helper(TreeNode root, int level) {
        if (level == ans.size()) ans.add(root.val);
        if (root.right != null) helper(root.right, level + 1);
        if (root.left != null) helper(root.left, level + 1);
    }

    /*
     // Algo:-
       Pseudo code :-
     - create a tree  with given preorder traversal and traverse to get all subpaths
     - check for current subpath size
        - if ss == (num-k) then add to ans
        - else keep on getting paths
     - ans array will hold all possible paths with size = nums.size()-k;
     - scan to get the min value
     - return min_value
     */

    public String removeKdigits(String num, int k) {
        int n = num.length();
        Stack<Integer> stk = new Stack<>();

        int elem = k;
        for (int i = 0; i < n; i++) {
            int curr = Integer.parseInt(String.valueOf(num.charAt(i)));
            while (!stk.isEmpty() && stk.peek() > curr && elem > 0) {
                stk.pop();
                elem--;
            }

            stk.push(curr);
        }

        while (stk.size() > n - k) stk.pop(); // It will remove all the trailing sequence of numbers

        if (stk.isEmpty()) return "0";
        StringBuilder sb = new StringBuilder("");
        while (!stk.isEmpty()) sb.append(stk.pop());

        String regex = "^0+(?!$)";
        return sb.reverse().toString().replaceAll(regex, "");
    }

    // TBD FROM 10 aug onwards

    public int[] mostCompetitive(int[] nums, int k) {
        Deque<Integer> queue = new ArrayDeque<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (!queue.isEmpty() && queue.peekLast() > nums[i] && queue.size() + n - i > k
            ) queue.pollLast();

            queue.addLast(nums[i]);
        }

        int[] ans = new int[k];
        for (int m = 0; m < k; m++) {
            ans[m] = queue.pollFirst();
        }

        return ans;
    }

    public String smallestSubsequence(String s) {
        int n = s.length();
        Stack<Character> stk = new Stack<>();

        int[] freq = new int[26];
        boolean[] exist = new boolean[26];
        Arrays.fill(exist, false);

        for (char ch : s.toCharArray()) freq[ch - 'a']++;
        for (int i = 0; i < n; i++) {

            char ch = s.charAt(i);
            freq[ch - 'a']--;
            if (exist[ch - 'a']) continue;

            while (!stk.isEmpty() && stk.peek() > ch && freq[stk.peek() - 'a'] > 0) {
                char rem = stk.pop();
                exist[rem - 'a'] = false;
            }

            stk.push(ch);
            exist[ch - 'a'] = true;
        }

        char[] ans = new char[stk.size()];
        int index = 0;
        while (!stk.isEmpty()) {
            ans[index] = stk.pop();
            index++;
        }
        return new StringBuilder(new String(ans)).reverse().toString();
    }

    public String removeDuplicateLetters(String s) {
        int n = s.length();
        Stack<Character> stk = new Stack<>();

        int[] freq = new int[26];
        boolean[] exist = new boolean[26];
        Arrays.fill(exist, false);

        for (char ch : s.toCharArray()) freq[ch - 'a']++;
        for (int i = 0; i < n; i++) {

            char ch = s.charAt(i);
            freq[ch - 'a']--;
            if (exist[ch - 'a']) continue;

            while (!stk.isEmpty() && stk.peek() > ch && freq[stk.peek() - 'a'] > 0) {
                char rem = stk.pop();
                exist[rem - 'a'] = false;
            }

            stk.push(ch);
            exist[ch - 'a'] = true;
        }

        char[] ans = new char[stk.size()];
        int index = 0;
        while (!stk.isEmpty()) {
            ans[index] = stk.pop();
            index++;
        }
        return new StringBuilder(new String(ans)).reverse().toString();
    }

    /*
    "abcd"
"aaaaa"
"aabcaabdaab"
     */


    public int longestRepeatingSubstring(String s) {
        Set<String> set = new HashSet<>();

        int max = 0;
        int n = s.length();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (set.contains(s.substring(i, j + 1))) max = Math.max(max, j + 1 - i);
                else set.add(s.substring(i, j));
            }
        }
        return max;
    }


    List<String> ansStr = new ArrayList<>();

    /*
"hello"
"ooolleoooleh"
"prosperity"
"properties"
"ab"
"eidbaooo"
"ab"
"eidboaoo"
 */
    // TC = O(N^2)
    public boolean checkInclusion(String s1, String s2) {
        substring(s2, s1.length());
        for (String a : ansStr) {
            if (exist(s1, a)) return true;
        }

        return false;
    }

    // Fn to generate all possible  substrings of a given string
    private void substring(String s, int l) {
        int n = s.length();
        for (int i = 0; i <= n - l; i++) ansStr.add(s.substring(i, i + l));
    }

    private boolean exist(String s1, String s2) {

        int[] freq = new int[26];

        // freq[] : stores the  frequency of each
        // character of a string
        for (int i = 0; i < s2.length(); i++) {
            freq[s2.charAt(i) - 'a']++;
        }

        for (int i = 0; i < s1.length(); i++) {
            char ch = s1.charAt(i);
            if (freq[ch - 'a'] > 0) freq[ch - 'a']--;
            else return false;
        }
        return true;
    }

    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        // base cases
        if (root1 == root2) return true;
        if (root1 == null || root2 == null || root1.val != root2.val) return false;

        return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) ||
                (flipEquiv(root1.right, root2.left)
                        && flipEquiv(root1.left, root2.right));
    }


    private boolean equals(TreeNode root1, TreeNode root2) {

        if (root1 == null && root2 == null) return true;

        if (root1 != null && root2 != null) {
            return root1.val == root2.val && equals(root1.left, root2.left) && equals(root1.right, root2.right);
        }

        return false;
    }


    private TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    private List<Integer> whoKnows(int[][] meetings, Integer personId) {
        HashMap<Integer, LinkedList<Integer[]>> map = new HashMap<>();

        for (int[] meeting : meetings) {
            map.put(meeting[0], map.getOrDefault(meeting[0], new LinkedList<>()));
            map.get(meeting[0]).add(new Integer[]{meeting[1], meeting[2]});
        }

        // Level order traversal to check nodes that knows the story
        Queue<Integer[]> queue = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        queue.offer(new Integer[]{meetings[0][0], meetings[0][2]});
        visited.add(meetings[0][0]); // adding the actual person

        while (!queue.isEmpty()) {
            Integer[] meeting = queue.poll();
            LinkedList<Integer[]> children = map.getOrDefault(meeting[0], new LinkedList<>());
            for (Integer[] child : children) {
                Integer id = child[0];
                Integer time = child[1];
                if (time > meeting[1]) {
                    visited.add(id);
                    queue.offer(child);
                }
            }
        }
        return new ArrayList<>(visited);
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] ans = new int[2];
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (map.containsKey(target - nums[i])) {
                ans[0] = i;
                ans[1] = map.get(target - nums[i]);
                break;
            }
            map.put(nums[i], i);
        }
        return ans;
    }

    public int twoSumLessThanK(int[] nums, int k) {
        Arrays.sort(nums);
        int ans = Integer.MIN_VALUE;
        int n = nums.length;
        int i = 0, j = n - 1;
        while (i < j) {
            if (nums[i] + nums[j] < k) {
                ans = Math.max(ans, nums[i] + nums[j]);
                i++;
            } else {
                j--;
            }
        }
        return ans == Integer.MIN_VALUE ? -1 : ans;
    }

    // TC  = O(N), SC = O(N)
    public int trap(int[] height) {
        int n = height.length;
        int ans = 0;
        int[] left_max = new int[n], right_max = new int[n];

        left_max[0] = height[0];
        right_max[n - 1] = height[n - 1];

        for (int i = 1; i < n; i++) left_max[i] = Math.max(left_max[i - 1], height[i]);

        for (int i = n - 2; i >= 0; i--) right_max[i] = Math.max(right_max[i + 1], height[i]);

        for (int i = 1; i < n - 1; i++) {
            ans += Math.min(left_max[i], right_max[i]) - height[i];
        }
        return ans;
    }


    public int trapStack(int[] height) {

        /*
        // More simplified
        int n = height.length;
        int ans = 0, current = 0;
        Stack<Integer> stk = new Stack<>();// to store the indecies of smaller heights bars

        while (current < n) {
            while (!stk.isEmpty() && height[stk.peek()] < height[current]) {
                int top = stk.pop();
                if (stk.isEmpty()) break;
                int distance = current - stk.peek() - 1;
                int bounded_height = Math.min(height[current], height[stk.peek()]) - height[top];
                ans += bounded_height * distance;
            }

            stk.push(current++);
        }

        return ans;

         */

        int n = height.length, left = 0, right = n - 1, left_max = 0, right_max = 0, ans = 0;
        while (left < right) {
            // calculate the ans from left towards right
            if (height[left] < height[right]) {
                if (height[left] >= left_max) left_max = height[left];
                else ans += left_max - height[left];
                left++;
            }

            // calculate water concentration from right towards left
            else {
                if (height[right] >= right_max) right_max = height[right];
                else ans += right_max - height[right];
                right--;
            }
        }
        return ans;
    }

    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            ans[i] = ans[i & (i - 1)] + 1;
        }
        return ans;
    }

    /*
    // Algorithm:-
    -  get the left_max, right_max for current index
    -  Iterate towards left till ht is lesser than current and eventually fill water
    -  Iterate towards right till ht is lesser than current and eventually fill water
    -  return the updated array
    */
    public int[] pourWater(int[] heights, int volume, int k) {
        int n = heights.length;
        // build the condition for eventually fall
        heights[k] += volume;

        int[] left_max = new int[n], right_max = new int[n];

        left_max[0] = heights[0];
        right_max[n - 1] = heights[n - 1];

        for (int i = 1; i < n; i++) left_max[i] = Math.max(left_max[i - 1], heights[i]);

        for (int i = n - 2; i >= 0; i--) right_max[i] = Math.max(right_max[i + 1], heights[i]);


        int l = k, r = k;
        while (l-- >= 0) {
            if (left_max[l] - heights[l] < heights[k]) {
                heights[l]++;
                heights[k]--;
            }
        }


        while (r++ < n) {
            if (right_max[r] - heights[r] < heights[k]) {
                heights[r]++;
                heights[k]--;
            }
        }


        return heights;
    }


    // TC = O(N^2), SC = O(1)
    public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int ans = 0;
        for (int i = 0; i < n - 2; i++) ans += twoSumSmallerInRange(nums, i + 1, target - nums[i]);

        return ans;
    }

    private int twoSumSmallerInRange(int[] nums, int startIndex, int target) {
        int sum = 0;
        int left = startIndex;
        int right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] < target) {
                sum += right - left;
                left++;
            } else {
                right--;
            }
        }

        return sum;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<List<Integer>> res = twoSumZero(nums, i + 1, -nums[i]);
            for (List<Integer> l : res) {
                List<Integer> triplet = new ArrayList<>();
                triplet.add(nums[i]);
                triplet.addAll(l);
                ans.add(triplet);
            }
        }
        return ans;
    }

    private List<List<Integer>> twoSumZero(int[] nums, int startIndex, int target) {
        int left = startIndex;
        int right = nums.length - 1;
        List<List<Integer>> res = new ArrayList<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = left; i < right; i++) {
            if (map.containsKey(target - nums[i])) {
                List<Integer> ans = new ArrayList<>();
                ans.add(nums[i]);
                ans.add(target - nums[i]);
                res.add(ans);
            }
            map.put(nums[i], i);
        }
        return res;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] < target) twoSum(nums, j, res, target - (nums[i] + nums[j]), i);
            }
        }
        return res;
    }

    private void twoSum(int[] nums, int i, List<List<Integer>> res, int target, int startIndex) {
        Set<Integer> seen = new HashSet<>();
        int n = nums.length;
        for (int j = i + 1; j < n; ++j) {
            int complement = target - nums[j];
            if (seen.contains(complement)) {
                res.add(Arrays.asList(nums[startIndex], nums[i], nums[j], complement));
            }
            seen.add(nums[j]);
        }
    }

    class KSum {
        public List<List<Integer>> fourSum(int[] nums, int target) {
            Arrays.sort(nums);
            return kSum(nums, target, 0, 4);
        }

        public List<List<Integer>> kSum(int[] nums, int target, int start, int k) {
            List<List<Integer>> res = new ArrayList<>();
            if (start == nums.length || nums[start] * k > target || target > nums[nums.length - 1] * k)
                return res;
            if (k == 2)
                return twoSum(nums, target, start);

            for (int i = start; i < nums.length; ++i)
                if (i == start || nums[i - 1] != nums[i])
                    for (List<Integer> subset : kSum(nums, target - nums[i], i + 1, k - 1)) {
                        res.add(new ArrayList<>(Collections.singletonList(nums[i])));
                        res.get(res.size() - 1).addAll(subset);
                    }

            return res;
        }

        public List<List<Integer>> twoSum(int[] nums, int target, int start) {
            List<List<Integer>> res = new ArrayList<>();
            Set<Integer> s = new HashSet<>();

            for (int i = start; i < nums.length; ++i) {
                if (res.isEmpty() || res.get(res.size() - 1).get(1) != nums[i])
                    if (s.contains(target - nums[i]))
                        res.add(Arrays.asList(target - nums[i], nums[i]));
                s.add(nums[i]);
            }

            return res;
        }
    }


    public int findPeakElement(int[] nums) {
        int n = nums.length;
        return search(nums, 0, n - 1);
    }

    // recursive binary search
    private int search(int[] nums, int l, int r) {
        if (l == r) return l;
        int mid = l + (r - l) / 2;
        if (nums[mid] > nums[mid + 1])
            return search(nums, l, mid);
        return search(nums, mid + 1, r);
    }

    // Iterative sollution
    public int peakIndexInMountainArray(int[] nums) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (nums[m] < nums[m + 1]) l = m + 1;
            else r = m;
        }
        return l;
    }

    // TC=  O(N2) SC=O(N)
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int cnt = 0;
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int n1 : nums1) {
            for (int n2 : nums2) {
                int key = n1 + n2;
                map.put(key, map.getOrDefault(key, 0) + 1);
            }
        }

        for (int n3 : nums3) {
            for (int n4 : nums4) {
                int key = -(n3 + n4);
                cnt += map.getOrDefault(key, 0);
            }
        }
        return cnt;
    }

    public int kNumberSumCount(int[] A, int[] B, int[] C, int[] D) {
        return kSumCount(new int[][]{A, B, C, D});
    }

    public int kSumCount(int[][] lists) {
        Map<Integer, Integer> m = new HashMap<>();
        addToHash(lists, m, 0, 0);
        return countComplements(lists, m, lists.length / 2, 0);
    }

    void addToHash(int[][] lists, Map<Integer, Integer> m, int i, int sum) {
        if (i == lists.length / 2)
            m.put(sum, m.getOrDefault(sum, 0) + 1);
        else
            for (int a : lists[i])
                addToHash(lists, m, i + 1, sum + a);
    }

    int countComplements(int[][] lists, Map<Integer, Integer> m, int i, int complement) {
        if (i == lists.length)
            return m.getOrDefault(complement, 0);
        int cnt = 0;
        for (int a : lists[i])
            cnt += countComplements(lists, m, i + 1, complement - a);
        return cnt;
    }

    boolean flip = false;
    int maxOne = Integer.MIN_VALUE;

    public int findMaxConsecutiveOnes(int[] nums) {
        consective(nums, 0);
        return maxOne;
    }

    private void consective(int[] nums, int index) {
        if (index == nums.length) return;

        int cnt = 0;
        if (nums[index] == 1) {
            cnt += 1;
            consective(nums, index + 1);
        } else if (!flip) {
            flip = true;
            cnt++;
        } else {
            cnt = 1;
        }

        maxOne = Math.max(maxOne, cnt);
    }

    public int findTargetSumWays(int[] nums, int target) {
        int[][] memo = new int[nums.length][2001];
        for (int[] rows : memo) Arrays.fill(rows, -1);
        return calculate(nums, 0, 0, target, memo);
    }

    private int calculate(int[] nums, int index, int sum, int target, int[][] memo) {
        if (index == nums.length) {
            if (sum == target)
                return 1;
            return 0;
        } else {

            if (memo[index][sum + 1000] != -1) return memo[index][sum + 1000];
            int add = calculate(nums, index + 1, sum + nums[index], target, memo);
            int subtract = calculate(nums, index + 1, sum - nums[index], target, memo);
            memo[index][sum + 1000] = add + subtract;

            return memo[index][sum + 1000];
        }
    }

    // sliding window technique
    public int findMaxConsecutiveOneSliding(int[] nums) {
        int n = nums.length;
        int left = 0, right = 0, longest = 0, zeros = 0;

        while (right < n) {
            // move right ptr towards right direction
            if (nums[right] == 0) zeros++;

            // check for invalid state
            while (zeros == 2) {
                if (nums[left] == 0) zeros--;
                left++;
            }

            // update out longest sequnce
            longest = Math.max(longest, right - left + 1);

            // expand out window
            right++;
        }
        return longest;
    }

    // cheer up solution for placing operators b/w numbers
    public ArrayList<String> answer;
    public String digits;
    public long target;

    public void recurse(int index, long previousOperand, long currentOperand, long value, ArrayList<String> ops) {
        String nums = this.digits;

        // Done processing all the digits in num
        if (index == nums.length()) {

            // If the final value == target expected AND
            // no operand is left unprocessed
            if (value == this.target && currentOperand == 0) {
                StringBuilder sb = new StringBuilder();
                ops.subList(1, ops.size()).forEach(v -> sb.append(v));
                this.answer.add(sb.toString());
            }
            return;
        }

        // Extending the current operand by one digit
        currentOperand = currentOperand * 10 + Character.getNumericValue(nums.charAt(index));
        String current_val_rep = Long.toString(currentOperand);
        int length = nums.length();

        // To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
        // valid operand. Hence this check
        if (currentOperand > 0) {

            // NO OP recursion
            recurse(index + 1, previousOperand, currentOperand, value, ops);
        }

        // ADDITION
        ops.add("+");
        ops.add(current_val_rep);
        recurse(index + 1, currentOperand, 0, value + currentOperand, ops);
        ops.remove(ops.size() - 1);
        ops.remove(ops.size() - 1);

        if (ops.size() > 0) {

            // SUBTRACTION
            ops.add("-");
            ops.add(current_val_rep);
            recurse(index + 1, -currentOperand, 0, value - currentOperand, ops);
            ops.remove(ops.size() - 1);
            ops.remove(ops.size() - 1);

            // MULTIPLICATION
            ops.add("*");
            ops.add(current_val_rep);
            recurse(
                    index + 1,
                    currentOperand * previousOperand,
                    0,
                    value - previousOperand + (currentOperand * previousOperand),
                    ops);
            ops.remove(ops.size() - 1);
            ops.remove(ops.size() - 1);
        }
    }

    public List<String> addOperators(String num, int target) {

        if (num.length() == 0) {
            return new ArrayList<String>();
        }

        this.target = target;
        this.digits = num;
        this.answer = new ArrayList<String>();
        this.recurse(0, 0, 0, 0, new ArrayList<String>());
        return this.answer;
    }

    public int maxScore(int[] cardPoints, int k) {
        int n = cardPoints.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) Arrays.fill(dp[i], -1);
        return cards(cardPoints, k, 0, n - 1, dp);
    }

    private int cards(int[] cardPoints, int k, int l, int r, int[][] dp) {
        // base case
        if (l > r || k <= 0) {
            return 0;
        }

        if (dp[l][r] != -1) return dp[l][r];

        // take a card from left
        int left = cardPoints[l] + cards(cardPoints, k - 1, l + 1, r, dp);
        // take a card from right
        int right = cardPoints[r] + cards(cardPoints, k - 1, l, r - 1, dp);

        dp[l][r] = Math.max(left, right);

        return dp[l][r];
    }
    // using prefix sum approach

    public int maxScorePrefixSum(int[] cardPoints, int k) {
        int n = cardPoints.length;
        int[] first = new int[k + 1], last = new int[k + 1];

        for (int i = 0; i < k; i++) {
            first[i + 1] = first[i] + cardPoints[i];
            last[i + 1] = last[i] + cardPoints[n - 1 - i];
        }

        int maxScore = 0;
        for (int i = 0; i <= k; i++) {
            int currSum = first[i] + last[k - i];
            maxScore = Math.max(maxScore, currSum);
        }
        return maxScore;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return bst(nums, 0, nums.length - 1);
    }

    private TreeNode bst(int[] nums, int l, int r) {
        // base case
        if (l > r) return null;

        int mid = l + (r - l) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = bst(nums, l, mid - 1);
        node.right = bst(nums, mid + 1, r);
        return node;
    }

    public int maxSubArrayLen(int[] nums, int target) {
        int n = nums.length;

        int ans = 0;
        HashMap<Integer, Integer> map = new HashMap<>(); //map<sum, index>
        map.put(0, -1);

        for (int i = 0, sum = 0; i < n; i++) {
            sum += nums[i];
            if (map.containsKey(target - sum)) {
                ans = Math.max(ans, i - map.get(target - sum));
            }
            map.putIfAbsent(sum, i);
        }
        return ans;
    }

    static class Edge {
        int v;
        int nbr;
        int wt;

        Edge(int v, int nbr, int wt) {
            this.v = v;
            this.nbr = nbr;
            this.wt = wt;
        }
    }

    static class Pair {
        int v;
        int av; //acquiring vertex (As Prims works on considering next connected min edge)
        int wt;

        Pair(int v, int av, int wt) {
            this.v = v;
            this.av = av;
            this.wt = wt;
        }
    }

    public int minCostConnectPoints(int[][] points) {
        //Construct Graph
        int n = points.length;
        ArrayList<Edge> graph[] = new ArrayList[n];

        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        //Since this graph will be a complete graph so have to connect each and every vertex with each other
        for (int i = 0; i < n; i++) {
            int u = i;
            int xi = points[i][0];
            int yi = points[i][1];
            for (int j = i + 1; j < n; j++) {
                int v = j;
                int xj = points[j][0];
                int yj = points[j][1];

                int wt = Math.abs(xi - xj) + Math.abs(yi - yj);
                graph[u].add(new Edge(u, v, wt));
                graph[v].add(new Edge(v, u, wt));
            }
        }
        //Now run your Prim's Alogrithm
        PriorityQueue<Pair> q = new PriorityQueue<>((a, b) -> {
            return a.wt - b.wt;
        });
        q.add(new Pair(0, -1, 0));
        int cost = 0;
        boolean visited[] = new boolean[n];
        while (q.size() > 0) {
            Pair rem = q.poll();

            if (visited[rem.v]) continue;
            visited[rem.v] = true;

            if (rem.av != -1) {
                cost = cost + rem.wt;
            }
            for (Edge e : graph[rem.v]) {
                if (!visited[e.nbr]) {
                    q.add(new Pair(e.nbr, rem.v, e.wt));
                }
            }
        }
        return cost;
    }


    public int maxSatisfaction(int[] satisfaction) {
        Arrays.sort(satisfaction);
        int n = satisfaction.length;

        int max = 0;
        for (int i = 0; i < n; i++) {
            int sum = 0, count = 1;
            for (int j = i; j < n; j++) {
                sum += count * satisfaction[j];
            }
            max = Math.max(sum, max);
        }
        return max;
    }

    // Recursion with memoisation
// 2 choices 1 take it and skip it (overall sum will not increase)
    // TC = O(2^N)
    // O(N^2)
    public int maxSatisfactionRecurse(int[] satisfaction) {
        Arrays.sort(satisfaction);
        int n = satisfaction.length;
        int[][] dp = new int[n + 1][n + 1];
        for (int i = 0; i < n + 1; i++) Arrays.fill(dp[i], -1);
        return maxsum(satisfaction, dp, 0, 1);
    }

    private int maxsum(int[] satisfaction, int[][] dp, int index, int coef) {

        // base case
        if (index == satisfaction.length) return 0;

        if (dp[index][coef] != -1) return dp[index][coef];

        // select
        int taken = coef * satisfaction[index] + maxsum(satisfaction, dp, index + 1, coef + 1);

        // number is not taken
        int notTaken = maxsum(satisfaction, dp, index + 1, coef);

        // return max of 2 choices
        return dp[index][coef] = Math.max(taken, notTaken);

    }

    public int validSubarrays(int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            for (int j = i; j < nums.length; j++) {
                if (nums[i] <= nums[j]) {
                    count++;
                    continue;
                }
                break;
            }
        }
        return count;
    }

    public int minSwap(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;

        // initialize dp table
        int[][] memo = new int[2][n1];
        Arrays.fill(memo[0], -1);
        Arrays.fill(memo[1], -1);
        memo[0][0] = 0;
        memo[1][0] = 1;

        return Math.min(recurse(nums1, nums2, n1 - 1, 0, memo), recurse(nums1, nums2, n1 - 1, 1, memo));
    }

    private int recurse(int[] nums1, int[] nums2, int i, int swap, int[][] memo) {
        // base cases
        if (i == 0) return swap;

        //check dp table
        if (memo[swap][i] != -1)
            return memo[swap][i];


        int res = Integer.MAX_VALUE;
        // no swap
        if (nums1[i] > nums1[i - 1] && nums2[i] > nums2[i - 1])
            res = recurse(nums1, nums2, i - 1, swap, memo);

        // swap
        if (nums1[i - 1] < nums2[i] && nums2[i - 1] < nums1[i])
            res = Math.min(res, recurse(nums1, nums2, i - 1, 1 - swap, memo));

        memo[swap][i] = swap == 0 ? res : res + 1;
        return memo[swap][i];
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        bfs(new ArrayList<>(), 0, 0, ans, target, candidates);
        return ans;
    }

    private void bfs(List<Integer> num, int sum, int index, List<List<Integer>> ans, int k, int[] candidates) {
        // base case
        if (sum == k) {
            if (!ans.contains(num)) ans.add(new ArrayList<>(num));
        } else if (sum > k) {
            return;
        } else {
            // For all position place numbers
            for (int p = index; p < candidates.length; p++) {
                //skip duplicates
                if (p > index && candidates[p] == candidates[p - 1]) {
                    continue;
                }

                num.add(candidates[p]);
                sum += candidates[p];
                bfs(num, sum, p + 1, ans, k, candidates);
                num.remove(new Integer(candidates[p]));
                sum -= candidates[p];
            }
        }
    }
}

// TOP_DOWN
// Backtracking
class CombinationSum4 {
    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        int[] memo = new int[target + 1];
        Arrays.fill(memo, -1);
        return backtrack(nums, target, memo);
    }

    private int backtrack(int[] nums, int target, int[] memo) {
        // base cases
        if (target == 0) return 1;
        if (target < 0) return 0;

        if (memo[target] != -1) return memo[target];

        int total = 0;
        for (int i = 0; i < nums.length; i++) {
            total += backtrack(nums, target - nums[i], memo);
        }
        memo[target] = total;
        return total;
    }
}

class CombinationSum {
    // TC = O(2^n), SC = O(2^n)
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        bfs(new ArrayList<>(), 0, 0, ans, target, candidates);
        return ans;
    }

    private void bfs(List<Integer> num, int sum, int index, List<List<Integer>> ans, int k, int[] candidates) {
        // base case
        if (sum == k) {
            ans.add(new ArrayList<>(num));
        } else if (sum > k) {
            return;
        } else {
            // For all position place numbers
            for (int p = index; p < candidates.length; p++) {
                num.add(candidates[p]);
                sum += candidates[p];
                bfs(num, sum, p, ans, k, candidates);
                num.remove(new Integer(candidates[p]));
                sum -= candidates[p];
            }
        }
    }
}

class LargetRectangleArea {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0)
            return 0;
        int[] heights = new int[matrix[0].length];

        for (int i = 0; i < matrix[0].length; i++)
            heights[i] = matrix[0][i] - '0';

        int maxArea = largestRectangleArea(heights);

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1')
                    heights[j]++;
                else heights[j] = 0;
            }
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }
        return maxArea;

    }

    int largestRectangleArea(int[] heights) {
        Stack<Integer> st = new Stack<>();
        int maxArea = 0;
        st.push(-1);
        for (int i = 0; i <= heights.length; i++) {
            int val = i == heights.length ? 0 : heights[i];

            while (st.peek() != -1 && heights[st.peek()] >= val) {
                int rightMin = i;
                int height = heights[st.pop()];
                int leftMin = st.peek();
                maxArea = Math.max(maxArea, height * (rightMin - leftMin - 1));
            }
            st.push(i);
        }
        return maxArea;
    }
}

class LinkedListRandomNode {

    /**
     * @param head The linked list's head.
     * Note that the head is guaranteed to be not null, so it contains at least one node.
     */

    ListNode head;
    Random generator;

    public LinkedListRandomNode(ListNode head) {
        this.head = head;
        this.generator = new Random();
    }

    /**
     * Returns a random node's value.
     */
    public int getRandom() {
        int elem = -1, index = -1;
        ListNode temp = head;
        while (temp != null) {
            index++;
            if (index == 0) {
                elem = temp.val;
            } else {
                int random = generator.nextInt(index + 1) + 1;
                if (random == index) {
                    elem = temp.val;
                }
            }
            temp = temp.next;
        }
        return elem;
    }
}

class FourSumCount {

    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int n = nums1.length;
        if (n == 1) {
            return (nums1[0] + nums2[0] + nums3[0] + nums4[0] == 0) ?
                    1 : 0;
        }

        // Put nums3 + nums4 into a HashMap, where
        // key = nums3[k] + nums4[l], value = count
        HashMap<Integer, Integer> m = new HashMap<Integer, Integer>();
        for (int k = 0; k < n; k++) {
            for (int l = 0; l < n; l++) {
                int s = nums3[k] + nums4[l];
                m.put(s, m.getOrDefault(s, 0) + 1);
            }
        }

        int r = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int comp = -nums1[i] - nums2[j];
                if (m.get(comp) != null) {
                    r += m.get(comp);
                }
            }
        }

        return r;
    }
}


// TODO:- This problem needs to be revisted again
class recoverArrayHard {
    public int[] recoverArray(int n, int[] sums) {
        List<Integer> sumList = new ArrayList<>();
        for (Integer sum : sums) sumList.add(sum);
        Collections.sort(sumList, (a, b) -> Integer.compare(b, a));
        int[] res = new int[n];
        int i = 0;

        while (sumList.size() > 2) {
            List<Integer> array1 = new ArrayList<>(), array2 = new ArrayList<>();
            int num = sumList.get(0) - sumList.get(1);
            Map<Integer, Integer> map = getCountMap(sumList);

            for (int elem : sumList) {
                if (map.containsKey(elem)) {
                    array2.add(elem);
                    array1.add(elem - num);
                    remove(map, elem);
                    remove(map, elem - num);
                }
            }

            int index = array2.indexOf(num);
            if (index != -1) {
                if (array1.get(index) == 0) {
                    res[i++] = num;
                    sumList = array1;
                    continue;
                }
            }
            res[i++] = -num;
            sumList = array2;
        }

        if (sumList.get(0) == 0) {
            res[i++] = sumList.get(1);
        } else {
            res[i++] = sumList.get(0);
        }

        return res;
    }

    // This function simply descrement the freq of element in map
    private void remove(Map<Integer, Integer> map, int element) {
        if (map.containsKey(element)) {
            if (map.get(element) > 1) {
                map.put(element, map.get(element) - 1);
            } else {
                map.remove(element);
            }
        }
    }

    // This function get the freq of sum in map and return it
    private Map<Integer, Integer> getCountMap(List<Integer> list) {
        Map<Integer, Integer> map = new HashMap<>();
        for (Integer num : list) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        return map;
    }
}

class SubSet2 {
    // TC = O(2^n), SC = exponential in nature
    // backtracking basesd solution
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        bfs(nums, 0, result, new ArrayList<Integer>());
        return result;
    }

    private void bfs(int[] nums, int index, List<List<Integer>> result, List<Integer> res) {
        // base case
        if (!result.contains(res)) result.add(new ArrayList<>(res));

        for (int i = index; i < nums.length; i++) {
            int num = nums[i];
            res.add(num);
            bfs(nums, i + 1, result, res);
            res.remove(new Integer(num));
        }
    }


    class Subsets {
        // TC = O(n), SC = O(n)

        public List<List<Integer>> subsets(int[] nums) {
            int n = nums.length;
            LinkedList<List<Integer>> result = new LinkedList<>();
            if (n == 0) return result;
            result.add(new ArrayList<>());
            permuate(nums, 0, result);
            return result;
        }

        private void permuate(int[] nums, int index, LinkedList<List<Integer>> result) {
            int n = nums.length;
            // base case
            if (index == n) {
                return;
            }

            int ch = nums[index];
            int size = result.size();
            LinkedList<Set<Integer>> list = new LinkedList<Set<Integer>>();

            while (size-- > 0) {
                List<Integer> elem = result.poll();
                list.add(new HashSet<>(elem));
                elem.add(ch);
                list.add(new HashSet<>(elem));
            }
            result.addAll(list.stream().map(x -> new ArrayList<>(x)).collect(Collectors.toList()));

            permuate(nums, index + 1, result);
        }
    }

// Iterative approach

// Algorithm :-

    /*
     - Idea is to take all paths by variation of a  single character (ie. uppercase or lowercase)
     - Intialise list with first given variation of string
     - Iterate two loops
       - i = 0 unitll n (n = size of string)
       - check for current char ie. if its character then its liable for changes else not
       - poll elements out of list untill it becomes empty
       - create both variation and update the stirng to list
       - Repeat the process untill while string gets exhausted
      - Return all possible paths
    */
// Recursive approach
    class LetterCasePermutation {
        // TC = O(2*n), SC = O(n!)
        public List<String> letterCasePermutation(String s) {
            int n = s.length();
            List<String> ans = new ArrayList<>();
            if (n == 0) return ans;
            StringBuilder sb = new StringBuilder("");
            permuate(s, sb, 0, ans);
            return ans;
        }

        private void permuate(String s, StringBuilder sb, int index, List<String> ans) {
            int n = s.length();
            // base case
            if (index == n) {
                ans.add(sb.toString());
                return;
            }

            char c = s.charAt(index);

            // if char is letter
            if (Character.isLetter(c)) {
                Character[] array = new Character[]{Character.toUpperCase(c), Character.toLowerCase(c)};
                for (char ch : array) {
                    sb.append(ch);
                    permuate(s, sb, index + 1, ans);
                    sb.setLength(sb.length() - 1);
                }
            }
            // else do not make any variation
            else {
                sb.append(c);
                permuate(s, sb, index + 1, ans);
                sb.setLength(sb.length() - 1);
            }
        }

    /*

    // Iterative solution
      public List<String> letterCasePermutation(String s) {
        int n = s.length();
        LinkedList<String> result = new LinkedList<>();
        if (n == 0) return result;
        result.add(s);
        for (int i = 0; i<n; i++) {
            char ch = s.charAt(i);
            if (Character.isLetter(ch)) {
                int size = result.size();
                while (size--> 0) {
                    String str = result.poll();
                    String left = str.substring(0, i);
                    String right = str.substring(i + 1);
                    result.add(left + String.valueOf(ch).toUpperCase() + right);
                    result.add(left + String.valueOf(ch).toLowerCase() + right);
                }
            }
        }


        // String[] arr = new String[result.size()];
        // arr = result.toArray(arr);
        // // Get lexicographically sorted order results
        // Arrays.sort(arr);
        // return Arrays.stream(arr).map(x->x).collect(Collectors.toList());
        return new ArrayList<>(result);
    }
     */
    }

    class PickRandomBlacklist {
        HashMap<Integer, Integer> map;
        Random generator;
        int ul;

        public PickRandomBlacklist(int n, int[] blacklist) {
            this.generator = new Random();
            this.map = new HashMap<>();
            for (Integer num : blacklist)
                this.map.put(num, -1);

            this.ul = n - blacklist.length;
            int j = n - 1;
            for (Integer num : blacklist) {
                if (num < ul) {
                    while (map.containsKey(j)) j--;
                    map.put(num, j);
                    j--;
                }
            }
        }

        public int pick() {
            int random = generator.nextInt(ul);
            if (map.containsKey(random))
                return map.get(random);
            return random;
        }
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * Solution obj = new Solution(n, blacklist);
     * int param_1 = obj.pick();
     */

    class FindDifferentBinaryString {
        Set<String> set;
        String res;
        int len;
        String[] d = {
                "1", "0"
        };

        public String findDifferentBinaryString(String[] nums) {
            len = nums.length;
            set = new HashSet<>();
            for (String s : nums) set.add(s);
            diff(0, "");
            return res;
        }

        private Boolean diff(int index, String str) {
            // base case
            if (index == set.size()) {
                if (!set.contains(str)) {
                    res = str;
                    return true;
                }
                return false;
            }
            // For all position place string characters
            for (int k = 0; k < len; k++) {
                for (int i = 0; i < d.length; i++) {
                    str += d[i];
                    if (diff(index + 1, str)) return true;
                    str = str.substring(0, str.length() - 1);
                }
            }
            return false;
        }
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
    class ReverseLinkedList {
        public ListNode reverseList(ListNode head) {

            if (head == null) return head;
            ListNode curr = head, prev = null, next;
            while (curr != null) {
                next = curr.next;
                curr.next = prev;
                prev = curr;
                curr = next;
            }
            head = prev;
            return head;
        }
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
    class ReverseLinkedListKGroup {
        public ListNode reverseKGroup(ListNode head, int k) {
            if (head == null) return head;
            int len = getLength(head);
            return reverse(head, k, len);
        }

        private int getLength(ListNode curr) {
            int cnt = 0;
            while (curr != null) {
                cnt++;
                curr = curr.next;
            }
            return cnt;
        }

        // Recursive function to reverse in groups
        private ListNode reverse(ListNode head, int k, int len) {
            // base case
            if (len < k) return head;

            ListNode curr = head, prev = null, next = null;
            for (int i = 0; i < k; i++) {
                // reverse linked list code
                next = curr.next;
                curr.next = prev;
                prev = curr;
                curr = next;
            }

            ListNode nextNode = reverse(curr, k, len - k);
            head.next = nextNode;
            return prev;
        }
    }

    class MaxTaxiEarningsDP {

        // DP + Iteration based solution
        public long maxTaxiEarnings(int n, int[][] rides) {
            // sort rides  matrix based on  start time
            Arrays.sort(rides, (a, b) -> a[0] - b[0]);
		/*
		   [[2,5,4],[1,5,1]]
		   sort ------> [[1 5 1], [2 5 4]]
		   dp[0]=5-1+1=5
		   dp[1]=max(7,5)=7

         - dp max for every switch to new point
         - For every next switch check wrt to prev path and curr and take
         - Return max value ie. dp[n]
		   Complexity -->  TC = O(n+k) , SC = O(n);
		*/
            long[] dp = new long[n + 1];
            int j = 0;
            for (int i = 1; i <= n; i++) {
                dp[i] = Math.max(dp[i - 1], dp[i]);
                while (j < rides.length && rides[j][0] == i) {
                    int d = rides[j][1] - rides[j][0] + rides[j][2];
                    dp[rides[j][1]] = Math.max(dp[rides[j][1]], dp[i] + d);
                    ++j;
                }
            }
            return dp[n];
        }
    }


    class FindOriginalArray {
        public int[] findOriginalArray(int[] changed) {
            int n = changed.length, cnt = 0;
            List<Integer> ans = new ArrayList<>();
            HashMap<Integer, List<Integer>> map = new HashMap<>();
            Arrays.sort(changed);

            // for (Integer ll: changed) System.out.println(ll);

            if (n == 1 || n == 0 || n % 2 != 0) return new int[]{};

            for (int i = 0; i < n; i++) {
                int e = changed[i];
                if (e % 2 == 0 && map.containsKey(e / 2) && map.get(e / 2).size() > 0) {
                    ans.add(e / 2);
                    cnt++;
                    List<Integer> list = map.get(e / 2);
                    list.remove(0);
                    if (list.size() > 0) map.put(e / 2, list);
                    else map.remove(e / 2);
                    continue;
                }
                if (map.containsKey(e)) {
                    List<Integer> l = map.get(e);
                    l.add(i);
                    map.put(e, l);
                } else {
                    List<Integer> l = new ArrayList<>();
                    l.add(i);
                    map.put(e, l);
                }
            }
            if (cnt == n / 2) return ans.stream().mapToInt(x -> x).toArray();
            return new int[]{};
        }
    }

    // TODO: Solve again
    class SplitIntoFibonacci {
        List<Integer> al = new ArrayList<>();

        public List<Integer> splitIntoFibonacci(String num) {
            for (int i = 0; i < num.length(); i++) {
                if (i > 0 && num.charAt(0) == '0') return new ArrayList<Integer>();
                long a = Long.parseLong(num.substring(0, i + 1));
                if (a > Integer.MAX_VALUE) break;
                for (int j = i + 1; j < num.length(); j++) {
                    long b = Long.parseLong(num.substring(i + 1, j + 1));
                    if (b > Integer.MAX_VALUE) break;
                    al.add((int) a);
                    al.add((int) b);
                    if (j > i + 1 && num.charAt(i + 1) == '0') break;
                    if (j + 1 < num.length() && solve((int) a, (int) b, num, j + 1)) return al;
                    else al.remove(al.size() - 1);
                }
                al.clear();
            }
            return new ArrayList<Integer>();
        }

        public boolean solve(int a, int b, String num, int index) {
            if (index == num.length()) return true;
            String sum = String.valueOf(a + b);
            if (index + sum.length() > num.length()) {
                al.remove(al.size() - 1);
                return false;
            }
            String actualSum = num.substring(index, index + sum.length());
            if (!sum.equals(actualSum)) {
                al.remove(al.size() - 1);
                return false;
            } else {
                al.add(a + b);
                return solve(b, a + b, num, index + sum.length());
            }
        }
    }


    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     * int val;
     * TreeNode left;
     * TreeNode right;
     * TreeNode() {}
     * TreeNode(int val) { this.val = val; }
     * TreeNode(int val, TreeNode left, TreeNode right) {
     * this.val = val;
     * this.left = left;
     * this.right = right;
     * }
     * }
     */
// Algo:-

// - if tree is null retrun empty array list
// - Apply preorder traversal recursivelyand check of node had beee already taken in map or not
// - if its taken first time for that level fetch it else skip it
// - return the global ans

/*

[1,2,3,4,5,6,7]
   [(0, [1]),(-1, [2]), (-2, [4]),  ]

TC = O(Nlogn) + O(N) ~= O(nlogn)
SC = O(N)
*/

    class VerticalTraversal {
        public class Pair implements Comparable<Pair> {
            int key;
            int value;

            public Pair(int key, int value) //Constructor of the class
            {
                this.key = key;
                this.value = value;
            }

            public int compareTo(Pair o) {
                return this.key - o.key;
            }
        }

        // To store the vline and nodes with level on that vline
        HashMap<Integer, PriorityQueue<Pair>> map = new HashMap<>();

        public List<List<Integer>> verticalTraversal(TreeNode root) {
            if (root == null) return null;
            inorder(root, 0, 0);
            List<List<Integer>> ans = new ArrayList<>();
            Map<Integer, PriorityQueue<Pair>> treemap = new TreeMap<>(map);

            // Iterate through map and get the result in desired format
            for (Map.Entry<Integer, PriorityQueue<Pair>> entry : treemap.entrySet()) {
                PriorityQueue<Pair> pq = entry.getValue();
                // to store level along iwth nodes on that level
                // grouping the nodes based on level
                Map<Integer, List<Integer>> hm = new TreeMap<>();
                List<Integer> colList = new ArrayList<>();

                while (!pq.isEmpty()) {
                    Pair res = pq.poll();
                    if (hm.containsKey(res.key)) {
                        List<Integer> list = new ArrayList<>();
                        List<Integer> l = hm.get(res.key);
                        l.add(res.value);
                        hm.put(res.key, l);
                    } else hm.put(res.key, new ArrayList<>(Arrays.asList(res.value)));
                }

                for (Map.Entry<Integer, List<Integer>> e : hm.entrySet()) {
                    List<Integer> pd = e.getValue();
                    Collections.sort(pd);
                    for (Integer data : pd) colList.add(data);
                }
                ans.add(colList);
            }
            return ans;
        }

        private void inorder(TreeNode root, int level, int vline) {
            if (root == null) return;

            vline -= 1;
            inorder(root.left, level + 1, vline);
            vline += 1;
            // node on this vline is already present then add new node in list of this vline
            if (map.get(vline) != null) {
                PriorityQueue<Pair> existingpq = map.get(vline);
                existingpq.add(new Pair(level, root.val));
                map.put(vline, existingpq);
            } else {
                // Else if node on this vline is found first time then simply add it
                PriorityQueue<Pair> newNode = new PriorityQueue<>();
                newNode.add(new Pair(level, root.val));
                map.put(vline, newNode);
            }
            inorder(root.right, level + 1, vline + 1);
        }
    }

    class CanPartitionKSubsets {
        int[] nums;
        int n;
        boolean[] vis;

        public boolean canPartitionKSubsets(int[] nums, int k) {
            this.n = nums.length;
            this.nums = nums;
            vis = new boolean[n];
            int t = 0;
            int max = Integer.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                t += nums[i];
                max = Math.max(max, nums[i]);
            }

            if (t % k != 0) return false;

            for (int i = 0; i < n; i++) if (nums[i] > max) return false;

            int target = t / k;
            return dfs(0, target, 0, k);
        }

        private boolean dfs(int sum, int target, int idx, int left) {
            if (left == 1) return true;

            if (sum == target) {
                return dfs(0, target, 0, left - 1);
            }

            if (sum > target || idx >= n) return false;

            for (int i = idx; i < n; i++) {
                if (!vis[i]) {
                    vis[i] = true;
                    if (dfs(sum + nums[i], target, idx + 1, left)) return true;
                    vis[i] = false;
                }
            }
            return false;
        }
    }


// Intial thoughts:-
// BFS algorithm


    //  DP
//  - create a map to adjancent nodes alogn with distance
//  - iterate through all nodes and calculate cost of each path recursively
//  - update minCost path if currNode reaches to destination
//  - returm minCost;
// TC  = (2^n)
// SC = O(n)
    class FindCheapestPriceGraphBFS {

        /*
        class Solution {
      int n;
      int[][] flights;
      int src;
      int dst;
      // create a graph
      Map<Integer, List < pair >> graph;
      int minCost = Integer.MAX_VALUE;
      public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        this.n = n;
        this.flights = flights;
        this.src = src;
        this.dst = dst;
        this.graph = new HashMap<>();
        //Build graph used for adjancy list and distance
        for (int i = 0; i < flights.length; i++) {
          int srcF = flights[i][0];
          int dest = flights[i][1];
          int dist = flights[i][2];
          if (graph.containsKey(srcF)) {
            List < pair > existing = graph.get(srcF);
            existing.add(new pair(dest, dist));
            graph.put(srcF, existing);
          } else {
            List < pair > newNodes = new ArrayList < > ();
            newNodes.add(new pair(dest, dist));
            graph.put(srcF, newNodes);
           }
        }

        int  result  =  cheapest(0, k, src) ;
        return result == Integer.MAX_VALUE ? -1 : res;
      }

      private int cheapest(int currStop, int maxStops, int currNode) {
        // base case

        if (maxStops == 0) {
         for (pair p :  graph.get(currNode)){
              if (p.node == dst) return p.dist;
          }
        }
        if (currNode == dst) return 0;
        if (currStop == maxStops) return -1;

        int currCost = 0;
        for (pair p :  graph.get(currNode)){
            int price = cheapest(currStop + 1, maxStops, p.node);
            if  (price != -1){
              currCost  = p.dist + price;
              currCost += p.dist; // add last edge cost
              minCost = Math.min(minCost, currCost);
            }
        }
        return minCost;
      }

      class pair {
        int node;
        int dist;
        pair(int node, int dist) {
          this.node = node;
          this.dist = dist;
        }

      }
    }
         */
        Map<Integer, List<int[]>> map;
        Integer dp[][];

        int find(int src, int dest, int k) {
            if (k < 0) return 1000_000_00;
            if (src == dest) return 0;
            if (dp[src][k] != null) return dp[src][k];
            int max = 1000_000_00;
            for (int arr[] : map.getOrDefault(src, new ArrayList<>())) {
                max = Math.min(max, arr[1] + find(arr[0], dest, k - 1));
            }
            return dp[src][k] = max;
        }

        public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
            map = new HashMap<>();
            dp = new Integer[n + 1][K + 2];
            for (int a[] : flights) {
                map.computeIfAbsent(a[0], k -> new ArrayList<>());
                map.get(a[0]).add(new int[]{a[1], a[2]});
            }
            int temp = find(src, dst, K + 1);
            return temp >= 1000_000_00 ? -1 : temp;
        }
    }

    // TODO: Solve it again
    class WaysToArriveAtDestnation {
        public int countPaths(int n, int[][] roads) {
            final List<List<Node>> graph = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList());
            }
            for (final int[] arr : roads) {
                graph.get(arr[0]).add(new Node(arr[1], arr[2]));
                graph.get(arr[1]).add(new Node(arr[0], arr[2]));
            }
            return this.dfs(graph, n);
        }

        public int dfs(final List<List<Node>> adj, int n) {
            final int mod = 1_000_000_007;
            final Queue<Node> queue = new PriorityQueue<>(n);
            final long[] costs = new long[n];
            final long[] ways = new long[n];
            final boolean[] cache = new boolean[n];
            queue.add(new Node(0, 0));
            Arrays.fill(costs, Long.MAX_VALUE);
            costs[0] = 0;
            //one way to visit first node
            ways[0] = 1;
            while (!queue.isEmpty()) {
                final Node currentNode = queue.poll();
                if (currentNode.cost > costs[currentNode.position] || cache[currentNode.position]) {
                    continue;
                }
                for (final Node vertex : adj.get(currentNode.position)) {
                    if (costs[currentNode.position] + vertex.cost < costs[vertex.position]) {
                        costs[vertex.position] = costs[currentNode.position] + vertex.cost;
                        ways[vertex.position] = ways[currentNode.position] % mod;
                        queue.add(new Node(vertex.position, costs[vertex.position]));
                    } else if (costs[currentNode.position] + vertex.cost == costs[vertex.position]) {
                        ways[vertex.position] = (ways[vertex.position] + ways[currentNode.position]) % mod;
                    }
                }
            }
            return (int) ways[n - 1];
        }

        @SuppressWarnings("ALL")
        private  class Node implements Comparable<Node> {
            int position;
            long cost;

            public Node(int dis, long val) {
                this.position = dis;
                this.cost = val;
            }


            @Override
            public int compareTo(final Node o) {
                return Long.compare(this.cost, o.cost);
            }
        }
    }


    // TODO: Revisit
    class MaxMatrixSum {
        public long maxMatrixSum(int[][] matrix) {
            int m = matrix.length;
            int n = matrix[0].length;
            int mini = Integer.MAX_VALUE;
            int cnt = 0;
            long sum = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    sum += Math.abs(matrix[i][j]);
                    if (matrix[i][j] < 0) cnt++;
                    mini = Math.min(mini, Math.abs(matrix[i][j]));
                }
            }
            if (cnt % 2 == 0) return sum;
            return sum - 2 * mini;
        }
    }

    class WordladderLengthBFS {
        public int ladderLength(String start, String end, List<String> wordList) {
            Set<String> dict = new HashSet<>(wordList);

            dict.add(start);
            Map<String, Integer> distances = new HashMap<>();
            return bfsHelper(start, end, dict, distances);

        }

        // use BFS, and traverse each word and its neighbors
        // if we find any word equals to end, we've return the "level" -- distances -- length
        private int bfsHelper(String start,
                              String end,
                              Set<String> dict,
                              Map<String, Integer> distances) {
            Queue<String> queue = new LinkedList<>();
            queue.offer(start);
            distances.put(start, 0); // distance keep track of how far it is from start

            while (!queue.isEmpty()) {
                String word = queue.poll();
                int distance = distances.get(word);

                for (String nextWord : getNextWords(word, dict)) {
                    if (!distances.containsKey(nextWord)) {
                        queue.offer(nextWord);
                        distances.put(nextWord, distance + 1);
                    }
                }
            }

            printMap(distances);

            if (distances.containsKey(end)) {
                return distances.get(end) + 1;
            }
            return 0;
        }

        private void printMap(Map<String, Integer> visited) {
            for (String s : visited.keySet()) {
                System.out.println(s + " : " + visited.get(s));
            }
        }

        private List<String> getNextWords(String word, Set<String> dict) {
            List<String> res = new ArrayList<>();
            for (int i = 0; i < word.length(); i++) {
                char ch = word.charAt(i);
                for (int j = 0; j < 26; j++) {
                    if (ch != (char) ('a' + j)) {
                        // substitute word.charAt(i) using (char)('a' + j)
                        String nextWord = word.substring(0, i) + (char) ('a' + j) + word.substring(i + 1);
                        if (dict.contains(nextWord)) {
                            res.add(nextWord);
                        }
                    }
                }
            }
            return res;
        }
    }

    class BoundaryPathsTree {

        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};
        int M = 1000000000 + 7;
        int m, n;

        public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
            this.m = m;
            this.n = n;
            int[][][] dp = new int[m][n][maxMove + 1];
            for (int[][] l : dp)
                for (int[] sl : l) Arrays.fill(sl, -1);

            return recursive(m, n, startRow, startColumn, dp, maxMove);
        }

        // this function will tell that the current node is safe to be visited or not
        private boolean isSafe(int r, int c) {
            return (r < m && c < n && r >= 0 && c >= 0);
        }

        private int recursive(int m, int n, int r, int c, int[][][] dp, int maxMove) {
            // base cases
            if (!isSafe(r, c)) return 1;
            if (maxMove == 0) return 0;

            if (dp[r][c][maxMove] >= 0) return dp[r][c][maxMove];

            dp[r][c][maxMove] = (
                    (recursive(m, n, r + 1, c, dp, maxMove - 1)
                            + recursive(m, n, r - 1, c, dp, maxMove - 1)) % M
                            + (recursive(m, n, r, c + 1, dp, maxMove - 1)
                            + recursive(m, n, r, c - 1, dp, maxMove - 1)) % M
            ) % M;

            return dp[r][c][maxMove];
        }
    }


//TODO: Revisit

    class UniquePathsIIIBFSGrid {
        int[] dx = {0, 1, 0, -1};
        int[] dy = {1, 0, -1, 0};
        boolean[][] visited;
        int[][] obstacleGrid;
        int m, n, space = 0;
        int paths = 0;
        // find src and dest points
        Point src, dest;

        public int uniquePathsIII(int[][] obstacleGrid) {
            m = obstacleGrid.length;
            n = obstacleGrid[0].length;

            if (m == 1 && n == 1) {
                if (obstacleGrid[0][0] == 0) return 1;
                else return 0;
            }
            this.src = new Point();
            this.dest = new Point();

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (obstacleGrid[i][j] == 1) {
                        this.src.x = i;
                        this.src.y = j;
                    } else if (obstacleGrid[i][j] == 2) {
                        this.dest.x = i;
                        this.dest.y = j;
                    } else if (obstacleGrid[i][j] == 0) {
                        space++;
                    }
                }
            }

            this.obstacleGrid = obstacleGrid;
            visited = new boolean[m][n];
            int[][] dp = new int[m][n];
            for (int[] rows : dp) Arrays.fill(rows, -1);
            recursive(m, n, this.src.x, this.src.y, 0);
            // For a single path once found paths=0 is done to reset the path again and evaluate it
            int k = paths;
            paths = 0;
            return k;
        }

        // this function will tell that the current node is safe to be visited or not
        private boolean isSafe(int r, int c) {
            return (r < m && c < n && r >= 0 && c >= 0 && (obstacleGrid[r][c] != -1));
        }

        private void recursive(int m, int n, int r, int c, int cells) {
            // base cases
            if (r == this.dest.x && c == this.dest.y && cells == space) {
                paths++;
                return;
            } else if (!isSafe(r, c) || visited[r][c]) return;
            for (int i = 0; i < 4; i++) {
                int x = r + dx[i];
                int y = c + dy[i];
                if (isSafe(x, y) && !visited[x][y]) {
                    visited[r][c] = true;
                    if (obstacleGrid[x][y] == 0) cells += 1;
                    recursive(m, n, x, y, cells);
                    if (obstacleGrid[x][y] == 0) cells -= 1;
                    visited[r][c] = false; // backtrack
                }
            }
            return;
        }

        class Point {
            int x;
            int y;

            Point(int x, int y) {
                this.x = x;
                this.y = y;
            }

            Point() {
            }
        }
    }

    //TODO: Revisit
    class SQRTBinarySeach {
        public int mySqrt(int num) {

            if (num < 2) {
                return num;
            }

            long start = 2;
            long end = num;
            while (start <= end) {
                long mid = start + (end - start) / 2;
                if (mid * mid == num) return (int) mid;
                else if (mid * mid < num) start = mid + 1;
                else end = mid - 1;
            }
            return (int) (start - 1);
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


        // This is based on binary search of peak element
        // Input: nums = [1,2,3,1]
        // Output: 2
        // Explanation: 3 is a peak element and your function should return the index number 2.
        // TC = O(logn), SC = O(1)
        public int findPeakElement(int[] nums) {
            int n = nums.length;
            int l = 0, r = n - 1;
            while (l < r) {
                int m = l + (r - l) / 2;
                if (nums[m] < nums[m + 1]) l = m + 1;
                else r = m;
            }
            return l;
        }

        // O(logn)
        // We have to iteratively partition array into haves and evavluate which half is sorted
        // and then apply BS for the element
        public int search(int[] nums, int target) {
            int n = nums.length;
            int l = 0, r = n - 1;

            // Find 2 halves of array
            while (l <= r) {
                int mid = l + (r - l) / 2;
                if (nums[mid] == target) return mid;
                // First half is sorted and other half is unsorted
                if (nums[l] <= nums[mid]) {
                    if (target >= nums[l] && target < nums[mid]) {
                        r = mid - 1;
                    } else l = mid + 1;
                }
                // Second half is sorted and other half is unsorted
                else {
                    if (target > nums[mid] && target <= nums[r]) {
                        l = mid + 1;
                    } else r = mid - 1;
                }
            }
            return -1;

        }

        // PQ + BFS
        // This is a astandard application for PQ and application of BFS
        //[[u,v] = meeting times list]
        Map<Integer, Map<Integer, List<Integer>>> g = new HashMap<>();

        public List<Integer> findAllPeople(int n, int[][] meetings, int firstPerson) {

            int[] notified = new int[n];
            Arrays.fill(notified, -1);
            for (int i = 0; i < n; i++) g.put(i, new HashMap<>());
            for (int[] m : meetings) getMeetingTimes(m[0], m[1]).add(m[2]);
            List<Integer> res = new ArrayList<>();

            PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
            pq.offer(new int[]{
                    0,
                    0
            });
            pq.offer(new int[]{
                    firstPerson,
                    0
            });

            while (!pq.isEmpty()) {
                int[] cur = pq.poll();
                int u = cur[0];
                if (notified[u] > -1) continue; //already visited
                notified[u] = cur[1];
                res.add(u);
                for (Map.Entry<Integer, List<Integer>> entry : g.get(u).entrySet()) {
                    int v = entry.getKey();
                    if (notified[v] > -1) continue; // already visited
                    Collections.sort(entry.getValue());
                    for (int time : entry.getValue()) {
                        //get first time u and v met after person u learned the secret
                        if (time >= notified[u]) {
                            pq.offer(new int[]{
                                    v,
                                    time
                            });
                            break;
                        }
                    }
                }
                g.remove(u); //u was fully processed
            }

            return res;
        }

        private List<Integer> getMeetingTimes(int u, int v) {
            List<Integer> times = g.get(u).get(v);
            if (times == null) {
                times = new ArrayList<>();
                g.get(u).put(v, times);
                g.get(v).put(u, times);
            }
            return times;
        }


        // TC = O(2^n), SC = O(2^n)
        // this is based on generating all possivble permuatation of an array of desired length
//    via backtracking
        Set<Integer> set = new HashSet<>();

        public int[] findEvenNumbers(int[] digits) {
            int n = digits.length;
            boolean[] seen = new boolean[n];
            permute(0, digits, "", seen);
            return set.stream().mapToInt(x -> x).sorted().toArray();
        }

        private void permute(int idx, int[] digits, String curr, boolean[] seen) {

            //base cases
            if (idx == 3) {
                set.add(Integer.parseInt(curr));
                return;
            }

            for (int i = 0; i < digits.length; i++) {
                if (seen[i] || (idx == 0 && digits[i] == 0) || (idx == 2 && digits[i] % 2 != 0)) continue;
                seen[i] = true;
                permute(idx + 1, digits, curr + digits[i], seen);
                seen[i] = false;
            }
        }


        // TC = (n/2), SC = O(1)
        public ListNode deleteMiddle(ListNode head) {
            if (head.next == null) return null;
            ListNode slow = head;
            ListNode fast = head.next.next;
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
            }
            slow.next = slow.next.next;
            return head;
        }


        private ListNode deleteMiddleSimpleApproach(ListNode head) {
            if (head == null || head.next == null) return null;
            ListNode curr = head;
            ListNode prev = null;
            ListNode slow = head;
            ListNode fast = head;
            while (fast != null && fast.next != null) {
                prev = curr;
                slow = slow.next;
                fast = fast.next.next;
            }
            prev.next = curr.next;
            return head;
        }


        // LCA based solution
        public TreeNode findLCA(TreeNode node, int s, int d) {
            if (node == null) return null;

            if (node.val == s || node.val == d) return node;

            TreeNode left = findLCA(node.left, s, d);
            TreeNode right = findLCA(node.right, s, d);
            if (left != null && right != null) return node;
            if (left == null && right != null) return right;
            else return left;
        }

        public String getDirections(TreeNode root, int startValue, int destValue) {
            TreeNode CA = findLCA(root, startValue, destValue);
            StringBuilder ans = new StringBuilder();
            ArrayDeque<String> q1 = new ArrayDeque<>();
            helper(CA, startValue, q1);
            ArrayDeque<String> q2 = new ArrayDeque<>();
            helper(CA, destValue, q2);

            for (int i = 0; i < q1.size(); i++) ans.append("U");
            while (!q2.isEmpty()) ans.append(q2.poll());

            return ans.toString();
        }

        private boolean helper(TreeNode n, int v, ArrayDeque<String> q) {
            if (n == null) return false;
            if (n.val == v) return true;

            q.offer("L");
            boolean left = helper(n.left, v, q);
            if (left) return true;
            q.removeLast();

            q.offer("R");
            boolean right = helper(n.right, v, q);
            if (right) return true;
            q.removeLast();

            return false;
        }

        //Represent the head and tail of the singly linked list
        ListNode curr = null;
        ListNode tail = null;

        //addNode() will add a new node to the list
        public void addNode(int data) {
            //Create a new node
            ListNode newNode = new ListNode(data);

            //Checks if the list is empty
            if (curr == null) {
                //If list is empty, both head and tail will point to new node
                curr = newNode;
                tail = newNode;
            } else {
                //newNode will be added after tail such that tail's next will point to newNode
                tail.next = newNode;
                //newNode will become new tail of the list
                tail = newNode;
            }
        }

        // Author: Anand
        public ListNode mergeNodes(ListNode head) {
            int sum = 0;
            int zero = 0;

            while (head != null) {
                sum += head.val;
                if (head.val == 0) {
                    zero++;
                }
                if (zero == 2) {
                    addNode(sum);
                    zero = 1;
                    sum = 0;
                }
                head = head.next;
            }


            return curr;
        }
    }

    class combinationSum4BottomUpDP {
        // TC = O(n*target), SC = O(1)
        // Bottom up approach
        public int combinationSum4(int[] nums, int target) {
            int n = nums.length;
            int[] dp = new int[target + 1];
            dp[0] = 1; // Do not take any element at all

            for (int i = 1; i <= target; i++) {
                for (int num : nums)
                    if (i >= num) dp[i] += dp[i - num];
            }
            return dp[target];
        }
    }
}


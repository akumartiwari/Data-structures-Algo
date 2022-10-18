package com.company;

import java.util.*;

//import for Scanner and other utility classes
public class DynamicArray {
    int size; // max size of array
    int index; // count number of elements of array
    HashMap<Character, Integer> map = new HashMap<>();
    private int[] array; // original array

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
            if (size >= 0) System.arraycopy(array, 0, temp, 0, size);
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

            System.arraycopy(array, 0, temp, 0, array.length);
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
    Input: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
Output: 3
E
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
    [[0,1],[1,0]]
    [[1,0],[0,1]]
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
            System.arraycopy(nmatrix[i], 0, matrix[i], 0, n);
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

    //"111000"

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



    /*
       index =1
       index =2
       index=3
     */

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
}


/*
Input: arr = [1,2,3,4]
Output: "23:41"

 */


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



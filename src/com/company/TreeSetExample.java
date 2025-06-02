package com.company;

import java.util.HashMap;
import java.util.*;

public class TreeSetExample {

    /*
    Input
    ["NumberContainers", "find", "change", "change", "change", "change", "find", "change", "find"]
    [[], [10], [2, 10], [1, 10], [3, 10], [5, 10], [10], [1, 20], [10]]
    Output
    [null, -1, null, null, null, null, 1, null, 2]

    Explanation
    NumberContainers nc = new NumberContainers();
    nc.find(10); // There is no index that is filled with number 10. Therefore, we return -1.
    nc.change(2, 10); // Your container at index 2 will be filled with number 10.
    nc.change(1, 10); // Your container at index 1 will be filled with number 10.
    nc.change(3, 10); // Your container at index 3 will be filled with number 10.
    nc.change(5, 10); // Your container at index 5 will be filled with number 10.
    nc.find(10); // Number 10 is at the indices 1, 2, 3, and 5. Since the smallest index that is filled with 10 is 1, we return 1.
    nc.change(1, 20); // Your container at index 1 will be filled with number 20. Note that index 1 was filled with 10 and then replaced with 20.
    nc.find(10); // Number 10 is at the indices 2, 3, and 5. The smallest index that is filled with 10 is 2. Therefore, we return 2.

     */
    class NumberContainers {

        Map<Integer, TreeSet<Integer>> map;
        Map<Integer, Integer> iN;

        public NumberContainers() {
            map = new HashMap<>();
            iN = new HashMap<>();
        }

        public void change(int index, int number) {
            if (iN.containsKey(index)) {
                map.get(iN.get(index)).remove(index);
            }

            iN.put(index, number);

            if (!map.containsKey(number)) map.put(number, new TreeSet<>());
            map.get(number).add(index);
        }

        //O(1)
        public int find(int number) {
            return map.containsKey(number) && map.get(number).size() > 0 ? map.get(number).first() : -1;
        }
    }

    //Author: Anand
    class FoodRatings {

        Map<String, TreeMap<Integer, TreeSet<String>>> CRF = new HashMap<>();  // {c, (r,f)}
        Map<String, Integer> FR = new HashMap<>(); // {f,r}
        Map<String, String> FC = new HashMap<>(); // {f,c}

        public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
            for (String cuisine : cuisines) {
                CRF.put(cuisine, new TreeMap<>());
            }

            for (int i = 0; i < foods.length; i++) FR.put(foods[i], ratings[i]);

            for (int i = 0; i < cuisines.length; i++) {
                if (CRF.containsKey(cuisines[i])) {
                    if (CRF.get(cuisines[i]).get(ratings[i]) != null)
                        CRF.get(cuisines[i]).get(ratings[i]).add(foods[i]);
                    else CRF.get(cuisines[i]).put(ratings[i], new TreeSet<>(Collections.singleton(foods[i])));
                } else {
                    TreeMap<Integer, TreeSet<String>> tm = new TreeMap<>();
                    tm.put(ratings[i], new TreeSet<>(Collections.singleton(foods[i])));
                    CRF.put(cuisines[i], tm);
                }
            }
            for (int i = 0; i < foods.length; i++) FC.put(foods[i], cuisines[i]);

        }

        public void changeRating(String food, int newRating) {
            if (FR.containsKey(food) && CRF.get(FC.get(food)).get(FR.get(food)) != null)
                CRF.get(FC.get(food)).get(FR.get(food)).remove(food);

            if (CRF.get(FC.get(food)).get(FR.get(food)).size() == 0) CRF.get(FC.get(food)).remove(FR.get(food));

            FR.put(food, newRating);
            if (!CRF.containsKey(FC.get(food))) CRF.put(FC.get(food), new TreeMap<>());

            if (CRF.get(FC.get(food)).get(newRating) != null) CRF.get(FC.get(food)).put(newRating, new TreeSet<>());
            CRF.get(FC.get(food)).get(newRating).add(food);
        }

        public String highestRated(String cuisine) {
            return CRF.containsKey(cuisine) && CRF.get(cuisine).size() > 0 && CRF.get(cuisine).lastEntry().getValue().size() > 0 ? CRF.get(cuisine).lastEntry().getValue().first() : null;
        }
    }

    /**
     * Your FoodRatings object will be instantiated and called as such:
     * FoodRatings obj = new FoodRatings(foods, cuisines, ratings);
     * obj.changeRating(food,newRating);
     * String param_2 = obj.highestRated(cuisine);
     */


    /*
    Input: nums = [1,2,3,1], k = 3
    Output: 5
    Explanation: The excellent pairs are the following:
    - (3, 3). (3 AND 3) and (3 OR 3) are both equal to (11) in binary. The total number of set bits is 2 + 2 = 4, which is greater than or equal to k = 3.
    - (2, 3) and (3, 2). (2 AND 3) is equal to (10) in binary, and (2 OR 3) is equal to (11) in binary. The total number of set bits is 1 + 2 = 3.
    - (1, 3) and (3, 1). (1 AND 3) is equal to (01) in binary, and (1 OR 3) is equal to (11) in binary. The total number of set bits is 1 + 2 = 3.
    So the number of excellent pairs is 5.
     */
    // TC = O(32*n)
    // The idea is to avoid solving the problem in O(n2) by using map
    // We have used map to store nr of set bits of all numbers grouped together
    // then used vis Set to avoid duplication of them.
    public long countExcellentPairs(int[] nums, int k) {
        long ans = 0L;

        TreeMap<Integer, TreeSet<Integer>> map = new TreeMap<>(); // {nr. of set  bits, nr}

        for (int num : nums) {
            if (!map.containsKey(Integer.bitCount(num))) map.put(Integer.bitCount(num), new TreeSet<>());
            map.get(Integer.bitCount(num)).add(num);
        }

        Set<Integer> vis = new HashSet<>();

        for (int num : nums) {
            if (vis.contains(num)) continue;

            int need = k - Integer.bitCount(num);

            // O(32) as largest num,ber can be of 32 bits
            for (Map.Entry<Integer, TreeSet<Integer>> entry : map.entrySet())
                if (entry.getKey() >= need) ans += entry.getValue().size();

            vis.add(num);
        }
        return ans;
    }

    public long sumOfLargestPrimes(String s) {
        Set<Long> pq = new TreeSet<>(Collections.reverseOrder()); // max pq

        int n = s.length();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                String subString = s.substring(i, j);
                long number = Long.parseLong(subString);
                if (isPrime(number)) pq.add(number);
            }
        }

        long sum = 0, cnt = 0;
        for (long e : pq) {
            sum += e;
            if (++cnt == 3) break;
        }
        return sum;
    }

    public boolean isPrime(long n) {
        if (n == 1) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i <= Math.sqrt(n); i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    public int maxSubstrings(String word) {
        Set<Long> pq = new TreeSet<>(Collections.reverseOrder()); // max pq

        int n = word.length();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                String subString = word.substring(i, j);
                long number = Long.parseLong(subString);
                if (isPrime(number)) pq.add(number);
            }
        }

        long sum = 0, cnt = 0;
        for (long e : pq) {
            sum += e;
            if (++cnt == 3) break;
        }
        return (int) sum;
    }


    /*

    abcdeafdef

     */

    //TODO
    class Solution {
        public int maxSubstrings(String word) {
            Map<Character, List<Integer>> map = new LinkedHashMap<>();
            int cnt = 0;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (!map.containsKey(c)) map.put(c, new ArrayList<>(Collections.singletonList(i)));
                else {
                    int li = map.get(c).get(0);
                    if (i-li >= 4) {
                        cnt++;
                        System.out.println(word.substring(li,i+1));
                        map.put(c, new ArrayList<>(Arrays.asList()));
                    }
                }
            }
            return cnt;
        }
    }
}

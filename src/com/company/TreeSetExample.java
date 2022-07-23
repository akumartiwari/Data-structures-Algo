package com.company;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;

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

            if (map.containsKey(number)) map.put(number, new TreeSet<>());
            map.get(number).add(index);
        }

        //O(1)
        public int find(int number) {
            return map.containsKey(number) && map.get(number).size() > 0 ? map.get(number).first() : -1;
        }
    }

    /**
     * Your NumberContainers object will be instantiated and called as such:
     * NumberContainers obj = new NumberContainers();
     * obj.change(index,number);
     * int param_2 = obj.find(number);
     */
}

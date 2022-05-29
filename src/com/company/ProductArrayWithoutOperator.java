package com.company;

import java.util.HashMap;

public class ProductArrayWithoutOperator {
    /*
     {1, 2, 3, 4};
     */
    public static void productArray(int[] arr) {
        int n = arr.length;
        if (n == 0) return;
        HashMap<Integer, Integer> map = new HashMap<>();
        // initialise the array
        map.put(arr[0], 1);
        int prod = 1;
        // store left half product
        for (int i = 1; i < n; i++) {
            prod *= arr[i - 1];
            map.put(arr[i], prod);
        }
        prod = 1;

        map.put(arr[n - 1], map.get(arr[n - 1]) * prod);
        // store right half product
        for (int i = n - 2; i >= 0; i--) {
            prod *= arr[i + 1];
            if (map.get(arr[i]) != null) {
                map.put(arr[i], map.get(arr[i]) * prod);
            } else map.put(arr[i], prod);
        }

        map.values().forEach(System.out::println);
    }
}

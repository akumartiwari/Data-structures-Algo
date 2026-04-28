package com.company;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentMap {

        /*
     * PROBLEM: Number of People Aware of a Secret (LeetCode 2327)
     * Simulate secret spreading over n days where each person shares after delay days and forgets after forget days.
     *
     * ALGORITHM: Simulation with ConcurrentHashMap
     * TC: O(n^2) | SC: O(n)
     */
    public int peopleAwareOfSecret(int n, int delay, int forget) {
        final int mod = 1_000_000_007;

        int i = 1;
        int discovery = 1;
        Map<Integer, String> store = new ConcurrentHashMap<Integer, String>();

        String key = discovery + "-" + (discovery + delay) + "-" + (discovery + forget);
        store.put(i, key);

        int day = 1;
        while (day <= n) {

            Iterator<Integer> it1 = store.keySet().iterator();

            while (it1.hasNext()) {
                Integer person = it1.next();
                System.out.println("Map Value:" + store.get(person));
                int[] arr = Arrays.stream(store.get(person).split("-")).mapToInt(Integer::parseInt).toArray();
                String nk = day + "-" + (day + delay) + "-" + (day + forget);

                if (day >= arr[1] && day < arr[2]) {
                    store.put(((i + 1) % mod), nk);
                }
                // if  days passed then this person will never able to generate new people
                if (day >= arr[2]) store.remove(person);
            }
            day++;
        }

        int max = -1;
        for (Map.Entry<Integer, String> entry : store.entrySet()) {
            max = Math.max(max, (int) entry.getKey());
            System.out.println(entry.getValue());
        }
        return max;
    }
}

package com.company;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentMap {

    /*
    To Avoid ConcurrentModificationException in multi-threaded environment

     - You can convert the list to an array and then iterate on the array.
     - This approach works well for small or medium size list but if the list is large then it will affect the performance a lot.
     - You can lock the list while iterating by putting it in a synchronized block. This approach is not recommended because it will cease the benefits of multithreading.
     - Concurrent Collection classes can be modified safely, they will not throw ConcurrentModificationException.
     - In case of CopyOnWriteArrayList, iterator doesn’t accommodate the changes in the list and works on the original list.

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
                int[] arr = Arrays.stream(store.get(person).split("-"))
                        .mapToInt(Integer::parseInt).toArray();
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

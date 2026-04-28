package com.company;

import java.util.LinkedHashMap;
import java.util.Map;

// TODO
class LRUCache extends LinkedHashMap<Integer, Integer> {
    private int capacity;

    /*
     * PROBLEM: LRU Cache (LeetCode 146)
     * Initialize LRU cache with given capacity using access-ordered LinkedHashMap.
     *
     * ALGORITHM: LinkedHashMap with access-order
     * TC: O(1) | SC: O(capacity)
     */
    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    /*
     * PROBLEM: LRU Cache get (LeetCode 146)
     * Retrieve value for key; returns -1 if absent.
     *
     * ALGORITHM: LinkedHashMap getOrDefault
     * TC: O(1) | SC: O(1)
     */
    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    /*
     * PROBLEM: LRU Cache put (LeetCode 146)
     * Insert/update key-value pair, evicting the LRU entry if over capacity.
     *
     * ALGORITHM: LinkedHashMap put
     * TC: O(1) | SC: O(1)
     */
    public void put(int key, int value) {
        super.put(key, value);
    }

    /*
     * PROBLEM: LRU Cache Eviction Policy (Helper)
     * Determines if the eldest entry should be removed when over capacity.
     *
     * ALGORITHM: Size check
     * TC: O(1) | SC: O(1)
     */
    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }

    // Solve it using doubly linked list
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */

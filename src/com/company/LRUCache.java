package com.company;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.Map;

// TODO
public class LRUCache {
    int capacity;
    Deque<Integer> cache;
    Map<Integer, Integer> map;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new LinkedHashMap<>();
        this.cache = new ArrayDeque<>();
    }

    public int get(int key) {
        if (map.containsKey(key)) {
            cache.remove(key);
            cache.offerLast(key);
            return map.get(key);
        }
        return -1;
    }

    public void put(int key, int value) {
        map.remove(key);
        cache.add(value);
        if (cache.size() > capacity) {
            int first = cache.stream().findFirst().get();
            cache.removeFirst();
            cache.offerLast(value);
            map.remove(first);
        }
        map.put(key, value);
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */

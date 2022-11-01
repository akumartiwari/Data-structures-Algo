package com.company;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

/*

     Problem statement :-
     ## Tree is given you need to tell the height of modified tree
       (after deletion iof subTree node for each query)

     ALGO-
     First of All you need to find the level, and height of each node of a tree and store them in a map
     Then store the 2 maximum node of each level

     Now start traversing in the queries array, suppose
     the node you wants to remove is not having the maximum height in its level,
     then the removal of that node, doesn't impact in height of tree,

     the node you wants to remove is having the maximum height in its level,
     then the removal of that node, impact in height of tree,
     it decreases the height by
     (height of that node - 2nd maximum height of that level)

     so in that case queries[i]=heightOf(queries[i])-2nd maximum height of that level;

  */
public class TreeHeightAfterSubtreeDeletion {
    Map<Integer, Integer> ht = new HashMap<>(); // Height of each node
    Map<Integer, Integer> level = new HashMap<>(); // Level of each node
    Map<Integer, Queue<Integer>> max2OnLevel = new HashMap<>(); // 2 max heights on each level

    public int[] treeQueries(TreeNode root, int[] queries) {

        int h = height(root, 0);
        for (int i = 0; i < queries.length; i++) {

            int node = queries[i];
            int lev = level.get(node);
            int hh = ht.get(node);

            int a = -1, b = -1;
            // If curr is having max height in that level

            if (max2OnLevel.get(lev).size() == 2) {
                b = max2OnLevel.get(lev).poll();
                a = max2OnLevel.get(lev).poll();

                if (hh == a) queries[i] = h - (a - b);
                else queries[i] = h;


                max2OnLevel.get(lev).add(a);
                max2OnLevel.get(lev).add(b);
            } else queries[i] = (h - hh - 1);

        }

        return queries;
    }

    private int height(TreeNode root, int l) {
        // base case
        if (root == null) return -1;
        max2OnLevel.putIfAbsent(l, new PriorityQueue<>());

        int left = 1 + height(root.left, l + 1);
        int right = 1 + height(root.right, l + 1);
        int maxh = Math.max(left, right);
        ht.put(root.val, maxh);
        level.put(root.val, l);
        // Add max height from left and right subtree on that level
        max2OnLevel.get(l).add(maxh);

        // Ony retain max 2 heights on that level
        if (max2OnLevel.get(l).size() > 2) max2OnLevel.get(l).remove();

        return maxh;
    }
}

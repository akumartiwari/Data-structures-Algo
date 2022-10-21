package Graph;

import java.util.*;

public class DSUPlusGreedy {
    public int componentValue(int[] nums, int[][] edges) {
        int totalSum = 0;

        Map<Integer, List<Integer>> graph = new HashMap<>();
        int[] indegree = new int[nums.length];

        for (int num : nums) totalSum += num;

        for (int[] edge : edges) {
            if (!graph.containsKey(edge[0])) graph.put(edge[0], new ArrayList<>());
            if (!graph.containsKey(edge[1])) graph.put(edge[1], new ArrayList<>());

            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
            indegree[edge[0]]++;
            indegree[edge[1]]++;
        }


        // We will greadily start with max number of components ie, traverse from right to left
        // and check if such configuration is achievable
        for (int i = nums.length; i > 1; i--) {

            // If we split the tree into i components, the target value of each component is sum / i.
            // Note that we cannot split into i components if sum % i != 0.
            if (totalSum % i == 0) {

                if (bfs(nums.clone(), totalSum / i, graph, indegree.clone())) {
                    // return i-1 as edges to be deleted == number of partition - 1
                    return i - 1;
                }
            }
        }

        return 0;
    }

    private boolean bfs(int[] nums, int target, Map<Integer, List<Integer>> graph, int[] indegree) {

        Deque<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < indegree.length; i++) {
            // start from leaves
            if (indegree[i] == 1) queue.addLast(i);
        }

        while (!queue.isEmpty()) {
            // leaf nodes
            int curr = queue.removeFirst();

            for (int adj : graph.get(curr)) {

                // if curr node itself exceeds target then this configuration is indeed impossible to obtain
                if (nums[curr] > target) return false;

                // nums[curr] < target is true it means adj can be taken in same component else make a cut
                nums[adj] += nums[curr] < target ? nums[curr] : 0;

                // In either cases indegree will be reduced by 1
                indegree[adj]--;

                // Repeat the same process
                if (indegree[adj] == 1) {
                    queue.addLast(adj);
                }

            }
        }


        return true;
    }
}

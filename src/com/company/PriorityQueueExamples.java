package com.company;

import javafx.util.Pair;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class PriorityQueueExamples {

    /*
    Input: nums = [2,4,-2], k = 5
    Output: 2
    Explanation: All the possible subsequence sums that we can obtain are the following sorted in decreasing order:
    - 6, 4, 4, 2, 2, 0, 0, -2.
    The 5-Sum of the array is 2.

    Based on idea to either include/exclude a +/- number  from max sum.
     */
    public long kSum(int[] nums, int k) {
        long sum = 0L;
        for (int num : nums) sum += Math.max(num, 0);

        for (int i = 0; i < nums.length; ++i) nums[i] = Math.abs(nums[i]);
        Arrays.sort(nums);

        //max PQ
        Queue<Pair<Long, Integer>> queue = new PriorityQueue<>((a, b) -> Long.compare(b.getKey(), a.getKey()));
        queue.offer(new Pair<>(sum - nums[0], 0));
        long result = sum;
        while (--k > 0) {
            Pair<Long, Integer> pair = queue.poll();
            result = pair.getKey();
            int idx = pair.getValue();
            if (idx < nums.length - 1) {
                queue.offer(new Pair<>(sum - nums[idx + 1] + nums[idx], idx + 1));
                queue.offer(new Pair<>(sum - nums[idx + 1], idx + 1));
            }
        }

        return result;
    }

    // Author: Anand
    public int halveArray(int[] nums) {
        PriorityQueue<BigDecimal> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int num : nums) pq.add(BigDecimal.valueOf((double) num).setScale(2, RoundingMode.HALF_UP));

        long ls = 0;
        for (int num : nums) ls += num;
        BigDecimal sum = BigDecimal.valueOf(ls);
        BigDecimal ns = sum;
        int cnt = 0;
        while (!pq.isEmpty()) {
            if (ns.compareTo(sum.divide(BigDecimal.valueOf(2))) <= 0) {
                return cnt;
            }

            BigDecimal greatest = pq.poll();
            ns = ns.subtract(greatest.divide(BigDecimal.valueOf(2)));
            pq.offer(greatest.divide(BigDecimal.valueOf(2)));
            cnt++;
        }
        return cnt;
    }

    // Author: Anand
    public int maximumProduct(int[] nums, int k) {
        int MOD = 1_000_000_000 + 7;

        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int num : nums) pq.offer(num);
        while (k > 0) {
            int num = pq.poll();
            num += 1;
            k--;
            pq.offer(num);
        }

        long prod = 1;
        while (!pq.isEmpty()) {
            prod = mod_mul(prod, pq.poll(), MOD);
        }
        return (int) prod;
    }

    public long mod_mul(long a, long b, long m) {
        a = a % m;
        b = b % m;
        return (((a * b) % m) + m) % m;
    }

    //***********

    Map<Integer, Long> map = new TreeMap<>(); // {roomId, count}


    /*
    Input: n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
    Output: 0
    Explanation:
    - At time 0, both rooms are not being used. The first meeting starts in room 0.
    - At time 1, only room 1 is not being used. The second meeting starts in room 1.
    - At time 2, both rooms are being used. The third meeting is delayed.
    - At time 3, both rooms are being used. The fourth meeting is delayed.
    - At time 5, the meeting in room 1 finishes. The third meeting starts in room 1 for the time period [5,10).
    - At time 10, the meetings in both rooms finish. The fourth meeting starts in room 0 for the time period [10,11).
    Both rooms 0 and 1 held 2 meetings, so we return 0.
     */
    public int mostBooked(int n, int[][] meetings) {

        // Sort rooms based on endTime of the meeting and roomId
        PriorityQueue<Node> pq = new PriorityQueue<Node>(
                (a, b) -> {
                    if (a.endTime != b.endTime)
                        return a.endTime - b.endTime;
                    return a.roomId - b.roomId;
                }
        );

        TreeSet<Integer> ar = new TreeSet<>();
        for (int i = 0; i < n; ++i) ar.add(i);

        Arrays.sort(meetings, Comparator.comparingInt(a -> a[0]));
        for (int[] meeting : meetings) {

            int start = meeting[0];
            int end = meeting[1];
            int room;
            // freeup the rooms which are available now
            while (!pq.isEmpty() && start >= pq.peek().endTime) ar.add(pq.poll().roomId);

            //Delay the meeting (as now rooms available)
            if (ar.size() == 0) {
                Node choice = pq.poll();
                end += choice.endTime - start;
                room = choice.roomId;
                pq.offer(new Node(room, end));
            } else {
                room = ar.pollFirst();
                pq.offer(new Node(room, end));
            }

            map.put(room, map.getOrDefault(room, 0L) + 1);
        }

        return maxCount();
    }


    private int maxCount() {
        int room = -1;
        long count = Long.MIN_VALUE;
        for (Map.Entry<Integer, Long> entry : map.entrySet()) {
            if (count < entry.getValue()) {
                room = entry.getKey();
                count = entry.getValue();
            }
        }
        return room;
    }

    class Node {
        int roomId;
        int endTime;


        Node(int roomId, int endTime) {
            this.roomId = roomId;
            this.endTime = endTime;
        }
    }

}

package com.company;

import javafx.util.Pair;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;import java.util.*;

public class PriorityQueueExamples {

    Map<Integer, Long> map = new TreeMap<>(); // {roomId, count}

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

    //***********

    public long mod_mul(long a, long b, long m) {
        a = a % m;
        b = b % m;
        return (((a * b) % m) + m) % m;
    }

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

    //O(n*log(n)) time and O(n*log(n)) space

    /*
    Input: nums = [8,12,6], target = [2,14,10]
    Output: 2
    Explanation: It is possible to make nums similar to target in two operations:
    - Choose i = 0 and j = 2, nums = [10,12,4].
    - Choose i = 1 and j = 2, nums = [10,14,2].
    It can be shown that 2 is the minimum number of operations needed.

     */
    public long makeSimilar(int[] nums, int[] target) {

        PriorityQueue<Integer> numsEven = new PriorityQueue<>();
        PriorityQueue<Integer> numsOdd = new PriorityQueue<>();
        PriorityQueue<Integer> targetEven = new PriorityQueue<>();
        PriorityQueue<Integer> targetOdd = new PriorityQueue<>();

        for (int num : nums)
            if (num % 2 == 0) numsEven.add(num);
            else numsOdd.add(num);
        for (int t : target)
            if (t % 2 == 0) targetEven.add(t);
            else targetOdd.add(t);

        long[] even = helper(numsEven, targetEven, 0L, 0L);
        long[] odd = helper(numsOdd, targetOdd, even[1], even[2]);

        return even[0] + odd[0];
    }

    private long[] helper(PriorityQueue<Integer> pqNums, PriorityQueue<Integer> pqTarget, long add, long sub) {
        //greedy solution
        long out = 0;

        while (!pqNums.isEmpty()) {
            //get the smallest numbers from nums and target and see if we can match them
            int num = pqNums.poll();
            int target = pqTarget.poll();

            if (num < target) {
                //calculate number of operations we need to use to make numbers similar and apply free adds we gained by previously using subs
                long diff = (target - num) / 2 - add;

                if (diff > 0) {
                    //how many operations we need to use
                    out += diff;
                    //no free adds
                    add = 0;
                    //add free subs we can use later
                    sub += diff;
                } else {
                    //still have some adds to use in the future (or 0)
                    add = -diff;
                }
            } else {
                long diff = (num - target) / 2 - sub;

                if (diff > 0) {
                    out += diff;
                    sub = 0;
                    add += diff;
                }
            }
        }

        return new long[]{out, add, sub};
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

    public int minGroups(int[][] intervals) {

        Arrays.sort(intervals, (a, b) -> {
            if (a[0] != b[0])
                return a[0] - b[0];
            return Math.min(a[1], b[1]);
        });

        PriorityQueue<TimeInterval> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.end));

        for (int[] interval : intervals) {

            int start = interval[0];
            int end = interval[1];
            if (!pq.isEmpty() && pq.peek().end < start) {
                TimeInterval ti = pq.poll();
                pq.offer(new TimeInterval(ti.start, end));
            } else {
                pq.offer(new TimeInterval(start, end));
            }
        }

        return pq.size();
    }

    /*
    Input: nums = [3,7,8,1,1,5], space = 2
    Output: 1
    Explanation: If we seed the machine with nums[3], then we destroy all targets equal to 1,3,5,7,9,...
    In this case, we would destroy 5 total targets (all except for nums[2]).
    It is impossible to destroy more than 5 targets, so we return nums[3].
     */
    public int destroyTargets(int[] nums, int space) {
        Arrays.sort(nums);
        Map<Integer, List<Integer>> freq = new HashMap<>(); // remainder, elements
        for (int num : nums) {
            int key = num % space;
            if (!freq.containsKey(key)) freq.put(key, new ArrayList<>());
            freq.get(key).add(num);
        }

        PriorityQueue<Pair<Integer, List<Integer>>> pq = new PriorityQueue<>((o1, o2) -> {
            if (o1.getValue().size() < o2.getValue().size()) return 1;
            if (o1.getValue().size() > o2.getValue().size()) return -1;
            return o1.getValue().get(0).compareTo(o2.getValue().get(0));
        }); //  remainder -> elements

        for (Map.Entry<Integer, List<Integer>> entry : freq.entrySet()) {
            pq.offer(new Pair<>(entry.getKey(), entry.getValue()));
        }

        return pq.poll().getValue().get(0);
    }

    public long totalCost(int[] costs, int k, int candidates) {
        // element, index
        PriorityQueue<Pair<Pair<Integer, Integer>, Boolean>> pq = new PriorityQueue<>((o1, o2) -> {
            if (o1.getKey().getKey() < o2.getKey().getKey()) return -1;
            if (o1.getKey().getKey() > o2.getKey().getKey()) return 1;
            return o1.getKey().getValue().compareTo(o2.getKey().getValue());
        }); //  element  -> index
        long minCost = 0L;

        for (int i = 0; i < costs.length; i++) {
            if (i >= candidates) break;

            pq.offer(new Pair<>(new Pair<>(costs[i], i), true));

            if (costs.length - 1 - i >= 0 && i < (costs.length - 1 - i)) {
                pq.offer(new Pair<>(new Pair<>(costs[costs.length - 1 - i], costs.length - 1 - i), false));
            }
        }

        int lif = candidates - 1;
        int lie = costs.length - candidates;

        while (k-- > 0 && !pq.isEmpty()) {
            Pair<Pair<Integer, Integer>, Boolean> elem = pq.poll();
            minCost += elem.getKey().getKey();

            int ni;

            // First half
            if (elem.getValue()) {
                ni = lif + 1;

                if (ni >= lie) continue;
                if (ni < costs.length) {
                    lif = ni;
                    pq.offer(new Pair<>(new Pair<>(costs[ni], ni), true));
                }
            } else {
                ni = lie - 1;
                if (ni <= lif) continue;
                if (ni >= 0) {
                    lie = ni;
                    pq.offer(new Pair<>(new Pair<>(costs[ni], ni), false));
                }
            }
        }

        return minCost;
    }

    class Node {
        int roomId;
        int endTime;


        Node(int roomId, int endTime) {
            this.roomId = roomId;
            this.endTime = endTime;
        }
    }

    /*
    Input: intervals = [[5,10],[6,8],[1,5],[2,3],[1,10]]
    Output: 3
    Explanation: We can divide the intervals into the following groups:
    - Group 1: [1, 5], [6, 8].
    - Group 2: [2, 3], [5, 10].
    - Group 3: [1, 10].
    It can be proven that it is not possible to divide the intervals into fewer than 3 groups.
     */
    //Author: Anand
    class TimeInterval {
        int start;
        int end;


        TimeInterval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }
}

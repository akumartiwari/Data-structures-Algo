package com.company;

public class GCD {

    /*

    There exists an infinitely large grid. You are currently at point (1, 1),
    and you need to reach the point (targetX, targetY) using a finite number of steps.

    In one step, you can move from point (x, y) to any one of the following points:

    (x, y - x)
    (x - y, y)
    (2 * x, y)
    (x, 2 * y)


    Input: targetX = 6, targetY = 9
    Output: false
    Explanation: It is impossible to reach (6,9) from (1,1) using any sequence of moves,
                 so false is returned.

     */
    public boolean isReachable(int targetX, int targetY) {
        while (targetX % 2 == 0) targetX /= 2;
        while (targetY % 2 == 0) targetY /= 2;
        return _gcd(targetX, targetY) == 1;
    }

    public long _gcd(long a, long b) {
        if (b == 0) return a;
        return _gcd(b, a % b);
    }
}

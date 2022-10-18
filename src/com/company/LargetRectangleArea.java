package com.company;

import java.util.Stack;

class LargetRectangleArea {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0)
            return 0;
        int[] heights = new int[matrix[0].length];

        for (int i = 0; i < matrix[0].length; i++)
            heights[i] = matrix[0][i] - '0';

        int maxArea = largestRectangleArea(heights);

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1')
                    heights[j]++;
                else heights[j] = 0;
            }
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }
        return maxArea;

    }

    int largestRectangleArea(int[] heights) {
        Stack<Integer> st = new Stack<>();
        int maxArea = 0;
        st.push(-1);
        for (int i = 0; i <= heights.length; i++) {
            int val = i == heights.length ? 0 : heights[i];

            while (st.peek() != -1 && heights[st.peek()] >= val) {
                int rightMin = i;
                int height = heights[st.pop()];
                int leftMin = st.peek();
                maxArea = Math.max(maxArea, height * (rightMin - leftMin - 1));
            }
            st.push(i);
        }
        return maxArea;
    }
}

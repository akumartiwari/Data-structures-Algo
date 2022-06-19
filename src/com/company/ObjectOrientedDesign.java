package com.company;

public class ObjectOrientedDesign {
    //Author: Anand
    class TextEditor {

        int cursorPos;
        StringBuilder sb;

        public TextEditor() {
            cursorPos = 0;
            sb = new StringBuilder();
        }

        public void addText(String text) {
            sb.insert(cursorPos, text);
            cursorPos += text.length();
        }

        public int deleteText(int k) {
            int min = Math.min(k, cursorPos);
            cursorPos -= min;
            sb.delete(cursorPos, cursorPos + min);
            return min;
        }

        public String cursorLeft(int k) {
            int min = Math.min(k, cursorPos);
            cursorPos -= min;
            return cursorPos < 10 ? sb.substring(0, cursorPos) : sb.substring(cursorPos - 10, cursorPos);
        }

        public String cursorRight(int k) {
            cursorPos = Math.min(sb.length(), cursorPos + k);
            return cursorPos < 10 ? sb.substring(0, cursorPos) : sb.substring(cursorPos - 10, cursorPos);
        }
    }

    /**
     * Your TextEditor object will be instantiated and called as such:
     * TextEditor obj = new TextEditor();
     * obj.addText(text);
     * int param_2 = obj.deleteText(k);
     * String param_3 = obj.cursorLeft(k);
     * String param_4 = obj.cursorRight(k);
     */
}

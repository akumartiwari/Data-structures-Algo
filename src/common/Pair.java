package common;

/**
 * Generic immutable key-value pair.
 * Replaces javafx.util.Pair which is unavailable without the JavaFX SDK.
 */
public class Pair<K, V> {
    private final K key;
    private final V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey()   { return key; }
    public V getValue() { return value; }

    @Override
    public String toString() { return "(" + key + ", " + value + ")"; }
}

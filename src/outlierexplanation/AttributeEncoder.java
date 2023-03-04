package outlierexplanation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import outlierexplanation.macrobase.util.ModBitSet;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Encode every combination of attribute names and values into a distinct integer.
 * This class assumes that attributes are stored in String columns in dataframes
 * and is used inside of the explanation operators to search for explanatory
 * column values.
 */
public class AttributeEncoder {
    private Logger log = LoggerFactory.getLogger("AttributeEncoder");
    // An encoding for values which do not satisfy the minimum support threshold in encodeAttributesWithSupport.
    public static int noSupport = Integer.MAX_VALUE;
    public static int cardinalityThreshold = 128;

    private HashMap<Integer, Map<String, Integer>> encoder;
    private int nextKey;

    private HashMap<Integer, String> valueDecoder;
    private HashMap<Integer, Integer> columnDecoder;
    private List<String> colNames;
    private HashMap<Integer, ModBitSet>[][] bitmap;
    private int[] colCardinalities;
    private ArrayList<Integer> outlierList[];

    public AttributeEncoder() {
        encoder = new HashMap<>();
        // Keys must start at 1 because IntSetAsLong does not accept zero values.
        nextKey = 1;
        valueDecoder = new HashMap<>();
        columnDecoder = new HashMap<>();
    }
    public void setColumnNames(List<String> colNames) {
        this.colNames = colNames;
    }

    public int decodeColumn(int i) {return columnDecoder.get(i);}
    public String decodeColumnName(int i) {return colNames.get(columnDecoder.get(i));}
    public String decodeValue(int i) {return valueDecoder.get(i);}
    public HashMap<Integer, Integer> getColumnDecoder() {return columnDecoder;}
    public HashMap<Integer, ModBitSet>[][] getBitmap() {return bitmap;}
    public ArrayList<Integer>[] getOutlierList() {return outlierList;}
    public int[] getColCardinalities() {return colCardinalities;}

    /**
     * Encode as integers all attributes satisfying a minimum support threshold.  Also
     * encode columns of attributes as bitmaps if their cardinalities are sufficiently
     * low.
     * @param columns A list of columns of attributes.
     * @param minSupport The minimal support an attribute must have to be encoded.
     * @param outlierColumn A column indicating whether a row of attributes is an inlier
     *                      our outlier.
     * @param useBitmaps Whether to encode any columns as bitmaps.
     * @return The encoded matrix of attributes, stored as an array of arrays.
     */
    public int[][] encodeAttributesWithSupport(List<String[]> columns, double minSupport,
                                               double[] outlierColumn, boolean useBitmaps) {
        if (columns.isEmpty()) {
            return new int[0][0];
        }

        int numColumns = columns.size();
        int numRows = columns.get(0).length;

        for (int i = 0; i < numColumns; i++) {
            if (!encoder.containsKey(i)) {
                encoder.put(i, new HashMap<>());
            }
        }
        // Create a map from strings to the number of times
        // each string appears in an outlier.
        int numOutliers = 0;
        HashMap<String, Double> countMap = new HashMap<>();
        for (int colIdx = 0; colIdx < numColumns; colIdx++) {
            String[] curCol = columns.get(colIdx);
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                if (outlierColumn[rowIdx] > 0.0) {
                    if (colIdx == 0)
                        numOutliers += outlierColumn[rowIdx];
                    // prepend column index as String to column value to disambiguate
                    // between two identical values in different columns
                    String colVal = Integer.toString(colIdx) + curCol[rowIdx];
                    Double curCount = countMap.get(colVal);
                    if (curCount == null)
                        countMap.put(colVal, outlierColumn[rowIdx]);
                    else
                        countMap.put(colVal, curCount + outlierColumn[rowIdx]);
                }
            }
        }

        // Rank the strings that have minimum support among the outliers
        // by the amount of support they have.
        double minSupportThreshold = minSupport * numOutliers;
        List<String> filterOnMinSupport = countMap.keySet().stream()
                .filter(line -> countMap.get(line) >= minSupportThreshold)
                .collect(Collectors.toList());
        filterOnMinSupport.sort((s1, s2) -> countMap.get(s2).compareTo(countMap.get(s1)));

        HashMap<String, Integer> stringToRank = new HashMap<>(filterOnMinSupport.size());
        for (int i = 0; i < filterOnMinSupport.size(); i++) {
            // We must one-index ranks because IntSetAsLong does not accept zero values.
            stringToRank.put(filterOnMinSupport.get(i), i + 1);
        }

        // Encode the strings that have support with a key equal to their rank.
        int[][] encodedAttributes = new int[numRows][numColumns];
        bitmap = new HashMap[numColumns][2];
        for (int i = 0; i < numColumns; ++i) {
            for (int j = 0; j < 2; j++)
                bitmap[i][j] = new HashMap<>();
        }
        outlierList = new ArrayList[numColumns];
        for (int i = 0; i < numColumns; i++)
            outlierList[i] = new ArrayList<>();
        colCardinalities = new int[numColumns];

        for (int colIdx = 0; colIdx < numColumns; colIdx++) {
            Map<String, Integer> curColEncoder = encoder.get(colIdx);
            String[] curCol = columns.get(colIdx);
            HashSet<Integer> foundOutliers = new HashSet<>();
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                String colVal = curCol[rowIdx];
                // Again, prepend column index as String to column value to disambiguate
                // between two identical values in different columns
                String colNumAndVal = Integer.toString(colIdx) + colVal;
                int oidx = (outlierColumn[rowIdx] > 0.0) ? 1 : 0; //1 = outlier, 0 = inlier
                if (!curColEncoder.containsKey(colVal)) {
                    if (stringToRank.containsKey(colNumAndVal)) {
                        int newKey = stringToRank.get(colNumAndVal);
                        curColEncoder.put(colVal, newKey);
                        valueDecoder.put(newKey, colVal);
                        columnDecoder.put(newKey, colIdx);
                        nextKey++;
                    } else {
                        curColEncoder.put(colVal, noSupport);
                    }
                }
                int curKey = curColEncoder.get(colVal);
                encodedAttributes[rowIdx][colIdx] = curKey;

                if (oidx == 1 && curKey != noSupport && !foundOutliers.contains(curKey)) {
                    foundOutliers.add(curKey);
                    outlierList[colIdx].add(curKey);
                }
            }
            colCardinalities[colIdx] = outlierList[colIdx].size();
            if (!useBitmaps)
                colCardinalities[colIdx] = cardinalityThreshold + 1;
        }
        log.info("Column cardinalities: {}", Arrays.toString(colCardinalities));
        // Encode the bitmaps.  Store bitmaps as an array indexed first
        // by column and then by outlier/inlier.  Each entry in array
        // is a map from encoded attribute value to the bitmap
        // for that attribute among outliers or inliers.
        for (int colIdx = 0; colIdx < numColumns; colIdx++) {
            Map<String, Integer> curColEncoder = encoder.get(colIdx);
            String[] curCol = columns.get(colIdx);
            if (useBitmaps && colCardinalities[colIdx] < cardinalityThreshold) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    String colVal = curCol[rowIdx];
                    int oidx = (outlierColumn[rowIdx] > 0.0) ? 1 : 0; //1 = outlier, 0 = inlier
                    int curKey = curColEncoder.get(colVal);
                    if (curKey != noSupport) {
                        if (bitmap[colIdx][oidx].containsKey(curKey)) {
                            bitmap[colIdx][oidx].get(curKey).set(rowIdx);
                        } else {
                            bitmap[colIdx][oidx].put(curKey, new ModBitSet());
                            bitmap[colIdx][oidx].get(curKey).set(rowIdx);
                        }
                    }
                }
            }
        }
        return encodedAttributes;
    }

    /**
     * Encode as integers all attribute strings.
     * @param columns A list of attribute strings from each column of the original
     *                dataset.
     * @return A matrix of encoded attributes, stored as an array of arrays.
     */
    public int[][] encodeAttributesAsArray(List<String[]> columns) {
        if (columns.isEmpty()) {
            return new int[0][0];
        }

        int numColumns = columns.size();
        int numRows = columns.get(0).length;

        for (int i = 0; i < numColumns; i++) {
            if (!encoder.containsKey(i)) {
                encoder.put(i, new HashMap<>());
            }
        }

        colCardinalities = new int[numColumns];
        for (int i = 0; i < numColumns; i++)
            colCardinalities[i] = cardinalityThreshold + 1;

        int[][] encodedAttributes = new int[numRows][numColumns];

        for (int colIdx = 0; colIdx < numColumns; colIdx++) {
            Map<String, Integer> curColEncoder = encoder.get(colIdx);
            String[] curCol = columns.get(colIdx);
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                String colVal = curCol[rowIdx];
                if (!curColEncoder.containsKey(colVal)) {
                    curColEncoder.put(colVal, nextKey);
                    valueDecoder.put(nextKey, colVal);
                    columnDecoder.put(nextKey, colIdx);
                    nextKey++;
                }
                int curKey = curColEncoder.get(colVal);
                encodedAttributes[rowIdx][colIdx] = curKey;
            }
        }

        return encodedAttributes;
    }

    /**
     * Encode as integers all attribute strings.
     * @param columns A list of attribute strings from each column of the original
     *                dataset.
     * @return A matrix of encoded attributes, stored as a list of arrays.
     */
    public List<int[]> encodeAttributes(List<String[]> columns) {
        if (columns.isEmpty()) {
            return new ArrayList<>();
        }

        int[][] encodedArray = encodeAttributesAsArray(columns);
        int numRows = columns.get(0).length;

        ArrayList<int[]> encodedAttributes = new ArrayList<>(numRows);
        for (int i = 0; i < numRows; i++) {
            encodedAttributes.add(encodedArray[i]);
        }

        return encodedAttributes;
    }

    public List<Set<Integer>> encodeAttributesAsSets(List<String[]> columns) {
        List<int[]> arrays = encodeAttributes(columns);
        ArrayList<Set<Integer>> sets = new ArrayList<>(arrays.size());
        for (int[] row : arrays) {
            HashSet<Integer> curSet = new HashSet<>(row.length);
            for (int i : row) {
                curSet.add(i);
            }
            sets.add(curSet);
        }
        return sets;
    }

    public int getNextKey() {
        return nextKey;
    }

    public Map<String, String> decodeSet(Set<Integer> set) {
        HashMap<String, String> m = new HashMap<>(set.size());
        for (int i : set) {
            m.put(decodeColumnName(i), decodeValue(i));
        }
        return m;
    }

}

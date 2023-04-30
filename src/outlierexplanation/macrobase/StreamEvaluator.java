package outlierexplanation.macrobase;

import mtree.tests.Data;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import outlierexplanation.exstream.model.Reward;

import java.util.*;

public class StreamEvaluator {
    private double boundaryPercentiles[] = {10.0, 90.0};
    private LinkedList<Data> points;
    private int n;
    private List<double[]> boundaries;
    private Percentile pCalc;
    private String[] bucketNames;

    public StreamEvaluator(int n) {
        points = new LinkedList<>();
        boundaries = new ArrayList<>();
        pCalc = new Percentile();
        this.n = n;
    }

    public List<String[]> addDatas(List<Data> datas) {
        for(Data data : datas) {
            if(points.size() == n) points.removeFirst();
            points.addLast(data);
        }
        evaluate();
        List<String[]> res = new ArrayList<>();
        for(Data data : datas) {
            int m = data.values.length;
            String[] transformedColVal = new String[m];
            for(int k=0;k<m;k++) {
                double curVal = data.values[k];
                int searchIdx = Arrays.binarySearch(boundaries.get(k), curVal);
                if (searchIdx < 0) {
                    searchIdx = -searchIdx - 1;
                }
                transformedColVal[k] = k + ":" + searchIdx;
            }
            res.add(transformedColVal);
        }
        return res;
    }

    public List<String[]> addDatas(List<Data> datas, List<Reward> rewards) {
        for(Data data : datas) {
            if(points.size() == n) points.removeFirst();
            points.addLast(data);
        }
        List<String[]> res = new ArrayList<>();
        for(Data data : datas) {
            int m = data.values.length;
            String[] transformedColVal = new String[m];
            for(int k=0;k<m;k++) {
                double curVal = data.values[k];
                int searchRes = 0;
                for(Reward r : rewards) {
                    if(r.attrIdx == k) {
                        for(double[] range : r.explanations) {
                            if(curVal >= range[0] && curVal <= range[1]) {
                                searchRes = 1;
                                break;
                            }
                        }
                    }
                }
                transformedColVal[k] = searchRes + "";
            }
            res.add(transformedColVal);
        }
        return res;
    }

    private void evaluate() {
        int b = points.getFirst().values.length;
        while(boundaries.size() < b) {
            boundaries.add(new double[]{});
        }
        int k = boundaryPercentiles.length;
        for(int i=0;i<b;i++) {
            double[] vals = new double[n];
            for(int j=0;j<n;j++) {
                vals[i] = points.get(j).values[i];
            }
            pCalc.setData(vals);
            double[] curBoudaries = new double[k];
            for(int p=0;p<k;p++) {
                curBoudaries[p] = pCalc.evaluate(boundaryPercentiles);
            }
            boundaries.set(i, curBoudaries);
        }
    }

    private void evaluate(List<Reward> rewards) {

    }

    public void setBoundaryPercentiles(double[] boundaryPercentiles) {
        this.boundaryPercentiles = boundaryPercentiles;
    }
}

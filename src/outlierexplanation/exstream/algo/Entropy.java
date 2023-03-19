package outlierexplanation.exstream.algo;


import outlierexplanation.exstream.model.Reward;
import outlierexplanation.exstream.model.Segment;
import mtree.tests.Data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Entropy {
    public static List<Reward> rewards(List<Data> inliers, List<Data> outliers) {
        int m = inliers.size();
        int n = outliers.size();
        double hClass = entropy((double)m / (m + n), (double)n / (m + n));
        List<Reward> hSegments = segments(inliers, outliers);
        for(int i=0;i<hSegments.size();i++) {
            hSegments.get(i).reward = hClass / hSegments.get(i).reward;
        }
        Collections.sort(hSegments, (a, b) -> {
            if(a.reward > b.reward) return -1;
            else if(b.reward > a.reward) return 1;
            else return 0;
        });
//        double maxDiff = -1;
//        for(int i=1;i<hSegments.size();i++) {
//            if(hSegments.get(i).reward - hSegments.get(i-1).reward > maxDiff) {
//                maxDiff = hSegments.get(i).reward - hSegments.get(i-1).reward;
//            }
//        }
//        int i = 1;
//        for(;i<hSegments.size() && hSegments.get(i).reward - hSegments.get(i-1).reward < maxDiff;i++) {}
        List<Reward> res = new ArrayList<>();
        res.addAll(hSegments);
//        for(int j=0;j<i-1;j++) {
//            res.add(hSegments.get(j));
//        }
        return res;
    }

    private static List<Reward> segments(List<Data> inliers, List<Data> outliers) {
        int n = inliers.get(0).values.length;
        List<Reward> res = new ArrayList<>();
        for(int i=0;i<n;i++) {
            List<Double> inlierAttr = new ArrayList<>();
            for(Data inlier : inliers) {
                inlierAttr.add(inlier.values[i]);
            }
            Collections.sort(inlierAttr);
            List<Double> outlierAttr = new ArrayList<>();
            for(Data outlier : outliers) {
                outlierAttr.add(outlier.values[i]);
            }
            Collections.sort(outlierAttr);
            List<Segment> segments = new ArrayList<>();
            int p = 0, q = 0;
            while(p < inlierAttr.size() || q < outlierAttr.size()) {
                Segment segment = new Segment();
                if(p == inlierAttr.size()) {
                    segment.outlier = true;
                    while(q < outlierAttr.size()) {
                        segment.add(outlierAttr.get(q++));
                    }
                } else if(q == outlierAttr.size()) {
                    while(p < inlierAttr.size()) {
                        segment.add(inlierAttr.get(p++));
                    }
                } else {
                    if(inlierAttr.get(p) < outlierAttr.get(q)) {
                        while(p < inlierAttr.size() && inlierAttr.get(p) < outlierAttr.get(q)) {
                            segment.add(inlierAttr.get(p++));
                        }
                    } else if(inlierAttr.get(p) > outlierAttr.get(q)) {
                        segment.outlier = true;
                        while(q < outlierAttr.size() && inlierAttr.get(p) > outlierAttr.get(q)) {
                            segment.add(outlierAttr.get(q++));
                        }
                    } else {
                        while(p < inlierAttr.size() && q < outlierAttr.size() && inlierAttr.get(p).equals(outlierAttr.get(q))) {
                            segment.mixed = true;
                            segment.add(inlierAttr.get(p++));
                            q++;
                        }
                    }
                }
                segments.add(segment);
            }
            int sum = 0;
            for(Segment segment : segments) {
                sum += segment.points.size();
            }
            Reward reward = new Reward();
            reward.attrIdx = i;
            for(Segment segment : segments) {
                double per = (double)segment.points.size() / sum;
                reward.reward += per * Math.log(per);
                if(segment.outlier) {
                    double[] range = new double[2];
                    if(segment.points.size() == 1) {
                        range[0] = segment.points.get(0);
                        range[1] = segment.points.get(0);
                    } else if(segment.points.size() > 1) {
                        range[0] = segment.points.get(0);
                        range[1] = segment.points.get(segment.points.size() - 1);
                    }
                    reward.explanations.add(range);
                }
                if(segment.mixed) {
                    int mixSize = segment.points.size();
                    double mixPer = 1 / (double)mixSize;
                    reward.reward += Math.log(mixPer);
                }
            }
            res.add(reward);
        }
        return res;
    }

    private static double entropy(double... ps) {
        double res = 0;
        for(double p : ps) {
            res += p * Math.log(p);
        }
        return res;
    }
}

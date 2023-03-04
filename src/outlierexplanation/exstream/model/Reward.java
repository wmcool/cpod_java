package outlierexplanation.exstream.model;

import java.util.ArrayList;
import java.util.List;

public class Reward {
    public double reward;
    public int attrIdx;
    public List<double[]> explanations;

    public Reward() {
        explanations = new ArrayList<>();
    }
}

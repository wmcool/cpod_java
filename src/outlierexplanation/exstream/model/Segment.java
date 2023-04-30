package outlierexplanation.exstream.model;

import java.util.ArrayList;
import java.util.List;

public class Segment {
    public List<Double> points;
    public boolean mixed;
    public boolean outlier;
    public double prop;

    public Segment() {
        points = new ArrayList<>();
        mixed = false;
        outlier = false;
    }

    public void add(Double data) {
        points.add(data);
    }
}

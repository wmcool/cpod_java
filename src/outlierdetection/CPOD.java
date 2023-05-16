/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package outlierdetection;

import java.util.*;

import de.bwaldvogel.liblinear.*;
import mtree.ComposedSplitFunction;
import mtree.DistanceFunctions;
import mtree.MTree;
import mtree.PartitionFunctions;
import mtree.PromotionFunction;
import mtree.tests.Data;
import mtree.utils.Constants;
import mtree.utils.Pair;
import mtree.utils.Utils;

/**
 *
 * @author luan
 */
public class CPOD {

    public static int currentTime;

    public static int expiredSlideIndex = -1;

    public static HashMap<Integer, ArrayList<C_Data>> all_slides = new HashMap<>();

    public static HashMap<Integer, ArrayList<CorePoint>> all_core_points = new HashMap<>();
    public static MTreeCorePoint mtree = new MTreeCorePoint(); // mtree没有删除环节！！
    public static HashMap<String, Integer> gridCount = new HashMap<>();
    public static HashMap<String, CorePoint> gridCore = new HashMap<>();

    public static HashSet<CorePoint> all_distinct_cores = new HashSet<>();
    public static HashMap<Integer, HashSet<C_Data>> outlierList = new HashMap<>();
    public static HashSet<C_Data> outlierSet = new HashSet<>();

    // 监控左邻居的？
    public static HashMap<Integer, HashSet<C_Data>> neighborCountTrigger = new HashMap<>();

    public static double timeProcessExpiredSlide = 0;
    public static double timeCreatingCore = 0;
    public static double timeProbing = 0;
    public static double timeReProbing = 0;
//    public static int count = 0;
    public static double numPointNeedProb = 0;
    public static double avg_points_check = 0;
    public static int countPoint = 0;

    public double avgNumCorePointsPerWindows = 0;
    public double precision = 0;
    public double recall = 0;
    public double Fscore = 0;
    public int numOutlier = 0;

//    public double timeForFirstWindow = 0;
    public ArrayList<Data> detectOutlier(ArrayList<Data> data, ArrayList<int[]> lbs, int _currentTime, int W, int slide) {

        currentTime = _currentTime;

        ArrayList<C_Data> d_to_process = new ArrayList<>(data.size());
//        HashSet<Integer> slide_to_process = new HashSet<>();
//        System.out.println("Current time = "+ currentTime);
        int[] slide_to_process;
        if (currentTime == W) {
            slide_to_process = new int[W / slide];
            for (int i = 0; i < slide_to_process.length; i++) {
                slide_to_process[i] = i;
            }
        } else {
            slide_to_process = new int[]{(currentTime - 1) / slide};
        }

        for (int i = 0; i < data.size(); i++) {
            Data o = data.get(i);
            C_Data d = new C_Data(o);

            d_to_process.add(d);

            ArrayList<C_Data> idx = all_slides.get(d.sIndex);
            if (idx != null) {
                idx.add(d);
            } else {
                idx = new ArrayList<>(Constants.slide);
                idx.add(d);
                all_slides.put(d.sIndex, idx);
            }
        }
//        MTTest.start = Utils.getCPUTime();

//        long start = Utils.getCPUTime();
        ArrayList<Data> result = new ArrayList<>((int) (data.size() * 1.0 / 100));
        expiredSlideIndex = (currentTime - 1) / slide - Constants.W / slide;
//        System.out.println("Expire slide index = " + expiredSlideIndex);
//        long start = Utils.getCPUTime();
        processExpiredData(expiredSlideIndex);

        for (int sIdx : slide_to_process) {
            ArrayList<CorePoint> corePoints = selectCore(sIdx);
            all_core_points.put(sIdx, corePoints);
        }
        System.out.println("Core Point Num: " + all_distinct_cores.size());

        int newestSlide = (currentTime - 1) / Constants.slide;

        // UpdateHalfRCount
        if (currentTime == Constants.W) {
            for (CorePoint c : all_distinct_cores) {
                c.totalHalfRPoints = c.getTotalHalfRPoints();
            }
        } else if (data.size() == Constants.slide) {
            for (CorePoint c : all_distinct_cores) {
                if (c.closeNeighbors_halfR.get(newestSlide) != null) {
                    c.totalHalfRPoints += c.closeNeighbors_halfR.get(newestSlide).size();
                }
            }
        }


        for (int i = 0; i < d_to_process.size(); i++) {
            C_Data d = d_to_process.get(i);
            if (d.closeCoreMaps_halfR != null && d.closeCoreMaps_halfR.totalHalfRPoints >= Constants.k + 1) {
                continue;
            }
            if(gridCount.getOrDefault(d.cellBase, 0) >= Constants.k + 1) {
                continue;
            }
            if (d.neighborCount < Constants.k) {
                probe(d, newestSlide);
//                pointNeedProb += 1;
            }
//            System.out.println("Finished "+ i);
        }

        for (Map.Entry<Integer, HashSet<C_Data>> e : outlierList.entrySet()) {
//        for (int slideIndex : outlierList.keySet()) {
            for (C_Data d : e.getValue()) {

                if (d.closeCoreMaps_halfR != null && d.closeCoreMaps_halfR.totalHalfRPoints >= Constants.k + 1) {
                    continue;
                }
                if(gridCount.getOrDefault(d.cellBase, 0) >= Constants.k + 1) {
                    continue;
                }
                if (d.neighborCount < Constants.k && d.sIndex < newestSlide) {
                    if (d.lastProbRight < newestSlide) {
                        probe(d, newestSlide);
                    }
                }
                if (d.neighborCount < Constants.k && (currentTime == W || d.sIndex == newestSlide)) {
                    result.add(d);
                }
                if(Constants.explainSingleOutlier) {
                    for(int slideP : slide_to_process) {
                        if(slideP == d.sIndex) {
                            int lbIdx;
                            if(currentTime == W) {
                                lbIdx = (d.arrivalTime - 1 + Constants.W) % Constants.W;
                            } else {
                                lbIdx = (d.arrivalTime - 1 + Constants.slide) % Constants.slide;
                            }
                            int[] l = lbs.get(lbIdx);
                            List<FeatureNode[]> trainData = new ArrayList<>();
//                            FeatureNode[][] trainData = new FeatureNode[6][d.values.length];
//                            double[] labels = new double[6];
                            List<Double> labels = new ArrayList<>();
                            for(int k=-3;k<3;k++) {
                                int idx = lbIdx + k;
                                if(idx < 0 || idx >= d_to_process.size()) continue;
                                C_Data cur = d_to_process.get(idx);
                                if(cur != d && outlierSet.contains(cur)) continue;
                                FeatureNode[] curFt = new FeatureNode[d.values.length];
                                for(int j=1;j<=cur.values.length;j++) {
//                                    trainData[k+3][j-1] = new FeatureNode(j, cur.values[j-1]);
                                    curFt[j-1] = new FeatureNode(j, cur.values[j-1]);
                                }
                                trainData.add(curFt);
                                labels.add((double) (outlierSet.contains(cur)? -1 : 1));
//                                labels[k+3] = outlierSet.contains(cur)? -1 : 1;
                            }
                            FeatureNode[][] td = new FeatureNode[trainData.size()][d.values.length];
                            for(int k=0;k<trainData.size();k++) {
                                td[k] = trainData.get(k);
                            }
                            double[] lb = new double[labels.size()];
                            for(int k=0;k<labels.size();k++) {
                                lb[k] = labels.get(k);
                            }
                            // 创建问题对象
                            Problem problem = new Problem();
                            problem.l = lb.length; // 样本数量
                            problem.n = td[0].length; // 特征数量
                            problem.x = td; // 训练数据
                            problem.y = lb; // 标签

                            // 设置线性SVM的参数
                            SolverType solver = SolverType.L2R_L2LOSS_SVC;
                            double C = 100.0;
                            double eps = 0.001;
                            Parameter parameter = new Parameter(solver, C, eps);
                            // 训练模型
                            Model model = Linear.train(problem, parameter);
                            System.out.println("outlier " + d + " weight ");
                            for(double w : model.getFeatureWeights()) {
                                System.out.print(w + ", ");
                            }
                            computeMetrics(model.getFeatureWeights(), l);
                            System.out.println();
                        }
                    }
                }
            }
        }

//        if(Constants.explanSingleOutlier) {
//            for(Data d : result) {
////            MTreeCorePoint.Query query = mtree.getNearest(d, Double.MAX_VALUE, 5);
////            CorePoint c = null;
////            double distance = Double.MAX_VALUE;
//
//            }
//        }
        return result;
    }

    private static class Pair {
        int idx;
        double wi;

        Pair(int idx, double wi) {
            this.idx = idx;
            this.wi = wi;
        }
    }

    public void computeMetrics(double[] w, int[] label) {
        int numAttr = 0;
        for(int i=0;i<label.length;i++) {
            if(label[i] == 1) numAttr++;
        }
        if(numAttr == 0) return;
        ArrayList<Pair> pairs = new ArrayList<>();
        for(int i=0;i<w.length;i++) {
            Pair pair = new Pair(i, Math.abs(w[i]));
            pairs.add(pair);
        }
        Collections.sort(pairs, (a, b) -> {
            if(b.wi > a.wi) return 1;
            else if(b.wi < a.wi) return -1;
            else return 0;
        });
        int hit = 0;
        for(int k=0;k<numAttr;k++) {
            if(label[pairs.get(k).idx] == 1) {
                hit++;
            }
        }
        precision += (double)hit / numAttr;
        recall += (double)hit / numAttr;
        numOutlier++;
    }

    public double computeNumberActiveCorePoints() {
        HashSet<CorePoint> cores = new HashSet<>();
        for (int sidx : all_core_points.keySet()) {
            cores.addAll(all_core_points.get(sidx));
        }
        return cores.size() * 1.0 / Constants.W;
    }

    public void checkCoreCountPoints() {
        int countOutlier = 0;
        for (CorePoint c : all_distinct_cores) {
            if (c.isCoveredAllSlides() //                    && (c.getTotalHalfRPoints()+c.getTotal32RPoints()+c.getTotalRPoints() < Constants.k)
                    ) {
                countOutlier += 1;
            }
        }
        System.out.println("Num Core Contains all slides = " + countOutlier);
    }

    public boolean check_distance_neighbor_boolean(C_Data d, C_Data d2) {

        double exact_d = DistanceFunction.euclideanDistance(d, d2);
        return exact_d <= Constants.R;
    }

    public double check_distance_neighbor(C_Data d, C_Data d2) {

        double exact_d = DistanceFunction.euclideanDistance(d, d2);
        return exact_d;
    }

    private void processExpiredData(int expiredSlideIndex) {

        if(all_core_points.containsKey(expiredSlideIndex)) {
            for(CorePoint c : all_core_points.get(expiredSlideIndex)) {
                gridCore.remove(c.cellBase);
            }
        }
        all_core_points.remove(expiredSlideIndex);
//        all_indexed_cores.remove(expiredSlideIndex);
        for (CorePoint c : all_distinct_cores) {
            if (c.closeNeighbors_halfR.get(expiredSlideIndex) != null) {
                c.totalHalfRPoints -= c.closeNeighbors_halfR.get(expiredSlideIndex).size();
                c.closeNeighbors_halfR.remove(expiredSlideIndex);
            }
            c.closeNeighbors_R.remove(expiredSlideIndex);
            c.closeNeighbors_3halfR.remove(expiredSlideIndex);
            c.closeNeighbors_2R.remove(expiredSlideIndex);
        }

        if(all_slides.containsKey(expiredSlideIndex)) {
            for(C_Data d : all_slides.get(expiredSlideIndex)) {
                gridCount.put(d.cellBase, gridCount.getOrDefault(d.cellBase, 0) - 1);
                if(gridCount.get(d.cellBase) == 0) {
                    gridCount.remove(d.cellBase);
                }
                outlierSet.remove(d);
//                if(d.closeCoreMaps_halfR != null) {
//                    CorePoint c = d.closeCoreMaps_halfR;
//                    if(c.sIndex <= expiredSlideIndex && c.closeNeighbors_halfR.size() == 0 && c.closeNeighbors_R.size() == 0) {
//                        all_distinct_cores.remove(c);
//                        mtree.remove(c);
//                    }
//                }
//                if(d.closeCoreMaps_R != null) {
//                    for(CorePoint c : d.closeCoreMaps_R) {
//                        if(c.sIndex <= expiredSlideIndex && c.closeNeighbors_halfR.size() == 0 && c.closeNeighbors_R.size() == 0) {
//                            all_distinct_cores.remove(c);
//                            mtree.remove(c);
//                        }
//                    }
//                }
            }
        }
        all_slides.remove(expiredSlideIndex);
        outlierList.remove(expiredSlideIndex);
//        for(Integer sIdx: all_slides.keySet()){
//            for(C_Data d: all_slides.get(sIdx)){
//                if(d.pred_neighbor_count.containsKey(expiredSlideIndex)){
//                    d.neighborCount -= d.pred_neighbor_count.get(expiredSlideIndex);
//                    d.pred_neighbor_count.remove(expiredSlideIndex);
//                }
//            }
//        }
        if (neighborCountTrigger.containsKey(expiredSlideIndex)) {
            for (C_Data d : neighborCountTrigger.get(expiredSlideIndex)) {
                if (d.pred_neighbor_count.containsKey(expiredSlideIndex)) {
                    d.neighborCount -= d.pred_neighbor_count.get(expiredSlideIndex);
                    d.pred_neighbor_count.remove(expiredSlideIndex);
                }
            }
        }
        if (neighborCountTrigger.containsKey(expiredSlideIndex - 1)) {
            neighborCountTrigger.remove(expiredSlideIndex - 1);
        }

//        if(all_core_points.get(expiredSlideIndex) != null) {
//            for(CorePoint c : all_core_points.get(expiredSlideIndex)) {
//                mtree.remove(c);
//            }
//        }
    }

    private HashMap<Integer, ArrayList<C_Data>> indexByAtt(ArrayList<C_Data> datas, int i, double min_value) {

        HashMap<Integer, ArrayList<C_Data>> results = new HashMap<>();

        for (C_Data d : datas) {
            int bin = (int) ((d.values[i] - min_value) / (Constants.R / 2));
            if (results.containsKey(bin)) {
                results.get(bin).add(d);
            } else {
                ArrayList<C_Data> v = new ArrayList<>();
                v.add(d);
                results.put(bin, v);
            }
        }
        return results;
    }

    private void scanForCore(CorePoint c, int sIdx,
            HashMap<Integer, ArrayList<C_Data>> bin_map, int index_att, double min_value) {

        ArrayList<C_Data> neighborsInHalfR = new ArrayList<>();
        ArrayList<C_Data> neighborsInR = new ArrayList<>();
        ArrayList<C_Data> neighborsIn3HalfR = new ArrayList<>();
        ArrayList<C_Data> neighborsIn2R = new ArrayList<>();

        int bin_core = (int) ((c.values[index_att] - min_value) / (Constants.R / 2));
        if (bin_map.containsKey(bin_core)) {
            neighborsInHalfR.addAll(bin_map.get(bin_core));
        }
        int[] possible_bins = new int[]{bin_core - 4, bin_core - 3, bin_core - 2, bin_core - 1,
            bin_core + 1, bin_core + 2, bin_core + 3, bin_core + 4};

        for (int b : possible_bins) {
            if (bin_map.containsKey(b)) {
                for (C_Data d2 : bin_map.get(b)) {
                    double distance = DistanceFunction.euclideanDistance(c, d2);
                    if (distance <= Constants.R / 2) {
                        neighborsInHalfR.add(d2);
                        d2.closeCoreMaps_halfR = c;
                    } else if (distance <= Constants.R) {
                        neighborsInR.add(d2);
                        d2.closeCoreMaps_R.add(c);
                    } else if (distance <= Constants.R * 1.5) {
                        neighborsIn3HalfR.add(d2);

                    } else if (distance <= Constants.R * 2) {
                        neighborsIn2R.add(d2);

                    }

                }
            }
        }
//        }
        c.closeNeighbors_halfR.put(sIdx, neighborsInHalfR);
        c.closeNeighbors_R.put(sIdx, neighborsInR);
        c.closeNeighbors_3halfR.put(sIdx, neighborsIn3HalfR);
        c.closeNeighbors_2R.put(sIdx, neighborsIn2R);
    }

    private void scanForCore(CorePoint c, int sIdx) {

        ArrayList<C_Data> neighborsInHalfR = new ArrayList<>();
        ArrayList<C_Data> neighborsInR = new ArrayList<>();
        ArrayList<C_Data> neighborsIn3HalfR = new ArrayList<>();
        ArrayList<C_Data> neighborsIn2R = new ArrayList<>();

        for (C_Data d2 : all_slides.get(sIdx)) {
            double distance = DistanceFunction.euclideanDistance(c, d2);
            if (distance <= Constants.R / 2) {
                neighborsInHalfR.add(d2);
                d2.closeCoreMaps_halfR = c;
            } else if (distance <= Constants.R) {
                neighborsInR.add(d2);
                d2.closeCoreMaps_R.add(c);
            } else if (distance <= Constants.R * 1.5) {
                neighborsIn3HalfR.add(d2);

            } else if (distance <= Constants.R * 2) {
                neighborsIn2R.add(d2);

            }

        }
//        }

        c.closeNeighbors_halfR.put(sIdx, neighborsInHalfR);
        c.closeNeighbors_R.put(sIdx, neighborsInR);
        c.closeNeighbors_3halfR.put(sIdx, neighborsIn3HalfR);
        c.closeNeighbors_2R.put(sIdx, neighborsIn2R);
    }

    private ArrayList<CorePoint> selectCore(Integer sIdx) {

        // 支持该窗口的核心点
        ArrayList<CorePoint> corePoints = new ArrayList<>();
//        HashSet<CorePoint> corePoints = new HashSet<>();
        // 新创建的核心点
        ArrayList<CorePoint> newCores = new ArrayList<>();

//        ArrayList<C_Data> temp_list = new ArrayList<>();
//        MTreeCorePoint mt = new MTreeCorePoint();
        for (int i = 0; i < Constants.slide; i++) {
//            System.out.println("Slide index = "+ sIdx);

            C_Data d = all_slides.get(sIdx).get(i);
            gridCount.put(d.cellBase, gridCount.getOrDefault(d.cellBase, 0) + 1);

            if((d.closeCoreMaps_R.isEmpty() && d.closeCoreMaps_halfR == null)) {
                //scan with current cores first
                for (int j = corePoints.size() - 1; j >= 0; j--) {
                    CorePoint c = corePoints.get(j);
                    double distance = DistanceFunction.euclideanDistance(d, c);
//                if(count > 0){
//                    numDCS +=1;
//                }
//                if (MTTest.numberWindows > 1) {
//                    numDCSForIndexing += 1;
//                }

                    if (distance <= Constants.R / 2) {
                        ArrayList<C_Data> closeNeighbors = c.closeNeighbors_halfR.get(sIdx);
                        if (closeNeighbors == null) {
                            closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                        closeNeighbors.add(d);
                            c.closeNeighbors_halfR.put(sIdx, closeNeighbors);
                        } else {
                            closeNeighbors.add(d);
                        }
                        d.closeCoreMaps_halfR = c;
                        break;
                    } else if (distance <= Constants.R) {
                        ArrayList<C_Data> closeNeighbors = c.closeNeighbors_R.get(sIdx);
                        if (closeNeighbors == null) {
                            closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                        closeNeighbors.add(d);
                            c.closeNeighbors_R.put(sIdx, closeNeighbors);
                        } else {
                            closeNeighbors.add(d);
                        }
                        d.closeCoreMaps_R.add(c);
                        break;
                    }
                }
            }

            if((d.closeCoreMaps_R.isEmpty() && d.closeCoreMaps_halfR == null) && gridCore.get(d.cellBase) != null) {
                CorePoint cc = gridCore.get(d.cellBase);
                double distance = DistanceFunction.euclideanDistance(d, cc);
                corePoints.add(cc);
                if (distance <= Constants.R / 2) {
                    ArrayList<C_Data> closeNeighbors = cc.closeNeighbors_halfR.get(sIdx);
                    if (closeNeighbors == null) {
                        closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                        closeNeighbors.add(d);
                        cc.closeNeighbors_halfR.put(sIdx, closeNeighbors);
                    } else {
                        closeNeighbors.add(d);
                    }
                    d.closeCoreMaps_halfR = cc;
                } else if (distance <= Constants.R) {
                    ArrayList<C_Data> closeNeighbors = cc.closeNeighbors_R.get(sIdx);
                    if (closeNeighbors == null) {
                        closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                        closeNeighbors.add(d);
                        cc.closeNeighbors_R.put(sIdx, closeNeighbors);
                    } else {
                        closeNeighbors.add(d);
                    }
                    d.closeCoreMaps_R.add(cc);
                }
            }

            // scan in all core points
            if ((d.closeCoreMaps_R.isEmpty() && d.closeCoreMaps_halfR == null)) { // 加入到现有核心点

//                ArrayList<CorePoint> inRCore = new ArrayList<>();
                //using mtree
                MTreeCorePoint.Query query = mtree.getNearest(d, Constants.R, 1);
                CorePoint c = null;
                double distance = Double.MAX_VALUE;
                for (MTreeClass.ResultItem ri : query) {
                    c = (CorePoint) ri.data;
                    distance = ri.distance;
                }
                if (distance <= Constants.R) {
                    //add c to the core of this slide
                    corePoints.add(c);

                    if (distance <= Constants.R / 2) {
                        ArrayList<C_Data> closeNeighbors = c.closeNeighbors_halfR.get(sIdx);
                        if (closeNeighbors == null) {
                            closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                            closeNeighbors.add(d);
                            c.closeNeighbors_halfR.put(sIdx, closeNeighbors);
                        } else {
                            closeNeighbors.add(d);
                        }
                        d.closeCoreMaps_halfR = c;

                    } else {
                        ArrayList<C_Data> closeNeighbors = c.closeNeighbors_R.get(sIdx);
                        if (closeNeighbors == null) {
                            closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                            closeNeighbors.add(d);
                            c.closeNeighbors_R.put(sIdx, closeNeighbors);
                        } else {
                            closeNeighbors.add(d);
                        }
                        d.closeCoreMaps_R.add(c);

                    }
//                    scanForCore(c, sIdx);

                } else { // 创建新核心点
                    c = new CorePoint(d);
                    all_distinct_cores.add(c);
                    newCores.add(c);
//                    mtree.add(c);
                    corePoints.add(c);

                    ArrayList<C_Data> closeNeighbors = c.closeNeighbors_halfR.get(sIdx);
                    if (closeNeighbors == null) {
                        closeNeighbors = new ArrayList<>(Arrays.asList(d));
//                        closeNeighbors.add(d);
                        c.closeNeighbors_halfR.put(sIdx, closeNeighbors);
                    } else {
                        closeNeighbors.add(d);
                    }
                    d.closeCoreMaps_halfR = c;
                    gridCore.put(d.cellBase, c);

                    //probe neighbors for c
//                    scanForCore(c, sIdx);
                }

            }
        }

        //find scan for cores
        boolean[] checked = new boolean[Constants.slide]; // 避免同一个点被多次加入核心点索引
        for (CorePoint c : corePoints) {
            if (c.closeNeighbors_halfR.get(sIdx) == null) {
                c.closeNeighbors_halfR.put(sIdx, new ArrayList<>());
            }
            if (c.closeNeighbors_R.get(sIdx) == null) {
                c.closeNeighbors_R.put(sIdx, new ArrayList<>());
            }
            if (c.closeNeighbors_3halfR.get(sIdx) == null) {
                c.closeNeighbors_3halfR.put(sIdx, new ArrayList<>());
            }
            if (c.closeNeighbors_2R.get(sIdx) == null) {
                c.closeNeighbors_2R.put(sIdx, new ArrayList<>());
            }

            for (int i = 0; i < Constants.slide; i++) {
                checked[i] = false;
            }
            for (CorePoint c2 : corePoints) {

                if (c != c2) {
                    double distance = DistanceFunction.euclideanDistance(c, c2);

                    if (distance <= Constants.R * 3) {
                        checked = probCoreWithList(c, c2.closeNeighbors_halfR.get(sIdx), sIdx, checked, all_slides.get(sIdx).get(0).arrivalTime);
                        checked = probCoreWithList(c, c2.closeNeighbors_R.get(sIdx), sIdx, checked, all_slides.get(sIdx).get(0).arrivalTime);
                    }
                }
            }

        }
        for (CorePoint c : newCores) {
            mtree.add(c);
        }
        return corePoints;

    }

    private void probe_slide_right(C_Data d, int slideIndex) {
//        int countNeighbor = 0;

        //scan possible points 
        ArrayList<ArrayList<C_Data>> possibleCandidates = new ArrayList<>();

        //find close core
//        long start = Utils.getCPUTime();
        double distance = 0;
        ArrayList<CorePoint> cores;
        ResultFindCore rf = null;
        rf = findCloseCore(d, slideIndex);
        distance = rf.getDistance();
        cores = rf.getCore();

//        System.out.println("Time to find core = "+ (Utils.getCPUTime() - start) * 1.0 / 1000000000);
        int case_ = 0;
        if (cores != null) {
            if (distance <= Constants.R / 2) {
                CorePoint c = cores.get(0);

                //grab close neighbor in range R/2 of c
                d.neighborCount += c.closeNeighbors_halfR.get(slideIndex).size();
                d.numSucceedingNeighbor += c.closeNeighbors_halfR.get(slideIndex).size();
                if (d.numSucceedingNeighbor >= Constants.k) {
                    return;
                }
                possibleCandidates.add(c.closeNeighbors_R.get(slideIndex));
                possibleCandidates.add(c.closeNeighbors_3halfR.get(slideIndex));

            } else if (distance <= Constants.R) {

                possibleCandidates.add(cores.get(0).closeNeighbors_halfR.get(slideIndex));
                possibleCandidates.add(cores.get(0).closeNeighbors_R.get(slideIndex));
                possibleCandidates.add(cores.get(0).closeNeighbors_3halfR.get(slideIndex));
                possibleCandidates.add(cores.get(0).closeNeighbors_2R.get(slideIndex));

            } else if (distance <= Constants.R * 2) {
                case_ = 1;
                for (int i = 0; i < cores.size(); i++) {
                    CorePoint c = cores.get(i);
                    if (rf.distance_to_cores.get(i) <= Constants.R * 3 / 2) { // R不包括R/2，所以要单独算一下
                        possibleCandidates.add(c.closeNeighbors_halfR.get(slideIndex));
                    }

                }
                for (CorePoint c : cores) {
                    possibleCandidates.add(c.closeNeighbors_R.get(slideIndex));

                }
//                for (CorePoint c : cores) {
//                    possibleCandidates.add(c.closeNeighbors_3halfR.get(slideIndex));
//
//                }
//                for (CorePoint c : cores) {
//                    possibleCandidates.add(c.closeNeighbors_2R.get(slideIndex));
//
//                }
            }

            int min_arrival_time = all_slides.get(slideIndex).get(0).arrivalTime;

            // 避免重复计算，剪枝用
            boolean[] checked = null;
            if (case_ == 1) {
                checked = new boolean[Constants.slide];
            }

            int oldNumSucNeighbor = d.numSucceedingNeighbor;
//        outerloop:
            for (ArrayList<C_Data> ps : possibleCandidates) {
                if(ps == null) continue;

                for (int t = 0; t < ps.size(); t++) {
                    C_Data d2 = ps.get(t);
//                if (!checked.contains(d2)) {
                    if (case_ == 0 || (case_ == 1 && !checked[d2.arrivalTime - min_arrival_time])) {
                        if (check_distance_neighbor_boolean(d, d2)) {
                            //add for stats

//                        d.neighborCount += 1;
                            d.numSucceedingNeighbor += 1;

                            if (d.numSucceedingNeighbor >= Constants.k) {
                                //test remove preceding neighbor map 
                                d.pred_neighbor_count.clear();

                                d.neighborCount += d.numSucceedingNeighbor - oldNumSucNeighbor;
                                return;
                            }
                        }
//                    checked.add(d2);
                        if (case_ == 1) {
                            checked[d2.arrivalTime - min_arrival_time] = true;
                        }
                    }
                }

            }
            d.neighborCount += d.numSucceedingNeighbor - oldNumSucNeighbor;
        } else {
//            System.out.println("No core found!!!");
        }
    }

    // 和右探测逻辑基本相同，但左边需要neighbor_trigger
    private void probe_slide_left(C_Data d, int slideIndex) {

//        int countNeighbor = 0;
        int oldNumNeighbor = d.neighborCount;
        //scan possible points 
        ArrayList<ArrayList<C_Data>> possibleCandidates = new ArrayList<>();

        //find close core
//        long start = Utils.getCPUTime();
        double distance = 0;
        ArrayList<CorePoint> cores;
        ResultFindCore rf = null;
        rf = findCloseCore(d, slideIndex);
        distance = rf.getDistance();
        cores = rf.getCore();
//        System.out.println("Time to find core = "+ (Utils.getCPUTime() - start) * 1.0 / 1000000000);
        int case_ = 0;
        if (cores != null) {
            if (distance <= Constants.R / 2) {
                CorePoint c = cores.get(0);

                //grab close neighbor in range R/2 of c
//                countNeighbor += c.closeNeighbors_halfR.get(slideIndex).size();
                d.neighborCount += c.closeNeighbors_halfR.get(slideIndex).size();

                if (d.neighborCount >= Constants.k) {

                    d.pred_neighbor_count.put(slideIndex, c.closeNeighbors_halfR.get(slideIndex).size());
                    if (neighborCountTrigger.containsKey(slideIndex)) {
                        neighborCountTrigger.get(slideIndex).add(d);
                    } else {
                        HashSet<C_Data> hs = new HashSet<>();
                        hs.add(d);
                        neighborCountTrigger.put(slideIndex, hs);
                    }

                    return;

                }
                possibleCandidates.add(c.closeNeighbors_R.get(slideIndex));
                possibleCandidates.add(c.closeNeighbors_3halfR.get(slideIndex));

            } else if (distance <= Constants.R) {

//                Collections.sort(cores, new Comparator<CorePoint>() {
//                    @Override
//                    public int compare(CorePoint o1, CorePoint o2) {
//                        if(o1.getTotalRPoints()+o1.getTotalRPoints()+o1.getTotal32RPoints()+o1.getTotal2RPoints() > 
//                                o2.getTotalRPoints()+o2.getTotalRPoints()+o2.getTotal32RPoints()+o2.getTotal2RPoints() )
//                            return 1;
//                        else return -1;
//                    }
//                });
                possibleCandidates.add(cores.get(0).closeNeighbors_halfR.get(slideIndex));
                possibleCandidates.add(cores.get(0).closeNeighbors_R.get(slideIndex));
                possibleCandidates.add(cores.get(0).closeNeighbors_3halfR.get(slideIndex));
                possibleCandidates.add(cores.get(0).closeNeighbors_2R.get(slideIndex));

            } else if (distance <= Constants.R * 2) {
                case_ = 1;
                for (int i = 0; i < cores.size(); i++) {
                    CorePoint c = cores.get(i);
                    if (rf.distance_to_cores.get(i) <= Constants.R * 3 / 2) {
                        possibleCandidates.add(c.closeNeighbors_halfR.get(slideIndex));
                    }

                }

                for (CorePoint c : cores) {
                    possibleCandidates.add(c.closeNeighbors_R.get(slideIndex));

                }
//                for (CorePoint c : cores) {
//                    possibleCandidates.add(c.closeNeighbors_3halfR.get(slideIndex));
//
//                }
//                for (CorePoint c : cores) {
//                    possibleCandidates.add(c.closeNeighbors_2R.get(slideIndex));
//
//                }
            }

//        start = Utils.getCPUTime();
//        HashSet<C_Data> checked = new HashSet<>();
            int min_arrival_time = all_slides.get(slideIndex).get(0).arrivalTime;

            boolean[] checked = null;
            if (case_ == 1) {
                checked = new boolean[Constants.slide];
            }

            outerloop:
            for (ArrayList<C_Data> ps : possibleCandidates) {
                if(ps == null) continue;
                for (int t = 0; t < ps.size(); t++) {
                    C_Data d2 = ps.get(t);
//                if (!checked.contains(d2)) {
                    if (case_ == 0 || (case_ == 1 && !checked[d2.arrivalTime - min_arrival_time])) {

//                    if (count >= 1) {
//                        avg_points_check += 1;
//                    }
//                        if (count > 1) {
//                            numDCS += 1;
//                        }
                        if (check_distance_neighbor_boolean(d, d2)) {
                            d.neighborCount += 1;
//                        countNeighbor += 1;

                            if (d.neighborCount >= Constants.k) {

                                break outerloop;
                            }
                        }
//                    checked.add(d2);
                        if (case_ == 1) {
                            checked[d2.arrivalTime - min_arrival_time] = true;
                        }
                    }
                }

            }
//        timeCheckingCandidates += (Utils.getCPUTime() - start) * 1.0 / 1000000000;
//        d.neighborCount += countNeighbor;

//            d.numSucceedingNeighbor += countNeighbor;
            d.pred_neighbor_count.put(slideIndex, d.neighborCount - oldNumNeighbor);
//            start = Utils.getCPUTime();
            if (neighborCountTrigger.containsKey(slideIndex)) {
                neighborCountTrigger.get(slideIndex).add(d);
            } else {
                HashSet<C_Data> hs = new HashSet<>();
                hs.add(d);
                neighborCountTrigger.put(slideIndex, hs);
            }
//            timeAddingToNeighborCount += (Utils.getCPUTime() - start) * 1.0 / 1000000000;

//        System.out.println("Time looping through candidates = "+ ((Utils.getCPUTime() - start) * 1.0 / 1000000000));
        }

    }

    private void probe(C_Data d, int newestSlide) {


        boolean counted = false;
        if (d.lastProbRight < newestSlide) {
            //prob right first
            int slideIndex = d.lastProbRight + 1;
            if (d.lastProbRight == -1) {
                slideIndex = d.sIndex;
            }
            while (slideIndex <= newestSlide && d.neighborCount < Constants.k) {
//                    if (d.closeCoreMaps_halfR != null
//                            && d.closeCoreMaps_halfR.closeNeighbors_halfR.containsKey(slideIndex)) {
//                        d.neighborCount -= d.closeCoreMaps_halfR.closeNeighbors_halfR.get(slideIndex).size();
//                    }
                if (!counted) {
//                    if (count >= 1) {
//                        numPointNeedNS += 1;
//                    }
                    counted = true;
                }
                probe_slide_right(d, slideIndex);
                d.lastProbRight = slideIndex;
                slideIndex++;
            }
        }
        //prob left
        if (d.neighborCount < Constants.k) {
            int slideIndex = d.lastProbLeft - 1;
            if (d.lastProbLeft == -1) {
                slideIndex = d.sIndex - 1;
            }

            while (slideIndex > expiredSlideIndex && slideIndex >= 0
                    && d.neighborCount < Constants.k) {
//                  while(d.lastProbLeft == -1 || (d.lastProbLeft >= 0 && d.lastProbLeft > expiredSlideIndex))
//                    if (d.closeCoreMaps_halfR != null
//                            && d.closeCoreMaps_halfR.closeNeighbors_halfR.containsKey(slideIndex)) {
//                        d.neighborCount -= d.closeCoreMaps_halfR.closeNeighbors_halfR.get(slideIndex).size();
//                    }
                if (!counted) {
//                    if (count > 1) {
//                        numPointNeedNS += 1;
//                    }
                    counted = true;
                }
                probe_slide_left(d, slideIndex);
                d.lastProbLeft = slideIndex;
                slideIndex--;
            }
        }
//        }
//        long start = Utils.getCPUTime();
        if (d.neighborCount < Constants.k) {
            //add to outlier List
            if (outlierList.containsKey(d.sIndex)) {
                outlierList.get(d.sIndex).add(d);
                outlierSet.add(d);
            } else {
                HashSet hs = new HashSet();
                hs.add(d);
                outlierList.put(d.sIndex, hs);
                outlierSet.add(d);
            }
        }
//        timeForAddingToOutlierList += (Utils.getCPUTime() - start) * 1.0 / 1000000000;
    }

    private boolean[] probCoreWithList(CorePoint c, ArrayList<C_Data> candidates, int sIdx, boolean[] checked, int start_time) {
        if (candidates != null) {

            for (C_Data d2 : candidates) {
                if (!checked[d2.arrivalTime - start_time]) {
                    double distance = DistanceFunction.euclideanDistance(c, d2);
//                if(MTTest.numberWindows > 1){
//                    numDCSForIndexing +=1;
//                }
                    if (distance <= Constants.R / 2) {
                        c.closeNeighbors_halfR.get(sIdx).add(d2);
                        d2.closeCoreMaps_halfR = c;
                    } else if (distance <= Constants.R) {
                        c.closeNeighbors_R.get(sIdx).add(d2);
                        d2.closeCoreMaps_R.add(c);
                    } else if (distance <= Constants.R * 1.5) {
                        c.closeNeighbors_3halfR.get(sIdx).add(d2);

                    } else if (distance <= Constants.R * 2) {
                        c.closeNeighbors_2R.get(sIdx).add(d2);

                    }
                    checked[d2.arrivalTime - start_time] = true;
                }

            }
//        }

//            c.closeNeighbors_halfR.put(sIdx, neighborsInHalfR);
//            c.closeNeighbors_R.put(sIdx, neighborsInR);
//            c.closeNeighbors_3halfR.put(sIdx, neighborsIn3HalfR);
//            c.closeNeighbors_2R.put(sIdx, neighborsIn2R);
        }
        return checked;
    }

    final class ResultFindCore {

        private final double distance;
        private final ArrayList<CorePoint> cores;
        public ArrayList<Double> distance_to_cores = new ArrayList<>();

        public ResultFindCore(double distance, ArrayList<CorePoint> cores) {
            this.distance = distance;
            this.cores = cores;
        }

        public ResultFindCore(double distance, ArrayList<CorePoint> cores, ArrayList<Double> all_distances) {
            this.distance = distance;
            this.cores = cores;
            this.distance_to_cores = all_distances;
        }

        public double getDistance() {
            return this.distance;
        }

        public ArrayList<CorePoint> getCore() {
            return this.cores;
        }
    }

    private ResultFindCore findCloseCore(C_Data d, int slideIndex) {

        ArrayList<CorePoint> resultCore = null;

        if (d.closeCoreMaps_halfR != null
                && d.closeCoreMaps_halfR.closeNeighbors_halfR.containsKey(slideIndex)) {
            resultCore = new ArrayList<>();
            resultCore.add(d.closeCoreMaps_halfR);
            return new ResultFindCore(Constants.R / 2, resultCore);
        } else if (!d.closeCoreMaps_R.isEmpty()) {
            for (CorePoint c : d.closeCoreMaps_R) {
                if (c.closeNeighbors_2R.containsKey(slideIndex)) {
                    resultCore = new ArrayList<>();
                    resultCore.add(c);
                    return new ResultFindCore(Constants.R, resultCore);
                }
            }

        }

        
        ArrayList<CorePoint> corePoints = all_core_points.get(slideIndex);

        ArrayList<CorePoint> inRangeRCores = new ArrayList<>();
        ArrayList<CorePoint> inRangeDoubleRCores = new ArrayList<>();
        ArrayList<Double> distance_to_cores = new ArrayList<>();
        if (corePoints != null) {
            for (int i = 0; i < corePoints.size(); i++) {
                CorePoint c = corePoints.get(i);
                double distance = DistanceFunction.euclideanDistance(d, c);
//                if (count > 1) {
//                    numDCS += 1;
//                }
//            if (distance <= Constants.R / 2) {
//                resultCore = new ArrayList<>();
//                resultCore.add(c);
//                return new ResultFindCore(Constants.R / 2, resultCore);
//            } else 
                if (distance <= Constants.R) {
                    inRangeRCores.add(c);
                    break;
                    //test
//                return new ResultFindCore(Constants.R, inRangeRCores);
                    //end test
                } else if (distance <= Constants.R * 2) {
                    inRangeDoubleRCores.add(c);
                    distance_to_cores.add(distance);
                }
            }
        }
        if (!inRangeRCores.isEmpty()) {
//            System.out.println("in Range R core = "+ inRangeRCores.size());

            return new ResultFindCore(Constants.R, inRangeRCores);
        } else if (!inRangeDoubleRCores.isEmpty()) {
//            System.out.println("AAAAAAAAAAAAAa");
            return new ResultFindCore(Constants.R * 2, inRangeDoubleRCores, distance_to_cores);
        } else {
            return new ResultFindCore(Constants.R * 2, null);
        }
    }

    class C_Data extends Data {

        private int numSucceedingNeighbor = 0;
//        private boolean isOutlier;
        public int lastProbRight = -1;
        public int lastProbLeft = -1;

        private HashMap<Integer, Integer> pred_neighbor_count = new HashMap<>();
        public int neighborCount = 0;

        private CorePoint closeCoreMaps_halfR;
        private ArrayList<CorePoint> closeCoreMaps_R = new ArrayList<>();
        public String cellBase;

        public int sIndex = -1;

        public C_Data(Data d) {
            this.arrivalTime = d.arrivalTime;
            this.values = d.values;
            this.hashCode = d.hashCode;
            this.cellBase = d.cellBase;

            this.sIndex = (arrivalTime - 1) / Constants.slide;
        }

        public C_Data() {

        }

        public int countNeighbor() {
            return neighborCount;
        }

    }

    class CorePoint extends C_Data implements Comparable<Data> {

        public HashMap<Integer, ArrayList<C_Data>> closeNeighbors_halfR = new HashMap<>();
        public HashMap<Integer, ArrayList<C_Data>> closeNeighbors_R = new HashMap<>();
        public HashMap<Integer, ArrayList<C_Data>> closeNeighbors_3halfR = new HashMap<>();
        public HashMap<Integer, ArrayList<C_Data>> closeNeighbors_2R = new HashMap<>();

        public int totalHalfRPoints = 0;


        public int getTotalHalfRPoints() {
            int t = 0;
            t = closeNeighbors_halfR.entrySet().parallelStream().map((e) -> e.getValue().size()).reduce(t, Integer::sum);
            return t;
        }

        public int getTotal32RPoints() {
            int t = 0;
            for (Map.Entry<Integer, ArrayList<C_Data>> e : closeNeighbors_3halfR.entrySet()) {
                t += e.getValue().size();
            }
            return t;
        }

        public int getTotal2RPoints() {
            int t = 0;
            for (Map.Entry<Integer, ArrayList<C_Data>> e : closeNeighbors_2R.entrySet()) {
                t += e.getValue().size();
            }
            return t;
        }

        public int getTotalRPoints() {
            int t = 0;
            for (Map.Entry<Integer, ArrayList<C_Data>> e : closeNeighbors_2R.entrySet()) {
                t += e.getValue().size();
            }
            return t;
        }

        public boolean isCoveredAllSlides() {
            for (Map.Entry<Integer, ArrayList<C_Data>> e : all_slides.entrySet()) {
                if (!closeNeighbors_halfR.containsKey(e.getKey())) {
                    return false;
                }
            }
            return true;
        }

//        public int getTotalCoverPoint() {
//            return closeNeighbors_2R.size() + closeNeighbors_3halfR.size()
//                    + closeNeighbors_R.size() + closeNeighbors_halfR.size();
//        }
//  
//        private int numPointsCovered = 0;
        public CorePoint(C_Data d) {
            this.values = d.values;
            this.hashCode = d.hashCode;
            this.arrivalTime = d.arrivalTime;
            this.sIndex = d.sIndex;
        }

    }
}

class MTreeClass extends MTree<Data> {

    private static final PromotionFunction<Data> nonRandomPromotion = (Set<Data> dataSet, mtree.DistanceFunction<? super Data> distanceFunction1) -> Utils.minMax(dataSet);

    MTreeClass() {
        super(25, DistanceFunctions.EUCLIDEAN, new ComposedSplitFunction<>(nonRandomPromotion,
                new PartitionFunctions.BalancedPartition<>()));
    }

    @Override
    public void add(Data data) {
        super.add(data);
        _check();
    }

    @Override
    public boolean remove(Data data) {
        boolean result = super.remove(data);
        _check();
        return result;
    }

    mtree.DistanceFunction<? super Data> getDistanceFunction() {
        return distanceFunction;
    }
};

class MTreeCorePoint extends MTree<Data> {

    private static final PromotionFunction<Data> nonRandomPromotion = new PromotionFunction<Data>() {
        @Override
        public Pair<Data> process(Set<Data> dataSet, mtree.DistanceFunction<? super Data> distanceFunction) {
            return mtree.utils.Utils.minMax(dataSet);
        }
    };

    MTreeCorePoint() {
        super(100, DistanceFunctions.EUCLIDEAN, new ComposedSplitFunction<Data>(nonRandomPromotion,
                new PartitionFunctions.BalancedPartition<Data>()));
    }

    @Override
    public void add(Data data) {
        super.add(data);
        _check();
    }

    @Override
    public boolean remove(Data data) {
        boolean result = super.remove(data);
        _check();
        return result;
    }
}



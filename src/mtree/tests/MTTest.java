package mtree.tests;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import org.checkerframework.checker.units.qual.C;
import outlierexplanation.FPGAttributeSet;
import outlierexplanation.FPGExplanation;
import outlierexplanation.WindowedOperator;
import outlierexplanation.exstream.algo.Entropy;
import outlierexplanation.exstream.model.Reward;
import mtree.utils.Constants;
import mtree.utils.Utils;

import outlierdetection.*;
import outlierexplanation.macrobase.IncrementalSummarizer;
import outlierexplanation.macrobase.StreamEvaluator;
import outlierexplanation.macrobase.model.DataFrame;

public class MTTest {

    public static int currentTime = 0;

    public static boolean stop = false;

    public static HashSet<Integer> idOutliers = new HashSet<>();

    public static String algorithm;
    public static int numberWindows = 0;

    public static double start;

    public static double prec_ex;
    public static double rec_ex;
    public static double Fscore_ex;

    public static double prec_mac;
    public static int mac_n = 0;
    public static double rec_mac;
    public static double Fscore_mac;

    public static int n;

    public static void main(String[] args) throws Exception {

        readArguments(args);
        for (String arg : args) {
            System.out.print(arg + " ");
        }
        System.out.println("");

        MesureMemoryThread mesureThread = new MesureMemoryThread();
        mesureThread.start();
        Stream s = Stream.getInstance("");

        CPOD cpod = new CPOD();
        MicroCluster_New mcod = new MicroCluster_New();
        StreamEvaluator se = new StreamEvaluator(Constants.W);

        IncrementalSummarizer outlierSummarizer = new IncrementalSummarizer();
        outlierSummarizer.setOutlierColumn("outlier");
        outlierSummarizer.setMinSupport(0.05);
        WindowedOperator<FPGExplanation> windowedSummarizer = new WindowedOperator<>(outlierSummarizer);
        windowedSummarizer.setWindowLength(Constants.W);
        windowedSummarizer.setTimeColumn("time");
        windowedSummarizer.setSlideLength(Constants.slide);
        windowedSummarizer.initialize();

        FPGExplanation lastExplanation = null;
        while (!stop) {

            numberWindows++;
            if (Constants.numberWindow != -1 && numberWindows > Constants.numberWindow) {
                break;
            }
            System.out.println("Num window = " + numberWindows);

            ArrayList<Data> incomingData;
            ArrayList<int[]> label = null;
            HashSet<String> rele = new HashSet<>();
            if(!Constants.labelFile.equals("")) {
                if (currentTime != 0) {
                    label = s.getIncomingLabel(currentTime, Constants.slide, Constants.labelFile);
//                System.out.println("Last idx time = " + (incomingData.get(incomingData.size()-1).arrivalTime-1));
                } else {
                    label = s.getIncomingLabel(currentTime, Constants.W, Constants.labelFile);
                }
            }
            if(!Constants.releFile.equals("")) {
                if (currentTime != 0) {
                    rele = s.getIncomingRele(currentTime, Constants.slide, Constants.releFile);
//                System.out.println("Last idx time = " + (incomingData.get(incomingData.size()-1).arrivalTime-1));
                } else {
                    rele = s.getIncomingRele(currentTime, Constants.W, Constants.releFile);
                }
            }
            if (currentTime != 0) {
                incomingData = s.getIncomingData(currentTime, Constants.slide, Constants.dataFile);
                currentTime = currentTime + Constants.slide;
//                System.out.println("Last idx time = " + (incomingData.get(incomingData.size()-1).arrivalTime-1));
            } else {
                incomingData = s.getIncomingData(currentTime, Constants.W, Constants.dataFile);
                currentTime = currentTime + Constants.W;
            }
            start = Utils.getCPUTime(); // requires java 1.5

            /**
             * do algorithm
             */
            ArrayList<Data> outliers;
            double elapsedTimeInSec;
            double elapsedTimeInMS;
            switch (algorithm) {
                case "cpod":
                    outliers = cpod.detectOutlier(incomingData, label, currentTime, Constants.W, Constants.slide);
                    Collections.sort(outliers, (a, b) -> (a.arrivalTime - b.arrivalTime));
                    if(outliers.isEmpty()) {
                        System.out.println("empty");
                        continue;
                    } else {
                        for(Data data : outliers) {
                            System.out.println("outlier: " + data.arrivalTime);
                        }
                    }
                    if(Constants.explainWindowOutlier) {
                        // EXStream
                        ArrayList<Data> inliers = new ArrayList<>(incomingData);
                        inliers.removeAll(outliers);
                        List<Reward> rewards = Entropy.rewards(inliers, outliers);
                        for(int i=0;i<rewards.size();i++) {
                            System.out.println("attr: " + rewards.get(i).attrIdx + " reward " + rewards.get(i).reward);
                            if(!rewards.get(i).explanations.isEmpty()) {
                                for(double[] range : rewards.get(i).explanations) {
                                    System.out.println("outlier range: [" + range[0] + ", " + range[1] + "]");
                                }
                            }
                        }
                        List<Integer> idxs = new ArrayList<>();
                        for(Reward reward : rewards) {
                            idxs.add(reward.attrIdx);
                        }
                        outlierSummarizer.setAttributes(idxs);
                        // Macrobase
                        List<String[]> scatter = se.addDatas(incomingData, rewards);
                        Set<Data> exist = new HashSet<>();
                        for(Data data : outliers) {
                            exist.add(data);
                        }
                        DataFrame df = new DataFrame();
                        int k = incomingData.get(0).values.length;
                        int n = incomingData.size();
                        for(int i=0;i<k;i++) {
                            String[] attrs = new String[n];
                            for(int j=0;j<n;j++) {
                                attrs[j] = scatter.get(j)[i];
                            }
                            df.addColumn("a" + i, attrs);
                        }

                        double[] isOutlier = new double[n];
                        double[] time = new double[n];
                        for(int j=0;j<n;j++) {
                            if(exist.contains(incomingData.get(j))) isOutlier[j] = 1.0;
                            else isOutlier[j] = 0.0;
                            time[j] = incomingData.get(j).arrivalTime;
                        }
                        df.addColumn("outlier", isOutlier);
                        df.addColumn("time", time);
                        windowedSummarizer.process(df);
                        FPGExplanation explanation = windowedSummarizer.getResults().prune();
//                        FPGExplanation explanation = windowedSummarizer.getResults();
                        System.out.println(explanation.prettyPrint());
                        validate(rewards, lastExplanation, explanation, rele);
                        lastExplanation = explanation;
                        elapsedTimeInSec = (Utils.getCPUTime() - start) * 1.0 / 1000000000;
                        elapsedTimeInMS = elapsedTimeInSec * 1000;
                        System.out.println("Num outliers = " + outliers.size());
//                    System.out.println("cur slide time: " + elapsedTimeInMS + "ms");
                        System.out.println("cur slide time: " + elapsedTimeInSec + "s");
                        if (numberWindows > 1) {
                            MesureMemoryThread.totalTime += elapsedTimeInSec;
                        }
                    }
//                    Thread.sleep(1000);
                    break;
                case "mcod":
                    outliers = mcod.detectOutlier(incomingData, currentTime, Constants.W, Constants.slide);
                    elapsedTimeInSec = (Utils.getCPUTime() - start) * 1.0 / 1000000000;
                    System.out.println("Num outliers = " + outliers.size());
                    if (numberWindows > 1) {
                        MesureMemoryThread.totalTime += elapsedTimeInSec;
                    }
//
//                    break;
//                    Thread.sleep(1000);
            }

            if (numberWindows == 1) {
                MesureMemoryThread.totalTime = 0;
                MesureMemoryThread.timeForIndexing = 0;
                MesureMemoryThread.timeForNewSlide = 0;
                MesureMemoryThread.timeForExpireSlide = 0;
                MesureMemoryThread.timeForQuerying = 0;

            }

        }

        mesureThread.averageTime = MesureMemoryThread.totalTime * 1.0 / (numberWindows - 1);
        mesureThread.writeResult();
        mesureThread.stop();
        mesureThread.interrupt();
        if(Constants.explainSingleOutlier) {
            double precision = cpod.precision / cpod.numOutlier;
            double recall = cpod.recall / cpod.numOutlier;
            System.out.println("single precision: " + precision);
            System.out.println("single recall: " + recall);
            System.out.println("single F score: " + 2 * precision * recall / (precision + recall));
        }
        if(Constants.explainWindowOutlier) {
            // ex
            double precision = prec_ex / n;
            double recall = rec_ex / n;
            System.out.println("window precision: " + precision);
            System.out.println("window recall: " + recall);
            System.out.println("window F score: " + 2 * precision * recall / (precision + recall));

            // mac
            precision = prec_mac / mac_n;
            recall = rec_mac / n;
            System.out.println("asso precision: " + precision);
            System.out.println("asso recall: " + recall);
            System.out.println("asso F score: " + 2 * precision * recall / (precision + recall));
        }
    }

    public static void readArguments(String[] args) {
        for (int i = 0; i < args.length; i++) {

            //check if arg starts with --
            String arg = args[i];
            if (arg.indexOf("--") == 0) {
                switch (arg) {
                    case "--algorithm":
                        algorithm = args[i + 1];
                        break;
                    case "--R":
                        Constants.R = Double.valueOf(args[i + 1]);
                        break;
                    case "--W":
                        Constants.W = Integer.valueOf(args[i + 1]);
                        break;
                    case "--k":
                        Constants.k = Integer.valueOf(args[i + 1]);
                        Constants.minSizeOfCluster = Constants.k + 1;
                        break;
                    case "--datafile":
                        Constants.dataFile = args[i + 1];
                        break;
                    case "--labelFile":
                        Constants.labelFile = args[i + 1];
                        break;
                    case "--releFile":
                        Constants.releFile = args[i + 1];
                        break;
                    case "--output":
                        Constants.outputFile = args[i + 1];
                        break;
                    case "--numberWindow":
                        Constants.numberWindow = Integer.valueOf(args[i + 1]);
                        break;
                    case "--slide":
                        Constants.slide = Integer.valueOf(args[i + 1]);
                        break;
                    case "--resultFile":
                        Constants.resultFile = args[i + 1];
                        break;
                    case "--samplingTime":
                        Constants.samplingPeriod = Integer.valueOf(args[i + 1]);
                        break;
                    case "--explainSingle":
                        Constants.explainSingleOutlier = true;
                        break;
                    case "--explainWindow":
                        Constants.explainWindowOutlier = true;
                        break;
                }
            }
        }
    }

    public static void writeResult() {

        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(Constants.resultFile, true)))) {
            for (Integer time : idOutliers) {
                out.println(time);
            }
        } catch (IOException e) {
        }

    }

    private static void validate(List<Reward> rewards, FPGExplanation lastExplanation, FPGExplanation explanation, HashSet<String> rele) {
        if(rele.isEmpty()) return;
        n++;
        // Exstream
        HashSet<Integer> attrs = new HashSet<>();
        for(String s : rele) {
            String[] sp = s.split(",");
            for(String idx : sp) {
                attrs.add(Integer.parseInt(idx));
            }
        }
        double prec_ex_hit = 0;
        for(Reward reward : rewards) {
            if(attrs.contains(reward.attrIdx)) prec_ex_hit++;
        }
        prec_ex += prec_ex_hit / (double) rewards.size();
        double rec_ex_hit = 0;
        for(Integer idx : attrs) {
            for(Reward reward : rewards) {
                if(reward.attrIdx == (int)idx) {
                    rec_ex_hit++;
                }
            }
        }
        rec_ex += rec_ex_hit / (double) attrs.size();
        double prec_mac_hit = 0;
        double m = 0;
        for(FPGAttributeSet fpa : explanation.getItemsets()) {
            boolean flag = false;
            if(lastExplanation != null) {
                for(FPGAttributeSet lastFpa : lastExplanation.getItemsets()) {
                    if(lastFpa.items.keySet().containsAll(fpa.items.keySet())){
                        flag = true;
                    }
                }
            }
            if(!flag) {
                m++;
                for(String s : rele) {
                    String[] sp = s.split(",");
                    HashSet<String> hs = new HashSet<>(Arrays.asList(sp));
                    if(fpa.items.keySet().containsAll(hs)) {
                        prec_mac_hit++;
                    }
                }
            }
        }
        if(m != 0) {
            prec_mac += prec_mac_hit / m;
            mac_n++;
        }
        double rec_mac_hit = 0;
        for(String s : rele) {
            String[] sp = s.split(",");
            HashSet<String> hs = new HashSet<>(Arrays.asList(sp));
            for(FPGAttributeSet fpa : explanation.getItemsets()) {
                if(fpa.items.keySet().containsAll(hs)) {
                    rec_mac_hit++;
                }
            }
        }
        rec_mac += rec_mac_hit / (double) rele.size();
    }
}

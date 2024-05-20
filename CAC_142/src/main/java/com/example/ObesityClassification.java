package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.title.PaintScaleLegend;
import org.jfree.chart.ui.RectangleEdge;
import org.jfree.chart.ui.RectangleInsets;
import org.jfree.data.xy.DefaultXYZDataset;
import org.jfree.data.xy.XYZDataset;
import org.jfree.chart.renderer.LookupPaintScale;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.Scanner;

public class ObesityClassification {

    private static Instances data;

    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("C:\\Users\\vikas\\OneDrive\\Desktop\\TRISEM III\\JAVA\\CAC1\\CAC1\\CAC_142\\src\\main\\java\\com\\example\\ObesityDataSet_raw_and_data.csv");
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            Scanner scanner = new Scanner(System.in);
            boolean exit = false;

            while (!exit) {
                System.out.println("\nMenu:");
                System.out.println("1. Perform Exploratory Analysis");
                System.out.println("2. Correlation Analysis and Heatmap");
                System.out.println("3. Build and Evaluate Classifier");
                System.out.println("4. Visualize Decision Tree");
                System.out.println("5. Exit");
                System.out.print("Enter your choice: ");
                int choice = scanner.nextInt();

                switch (choice) {
                    case 1:
                        performExploratoryAnalysis(data);
                        break;
                    case 2:
                        performCorrelationAnalysisAndDisplayHeatmap(data);
                        break;
                    case 3:
                        buildAndEvaluateClassifier(data);
                        break;
                    case 4:
                        visualizeTree();
                        break;
                    case 5:
                        exit = true;
                        break;
                    default:
                        System.out.println("Invalid choice. Please try again.");
                }
            }
            scanner.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void performExploratoryAnalysis(Instances data) {
        // Exploratory analysis
        System.out.println("Exploratory Analysis:");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println("**************************************************************************");
        System.out.println("Attributes: ");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println(data.attribute(i));
        }
        System.out.println("**************************************************************************");

        System.out.println("Class attribute: " + data.attribute(data.classIndex()));
        System.out.println("Class distribution:");
        System.out.println(data.attributeStats(data.classIndex()));
        System.out.println("**************************************************************************");
        // Summary statistics
        System.out.println("\nSummary Statistics:");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println("Attribute: " + data.attribute(i).name());
            System.out.println("Type: " + data.attribute(i).type());
            if (data.attribute(i).isNumeric()) {
                System.out.println("Mean: " + data.attributeStats(i).numericStats.mean);
                System.out.println("Std. Deviation: " + data.attributeStats(i).numericStats.stdDev);
                System.out.println("Minimum: " + data.attributeStats(i).numericStats.min);
                System.out.println("Maximum: " + data.attributeStats(i).numericStats.max);
                System.out.println("***********************************************************************");
            }
        }
    }

    private static double[][] calculateCorrelationMatrix(Instances data) {
        int numAttributes = data.numAttributes();
        double[][] correlationMatrix = new double[numAttributes][numAttributes];

        for (int i = 0; i < numAttributes; i++) {
            for (int j = i; j < numAttributes; j++) {
                correlationMatrix[i][j] = calculateCorrelation(data.attributeToDoubleArray(i), data.attributeToDoubleArray(j));
                correlationMatrix[j][i] = correlationMatrix[i][j];
            }
        }

        return correlationMatrix;
    }

    private static double calculateCorrelation(double[] x, double[] y) {   //FINDING THE CORRELATION FOR EACH OF THE VARIABLES 
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;

        for (int i = 0; i < x.length; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        int n = x.length;
        return (n * sumXY - sumX * sumY) / Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    }

    private static void performCorrelationAnalysisAndDisplayHeatmap(Instances data) {
        System.out.println("\nCorrelation Analysis:");
        double[][] correlationMatrix = calculateCorrelationMatrix(data);
        displayHeatmap(correlationMatrix, data);
    }

    private static void displayHeatmap(double[][] matrix, Instances data) {
        // Create dataset for heatmap
        XYZDataset dataset = createDataset(matrix, data.numAttributes());

        // Create heatmap chart
        JFreeChart chart = createChart(dataset, data.numAttributes(), matrix);

        // Create and display JFrame for heatmap
        JFrame frame = new JFrame("Correlation Heatmap");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

    private static XYZDataset createDataset(double[][] matrix, int numAttributes) {
        DefaultXYZDataset dataset = new DefaultXYZDataset();
        double[] xValues = new double[numAttributes * numAttributes];
        double[] yValues = new double[numAttributes * numAttributes];
        double[] zValues = new double[numAttributes * numAttributes];

        int index = 0;
        for (int i = 0; i < numAttributes; i++) {
            for (int j = 0; j < numAttributes; j++) {
                xValues[index] = i;
                yValues[index] = j;
                zValues[index] = matrix[i][j];
                index++;
            }
        }

        dataset.addSeries("Correlation", new double[][]{xValues, yValues, zValues});
        return dataset;
    }

    private static JFreeChart createChart(XYZDataset dataset, int numAttributes, double[][] correlationMatrix) {  //PLOTTING A HEATMAP WITH THE CORRELATION OF THE VARIABLES FOUND
        NumberAxis xAxis = new NumberAxis("Attributes");
        xAxis.setLowerMargin(0.0);
        xAxis.setUpperMargin(0.0);
        xAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        NumberAxis yAxis = new NumberAxis("Attributes");
        yAxis.setLowerMargin(0.0);
        yAxis.setUpperMargin(0.0);
        yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        XYBlockRenderer renderer = new XYBlockRenderer();
        LookupPaintScale paintScale = new LookupPaintScale(-1.0, 1.0, Color.white);
        for (double v = -1.0; v <= 1.0; v += 0.01) {
            if (v > 0) {
                paintScale.add(v, new Color(255, (int) (255 * (1 - v)), (int) (255 * (1 - v))));
            } else {
                paintScale.add(v, new Color((int) (255 * (1 + v)), (int) (255 * (1 + v)), 255));
            }
        }
        renderer.setPaintScale(paintScale);

        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer) {
            @Override
            public void drawAnnotations(Graphics2D g2, Rectangle2D dataArea, PlotRenderingInfo info) {
                super.drawAnnotations(g2, dataArea, info);
                FontMetrics fm = g2.getFontMetrics();
                for (int i = 0; i < numAttributes; i++) {
                    for (int j = 0; j < numAttributes; j++) {
                        String text = String.format("%.2f", correlationMatrix[i][j]);
                        int width = fm.stringWidth(text);
                        int height = fm.getHeight();
                        double x = dataArea.getX() + dataArea.getWidth() * i / numAttributes + (dataArea.getWidth() / numAttributes - width) / 2;
                        double y = dataArea.getY() + dataArea.getHeight() * (numAttributes - j - 1) / numAttributes + (dataArea.getHeight() / numAttributes + height) / 2;
                        g2.drawString(text, (float) x, (float) y);
                    }
                }
            }
        };
        plot.setBackgroundPaint(Color.white);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);
        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5));

        JFreeChart chart = new JFreeChart("Correlation Heatmap", plot);
        chart.removeLegend();
        chart.setBackgroundPaint(Color.white);

        NumberAxis scaleAxis = new NumberAxis("Correlation");
        scaleAxis.setRange(-1.0, 1.0);
        PaintScaleLegend legend = new PaintScaleLegend(paintScale, scaleAxis);
        legend.setPosition(RectangleEdge.RIGHT);
        chart.addSubtitle(legend);

        return chart;
    }

    private static void buildAndEvaluateClassifier(Instances data) {   //BUILDING AND EVALUATING THE MODEL USING A CONFUSION MATRIX
        try {
            Classifier classifier = new J48();
            classifier.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void visualizeTree() {   //DISPLAYING THE DECISION TREE
        try {
            Classifier classifier = new J48();
            classifier.buildClassifier(data);
            TreeVisualizer treeVisualizer = new TreeVisualizer(null, ((J48) classifier).graph(), new PlaceNode2());
            JFrame jFrame = new JFrame("Decision Tree Visualizer");
            jFrame.setSize(1800, 1200);
            jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            jFrame.getContentPane().setLayout(new BorderLayout());
            jFrame.getContentPane().add(treeVisualizer, BorderLayout.CENTER);
            jFrame.setVisible(true);
            treeVisualizer.fitToScreen();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

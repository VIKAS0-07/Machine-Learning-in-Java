# Machine-Learning-in-Java

Obesity Classification using Machine Learning in Java
This project implements a machine learning model to classify obesity levels using the WEKA library for data preprocessing and model building. It also includes data visualization functionalities using JFreeChart. The project covers exploratory analysis, correlation heatmaps, classifier evaluation, and decision tree visualization.

Features
Exploratory Data Analysis
Analyze dataset attributes, class distribution, and basic summary statistics.

Correlation Heatmap
Compute and visualize correlations between dataset attributes using a heatmap.

Model Building and Evaluation
Train and evaluate a J48 Decision Tree classifier. Evaluate model performance with metrics such as precision, recall, and a confusion matrix.

Decision Tree Visualization
Visualize the structure of the trained decision tree.

Technologies Used
Java: Core programming language
WEKA: Machine learning library for data preprocessing and classification
JFreeChart: Library for heatmap visualization
Swing: Java's GUI toolkit for displaying visualizations
Installation and Setup
Prerequisites
JDK 8 or higher
Apache Maven (optional, for dependency management)
WEKA library
JFreeChart library
Steps
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/obesity-classification-java.git
Add the WEKA and JFreeChart JAR files to your project build path.

Place the dataset (ObesityDataSet_raw_and_data.csv) in the specified location or modify the file path in the code:

java
Copy code
DataSource source = new DataSource("path/to/ObesityDataSet_raw_and_data.csv");
Compile and run the program:

bash
Copy code
javac ObesityClassification.java
java ObesityClassification
Dataset
The dataset contains attributes related to dietary habits, physical activity, and personal characteristics. The last attribute is the class label for obesity levels.
Source: [Add Dataset Source or Description]

Usage
Run the program and select options from the menu:

1: Perform Exploratory Data Analysis.
2: Generate a correlation heatmap.
3: Train and evaluate a decision tree classifier.
4: Visualize the decision tree.
Analyze the results printed in the console or displayed in visualizations.

Project Structure
bash
Copy code
src/
├── main/
│   ├── java/
│   │   └── com/example/
│   │       ├── ObesityClassification.java   # Main Java program
├── resources/
│   └── ObesityDataSet_raw_and_data.csv      # Dataset
Sample Output
Console Output
Exploratory Analysis: Summary statistics, class distribution.
Classifier Results: Accuracy, confusion matrix, precision, recall.
Visualizations
Heatmap: Displays correlations between attributes.
Decision Tree: Interactive visualization of the trained decision tree.
Improvements
Add support for other classifiers (e.g., RandomForest, NaiveBayes).
Extend the correlation analysis to handle missing data.
Include advanced visualizations for model evaluation (ROC curves, precision-recall plots).
License
This project is licensed under the MIT License.

Acknowledgments
WEKA library: https://www.cs.waikato.ac.nz/ml/weka/
JFreeChart library: https://www.jfree.org/jfreechart/

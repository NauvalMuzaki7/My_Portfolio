# Muhammad Nauval Muzaki - Portfolio
## Data Analyst | Data-Driven Solutions

As a final-year Statistics major, I bring a robust foundation in data analysis, visualization, and organization. My passion for data analytics is demonstrated through my strong analytical skills, having knowledge of developing Deep Learning and classic Machine Learning fundamentals, honed by hands-on experience in interpreting complex data sets. My academic journey has been marked by continuous growth in the use of advanced statistical tools, preparing me for challenging roles in the field of data analytics.

# Top Projects

## 1. SANO: Disease Condition Predictor
**Description**: My team was motivated by the prospect of leveraging machine learning techniques to develop a tool that could assist in the timely identification of these critical health conditions. The "Sano - Disease Condition Predictor" project focuses on using machine learning implementation to predict three major diseases: heart disease, stroke, and diabetes using ML classification algorithm.

- [Link to Repository](https://github.com/NauvalMuzaki7/SANO_Bangkit-Academy-Project): Repository of our Classification Project

**Requirements**: Python, Tensorflow, Numpy, Pandas, Matplotlib, Seaborn.

**Result**: The model demonstrates excellent performance across all conditions, with high accuracy, precision, recall, and F1-score for heart disease, effectively identifying cases with minimal false positives and negatives. It shows strong overall performance for diabetes, though with a slightly lower recall, indicating room for improvement in capturing all true positive cases. For stroke, the model excels in all metrics, showcasing a robust ability to distinguish between positive and negative cases, resulting in an exceptional predictive capability across all three conditions.

### Data Cleaning
**Description**: The dataset used in this project requires preprocessing to ensure the accuracy and reliability of the predictions. The data cleaning process involved several key steps:
- Handling Missing Data
- Outlier Detection
- Encoding Categorical Variables
- Splitting the Dataset

### Model Training
**Description**: The model was built using a classification algorithm in TensorFlow. The following steps were undertaken to train the model:

- Model Architecture:
A deep neural network (DNN) architecture was chosen, consisting of multiple dense layers, with ReLU activation for hidden layers and a softmax/sigmoid activation in the output layer based on the classification type.

- Loss Function:
A suitable loss function, such as binary cross-entropy or categorical cross-entropy, was chosen depending on the classification problem.

- Optimizer:
Adam optimizer was used to minimize the loss function, ensuring efficient and effective model training.

- Training:
The model was trained over multiple epochs (e.g., 50-100 epochs) to optimize weights and biases, while tracking performance on the validation set to prevent overfitting.

### Model Evaluation
The model was evaluated using several performance metrics:

- Accuracy: Measures the overall correct predictions of the model.
- Precision: Assesses the number of true positive predictions out of all positive predictions.
- Recall: Measures the ability of the model to identify all true positive cases.
- F1-score: A balanced measure combining precision and recall, useful when dealing with imbalanced datasets.

Results:
The model demonstrates excellent performance across all conditions, with high accuracy, precision, recall, and F1-score for heart disease, effectively identifying cases with minimal false positives and negatives.
The diabetes prediction shows good performance, though with a slightly lower recall, indicating potential improvements in capturing all true positive cases.
For stroke, the model excels in all metrics, showcasing a strong ability to distinguish between positive and negative cases.

- [Link to Model Result](https://github.com/mariown/C241-PS439_Sano_Bangkit/tree/ML): The result consists of all the model evaluation metrics.

## 2. Bike Sharing Data Analysis
**Description**: This dashboard is a powerful tool designed to provide comprehensive insights into bike sharing systems usage patterns and trends. This interactive dashboard leverages data visualization techniques to present key metrics and analytics in a user-friendly interface, catering to both casual users and data enthusiasts alike.  

- [Link to Dashboard](https://dashboard-bikesharing-nauval.streamlit.app/): Interactive dashbord for accessing the data visualization of the Bike Sharing Data Analysis Project.

**Requirements**: Python, Pandas, Matplotlib, Seaborn, Streamlit. 

**Result**: trend of total bike users in 2011 is having uptrend until June, and having down after June to December. For total bike users in 2012 we can see that the trend is still indicating uptrend until September and have some sharp downtrend until December.  

### Data Wrangling
**Description**: Data wrangling is the process of gathering, assessing, and cleaning raw data into a format suitable for analysis. I also change the variable name to the suitable name for easier analysis.  

**Requirements**: Python, Pandas, MySQL.  

**Result**: The clean data without missing values, duplicate data, and inaccurate values so data suitable to be explored and for further analysis.  

### Exploratory Data Analysis and Data Visualization
**Description**: Provides summary statistics and descriptive statistics about the dataset, identifying patterns and trends present in the data through visualizations.  

**Requirements**: Python, Pandas, Matplotlib, Seaborn.  

**Result**: There are many tables and chart shown in the [Dashboard](https://dashboard-bikesharing-nauval.streamlit.app/).

### Forecasting
**Description**: Used Exponential Smoothing (trying without using Machine Learning algorithm).  

**Requirements**: Python, Pandas, Matplotlib, Seaborn.  

**Result**: After december 2012, the trend of the total bike users tend to a bit uptrend. and the forecast pattern during following the actual data visually following the pattern.

## 3. K-Means Clustering of Poverty in Central Java
**Description**: Poverty poses a complex challenge and is a significant issue in Indonesia. The province of Central Java is characterized by a substantial population living in poverty. The aim of this research is to conduct an analysis of poverty clustering based on Districts/Cities in Central Java in 2021 using the K-Means Clustering method. As analytical material, secondary data comprising poverty indicators from the Central Statistics Agency of Central Java for the year 2021 has been utilized. The K-Means Clustering method is employed to group Districts/Cities based on similar characteristics of poverty.  

**Result**: The research results indicate the presence of two main clusters, namely Cluster 1 with an average percentage of the population in poverty at 10.75%, and Cluster 2 with an average percentage of the population in poverty at 13.97%. Cluster 2 involves several Districts/Cities with a higher poverty rate compared to Cluster 1.  

### Data Wrangling
**Description**: Gather data from BPS Website, Then assess the data, and clean raw data into a format suitable for analysis.  

**Requirements**: R, MySQL.  

**Result**: The clean data without missing values, duplicate data, and inaccurate values so data suitable to be explored and for further analysis.  

### K-Means Clustering
Based on the results of determining the number of clusters using the silhouette method, the ideal number of clusters obtained is 2 clusters. This means that the classification of poverty in the regions of Central Java Province in 2021 will be grouped into 2 clusters.  

## 4. JKSE Stock Price Prediction Using Long-Short Term Memory With and Without Stationarity Input
**Description**: In conducting LSTM (Long Short-Term Memory) analysis using JKSE (Jakarta Stock Exchange) stock prices, the focus lies on leveraging LSTM, a type of recurrent neural network (RNN), renowned for its capability to capture long-term dependencies in sequential data, to predict future stock price movements.  

**Requirements**: Python, Tensorflow, Numpy, Pandas, Matplotlib, Seaborn.  

**Result**: The best model for predicting validation data is using the Bidirectional LSTM method with an RMSE value of 26.26 and an MAE value of 20.44. Meanwhile, to predict data for the next 4 periods (days), the best model is the Vanilla LSTM model with an RMSE value of 37.21 and MAE of 28.72.  

### Data Wrangling
**Description**: Gather JKSE Stock Price data from Yahoo Finance, Then assess the data, and make sure the data in clean format so it's suitable for analysis.   

**Result**: The clean data without missing values, duplicate data, and inaccurate values so data suitable to be explored and for further analysis. 

### Data Stationarity Test
**Description**: Using the significance values of Augmented Dickey Fuller Test and Kwiatkowski-Phillips-Schmidt-Shin Test.  

**Result**: Based on the Augmented Dickey Fuller test before differencing is not stationary but after differencing it becomes stationary. In addition, the Kwiatkowski-Phillips-Schmidt-Shin test shows significant results both before differencing and after differencing, which means that the time series is not stationary in the level sense, which means that there is a consistent trend or pattern in the time series.

### 5 Types of LSTM Model
- Vanilla LSTM
- Stacked LSTM
- Bidirectional LSTM
- 2 Stacked Bidirectional LSTM
- 3 Stacked Bidirectional LSTM

### Forecasting
JKSE Stock Index Close Price data using stationarity handling modeled using the five methods, the best model for predicting validation data is to use the Bidirectional LSTM method with an RMSE value of 41.94 and an MAE value of 32.61.  

Meanwhile, JKSE Stock Index Close Price Data without using stationarity handling modeled using the five methods, the best model for predicting validation data is to use the Bidirectional LSTM method with an RMSE value of 26.26 and an MAE value of 20.44. Meanwhile, to predict the next 4 periods (days) of data, the best model is the Vanilla LSTM model with an RMSE value of 37.21 and MAE of 28.72.

## 5. Using Multidimensional Scaling for Grouping Social Media Influencers
**Description**: This project aims to group social media influencers based on multiple factors such as follower count, engagement rates, and post frequency, using Multidimensional Scaling (MDS). MDS is used to visualize the similarity or dissimilarity between influencers, providing insights into how different influencers compare and form clusters.

**Requirements**: Python, Pandas, scikit-learn, Matplotlib, Seaborn.

**Result**: The project successfully grouped influencers into distinct clusters based on their engagement metrics. These groupings help brands and marketing teams target the right influencers for their campaigns, optimizing outreach efforts and increasing audience engagement.

### Data Wrangling
**Description**: Gather social media influencer data from various platforms, assess and clean the data for analysis.

**Requirements**: R, Python, Pandas, MySQL.

**Result**: Clean and structured data, suitable for exploration and further analysis.

### Multidimensional Scaling (MDS) Implementation
**Description**: Using MDS to visualize the relative positioning of influencers based on similarity and clustering patterns.

**Result**: Visual representation of influencers in a 2D space, highlighting clear clusters based on engagement and reach.

### Link to Project



### Link to Project

- [SANO: Disease Condition Predictor](https://github.com/NauvalMuzaki7/SANO_Bangkit-Academy-Project): Repository of Classification Machine Learning Project
- [Bike Sharing Project](https://github.com/NauvalMuzaki7/Data_Analysis_Project): Showing all analysis steps of Bike Sharing Dataset.
- [K-Means Clustering Project](https://github.com/NauvalMuzaki7/Clustering_Project): Grouping Poverty by District/City in Central Java in 2021 Using K-Means Clustering.
- [Support Vector Regression Project](https://github.com/NauvalMuzaki7/LSTM_Project/blob/main/Univariat_Timeseries_Close_JKSE_LSTM_without_differencing_method_(1).ipynb): JKSE Stock Price Prediction Using Long-Short Term Memory With and Without Stationarity Input.
- [Multidimensional Scaling Project](https://github.com/NauvalMuzaki7/MDS_Project): Grouping Social Media Influencers using Multidimensional Scaling.

## Skills

- Programming Language: Python, SQL, R
- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn, TensorFlow
- Databases: MySQL
- Tools: Google Colab, Git, Tableau

## Education and Certification

- Bachelor of Statistics, Padjadjaran University, 2021-2025
- [Machine Learning Path, Bangkit Academy 2024 By Google, GoTo, Traveloka, 2024](https://drive.google.com/drive/folders/1DCkU5j_J5sQvrRWmoE3zBbU6M805KYz3?usp=drive_link)
- [Google Data Analytics Professional Certificate](https://coursera.org/verify/professional-cert/J3NXMPEVB6MY)
- [Machine Learning Specialization](https://coursera.org/share/177b657224b2626ee72391291a0ee48b)
- [HackerRank SQL Intermediate](https://www.hackerrank.com/certificates/a88e4b232962)

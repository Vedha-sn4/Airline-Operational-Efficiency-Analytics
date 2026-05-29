Project Overview
This project is an end-to-end data analytics and business intelligence platform designed to evaluate and optimize commercial airline performance. The pipeline integrates multiple synthetic aviation datasets—covering flight operations, weather metrics, airport traffic, fuel price volatility, and customer reviews—into a centralized, optimized Star Schema Data Warehouse. By processing raw operational inputs and integrating predictive machine learning outputs, the platform delivers actionable insights through an executive-level Power BI Dashboard and structural data reports, bridging the gap between technical data engineering and business stakeholders.

Core Objectives
Multi-Domain Data Ingestion: Systematically blend disparate data sources including Operations, Weather, Traffic, Finance, and Customer Feedback.

KPI Engineering: Formulate advanced diagnostic metrics like Route Efficiency Scores, Congestion Severity, and Customer Satisfaction Intensity.

Predictive Analytics Integration: Incorporate machine learning model forecasting outputs to conduct operational variance analysis.

Data Warehouse Modeling: Architect a high-performance Star Schema with decoupled Fact and Dimension tables optimized for BI tools.

Interactive Business Intelligence: Build a multi-page Power BI report to track airline operational health and customer sentiment trends.

Data Architecture & Star Schema Model
The pipeline transforms raw data into an optimized relational model to maximize query performance and reporting efficiency in Power BI.

Schema Relationships
Dim_Date (DateID) to Fact_Flights (DateID)

Dim_Route (RouteID) to Fact_Flights (RouteID)

Dim_Airport (AirportID) to Fact_Flights (AirportID)

Dim_Weather (WeatherID) to Fact_Flights (WeatherID)

Dim_Fuel (FuelID) to Fact_Flights (FuelID)

Fact_Flights (FlightID) to Fact_Delays (FlightID)

Dim_Sentiment (SentimentID) to Fact_Flights (SentimentScore/ID matching)

Data Dictionary & Outputs
1. Fact Tables
Fact_Flights.csv: The central operational star schema table containing keys to dimensions and core flight metrics such as DelayMinutes, DelayCost, RouteEfficiencyScore, and SentimentScore.

Fact_Delays.csv: Specialized granularity for delay analysis, housing DelayCategory, DelayDuration, ActualDelay, and the ML-generated PredictedDelay.

2. Dimension Tables
Dim_Date.csv: Time-intelligence fields including Date, Month, Quarter, Year, and DayOfWeek.

Dim_Route.csv: Structural route tracking including OriginAirport, DestinationAirport, and RouteDistance.

Dim_Airport.csv: Infrastructure metrics including AirportName, City, Country, and CongestionSeverity.

Dim_Weather.csv: Meteorological conditions including WeatherCondition, Temperature, Visibility, and WeatherSeverity.

Dim_Fuel.csv: Financial overhead tracking including FuelPrice, and FuelVolatilityIndex.

Dim_Sentiment.csv: Granular customer text-mining metadata including ReviewText, SentimentScore, and SentimentCategory.

Pipeline Workflow
The workflow begins with raw CSV inputs which are fed into a Python/Pandas Data Pipeline. From there, data undergoes machine learning artifact scoring to append predictions before exporting clean Star Schema CSV files. Finally, these files are imported into Power BI for data modeling and dashboard creation.

1. Data Engineering & Transformation (Python)
Unifies datetime fields, standardizes naming conventions, and handles missing values/outliers.

Eliminates noise by removing cancelled flights before evaluating operational costs.

Engineers advanced metrics using vectorization in Pandas.

2. Analytics & Predictive Modeling Integration
Leverages pre-trained machine learning artifacts (delay_classification_model.joblib and delay_regression_model.joblib) to append predictive delay scores to the data layer.

Provides the data structure necessary to evaluate model error rates and operational risk calculations within reports.

3. Power BI Implementation
Data Load: Import all CSV elements located in the exports directory.

Relationships: Connect the central Fact_Flights table to all surrounding dimension tables using their respective ID fields, and establish a 1-to-many relationship from Fact_Flights to Fact_Delays based on FlightID.

Dashboard Visuals & Insights
Page 1: Executive Operational Efficiency
KPI Cards: Average Delay Minutes, Total Delay Financial Cost, Fleet Route Efficiency.

Clustered Bar Chart: Top 10 Most Delayed Routes vs. Congestion Severity Index.

Line Chart: Fuel Price Volatility Trends mapped against Route Operational Costs.

Page 2: Predictive Delay & Variance Analysis
Scatter Plot: PredictedDelay vs. ActualDelay to visualize model prediction variance and flag operational bottlenecks.

Matrix Visual: Delay Categories breakdown (Minor, Major, Severe) cross-referenced by Weather Conditions and Airport Hubs.

Page 3: Passenger Sentiment & Experience
Donut Chart: Share of Sentiment Categories (Positive, Neutral, Negative).

Scatter Chart: Customer Satisfaction Intensity vs. Average Flight Delay Minutes to prove the correlation between operational lulls and brand detraction.

Table Visual: Detailed customer review text-mining filtered by low sentiment scores to flag urgent issues.

Technology Stack
Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-Learn, Joblib

Data Warehouse Modeling: Relational Star Schema Optimization

Business Intelligence: Microsoft Power BI (DAX, Power Query)

# Airline Operational Efficiency & Passenger Sentiment Analytics

## Project Overview
This project is an end-to-end data analytics and business intelligence platform designed to evaluate and optimize commercial airline performance. The pipeline integrates multiple synthetic aviation datasets—covering flight operations, weather metrics, airport traffic, fuel price volatility, and customer reviews—into a centralized, optimized **Star Schema Data Warehouse**. 

By processing raw operational inputs and integrating predictive machine learning outputs, the platform delivers actionable insights through an executive-level **Power BI Dashboard** and structural data reports, successfully bridging the gap between technical data engineering and business stakeholders.

---

## Core Objectives
* **Multi-Domain Data Ingestion:** Systematically blend disparate data sources including Operations, Weather, Traffic, Finance, and Customer Feedback.
* **KPI Engineering:** Formulate advanced diagnostic metrics like *Route Efficiency Scores*, *Congestion Severity*, and *Customer Satisfaction Intensity*.
* **Predictive Analytics Integration:** Incorporate machine learning model forecasting outputs to conduct operational variance analysis.
* **Data Warehouse Modeling:** Architect a high-performance Star Schema with decoupled Fact and Dimension tables optimized for BI tools.
* **Interactive Business Intelligence:** Build a multi-page Power BI report to track airline operational health and customer sentiment trends.

---

## Data Architecture & Star Schema Model
The pipeline transforms raw data into an optimized relational model to maximize query performance and reporting efficiency in Power BI.

### Schema Relationships
* `Dim_Date (DateID)` $\rightarrow$ `Fact_Flights (DateID)`
* `Dim_Route (RouteID)` $\rightarrow$ `Fact_Flights (RouteID)`
* `Dim_Airport (AirportID)` $\rightarrow$ `Fact_Flights (AirportID)`
* `Dim_Weather (WeatherID)` $\rightarrow$ `Fact_Flights (WeatherID)`
* `Dim_Fuel (FuelID)` $\rightarrow$ `Fact_Flights (FuelID)`
* `Dim_Sentiment (SentimentID)` $\rightarrow$ `Fact_Flights (SentimentScore/ID matching)`
* `Fact_Flights (FlightID)` $\rightarrow$ `Fact_Delays (FlightID)` *(1-to-many relationship)*

---

## Data Dictionary & Outputs

### Fact Tables
| File Name | Description | Key Fields & Metrics |
| :--- | :--- | :--- |
| **`Fact_Flights.csv`** | Central operational star schema table. | `DelayMinutes`, `DelayCost`, `RouteEfficiencyScore`, `SentimentScore` |
| **`Fact_Delays.csv`** | Specialized granularity for detailed delay analysis. | `DelayCategory`, `DelayDuration`, `ActualDelay`, ML-generated `PredictedDelay` |

### Dimension Tables
| File Name | Description | Attributes |
| :--- | :--- | :--- |
| **`Dim_Date.csv`** | Time-intelligence fields for temporal tracking. | `Date`, `Month`, `Quarter`, `Year`, `DayOfWeek` |
| **`Dim_Route.csv`** | Structural route tracking across the network. | `OriginAirport`, `DestinationAirport`, `RouteDistance` |
| **`Dim_Airport.csv`** | Infrastructure and facility metrics. | `AirportName`, `City`, `Country`, `CongestionSeverity` |
| **`Dim_Weather.csv`** | Meteorological conditions mapped to flight times. | `WeatherCondition`, `Temperature`, `Visibility`, `WeatherSeverity` |
| **`Dim_Fuel.csv`** | Financial overhead and economic tracking. | `FuelPrice`, `FuelVolatilityIndex` |
| **`Dim_Sentiment.csv`** | Granular customer text-mining metadata. | `ReviewText`, `SentimentScore`, `SentimentCategory` |

---

## Pipeline Workflow
The data pipeline follows a structured, modular workflow:
1. **Raw Ingestion:** Ingest raw CSV files into a Python/Pandas pipeline.
2. **ML Enrichment:** Apply pre-trained machine learning models to generate predictive data layers.
3. **Star Schema Export:** Structure and clean data into final schema CSVs.
4. **BI Powering:** Import and model files inside Power BI to render the interactive dashboard.

### 1. Data Engineering & Transformation (Python)
* **Data Cleaning:** Unifies datetime fields, standardizes naming conventions, and systematically handles missing values/outliers.
* **Noise Reduction:** Eliminates noise by filtering out cancelled flights before evaluating operational costs.
* **KPI Engineering:** Computes complex operational and financial metrics using high-performance vectorization in Pandas.

### 2. Analytics & Predictive Modeling Integration
* Leverages pre-trained machine learning artifacts (`delay_classification_model.joblib` and `delay_regression_model.joblib`) to append predictive delay scores to the data layer.
* Provides the necessary reporting data structures to evaluate model error rates and operational risk calculations.

### 3. Power BI Implementation
* **Data Load:** Seamless automated import of all processed CSV elements located in the `exports/` directory.
* **Data Modeling:** Connections established from the central `Fact_Flights` table to all surrounding dimension tables using their respective ID fields, with a 1-to-many relationship tracking down to `Fact_Delays`.

---

## Dashboard Visuals & Insights

### Page 1: Executive Operational Efficiency
* **KPI Cards:** `Average Delay Minutes`, `Total Delay Financial Cost`, and `Fleet Route Efficiency`.
* **Clustered Bar Chart:** *Top 10 Most Delayed Routes* vs. *Congestion Severity Index*.
* **Line Chart:** *Fuel Price Volatility Trends* mapped dynamically against *Route Operational Costs*.

### Page 2: Predictive Delay & Variance Analysis
* **Scatter Plot:** `PredictedDelay` vs. `ActualDelay` to visualize model prediction variance and flag operational bottlenecks.
* **Matrix Visual:** Detailed `Delay Categories` breakdown (*Minor, Major, Severe*) cross-referenced by `Weather Conditions` and major `Airport Hubs`.

### Page 3: Passenger Sentiment & Experience
* **Donut Chart:** Market share of `Sentiment Categories` (*Positive, Neutral, Negative*).
* **Scatter Chart:** *Customer Satisfaction Intensity* vs. *Average Flight Delay Minutes* to visually isolate the mathematical correlation between operational lulls and brand detraction.
* **Table Visual:** Granular customer review text-mining filtered specifically by low sentiment scores to flag urgent operational issues.

---

## Technology Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib
* **Data Warehouse Modeling:** Relational Star Schema Optimization
* **Business Intelligence:** Microsoft Power BI (DAX, Power Query)

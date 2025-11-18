# Gym Analytics Dashboard

An interactive web application built with Python (Streamlit) designed to analyze workout efficiency, member demographics, and health correlations. The project combines static CSV data analysis with dynamic data fetching (web scraping) from external sources.

## Key Features

### 1. Analytical Dashboard (Streamlit)
The application provides an interactive interface for data exploration:
* **Dynamic Filtering:** Select data based on gender and age range.
* **Key Performance Indicators (KPIs):** Instant view of total members, average age, BMI, and average calories burned.
* **Advanced Visualization (Plotly):**
    * Workout type distribution (Pie chart).
    * Calorie burn analysis (Box plot).
    * BMI vs. Experience Level relationship.
    * Age histograms and Correlation Heatmaps.
* **Custom Chart Module:** Allows users to select specific X and Y variables to analyze custom correlations.

### 2. Web Scraping Module
The script includes a module that fetches current calorie burn data:
* Automatically scrapes data tables from *bieganie.pl*.
* HTML parsing using `BeautifulSoup`.
* Data processing and conversion to Pandas DataFrame (Activity, kcal/h, Time per 100kcal).

### 3. Feature Engineering
The project goes beyond raw data display by calculating advanced metrics:
* **Intensity Score:** A calculated metric based on heart rate (Max/Resting) and calories burned per session duration.
* **Fitness Level:** Algorithm that classifies users (Beginner/Intermediate/Advanced) based on BMI, resting heart rate, and workout frequency.
* **Efficiency Score:** Ratio of calories burned to session duration.

## Technologies Used

* **Python 3.12**
* **Streamlit** - Web interface and dashboard framework.
* **Pandas & NumPy** - Data manipulation and analysis.
* **Plotly Express & Graph Objects** - Interactive charting and visualization.
* **BeautifulSoup4 & Requests** - Web scraping and HTTP requests.

## Setup and Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TWÓJ_NICK/gym-tracking-analysis.git](https://github.com/TWÓJ_NICK/gym-tracking-analysis.git)
    cd gym-tracking-analysis
    ```

2.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run main.py
    ```

## Project Structure

* `main.py` - Main application script containing Streamlit logic and scraping functions.
* `gym_members_exercise_tracking.csv` - Input dataset containing member and workout statistics.
* `requirements.txt` - List of Python dependencies.

---

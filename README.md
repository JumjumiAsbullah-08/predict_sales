# Predict Sales Project

This repository contains a Streamlit web application for predicting sales based on various features. The prediction model is built using scikit-learn and data manipulation is done using pandas and numpy. Visualization is handled with Altair, Matplotlib, and Plotly. The project also involves reading and writing Excel files using openpyxl.

## Libraries Used

- [Streamlit](https://streamlit.io/): For creating interactive web applications with Python.
- [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
- [NumPy](https://numpy.org/): For numerical computing in Python.
- [scikit-learn](https://scikit-learn.org/stable/): For machine learning modeling and predictive analytics.
- [Altair](https://altair-viz.github.io/): For declarative statistical visualization.
- [Matplotlib](https://matplotlib.org/): For creating static, animated, and interactive visualizations in Python.
- [Plotly](https://plotly.com/python/): For interactive and publication-quality graphs.

## Installation

To run this project locally, make sure you have Python installed. Clone the repository and install the required libraries using pip:

```bash
pip install -r requirements.txt
```
## Usage
Navigate to the project directory.
Install the required dependencies as mentioned above.
Run the Streamlit application:
```
streamlit run app.py
```
This will start the Streamlit server locally, and you can access the application in your web browser at http://localhost:8501.

## Folder Structure
app.py: Main Streamlit application file.
requirements.txt: List of Python packages required for this project.
data/: Directory containing datasets or input files.
models/: Directory for machine learning models (if applicable).
plots/: Directory for saving generated plots or visualizations.

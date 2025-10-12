California Housing Price Predictor
This project presents a comprehensive, end-to-end data science application designed to predict median house values in California. It features an interactive web application built with Streamlit that provides a seamless and intuitive user experience. The backend is powered by a sophisticated, tuned XGBoost machine learning model.

A key architectural feature of this project is its model chaining approach. Instead of requiring users to input complex statistical data, the application uses a series of "helper" models to automatically derive necessary features—such as neighborhood income level and ocean proximity—based on a simple map click. This design philosophy bridges the gap between a powerful, data-hungry model and a user-friendly, accessible interface.

Live Demo
(A link to your deployed Streamlit application will go here once you deploy it!)

Features
Interactive Map Input: The primary user interaction is a map of California, powered by Folium and rendered within Streamlit. Users can pan and zoom, but the view is constrained to the state's boundaries to ensure valid geographical inputs. A single click captures the precise latitude and longitude for the prediction.

User-Friendly Abstraction: The application intentionally abstracts away the complex features of the original dataset. Instead of asking for statistical inputs like "median income for block" or "total households," it prompts the user for simple, relatable details like the age of a house, the number of rooms, and a general description of the neighborhood (e.g., 'Suburban', 'Urban', 'Rural').

Automated Feature Prediction: This is the core of the model chaining strategy. When a user selects a location, two helper models run instantly in the background:

A K-Nearest Neighbors model predicts the area's ocean proximity category.

A highly-tuned Random Forest model predicts the median income level for that neighborhood.
This automated process enriches the data provided to the main model without requiring any expert knowledge from the user.

Dual Prediction Output: Upon successful prediction, the application presents two key pieces of information: the estimated median house value for that location and the predicted ocean proximity category, providing a more complete picture of the property.

Robust Input Validation: To ensure the model's predictions are reliable, the application enforces strict input validation. The interactive map is visually and functionally constrained to California's geographical boundaries. The main "Predict" button remains disabled until a valid location within these bounds is selected, preventing out-of-scope predictions.

Project Workflow
This project follows a complete machine learning workflow, from initial data exploration and feature engineering to model training, evaluation, and final application deployment.

Data Loading and Preprocessing:

The workflow begins by loading the housing.csv dataset, which is based on the 1990 California census data, using the Pandas library.

An initial exploratory data analysis reveals the presence of missing values, particularly in the total_bedrooms column. To maintain data integrity for the model, a simple and effective strategy of dropping all rows with any missing values is employed.

Feature Engineering:

To enhance the model's predictive power, new, more insightful features are engineered from the existing data. This step is based on the hypothesis that relative metrics can be more powerful than absolute counts.

bedroom_ratio: Calculated as total_bedrooms / total_rooms. This feature helps capture the property's composition (e.g., family homes vs. studio-like layouts).

household_rooms: Calculated as total_rooms / households. This feature provides a proxy for the average size of homes in a given block.

Model Training - A Chained Approach:
The core of the project's design is a two-stage training process, separating the helper models from the main prediction model.

Helper Models (Trainer_helper_model.ipynb): Two specialized models are trained to automate data generation.

Ocean Proximity Model: A KNeighborsClassifier is trained to classify a location into one of five ocean proximity categories based solely on its latitude and longitude. K-NN is well-suited for this task due to the inherent spatial nature of the data. It achieves a high accuracy, making it a reliable feature generator.

Income Level Model: A RandomForestRegressor is trained to predict a neighborhood's median_income (a numerical value) from its latitude and longitude. The choice of Random Forest was the result of a competitive evaluation where it was benchmarked against an XGBoostRegressor and a GradientBoostingRegressor. It was selected for its superior R² score on this specific sub-task.

Main Model (DMBI_Project_v2.ipynb): The primary XGBoost Regressor is trained to predict the final median_house_value.

It utilizes the full set of processed and engineered features.

To achieve maximum performance, the model undergoes extensive hyperparameter tuning using GridSearchCV. This process systematically searches through a predefined grid of parameter combinations (e.g., n_estimators, max_depth, learning_rate) using 3-fold cross-validation to find the optimal configuration that minimizes error and prevents overfitting.

Interactive Web Application (app.py):

The final stage is the user-facing application built with Streamlit.

The app's logic includes a crucial "translation layer." It converts the user's simple inputs (e.g., selecting 'Suburban' as the neighborhood type) into numerical estimates for population and households that the main model requires.

When the "Predict" button is clicked, the app gathers all the features—user-provided, translated, and auto-predicted by the helper models—into a final DataFrame.

This DataFrame is then passed through a saved StandardScaler (fit on the original training data) to ensure the data is in the correct scale and format before being fed into the main XGBoost model for the final prediction.

Tech Stack & Models
Framework: Streamlit

Core Libraries: Pandas, Scikit-learn, XGBoost, Folium, Matplotlib, Seaborn

Main Prediction Model: XGBoost Regressor (tuned with GridSearchCV)

Helper Models:

Income Prediction: Random Forest Regressor

Ocean Proximity: K-Nearest Neighbors Classifier

Project Structure
.
├── app.py                         # The main Streamlit application file
├── DMBI_Project_v2.ipynb          # Jupyter Notebook for training the main model
├── Trainer_helper_model.ipynb     # Jupyter Notebook for training helper models
├── housing.csv                    # The raw dataset used for training
├── requirements.txt               # Required Python packages for the project
└── ... (Model files are generated locally by the notebooks)

Setup and Installation
To run this project on your local machine, please follow these steps:

1. Prerequisites:

Python 3.8+

Git installed

2. Clone the Repository:

git clone [https://github.com/NirmayKadam/DMBI_project_mk2.git](https://github.com/NirmayKadam/DMBI_project_mk2.git)
cd DMBI_project_mk2

3. Create a Virtual Environment:
It's a strong best practice to use a virtual environment to manage project dependencies and avoid conflicts.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate

4. Install Dependencies:
Install all the required packages listed in the requirements.txt file using pip.

pip install -r requirements.txt

How to Run
Because the model files (.joblib) are not included in this repository, you must first train the models locally. This process will generate the necessary files for the application to run.

1. Train the Models:
Run the Jupyter Notebooks in the following order. This entire process may take several minutes, depending on your hardware.

First, open and run all cells in Trainer_helper_model.ipynb. This will create ocean_model.joblib, ocean_encoder.joblib, and income_model.joblib.

Second, open and run all cells in DMBI_Project_v2.ipynb. This will create housing_xgb_model.joblib and housing_scaler.joblib.

2. Launch the Streamlit App:
Once all five .joblib files have been successfully created, you can launch the application.

streamlit run app.py

This command starts a local web server, and your default browser should automatically open to the application's URL.

Potential Future Improvements
Deployment: Deploy the application to a cloud service like Streamlit Community Cloud or Hugging Face Spaces to make it publicly accessible.

Model Management with Git LFS: Implement Git Large File Storage (LFS) to include the pre-trained .joblib files directly in the repository. This would simplify the setup process for new users, as they would not need to re-train the models.

Advanced Feature Engineering: Experiment with more complex features, such as creating geographical clusters to represent distinct real estate sub-regions or incorporating polynomial features for key variables.

Data Visualization: Enhance the Streamlit app by adding a "Data Explorer" page with interactive charts and maps (e.g., a heatmap of house prices across California) to provide more insights from the dataset.

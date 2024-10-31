import pandas as pd
import streamlit as st
import pickle

# Let's define a function to load the machine learning models we trained.

def load_model(filename):
    with open(filename, "rb") as file:
          return pickle.load(file)

# Loading all the trained models

xgboost_model = load_model("xgb_model.pkl")

naive_bayes_model = load_model("nb_model.pkl")

random_forest_model = load_model("rf_model.pkl")

decision_tree_model = load_model("dt_model.pkl")

svm_model = load_model("svm_model.pkl")

knn_model = load_model("knn_model.pkl")

voting_classifier_model = load_model("voting_clf.pkl")

xgboost_SMOTE_model = load_model("xgboost-SMOTE.pkl")

xgboost_featuresEngineered_model = load_model("xgboost-featureEngineered.pkl")


# Now we create a function to prepare the inputs for the models.
# From onehotenconding to numerical values.

def prepare_input(credit_score, location, gender, age, tenure, balance,                               num_products, has_credit_card, is_active_member,                                     estimated_salary):

    input_dict = {
      "CreditScore": credit_score,
      "Age": age,
      "Tenure": tenure,
      "Balance": balance,
      "NumOfProducts": num_products,
      "HasCrCard": int(has_credit_card),
      "IsActiveMember": int(is_active_member),
      "EstimatedSalary": estimated_salary,
      "Geography_France": 1 if location == "France" else 0,
      "Geography_Germany": 1 if location == "Germany" else 0,
      "Geography_Spain": 1 if location == "Spain" else 0,
      "Gender_Male": 1 if gender == "Male" else 0,
      "Gender_Female": 1 if gender == "Female" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

#Now let's define a function to make predictions using the machine learning models we trained

def make_predictions(input_df, input_dict):

  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  st.markdown("### Model Probabilities")
  for model, prob in probabilities.items():
    st.write(f"{model} {prob}")
  st.write(f"Average Probability: {avg_probability}")



st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

# Now let's create a list of customers. We'll be using this list to create dropdown menu

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Now let's create a dropdown menu

selected_customer_option = st.selectbox("Select a customer", customers)

# splitting the selected_customer_option into CustomerId and Surname

if selected_customer_option:

  selected_customer_id = int(selected_customer_option.split("-")[0])

  print("Selected Customer ID:", selected_customer_id)

  selected_surname = selected_customer_option.split("-")[1]

  print("Selected Surname:", selected_surname)

  # Now we can filter to get the information about the selected customer
  
  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

  print("Selected Customer:", selected_customer)

  # creating two columns now

  col1, col2 = st.columns(2)

  with col1:

    credit_score = st.number_input(
      "Credit Scored",
      min_value=300,
      max_value=850,
      value=int(selected_customer['CreditScore'])
    )

    location = st.selectbox(
      "Location", ["Spain", "France", "Germany"],
      index=["Spain", "France", "Germany"].index(selected_customer['Geography'])
    )

    gender = st.radio("Gender" , ["Male", "Female"], 
                      index=0 if selected_customer['Gender'] == "Male" else 1)

    age = st.number_input(
      "Age",
      min_value=18,
      max_value=100,
      value=int(selected_customer['Age'])
    )

    tenure = st.number_input(
      "Tenure (years)",
      min_value=0,
      max_value=50,
      value=int(selected_customer['Tenure'])
    )


  with col2:

    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value=float(selected_customer['Balance'])
    )

    num_of_products = st.number_input(
      "Number of Products",
      min_value=1,
      max_value=10,
      value=int(selected_customer['NumOfProducts'])
    )

    has_credit_card = st.checkbox(
      "Has Credit Card",
      value=bool(selected_customer['HasCrCard'])
    )

    is_active_member = st.checkbox(
      "Is Active Member",
      value=bool(selected_customer['IsActiveMember'])
    )

    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value=float(selected_customer['EstimatedSalary'])
    )
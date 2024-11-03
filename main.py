import pandas as pd
import streamlit as st
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

#OVERVIEW OF PROJECT

# PART 1:
# - Downloading the dataset
# - Understanding the dataset
# - Preprocessing the dataset
# - Training the machine learning models
# - Evaluating the machine learning models

# PART 2:
# - Building the web app
#   - Creating the UI and charts
#   - Making predictions with machine learning models
#   - Generating personalized emails with Llama 3.1 via Groq


# The model prediciton explanation here will be generated using the LAMA 3.2 
# large language model, which was recently released by meta. We're going to use 
# this model through the GROQ API which allows us to access various open source
# models throught one API.

# Now what we need to de is to initialize a openai client with a GROQ API endpoint
# so the cool thing about the OpenAI library is that it provides a standard API
# that most other large language model API providers have adopted. So, by just 
# changing the base_url and the API key, we can use many other API providers such
# as fireworks, togetherai just through one simple interface here.
# In this case we're using the GROQ API because it's free and it's the fastest inference.
# which means it generates text the fastest and that's actually GROQ has built it's
# own hardware to run LLMS on. So it's called the LPU or Language Processing
# Unit and it's upto 10 times faster than Nvidia GPUs.

# An LLM or Large Language Model is a type of AI model designed to understand and 
# generate human-like text based on large amounts of language data.
# Examples of LLMs include GPT-4 from OpenAI, LLAMA from meta and Claude from 
# Anthropic.
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)



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

  # This code displays the probabilities of the models on the frontend of our 
  # web app.

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)
  
  # This was from BEFORE
  # st.markdown("### Model Probabilities")
  # for model, prob in probabilities.items():
  #   st.write(f"{model} {prob}")
  # st.write(f"Average Probability: {avg_probability}")

  return avg_probability


# Now let's define a function to explain the model predictions

def explain_prediction(probability, input_dict, surname):

  # This prompt guides LAMA 3.2 to deliver a clear explanation of the prediction in a way that is accessible to customers.
  prompt = f"""
  You are a data scientist at a bank, specializing in interpreting and explaining predictions from machine learning models
  to help customers understand their potential for churn and the factors that might influence it.

  For a customer named {surname}, the model has predicted a {round(probability * 100, 1)}% likelihood of churning.
  Here is the customer’s information:

  Customer Information:
  {input_dict}

  Model's Key Features for Churn Prediction:

    Feature            |   Importance
    -----------------------------------
    NumOfProducts      |  0.323888
    IsActiveMember     |  0.164146
    Age                |  0.109550
    Geography_Germany  |  0.091373
    Balance            |  0.052786
    Geography_France   |  0.046463
    Gender_Female      |  0.045283
    Geography_Spain    |  0.036855
    CreditScore        |  0.035005
    EstimatedSalary    |  0.032655
    HasCrCard          |  0.031940
    Tenure             |  0.030054
    Gender_Male        |  0.000000

  Profiles of Churned and Non-Churned Customers:
  - Statistics for Customers Who Churned:
    {df[df['Exited'] == 1].describe()}

  - Statistics for Customers Who Did Not Churn:
    {df[df['Exited'] == 0].describe()}

  Explanation Guidelines:
  - If the churn probability is over 40%, craft a 3-sentence explanation that clearly highlights why this customer may be at risk of churning. Focus on the most influential factors from the customer's profile that align with patterns seen in past customers who left, such as product usage, activity level, or geographical factors. Emphasize aspects that may feel most relevant to the customer’s personal situation and how their behaviors or circumstances compare to churned customers.

  - If the probability is under 20%, provide a 3-sentence explanation on why the customer is likely to stay with the bank. Highlight stable and positive factors in their profile—such as consistent engagement, favorable financial metrics, or loyalty indicators—that resemble traits of customers who typically remain. Draw comparisons to the profiles of non-churned customers, focusing on strengths or positive signs in the customer’s history that suggest satisfaction or stability.

  - Use empathetic language that acknowledges the customer’s potential concerns about this analysis. If risk factors are high, frame the information in a way that shows understanding and offers reassurance by describing how the bank is aware of these trends and committed to supporting customers in similar situations.

  - Make it clear how specific actions or conditions (e.g., increasing product usage, maintaining activity, or managing balances) have impacted other customers’ decisions in the past. Where possible, explain what might help reduce churn risk without sounding prescriptive, instead showing how others with similar profiles have benefited.

  - Avoid technical terms like "model," "probability," or jargon. Instead, create a conversational, straightforward explanation that feels approachable. Use the customer’s specific details to paint a picture that shows what factors are important for their unique profile and why these might matter based on patterns seen in similar customers.

  - Whenever possible, highlight how positive factors (like loyalty indicators, active engagement, or favorable financial history) help boost stability and customer satisfaction, especially if the customer’s churn risk is low. Reinforce why these positive signs might indicate that they’re well-suited to remain.

  - In cases where the risk is higher, balance the explanation by not only pointing out risk factors but also suggesting which positive aspects in the customer’s profile might mitigate some risk. For example, even if they have multiple risk indicators, highlight any positive engagement trends that show signs of satisfaction.

  - Consider the customer’s potential questions and address them preemptively. For example, clarify why certain factors, such as age or geography, may play a role by tying them to broader customer behavior trends. This helps make the explanation feel comprehensive and anticipates customer curiosity.

  - Make the language friendly and customer-focused, with a tone that feels helpful and supportive. The goal is to deliver a clear, insightful response that helps the customer understand their unique situation in relation to others without feeling overly technical or judgmental.

  Generate a friendly, intuitive response that explains why this customer might—or might not—be at risk, without mentioning the model or technical aspects.

  """

  print("EXPLANATION PROMPT", prompt)

  raw_response = client.chat.completions.create(
    model='llama-3.2-11b-text-preview',
    messages=[{
      "role": "user",
      "content": prompt
    }],
  )

  return raw_response.choices[0].message.content  # LLM's response


# Now, let's define a function to generate a personalized email send to the user
# to incentivize them to stay with the bank.

def generate_email(probability, input_dict, explanation, surname):

  # Determine the tone and level of incentives based on the churn probability
  if probability > 0.4:
      encouragement_message = "We truly value having you with us and would love to continue supporting your financial journey. To show our appreciation, we’re offering you some special incentives designed to enhance your experience with us."
  elif probability < 0.2:
      encouragement_message = "We’re pleased to see you’re satisfied with your banking experience. To make things even better, we’d like to extend a few exclusive offers to you as a valued member of the HS Bank family."
  else:
      encouragement_message = "As one of our valued customers, we’d like to ensure you’re fully supported in all your banking needs. We’re here for you and would love to make your experience even better with some tailored incentives."

  prompt = f"""
  You are a manager at HS Bank, responsible for ensuring customers stay with the bank and feel valued. Your role involves reaching out to customers who may be at risk of leaving, offering them personalized support and incentives to encourage their loyalty.

  You have identified that a customer named {surname} has indicators suggesting they might be considering leaving. Here is some context on their situation:

  Customer Information:
  {input_dict}

  Reasons Why the Customer Might Be at Risk of Churning:
  {explanation}

  Email Guidelines:
  - Start with a warm, personalized greeting addressing {surname} by name. Make them feel valued and acknowledged as a loyal customer of the bank.
  - {encouragement_message}
  - If they appear to be at risk, kindly ask them to stay by expressing appreciation for their loyalty and addressing any potential concerns they may have. Be understanding, and encourage open communication by inviting them to reach out if they have questions or feedback.
  - Offer a tailored set of incentives that would be meaningful to the customer, based on the information provided. Frame these incentives as a way to enhance their banking experience and address potential needs or interests. Here are examples of possible incentives:
    - Reduced fees or waived monthly charges
    - Personalized loan or credit offers
    - Higher interest rates on savings accounts
    - Points or rewards for specific transactions
    - Enhanced customer service options or dedicated account manager
    - Access to exclusive events or financial wellness resources
  - List these incentives in a clear, bullet-point format for easy readability and highlight how they align with the customer’s preferences or banking behaviors.
  - Be tactful and avoid mentioning the probability of churning or any technical terms related to predictions or models. Instead, focus on building a positive and supportive tone that encourages the customer to feel secure and appreciated.
  - Close the email by thanking the customer for their valued relationship with HS Bank, reaffirming your commitment to their satisfaction. Offer a direct line of communication if they wish to discuss further or need assistance with any aspect of their banking experience.

  Generate a warm, personalized email that makes the customer feel appreciated and encourages them to stay, without mentioning any predictive analysis or churn probability.

  """

  print("EMAIL GENERATION PROMPT", prompt)
    
#llama-3.2-11b-text-preview
  raw_response = client.chat.completions.create(
    model='llama-3.2-11b-text-preview',
    messages=[{
      "role": "user",
      "content": prompt
    }],
  )
  print("\n\nEMAIL PROMPT", prompt)
  return raw_response.choices[0].message.content   # LLM's response






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


  input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_of_products, has_credit_card,is_active_member, estimated_salary)

  avg_probability = make_predictions(input_df, input_dict)

  explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

  st.markdown("------")

  st.subheader("Explanation of Prediction")

  st.markdown(explanation)


  email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

  st.markdown("------")

  st.subheader("Personalized Email")

  st.markdown(email)
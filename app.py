import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
from io import StringIO
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime
import time
from tensorflow.keras.models import load_model

# def load_model():
#     with open('saved_steps.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)
model = load_model('model.h5')
scaler = data["scaler"]

gcp_cred = {
  "type": "service_account",
  "project_id": "deploying-apps-403014",
  "private_key_id": "8cc5a76b9fc1e35f175e00914dcf352f261521c0",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDD1P1LgpBfyyJI\nnUlWq0i/dcdyRtGWulixn7tG6T0P6MaGytPIy9/2vS/7kFP4SBe8a5r8xO7WzDcR\nRAF5bIO3pRlu1dqxE939IgZJQvl27nArTEQbF3jZ6CKSrVRW3bPrRsHdeh9GT4cB\n20IEq57KjYP/o7b2/j3RCvq9EGOW2iVPEzJRBRKmUbChr8oCf6x0+iSVA3C1TcBc\nnJXydQ/4iYUmovU3a86srIdVOTaqaH4Rmg+JC3dWpArPLlLnl4MkH/pEzzn79SVv\nWjIhWU5Xs1FH/8AZfvis6IW2vCpDVLKPedElQ2bkHs07OWgmti+4lvduzK+/MPYf\nm3dO+B2/AgMBAAECggEAVax3OhM/VWFZuNny7VNXKsf8vx6DcAVLaz4d4QqGN9EX\nFX5jxAuJink/JjgAvHTRyZFsB1yivt8t0kRlvrj+MRguNcKif+jYq/c4N9HwsJrl\nANRHUXDl4rM/eNsStEQfA/X7N4lHaGu/fUzup3trp2LdFjiKAam94pgClTc30pUO\nSDl0+ghoSyG7PUDP/wOcqXpD4+mkD2NDSOO+QC8fFb7T5VGoK9TnrB3U/mPFve/t\n+3bSZcL6TRo2WTAf4zu7/kfzjByydvzvic1cM7hmk1yDjS37qhCYHd/5uMDJ9nhw\nDa8toqiB3kjyQUssn/Gl1f+7giBYiVKKvir6jvK1yQKBgQDn3eMPdwb+K+GnSUb/\nhorxGXtCQrKfmnCswjJfnKWA6Q1OaByuXj0M+vVx1y/CBaGokZJIPKgpz1eRxgDi\n8PJhgePkszbh49hUr0N37oa0bDQNGfZlNbyHamUKaR7R58RUAFTMWO6PatfzHDb4\nN7k0jXBhGbFP89zJl2VLpHoEZwKBgQDYNvPmsb9O4XdL9VA68pGV8NaTA9dAn94T\nNqLYjY71A6xcWckimQwCrD3oIMGS4QD/FixTNNJXBgPdZUgyMh9nYDmhYO5s38g4\nUAHjPRuOI/0HOeTB8K57CqC8WxjPRcmkeXatkyxCqRZYq5jwyrjEgpI/8Hj4+6DI\nJ0io8hOE6QKBgQCK5mWLciaCRQ9dA4zArny1iipIu2P+MKqnE37RwCl1XCdYfQ4R\ndurjx8MZe2tks7LwJKSZGZ0zzr4K2a7WRLkuqH05GBMcpz2FHakxQ3b2xos3/gGZ\nB+P0y0vUPLz1yf3WxIwIDo564+qR/KkBcYBFdyWHRbjuyIvPSB6qfdGKqQKBgAz9\n8FXyZE53GdYEnZeNL9ZUrHAVEQAfGxcId2yPxQFQATFja8SacbBPbUDfhwIuZwLs\n8CjnyaPVBrJs/ZOWk3CAxbW+v1TndX58wEBJUbiOzQt8HRTSQ4m9L79hsfHyfZVW\nfx0a9NPkmSUm2n/NjqCwP917s6kyZYzhX2pXcXjRAoGBAMjuebNGrNnWSAv0Q8la\nCg/yE+dXh2Jg3SY6X7mPm6DpAL7Ks7JB2b01Ahs86aVUaXi4aB1ydbPQCnVNmp6s\nPdbZtMJS3B80ep/kvG8MnpWhf7Ik7MeL2029Q6hZQCNhJABjSSlTDE3eV6/ucPxX\n7G7nF9c5DGDoCfTcHVOFcesY\n-----END PRIVATE KEY-----\n",
  "client_email": "mlflow@deploying-apps-403014.iam.gserviceaccount.com",
  "client_id": "110921037524701885630",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mlflow%40deploying-apps-403014.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# Initialize a BigQuery client with the credentials from the dictionary
credentials = service_account.Credentials.from_service_account_info(gcp_cred)
client = bigquery.Client(credentials=credentials, project=gcp_cred['project_id'])

# data = load_model()
# model = data["model"]
# scaler = data["scaler"]
features = ['Open', 'High', 'Low', 'Close', 'Volume']
time_step = 30
st.title("Tesla Stock Price Prediction")
placeholder = st.empty()
while True:
    with placeholder.container():
        st.write("As on:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        query = f"SELECT * FROM `deploying-apps-403014.TeslaTrained.Teslatraining` ORDER BY Date"
        query_job = client.query(query)
        df = query_job.to_dataframe()
        # 1. Get the last time_step number of days from the df
        last_days = df[-time_step:]
        #print(last_days)
        # 2. Scale the last_days data
        last_days_scaled = scaler.transform(last_days[features])
        # 3. Reshape the data
        X_last_days_scaled = last_days_scaled.reshape(1, time_step, len(features))
        # 4. Make prediction
        next_day_open_price_scaled = model.predict(X_last_days_scaled)
        # 5. Inverse transform the prediction to get the actual price
        # We need to create a dummy array to inverse transform only the 'Open' price
        # which is typically at index 0, if the first feature in your features list is 'Open'
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = next_day_open_price_scaled

        next_day_open_price = round(scaler.inverse_transform(dummy_array)[0, 0], 2)
        print(next_day_open_price)

        st.write(f"Predicted next day's opening price: {next_day_open_price}")
        df_sorted = df.sort_values(by='Date')
        max_date_row = df_sorted.iloc[-1]
        print(max_date_row)

        # Prepare the row to be inserted with the prediction
        row_to_insert = [{
            "Date": max_date_row["Date"].strftime("%Y-%m-%d"),
            "Open": float(max_date_row["Open"]),  # or whatever the true value is if available
            "High": float(max_date_row["High"]),
            "Low": float(max_date_row["Low"]),
            "Close": float(max_date_row["Close"]),
            "Volume": int(max_date_row["Volume"]),
            "Next_Open_Prediction": float(next_day_open_price)
        }]
        print(next_day_open_price)
        # Specify your table name
        table_id = 'deploying-apps-403014.TeslaTrained.predictiontable1'

        # Make an API request to insert the row
        errors = client.insert_rows_json(table_id, row_to_insert)

        # If errors are returned, print them
        if errors:
            print("Errors occurred:", errors)

        # After inserting rows into the table, remove duplicates
        def remove_duplicates(client, table_id):
            # Create a temporary table
            temp_table_id =  'deploying-apps-403014.TeslaTrained.predictiontable_temp1'

            # Create a temporary table with distinct records
            deduplicate_query = f"""
            CREATE OR REPLACE TABLE `{temp_table_id}` AS
            SELECT DISTINCT * FROM `{table_id}`;
            """
            client.query(deduplicate_query).result()
            time.sleep(10)

        # Call the function to remove duplicates
        remove_duplicates(client, table_id)


        query_top_predictions = """
        SELECT *
        FROM `deploying-apps-403014.TeslaTrained.predictiontable_temp1`
        ORDER BY Date DESC
        LIMIT 10
        """
        
        query_job = client.query(query_top_predictions)
        predictions_df = query_job.to_dataframe()

        # Use st.dataframe to display the dataframe in the Streamlit app
        st.subheader('Recent Predictions')
        st.dataframe(predictions_df)
        time.sleep(3600)
        


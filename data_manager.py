import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, file_path):
        # Creation and Storing the  file Path
        self.file_path = file_path
        self._df = None

    # Getter for the Data Frame
    @property
    def dataframe(self):
        return self._df
    # Setter fo the Dataframe

    @dataframe.setter
    def dataframe(self, value):
        self._df = value

    def inspection_schema(self, num_rows=3):
        with open(self.file_path, mode="r", newline="", encoding="utf-8") as file:
            # creating a local reder object to read the csv file
            reader = csv.reader(file)

            # extracting and printing headers
            headers = next(reader)
            print(f" Headers: {len(headers)}\n")
            print("--- First Data Rows ---")

            # Tracking and the amount of rows that are being process
            for count, row in enumerate(reader, start=1):
                print(row)

                # Breaking out when  the  count reaches the requested number of rows
                if count == num_rows:
                    print("--- Limit reached. Stopping and exiting! ---")
                    break

    def clean_data(self):
        # Dataset Loading and printing the original shape

        df = pd.read_csv(self.file_path)
        print(f"Current Shape {df.shape}")

        # Dropping Duplicates Rows

        df.drop_duplicates(inplace=True)
        print(f"Shape after removing duplicate ROWS: {df.shape}")

        # Dropping Duplicates Columns Temporary! ‼️‼️
        # Function needs to be changed anytime the dataset layout changes

        duplicate_col_name = df.columns[4]
        df.drop(columns=[duplicate_col_name], inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"Shape after removing duplicate Columns: {df.shape}")

        # Checking and returning missing values
        print("\n -- Total Missing Values --")
        print(df.isnull().sum())

        self.dataframe = df

        return df

    def data_conversion(self):

        df = self.dataframe
        # ------
        print("---- Converting Data Types ----")

        # Raw Text String To True Pandas Datatime &
        # Downcast the target label to int8 to save RAM

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Is Laundering'] = df['Is Laundering'].astype("int8")
        # updated data types and memory usage
        print(df.dtypes)
        print(
            f" \nMemory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

        self.dataframe = df

        return df

    def explore_data(self):
        df = self.dataframe

        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        #  Target Distribution
        print("--- 1. Target Distribution (Is Laundering) ---")

        print(df['Is Laundering'].value_counts())

        print("\nPercentage:")

        print(df['Is Laundering'].value_counts(normalize=True)*100)

        #  Payment Format Distribution
        print("\n--- 2. Payment Formats ---")

        print(df['Payment Format'].value_counts())

        # Transaction Amount Statistics by Class
        print("\n--- 3. Amount Paid Stats by Class ---")

        print(df.groupby('Is Laundering')['Amount Paid'].describe())

    def feature_engineering(self):
        df = self.dataframe
        print("\n--- Phase 4: Feature Engineering ---")
        print("Extracting time-based signals...")

        # Explicitly cast Timestamp to datetime so .dt works natively
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # 1. Hour of the Day (0-23)
        df['Hour_Of_Day'] = df['Timestamp'].dt.hour

        # 2. Day of the Week (0 = Monday, 6 = Sunday)
        df['Day_Of_Week'] = df['Timestamp'].dt.dayofweek

        # 3#  Month of the Year (1-12)
        df['Month'] = df['Timestamp'].dt.month

        # Checking New columns
        # 5 rows of just new columns and the target
        print(df[['Timestamp',
                  'Hour_Of_Day',
                  'Day_Of_Week',
                  'Month',
                 'Is Laundering']].head())
        self.dataframe = df

        return df

    def engineer_numerical_feature(self):

        df = self.dataframe

        print("--- Engineering Numerical Signals ---")
        # 1. Log Transformation to compress extreme outliers
        df['Amount_Paid'] = np.log1p(df['Amount Paid'])

        df['Amount_Received'] = np.log1p(df['Amount Received'])

        # 2. Value Loss / Exchange Rate anomaly signal
        df['Net_Cash_Flow'] = df['Amount Paid']-df['Amount Received']

        self.dataframe = df

        return df

    def encode_clean(self):
        df = self.dataframe
        # 1. One-Hot Encode the Payment Format (turns 'Wire' into a 1 or 0 column)
        # --.drop_first=True to avoid the dummy variable trap (multicollinearity)
        df = pd.get_dummies(df, columns=['Payment Format'], drop_first=True)

    #  Rescue columns into reference_df
        self.reference_df = df[['Account', 'From Bank', 'To Bank']].copy()

        print("🗺️ Mapping accounts and banks to continuous integers...")

        unique_acc = self.reference_df['Account'].unique()
        unique_from = self.reference_df['From Bank'].unique()
        unique_to = self.reference_df['To Bank'].unique()

        #  Combination into one set

        all_nodes = set(unique_acc) | set(unique_from) | set(unique_to)

        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node,
                            idx in node_to_idx.items()}  # Saved for Stage 4

        # New Mapped Integers Columns
        self.reference_df['Account_Node_ID'] = self.reference_df['Account'].map(
            node_to_idx)
        self.reference_df['From_Bank_Node_ID'] = self.reference_df['From Bank'].map(
            node_to_idx)
        self.reference_df['To_Bank_Node_ID'] = self.reference_df['To Bank'].map(
            node_to_idx)

        # 2. Dropping raw text columns and redundant data
        columns_to_drop = ['Timestamp',
                           'Account',
                           'From Bank',
                           'To Bank',
                           'Receiving Currency',
                           'Payment Currency',
                           'Amount Paid',
                           'Amount Received']

        df = df.drop(columns=columns_to_drop, errors='ignore')

        self.dataframe = df
        return df

    def split_scale(self):
        df = self.dataframe

        # 1. Separate Features (X) from Target (y)
        y = df['Is Laundering']

        X = df.drop(columns=['Is Laundering'])

        # 2. Train/Test Split (80% training, 20% testing)

        X_train, X_temp, y_train, y_temp, ref_train, ref_temp = train_test_split(
            X, y,  self.reference_df, test_size=0.3, random_state=42, stratify=y)

        X_cv, X_test, y_cv, y_test, ref_cv, ref_test = train_test_split(
            X_temp, y_temp, ref_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # 2. Force independent memory copies so scaling works safely
        X_train = X_train.copy()
        X_cv = X_cv.copy()
        X_test = X_test.copy()

        # 3. Scale numerical features
        scaler = StandardScaler()
        numerical_columns = ['Amount_Paid', 'Amount_Received', 'Net_Cash_Flow',
                             'Hour_Of_Day', 'Day_Of_Week', 'Month']

        X_train[numerical_columns] = scaler.fit_transform(
            X_train[numerical_columns])

        X_cv[numerical_columns] = scaler.transform(
            X_cv[numerical_columns])
        X_test[numerical_columns] = scaler.transform(
            X_test[numerical_columns])

      #  Final Training Data Shape
        print(f"Training set's shape{X_train.shape}")

        return X_train, X_cv, X_test, y_train, y_cv, y_test, ref_train,  ref_cv, ref_test

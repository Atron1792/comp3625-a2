import pandas as pd
import numpy as np
from skops.io import load



# At grading time, I will call the below function with a dataframe of new potential customers
# The dataframe I pass in will have the same columns as customer_data.csv, but will not include any of the customers in customer_data.csv
# You will not be given any of the historical customer data or claims data at grading time, 
# so you will need to save any models you train in model_dev.py and load them here.

def set_premium(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a dataframe of data on potential customers, identifies which customers to take on, and sets premiums for them.
    This function may load a saved model (previously trained using supervised learning) to save time.
    :param customer_data: a dataframe with the same columns as customer_data.csv
    :return: a DataFrame of the same lenght as customer_data, with the following columns:
            p_claim:  your estimated probability that each customer in customer_data will make a claim
            desired_customer:  a boolean column indicating which customers you want to take on
            premium:    the amount of the premium to charge each customer
    """
    model = load('model.skops', trusted=['sklearn.tree._classes.DecisionTreeClassifier'])

     # 1. estimate claim probabilities
    p_claim = claim_probability(model, customer_data)
 
    # 2. set premiums (NaN = customer fails the utility check)
    premiums = calculate_premium(customer_data, p_claim)
 
    # 3. choose customers within the $100M coverage limit
    desired = select_customers(customer_data, premiums, p_claim)
 
    # customers we don't want still need a numeric premium in the output;
    # use the plain expected cost where the utility check produced NaN
    coverage = customer_data['coverage_amount'].to_numpy(dtype=float)
    premiums = np.where(np.isnan(premiums), p_claim * coverage, premiums)
 
    return pd.DataFrame({
        'p_claim': p_claim,
        'desired_customer': desired.astype(bool),
        'premium': premiums,
    }, index=customer_data.index)


if __name__ == '__main__':
    df = pd.read_csv('./data/customer_data.csv').sample(5000, random_state=1)
    out = set_premium(df)
    print(out.head())
    print(out['desired_customer'].sum(), df.loc[out['desired_customer'], 'coverage_amount'].sum())
    print(out['p_claim'].nunique(), "distinct probabilities")

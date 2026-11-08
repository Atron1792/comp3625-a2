import pandas as pd
import numpy as np



# load historical customer data
customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)


# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

def claim_probability(model: object, customer_data: pd.DataFrame) -> np.ndarray:
    """
    Estimate the probability that each customer will make an insurance claim.

    :param model: A trained supervised-learning model with a predict_proba method
    :param customer_data: DataFrame containing the customer features
    :return: claim_probabilities, an array containing one claim probability for each customer
    """
    # TODO
    pass

def calculate_premium(customer_data: pd.DataFrame, claim_probability: np.ndarray) -> np.ndarray:
    """
    Calculate a premium for each customer that benefits the customer and gives the company positive expected profit.

    :param customer_data: DataFrame containing customer coverage amounts
    :param claim_probability: Estimated claim probability for each customer
    :return: premiums, an array containing one premium for each customer
    """
    # TODO
    pass

def calculate_utility(customer_money: np.ndarray) -> np.ndarray:
    """
    Calculate utility from monetary values derived from the customer data using the utility curve U(x) = -263.31 + 22.09 ln(x + 150,000).

    :param customer_data: DataFrame containing the monetary values needed for the utility calculation
    :return: Array containing one utility value for each customer
    """
    # TODO

    customer_utilities = np.zeroes(customer_money.length)

    for money in customer_money:
        utility = -263.31 + 22.09 * np.log(money + 150,000)
        np.append(customer_utilities, utility)

    return customer_utilities


def select_customers(customer_data: pd.DataFrame, premiums: np.ndarray, claim_probabilities: np.ndarray) -> np.ndarray:
    """
    Search for a set of customers to insure while keeping their total
    coverage within the $100 million startup limit.

    :param customer_data: DataFrame containing potential customers
    :param premiums: Array containing the proposed premium for each customer
    :param claim_probabilities: Array containing the estimated claim probability for each customer
    :return: Boolean array indicating which customers should be accepted
    """
    # TODO
    pass

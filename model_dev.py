import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression


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
    return model.predict_proba(customer_data)[:, 1]

def calculate_premium(customer_data: pd.DataFrame, claim_probability: np.ndarray) -> np.ndarray:
    """
    Calculate a premium for each customer that benefits the customer and gives the company positive expected profit.

    :param customer_data: DataFrame containing customer coverage amounts
    :param claim_probability: Estimated claim probability for each customer
    :return: premiums, an array containing one premium for each customer
    """
    margin = 0.1
    # margin is a place holder value for now, once the model is finished can test a few values to fine tune it.
    coverage = customer_data['coverage_amount'].to_numpy()

    p = np.asarray(claim_probability)

    # Company side: expected payout is p * coverage, so charge a markup over it
    expected_cost = p * coverage
    premiums = expected_cost * (1 + margin)

    # Customer side: expected utility of going uninsured (lottery)
    # claim -> lose the full coverage amount, no claim -> lose nothing
    expected_utility_uninsured = (p * calculate_utility(-coverage)+ (1 - p) * calculate_utility(np.zeros(len(coverage))))

    # Utility of insuring: pay the premium for certain, claims are fully covered
    utility_insured = calculate_utility(-premiums)

    # Premium must give the customer strictly better utility than being uninsured
    benefits_customer = utility_insured > expected_utility_uninsured

    return np.where(benefits_customer, premiums, np.nan)


def calculate_utility(customer_money: np.ndarray) -> np.ndarray:
    """
    Calculate utility from monetary values derived from the customer data using the utility curve U(x) = -263.31 + 22.09 ln(x + 150,000).

    :param customer_money: Array of monetary amounts in dollars
    :return: Array containing one utility value for each monetary amount
    """
    customer_utilities = np.zeros(customer_money.size)
    i = 0

    for money in customer_money:
        customer_utilities[i] = (-263.31 + 22.09 * np.log(money + 150000))
        i+=1

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

if __name__ == '__main__':
    model = LogisticRegression(max_iter=1000)
    model.fit(X=customer_data, y=customer_claims['claim'])

    claim_probabilities = claim_probability(model, customer_data)
    
    utilities = calculate_utility(customer_data['coverage_amount'])
    print(utilities)
    print(claim_probabilities)
    premiums = calculate_premium(customer_data, claim_probabilities)
    print(np.isnan(premiums).sum(), "customers fail the utility check")
    print("avg expected profit:", np.nanmean(premiums - claim_probabilities * customer_data['coverage_amount']))
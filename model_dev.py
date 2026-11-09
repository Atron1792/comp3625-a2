import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression


# load historical customer data
customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)


# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

def hyperparamater_selection(customer_data: pd.DataFrame, customer_claims: pd.DataFrame) -> np.array:

    # 66% of the data is better for training compared to 50%, as per the lab 
    train_idx = np.random.random(size=customer_data.shape[0]) < 0.66


    x_train = customer_data.loc[train_idx]
    y_train = customer_claims.loc[train_idx]
    x_test = customer_data.loc[~train_idx]
    y_test = customer_claims.loc[~train_idx]


    best_acc = 0

    for criterion in ['gini', 'entropy', 'log_loss']:
        for max_depth in range(1, 6):
            for ccp_alpha in np.arange(0, 0.25, 0.01):
            
                tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, ccp_alpha=ccp_alpha)
                tree.fit(x_train, y_train)
            
                x_predict = tree.predict(x_test)
                acc = (x_predict == y_test).mean()
            
                if(acc >= best_acc):
                    criterion_best = criterion
                    max_depth_best = max_depth
                    ccp_alpha_best = ccp_alpha
                    best_acc = acc
                
    best_hyperparameters = np.array([criterion_best,max_depth_best,ccp_alpha_best])

    return best_hyperparameters

def claim_probability(model: object, customer_data: pd.DataFrame) -> np.ndarray:
    """
    Estimate the probability that each customer will make an insurance claim.

    :param model: A trained supervised-learning model with a predict_proba method
    :param customer_data: DataFrame containing the customer features
    :return: claim_probabilities, an array containing one claim probability for each customer
    """
    return model.predict_proba(customer_data)[:, 1]

def calculate_premium(customer_data: pd.DataFrame, claim_probability: np.ndarray, margin: float = 0.05, z: float = 1.0, sigma: float = 0.01) -> np.ndarray:
    """
    Calculate a premium for each customer that benefits the customer and gives the company positive expected profit.
    Taking into account the uncertainty in the claim probability, z-score is used to adjust the premium to ensure that we are charging enough to cover potential claims.
    The more confident in our claim probability estimate, the lower the z-score we can use, and the lower the premium we can charge.

    :param customer_data: DataFrame containing customer coverage amounts
    :param claim_probability: Estimated claim probability for each customer
    :param margin: Thin markup over expected cost, keep small so we can undercut competitors
    :param z: The z-score for the desired confidence level (default is 1.0 for 68% confidence)
    :param sigma: The standard deviation of the claim probability (default is 0.01
    :return: premiums, an array containing one premium for each customer
    """
    coverage = customer_data['coverage_amount'].to_numpy(dtype=float)
    p = np.clip(np.asarray(claim_probability, dtype=float), 0, 1)
 
    p_padded = np.clip(p + z * sigma, 0, 1)
    premiums = coverage * p_padded * (1 + margin)
 
    # Customer side: lottery of going uninsured vs paying the premium for certain
    with np.errstate(all='ignore'):
        eu_uninsured = p * calculate_utility(-coverage) + (1 - p) * calculate_utility(np.zeros(len(coverage)))
        u_insured = calculate_utility(-premiums)
 
    benefits_customer = u_insured > eu_uninsured   # NaN comparisons are False -> excluded
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
    coverage = customer_data['coverage_amount'].to_numpy()
    expected_profit = premiums - claim_probabilities * coverage
    value = expected_profit / coverage

    eligible = ~np.isnan(premiums) & (expected_profit > 0)

    selected = np.zeros(len(coverage), dtype=bool)
    total_coverage = 0

    # TODO: Implement a selection algorithm to choose customers while keeping total coverage under $100 million.

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

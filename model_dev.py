import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

CAPACITY = 100000000

# write your code to develop your agent program here.
# you can, of course, add additional .py files as needed.

def hyperparameter_selection(customer_data: pd.DataFrame, customer_claims: pd.DataFrame) -> np.ndarray:

    # 66% of the data is better for training compared to 50%, as per the lab 
    train_idx = np.random.random(size=customer_data.shape[0]) < 0.66

    y = customer_claims['claim']
    x_train = customer_data.loc[train_idx]
    y_train = y.loc[train_idx]
    x_test = customer_data.loc[~train_idx]
    y_test = y.loc[~train_idx]


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
    :param sigma: The standard deviation of the claim probability (default is 0.01)
    :return: premiums, an array containing one premium for each customer
    """
    coverage = customer_data['coverage_amount'].to_numpy(dtype=float)
    p = np.clip(np.asarray(claim_probability, dtype=float), 0, 1)
 
    p_padded = np.clip(p + z * sigma, 0, 1)
    premiums = coverage * p_padded * (1 + margin)
 
    # Customer side: lottery of going uninsured vs paying the premium for certain
    with np.errstate(all='ignore'):
        expected_utility_uninsured = (p * calculate_utility(-coverage) + (1 - p) * calculate_utility(np.zeros(len(coverage))))
        utility_insured = (p * calculate_utility(coverage - premiums) + ((1 - p) * calculate_utility(-premiums)))

    benefits_customer = utility_insured > expected_utility_uninsured   # NaN comparisons are False -> excluded
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
    eligible = ~np.isnan(premiums) & (expected_profit > 0)
    n_iterations = 20000  # number of iterations for hill climbing, can adjust depending on how long program runs

    # --- greedy initial solution ---
    score = np.where(eligible, expected_profit / coverage, -np.inf) # replaces an if-else loop
    selected = np.zeros(len(coverage), dtype=bool)
    total = 0.0
    for i in np.argsort(-score):
        if not eligible[i]:
            break                      # sorted, so everything after is ineligible too
        if total + coverage[i] <= CAPACITY:
            selected[i] = True
            total += coverage[i]

    # --- hill climbing: swap one selected customer for one unselected customer, helps gets max value after greedy ---
    rng = np.random.default_rng(0)#seed set to 0 for reproducibility for now
    in_idx = np.flatnonzero(selected)
    out_idx = np.flatnonzero(eligible & ~selected)
    if len(in_idx) and len(out_idx): # makes sure there are customers to swap in and out
        for _ in range(n_iterations):
            a = rng.integers(len(in_idx)) # random index number of selected customer
            b = rng.integers(len(out_idx)) # random index number of unselected customer
            i = in_idx[a] 
            j = out_idx[b]
            gain = expected_profit[j] - expected_profit[i]
            if gain > 0 and total - coverage[i] + coverage[j] <= CAPACITY:
                selected[i], selected[j] = False, True
                total += coverage[j] - coverage[i]
                in_idx[a], out_idx[b] = j, i
 
    return selected

if __name__ == '__main__':
    
# load historical customer data
    customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
    customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)
    customer_data.columns = customer_data.columns.str.strip()
    customer_claims.columns = customer_claims.columns.str.strip()
     
    best = hyperparamater_selection(customer_data, customer_claims)

    model = DecisionTreeClassifier(
    criterion=best[0],
    max_depth=int(best[1]),
    ccp_alpha=float(best[2]),
    random_state=0,
)
    model.fit(X=customer_data, y=customer_claims['claim'])

    #model = LogisticRegression(max_iter=1000)
    #model.fit(X=customer_data, y=customer_claims['claim'])

    claim_probabilities = claim_probability(model, customer_data)
    
    premiums = calculate_premium(customer_data, claim_probabilities)
    print(np.isnan(premiums).sum(), "customers fail the utility check")
    expected_profit = premiums - claim_probabilities * customer_data['coverage_amount']
    print("accepted customers with non-positive expected profit:",
          np.sum((~np.isnan(premiums)) & (expected_profit <= 0)))
    print("avg expected profit:", np.nanmean(expected_profit))

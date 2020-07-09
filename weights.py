import pandas as pd
import numpy as np

# weights
def get_weight_df(train_df, calendar, prices, weight_columns) -> pd.DataFrame:
    """
    returns weights for each series in a dataFrame
    """
    
    id_columns = [i for i in train_df.columns if not i.startswith('d_')]
    
    day_to_week = calendar.set_index("d")["wm_yr_wk"].to_dict()
    weight_df = train_df[["item_id", "store_id"] + weight_columns].set_index(
        ["item_id", "store_id"]
    )
    weight_df = (
        weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
    )
    weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
    weight_df = weight_df.merge(
        prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
    )
    weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
    weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
        "value"
    ]
    weight_df = weight_df.loc[
        zip(train_df.item_id, train_df.store_id), :
    ].reset_index(drop=True)
    weight_df = pd.concat(
        [train_df[id_columns], weight_df], axis=1, sort=False
    )

    return weight_df

calendar = pd.read_csv('data/raw/calendar.csv')
prices = pd.read_csv('data/raw/sell_prices.csv')
train = pd.read_csv('data/raw/sales_train_validation.csv') 

train_fold = train.iloc[:, :-28] 
valid_fold = train.iloc[:, -28:]

weight_columns = train_fold.iloc[:, -365:].columns.tolist()
weight_df = get_weight_df(train_fold, calendar, prices, weight_columns)  # train_fold 这里应该改成 train 

weights = weight_df.loc[:, 'd_1521':].sum(axis=1)
weights = (weights / weights.sum()).values
np.save('data/processed/weights.npy', weights)

# df['weights'] = weights
# weights_validation = pd.read_csv(os.path.join(base_dir, 'data/raw/weights_validation.csv'))
# weights_validation = weights_validation[weights_validation.Level_id == 'Level12']

# df = df.merge(weights_validation, 
#          left_on=['item_id', 'store_id'],
#          right_on=['Agg_Level_1', 'Agg_Level_2'],
#          how='left')
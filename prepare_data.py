import os
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')
    
df = pd.read_csv('data/raw/sales_train_evaluation.csv') 

pred_cols = ['d_'+str(i) for i in range(1942, 1942+28)]
for col in pred_cols:
    df[col] = 0
    
df.columns = [int(c.split('_')[1]) if 'd_' in c else c for c in df.columns]

id_cols = [c for c in df.columns if 'id' in str(c)]
date_cols = [c for c in df.columns if c not in id_cols]
df_idx = df[id_cols]
print('loaded data')

# non-temporal features
le = LabelEncoder()
item_id = le.fit_transform(df['item_id'])
dept_id = le.fit_transform(df['dept_id'])
cat_id = le.fit_transform(df['cat_id'])
store_id = le.fit_transform(df['store_id'])
state_id = le.fit_transform(df['state_id'])
df['all_id'] = le.fit_transform(df['id'])

np.save('data/processed/item_id.npy', item_id.astype(np.int16))
np.save('data/processed/dept_id.npy', dept_id.astype(np.int8))
np.save('data/processed/cat_id.npy', cat_id.astype(np.int8))
np.save('data/processed/store_id.npy', store_id.astype(np.int8))
np.save('data/processed/state_id.npy', state_id.astype(np.int8))
np.save('data/processed/all_id.npy', df['all_id'].values.astype(np.int16))

df[['all_id', 'id']].to_csv('data/processed/all_ids.csv', encoding='utf-8', index=False)

df[date_cols] = np.log(df[date_cols].values + 1)
df[date_cols] = df[date_cols].round(8) # added
df[date_cols] = df[date_cols].astype(np.float32)
np.save('data/processed/x.npy', df[date_cols].values)
print('non-temporal features')

del item_id, dept_id, cat_id, store_id, state_id
df.drop('all_id', axis=1, inplace=True)
gc.collect()

# hierarchy
hierarchy_id = [['state_id', 'cat_id'], ['state_id', 'dept_id'],
                ['store_id', 'cat_id'], ['store_id', 'dept_id'], 
                ['item_id', 'state_id'], ['item_id', 'store_id']] # change the last item to all_id

hierarchy_cols = []
for id__ in hierarchy_id:
    new_col__ = '-'.join(id__)
    hierarchy_cols.append(new_col__)
    df[new_col__] = df.loc[:, id__].sum(axis=1)

hierarchy_data = df[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + hierarchy_cols].apply(LabelEncoder().fit_transform)
np.save('data/processed/hierarchy_data.npy', hierarchy_data.values.astype(np.int16))

del hierarchy_data
df.drop(hierarchy_cols, axis=1, inplace=True)
print('hierarchy')

# lags
x = df[date_cols].values

x_lags = [1, 7, 14, 16, 21]
lag_data = np.zeros([x.shape[0], x.shape[1], len(x_lags)], dtype=np.float16)

for i, lag in enumerate(x_lags):
    lag_data[:, lag:, i] = x[:, :-lag]

np.save('data/processed/x_lags.npy', lag_data)
del lag_data

xy_lags = [28, 35, 365/4, 365/2, 365, 365*2, 365*3, 365*4]
xy_lags = [int(n) for n in xy_lags]
lag_data = np.zeros([x.shape[0], x.shape[1], len(xy_lags)], dtype=np.float16)

for i, lag in enumerate(xy_lags):
    lag_data[:, lag:, i] = x[:, :-lag]

np.save('data/processed/xy_lags.npy', lag_data)
del lag_data
print('lags')

# aggregate time series
hierarchy_id.remove(['item_id', 'store_id'])
groups = [['item_id'], ['dept_id'], ['cat_id'], ['store_id'], ['state_id']] + hierarchy_id
aux_ts = np.zeros([df.shape[0], len(date_cols), len(groups)], dtype=np.float16)

for i, group in enumerate(groups):
    ts = df.groupby(group)[date_cols].mean().reset_index()
    print(ts.shape)
    ts = df_idx.merge(ts, how='left', on=group)
    aux_ts[:, :, i] = ts[date_cols].values
    
np.save('data/processed/ts.npy', aux_ts)
del aux_ts
gc.collect()
print('ts')

# start_date
df_melted = pd.melt(df, id_vars=id_cols, value_vars=date_cols)
start_date = df_melted[df_melted.value != 0].groupby(['item_id', 'store_id'])['variable'].min()
start_date = start_date.reset_index().rename({'variable':'start_date'}, axis=1)

df_idx = df_idx.merge(start_date, on=['item_id', 'store_id'], how='left')
np.save('data/processed/start_date.npy', df_idx['start_date'].values.astype(np.int16))
df_idx.drop(['start_date'], axis=1, inplace=True)
df_melted.drop('value', axis=1, inplace=True)
del start_date
print('start_date')

# calendar
calendar = pd.read_csv('data/raw/calendar.csv')
calendar['d'] = calendar['d'].str.extract('(\d+)').astype(int)
calendar.drop(['date', 'weekday', 'year', 'event_name_2', 'event_type_2'], axis=1, inplace=True)
event_cols = ['event_name_1', 'event_type_1']
calendar[event_cols] = calendar[event_cols].fillna('unknown')
calendar[event_cols] = calendar[event_cols].apply(LabelEncoder().fit_transform)
calendar.set_index('d', inplace=True)

for c in calendar.columns:
    print(c)
    df_melted[c] = df_melted.variable.map(calendar[c])

del calendar
gc.collect()

df_melted['snap'] = 0
df_melted[['snap', 'snap_CA', 'snap_TX', 'snap_WI']] = df_melted[['snap', 'snap_CA', 'snap_TX', 'snap_WI']].astype(np.int8)
df_melted.loc[df_melted.state_id == 'CA', 'snap'] = df_melted.loc[df_melted.state_id == 'CA', 'snap_CA']
df_melted.loc[df_melted.state_id == 'TX', 'snap'] = df_melted.loc[df_melted.state_id == 'TX', 'snap_TX']
df_melted.loc[df_melted.state_id == 'WI', 'snap'] = df_melted.loc[df_melted.state_id == 'WI', 'snap_WI']

for col in ['wday', 'month', 'event_name_1', 'event_type_1', 'snap']:
    print(col)
    df_pivoted = df_melted.pivot_table(index=['item_id', 'store_id'], columns='variable', values=col).reset_index()
    df_pivoted = df_idx.merge(df_pivoted, on=['item_id', 'store_id'], how='left')
    df_pivoted = df_pivoted[date_cols].values.astype(np.int8)
    np.save('data/processed/{}.npy'.format(col), df_pivoted)
print('calendar')

   
# prices
def replace_zero(x):
    if x == '0':
        return '00'
    else:
        return x

prices = pd.read_csv('data/raw/sell_prices.csv')
prices_splitted = prices.sell_price.astype(str).str.split('.', expand=True)[1]
prices_splitted = prices_splitted.map(lambda x: replace_zero(x))
prices['sell_price_first_digit'] = prices_splitted.map(lambda x: x[0]).astype(int)
prices['sell_price_last_digit'] = prices_splitted.map(lambda x: x[-1]).astype(int)
prices.sell_price = prices.sell_price.round(0).astype(int)
prices.wm_yr_wk = prices.wm_yr_wk + 1
df_melted = df_melted.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

del prices 
gc.collect()

df_melted[['sell_price', 'sell_price_first_digit', 'sell_price_last_digit']] = df_melted[['sell_price', 'sell_price_first_digit', 'sell_price_last_digit']].fillna(0)

for c in ['sell_price', 'sell_price_first_digit', 'sell_price_last_digit']:
    print(c)
    df_pivoted = df_melted.pivot_table(index=['item_id', 'store_id'], columns='variable', values=c).reset_index()
    df_pivoted = df_idx.merge(df_pivoted, on=['item_id', 'store_id'], how='left')
    df_pivoted = df_pivoted[date_cols].values.astype(np.int8)
    gc.collect()
    np.save('data/processed/{}.npy'.format(c), df_pivoted)    
print('prices')
    
   
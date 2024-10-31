
# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from joblib import Parallel, delayed
from time import time

# specify filepaths and files to be imported
path_ar6_data = '/Users/rubenprutz/Documents/PhD/Primary workspace/Studies/Land sequestration prediction/AR6_R10/'
meta_file = 'AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx'
filename = 'AR6_Scenarios_Database_R10_regions_v1.1.csv'
ar6_db = pd.read_csv(path_ar6_data + filename)
df_ar6_meta = pd.read_excel(path_ar6_data + meta_file,
                            sheet_name='meta_Ch3vetted_withclimate')
ar6_db = pd.merge(ar6_db,
                  df_ar6_meta,
                  how='left', on=['Model', 'Scenario'])
scenario_category = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
ar6_db = ar6_db.loc[ar6_db['Category'].isin(scenario_category)]

# filter required variables and years for selected scenarios
var_df = ar6_db.query(
    'Variable == "Carbon Sequestration|Land Use" | Variable == "Emissions|CO2|AFOLU"').reset_index(drop=True)
numeric_cols = [str(year) for year in range(2020, 2110, 10)]
var_df = var_df[['Model', 'Scenario', 'Region', 'Variable'] + numeric_cols].copy()

# only keep scenarios for which both variables are available
clean_data = var_df.dropna()
clean_data = clean_data.groupby(
    ['Model', 'Scenario', 'Region']).filter(lambda x: len(x) == 2)

# split the data into train and test sets while maintaining balance
train_data_X = clean_data.query(
    'Variable == "Emissions|CO2|AFOLU"').reset_index(drop=True)
train_data_y = clean_data.query(
    'Variable == "Carbon Sequestration|Land Use"').reset_index(drop=True)

train_data_X.set_index(['Model', 'Scenario', 'Region', 'Variable'], inplace=True)
train_data_y.set_index(['Model', 'Scenario', 'Region', 'Variable'], inplace=True)
train_data_y = train_data_y.abs()

# %% use train test split to compare different regression models ############### update this in the clean script

start = time()  # runtime monitoring

# step 1: split the data in training and testing sets for initial grid search
X_train, X_test, y_train, y_test = train_test_split(train_data_X,
                                                    train_data_y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=train_data_X.index.get_level_values('Variable'))

# initialize considered regression models
gbr = HistGradientBoostingRegressor()
models = {
    'Gradient Boosting': MultiOutputRegressor(gbr),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'k-nearest neighbors': KNeighborsRegressor()
}

# define parameters for grid search tuning for the individual models
gb_model_params = {'estimator__min_samples_leaf': [1],
                   'estimator__max_depth': [10, 11]}

dt_model_params = {'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'max_features': [None, 'sqrt', 'log2'],
                   'max_leaf_nodes': [None, 10, 20, 30]}

rf_model_params = {'n_estimators': [100],
                   'max_features': [None, 'sqrt'],
                   'min_samples_split': [2, 3, 4]}

knn_model_params = {'n_neighbors': list(range(1, 10, 1)),
                    'weights': ['uniform', 'distance'],
                    'leaf_size': list(range(1, 5, 1)),
                    'metric': ['euclidean', 'manhattan']}

# step 2: perform grid search for each model and save the best parameters
best_params = {}
for model_name, model in models.items():
    if model_name == 'Gradient Boosting':
        grid_search = GridSearchCV(model, gb_model_params, cv=5, n_jobs=-1)
    elif model_name == 'Decision Tree':
        grid_search = GridSearchCV(model, dt_model_params, cv=5, n_jobs=-1)
    elif model_name == 'Random Forest':
        grid_search = GridSearchCV(model, rf_model_params, cv=5, n_jobs=-1)
    elif model_name == 'k-nearest neighbors':
        grid_search = GridSearchCV(model, knn_model_params, cv=5, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_params[model_name] = grid_search.best_params_
    print(f'Completed parameter tuning for {model_name}')

# step 3: use bootstrapping to test the model's performance using best params
n_bootstraps = 1000

# initialize lists to store results
y_test_dfs = []
y_pred_dfs = []
r2_dfs = []
mae_dfs = []
median_ae_dfs = []
max_ae_dfs = []

def bootstrap_evaluation(model_name, model, best_params, X_train, y_train, X_test, y_test, numeric_cols, i):
    model.set_params(**best_params[model_name])

    # resample with replacement from the training set
    X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, replace=True, random_state=i)
    model.fit(X_train_bootstrap, y_train_bootstrap)
    y_pred = model.predict(X_test)

    # prepare data for plotting evaluation metrics
    y_test_df = y_test.reset_index()

    y_pred_df = pd.DataFrame(y_pred, columns=numeric_cols)
    y_pred_df[['Model', 'Scenario', 'Region']] = y_test_df[['Model', 'Scenario', 'Region']]
    y_pred_df['Reg_model'] = f'{model_name}'
    y_pred_df['Bootstrap'] = i
    y_test_df['Reg_model'] = f'{model_name}'
    y_test_df['Bootstrap'] = i

    r2_by_yr = {}  # r-squared
    mae_by_yr = {}  # mean absolute error
    median_ae_by_yr = {}  # median absolute error
    max_ae_by_yr = {}  # maximum absolute error

    for yr in numeric_cols:
        r2 = r2_score(y_test_df[yr], y_pred_df[yr])
        r2_by_yr[yr] = r2
        mae = mean_absolute_error(y_test_df[yr], y_pred_df[yr])
        mae_by_yr[yr] = mae
        median_ae = median_absolute_error(y_test_df[yr], y_pred_df[yr])
        median_ae_by_yr[yr] = median_ae
        max_ae = max(abs(y_test_df[yr] - y_pred_df[yr]))
        max_ae_by_yr[yr] = max_ae

    r2_df = pd.DataFrame(list(r2_by_yr.items()), columns=['Year', 'r2'])
    r2_df['Reg_model'] = f'{model_name}'
    r2_df['Bootstrap'] = i
    mae_df = pd.DataFrame(list(mae_by_yr.items()), columns=['Year', 'MAE'])
    mae_df['Reg_model'] = f'{model_name}'
    mae_df['Bootstrap'] = i
    median_ae_df = pd.DataFrame(list(median_ae_by_yr.items()), columns=['Year', 'MedianAE'])
    median_ae_df['Reg_model'] = f'{model_name}'
    median_ae_df['Bootstrap'] = i
    max_ae_df = pd.DataFrame(list(max_ae_by_yr.items()), columns=['Year', 'MaxAE'])
    max_ae_df['Reg_model'] = f'{model_name}'
    max_ae_df['Bootstrap'] = i

    return y_test_df, y_pred_df, r2_df, mae_df, median_ae_df, max_ae_df

# use joblib for parallel processing
start = time()
results = Parallel(n_jobs=-1)(delayed(bootstrap_evaluation)(
    model_name, model, best_params, X_train, y_train, X_test, y_test, numeric_cols, i
) for model_name, model in models.items() for i in range(n_bootstraps))

# aggregate results
for y_test_df, y_pred_df, r2_df, mae_df, median_ae_df, max_ae_df in results:
    y_test_dfs.append(y_test_df)
    y_pred_dfs.append(y_pred_df)
    r2_dfs.append(r2_df)
    mae_dfs.append(mae_df)
    median_ae_dfs.append(median_ae_df)
    max_ae_dfs.append(max_ae_df)

# compile evaluation dataframes
y_pred_df = pd.concat(y_pred_dfs, ignore_index=True)
y_test_df = pd.concat(y_test_dfs, ignore_index=True)
r2_df = pd.concat(r2_dfs, ignore_index=True)
mae_df = pd.concat(mae_dfs, ignore_index=True)
median_ae_df = pd.concat(median_ae_dfs, ignore_index=True)
max_ae_df = pd.concat(max_ae_dfs, ignore_index=True)

end = time()
print(f'Runtime {(end - start) / 60} min')

# %% plot performance of tested models
plt.rcParams.update({'figure.dpi': 600})
pal = ['orange', 'slateblue', 'green', 'crimson']

fig, axs = plt.subplots(1, 4, figsize=(10, 2.3), sharex=True)

sns.lineplot(data=r2_df, x='Year', y='r2', hue='Reg_model', estimator='median',
             errorbar=('pi', 90), legend=True, palette=pal, ax=axs[0])  # R-squared
sns.lineplot(data=mae_df, x='Year', y='MAE', hue='Reg_model', estimator='median',
             errorbar=('pi', 90), legend=False, palette=pal, ax=axs[1])  # mean absolut error
sns.lineplot(data=median_ae_df, x='Year', y='MedianAE', hue='Reg_model', estimator='median',
             errorbar=('pi', 90), legend=False, palette=pal, ax=axs[2])  # median absolut error
sns.lineplot(data=max_ae_df, x='Year', y='MaxAE', hue='Reg_model', estimator='median',
             errorbar=('pi', 90), legend=False, palette=pal, ax=axs[3])  # max absolute error

axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('')
axs[3].set_xlabel('')

axs[0].set_ylabel('R-squared')
axs[1].set_ylabel('Mean absolute error\n[MtCO$_2$ yr$^{-1}$]')
axs[2].set_ylabel('Median absolute error\n[MtCO$_2$ yr$^{-1}$]')
axs[3].set_ylabel('Maximum absolute error\n[MtCO$_2$ yr$^{-1}$]')

axs[0].set_xlim(0, 8)
axs[1].set_xlim(0, 8)
axs[2].set_xlim(0, 8)
axs[3].set_xlim(0, 8)

axs[0].set_ylim(0.5, 1)
axs[1].set_ylim(0, 25)
axs[2].set_ylim(0, 8)
axs[3].set_ylim(0, 2500)

sns.despine(ax=axs[0])
sns.despine(ax=axs[1])
sns.despine(ax=axs[2])
sns.despine(ax=axs[3])

plt.xticks([0, 4, 8], ['2020', '2060', '2100'])
plt.subplots_adjust(wspace=0.8)
sns.move_legend(axs[0], 'lower left',
                bbox_to_anchor=(-0.05, 1.02), ncols=5, title='', frameon=False)

# %% plot predicted versus actual of best performing model

# run best model again and plot predictions
model = KNeighborsRegressor()
modelCV = GridSearchCV(model, knn_model_params, cv=5, n_jobs=-1)
modelCV.fit(X_train, y_train)
y_pred = modelCV.predict(X_test)

# prepare data for plotting evaluation metrics
X_test_neg = X_test.applymap(lambda x: 0 if x > 0 else x).abs()
X_test_df = X_test_neg.reset_index()
y_test_df = y_test.reset_index()

y_pred_df = pd.DataFrame(y_pred, columns=numeric_cols)
y_pred_df[['Model', 'Scenario', 'Region']] = y_test_df[['Model', 'Scenario', 'Region']]
y_pred_df['Variable'] = 'Predicted'

X_test_df['Variable'] = 'Net negative AFOLU CO$_2$'
y_test_df['Variable'] = 'Actual'

eval_df = pd.concat([y_test_df, y_pred_df])
eval_df = pd.concat([eval_df, X_test_df])

eval_df = pd.merge(eval_df, df_ar6_meta[['Model', 'Scenario', 'Category']],
                   on=['Model', 'Scenario'])

eval_df = eval_df.sort_values(by='Category')

plot_df = pd.melt(eval_df, id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Category'],
                  var_name='Year',
                  value_vars=numeric_cols)

ax = sns.relplot(data=plot_df,
                 kind='line',
                 x='Year',
                 y='value',
                 col='Region',
                 col_wrap=4,
                 hue='Variable',
                 hue_order=['Actual', 'Predicted',
                            'Net negative AFOLU CO$_2$'],
                 palette=['blue', 'red', 'orange'],
                 estimator='median',
                 errorbar=('pi', 90),
                 height=2.1,
                 aspect=1)

sns.move_legend(ax, 'upper right', bbox_to_anchor=(
    0.81, 1.035), ncols=3, title='')

plt.xlim(0, 8)
plt.ylim(0, 3000)
ax.set(xlabel=None,
       ylabel='Median land CDR with\n5-95 percentile range\n[MtCO$_2$ yr$^{-1}$]')

plt.xticks([0, 4, 8], ['2020', '2060', '2100'])
plt.subplots_adjust(hspace=0.25)
plt.subplots_adjust(wspace=0.3)

scen_count_dict = plot_df.groupby('Region').apply(
    lambda df: df[['Model', 'Scenario']].drop_duplicates().shape[0]
).to_dict()
for ax_i, region_name in zip(ax.axes.flat, plot_df['Region'].unique()):
    scen_count = scen_count_dict[region_name]
    ax_i.set_title(f'{region_name} (n={scen_count})', fontsize=10)

sns.despine()
plt.show()

# %% create imputed dataset for missing data using the best performing model

# filter models that only have AFOLU CO2 but no land sequestration
select_for_imput = var_df.groupby(['Model', 'Scenario', 'Region']).filter(lambda x: len(x) == 1)
select_for_imput.set_index(['Model', 'Scenario', 'Region', 'Variable'], inplace=True)

# set up best model based on training data
model = KNeighborsRegressor()
modelCV = GridSearchCV(model, knn_model_params, cv=5, n_jobs=-1)
modelCV.fit(X_train, y_train)
imputed_data = modelCV.predict(select_for_imput)
imputed_data_df = pd.DataFrame(imputed_data, columns=numeric_cols)

# assign model and scenario
select_for_imput.reset_index(['Model', 'Scenario', 'Region', 'Variable'], inplace=True)
imputed_data_df[['Model', 'Scenario', 'Region']
                ] = select_for_imput[['Model', 'Scenario', 'Region']]

imputed_data_df['Variable'] = 'Imputed|Carbon Sequestration|Land Use'
imputed_data_df['Unit'] = 'Mt CO2/yr'
imputed_data_df['Source'] = 'This study'

imputed_data_df = imputed_data_df[['Model',
                                   'Scenario',
                                   'Region',
                                   'Variable',
                                   'Source',
                                   'Unit'] + numeric_cols].copy()

# small negative values are set to zero tolerance threshold = 10 Mt
imputed_data_df[numeric_cols] = imputed_data_df[numeric_cols].applymap(
    lambda x: 0 if x <= 0 and x >= -10 else x)


imputed_data_adjust = imputed_data_df[['Model',
                                       'Scenario',
                                       'Region',
                                       'Variable',
                                       'Source',
                                       'Unit'] + numeric_cols].copy()
imputed_data_adjust['Variable'] = 'Imputed & Proxy|Carbon Sequestration|Land Use'

# replace entire rows if neg AFOLU is larger than predicted land sequestration
neg_afolu = select_for_imput[['Model', 'Scenario', 'Region'] + numeric_cols].copy()
neg_afolu[numeric_cols] = neg_afolu[numeric_cols].applymap(
    lambda x: 0 if x > 0 else x).abs()

# double check
row_mask = neg_afolu[numeric_cols] > imputed_data_df[numeric_cols]
any_row_true = row_mask.any(axis=1)
imputed_data_adjust.loc[any_row_true,
                       numeric_cols] = neg_afolu.loc[any_row_true, numeric_cols]

imputed_data_adjust['Net negative AFOLU CO2'] = (
    imputed_data_adjust[numeric_cols] != imputed_data_df[numeric_cols]).any(axis=1)

# retrieve available land sequestation in ar6 R10 and add to imputation dataset
ar6_landseq_nativ = ar6_db.loc[ar6_db['Variable'] == 'Carbon Sequestration|Land Use']
ar6_landseq_nativ.reset_index(drop=True, inplace=True)
ar6_landseq_nativ['Variable'] = 'Carbon Sequestration|Land Use'
ar6_landseq_nativ['Unit'] = 'Mt CO2/yr'
ar6_landseq_nativ['Source'] = 'AR6 Scenario Database'
ar6_landseq_nativ = ar6_landseq_nativ[['Model',
                                       'Scenario',
                                       'Region',
                                       'Variable',
                                       'Unit',
                                       'Source'] + numeric_cols].copy()

n_imputed = imputed_data_df[['Model', 'Scenario']].drop_duplicates().shape[0]
n_ar6_nativ = ar6_landseq_nativ[['Model', 'Scenario']].drop_duplicates().shape[0]
n_imputed_r = imputed_data_df[['Model', 'Scenario', 'Region']].drop_duplicates().shape[0]
n_ar6_nativ_r = ar6_landseq_nativ[['Model', 'Scenario', 'Region']].drop_duplicates().shape[0]

imputed_data_df = pd.concat([imputed_data_df, ar6_landseq_nativ])
imputed_data_df[numeric_cols] = imputed_data_df[numeric_cols].abs()  # abs negative data in original variable

# replace available land sequestation with proxy where net>gross removal
ar6_landseq_netAFOLU = ar6_landseq_nativ.copy()
afolu_co2 = ar6_db.loc[ar6_db['Variable'] == 'Emissions|CO2|AFOLU']
afolu_co2[numeric_cols] = afolu_co2[numeric_cols].applymap(
    lambda x: 0 if x > 0 else x).abs()
afolu_co2 = pd.merge(afolu_co2, ar6_landseq_netAFOLU[['Model', 'Scenario', 'Region']],
                     on=['Model', 'Scenario', 'Region'], how='inner')
row_mask = afolu_co2[numeric_cols] > ar6_landseq_netAFOLU[numeric_cols]
any_row_true = row_mask.any(axis=1)
ar6_landseq_netAFOLU.loc[any_row_true,
                         numeric_cols] = afolu_co2.loc[any_row_true,
                                                       numeric_cols]

# specify which rows have been adjusted
ar6_landseq_netAFOLU['Net negative AFOLU CO2'] = (
    ar6_landseq_netAFOLU[numeric_cols] != ar6_landseq_nativ[numeric_cols]).any(axis=1)

# adjust variable name depending on whether row has been adjusted or not
for index, row in ar6_landseq_netAFOLU.iterrows():
    if row['Net negative AFOLU CO2'] is True:
        ar6_landseq_netAFOLU.at[index, 'Variable'] = 'Net negative AFOLU CO2'
imputed_data_adjust = pd.concat([imputed_data_adjust, ar6_landseq_netAFOLU])

# write data to excel
path = '/Users/rubenprutz/Documents/PhD/Primary workspace/Studies/Land sequestration prediction/Paper draft/Data/'

with pd.ExcelWriter(path + 'Prütz_et_al_2024_ar6_land_sequestration_imputation_R10_regions_v1.1.xlsx',
                    engine='openpyxl') as writer:
    imputed_data_df.to_excel(
        writer, sheet_name='Imputation_output', index=False)
    imputed_data_adjust.to_excel(
        writer, sheet_name='Imputation_adjusted', index=False)

# print count of included scenarios
total_count = n_imputed + n_ar6_nativ
total_count = n_imputed + n_ar6_nativ
total_count_r = n_imputed_r + n_ar6_nativ_r
print(f'Imputed scenarios: {n_imputed}')
print(f'Original AR6 scenarios: {n_ar6_nativ}')
print(f'Total number of usable: {total_count}')
print(f'Imputed scenarios across R10: {n_imputed_r}')
print(f'Original AR6 scenarios across R10: {n_ar6_nativ_r}')
print(f'Total number of usable across R10: {total_count_r}')
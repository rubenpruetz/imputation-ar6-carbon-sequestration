
# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# specify filepaths and files to be imported
filepath = '/Users/rubenprutz/Documents/phd/datasets/'  # specify filepath to ar6 db
ar6_data_file = 'AR6_Scenarios_Database_World_v1.1.csv'
meta_file = 'AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx'
gidden_etal_file = '10.5281_zenodo.10158920_gidden_et_al_2023_ar6_reanalysis_data.xlsx'

numeric_cols = [str(year) for year in range(2020, 2110, 10)]

# load datasets
ar6_data = pd.read_csv(filepath + ar6_data_file)
gidden_etal = pd.read_excel(filepath + gidden_etal_file,
                            sheet_name='data')
gidden_etal.columns = gidden_etal.columns.astype(str)

df_ar6_meta = pd.read_excel(filepath + meta_file,
                            sheet_name='meta_Ch3vetted_withclimate')

# filter C1-8 scenarios and select columns
scenario_category = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
ar6_data = pd.merge(ar6_data,
                    df_ar6_meta[['Model', 'Scenario', 'Category']],
                    how='left', on=['Model', 'Scenario'])
ar6_data = ar6_data.loc[ar6_data['Category'].isin(scenario_category)]

gidden_etal = pd.merge(gidden_etal,
                       df_ar6_meta[['Model', 'Scenario', 'Category']],
                       how='left', on=['Model', 'Scenario'])
gidden_etal = gidden_etal.loc[gidden_etal['Category'].isin(scenario_category)]
gidden_etal = gidden_etal.loc[gidden_etal['Region'].isin(['World'])]

ar6_data = ar6_data[['Model',
                     'Scenario',
                     'Region',
                     'Variable',
                     'Unit',
                     'Category'] + numeric_cols].copy()

gidden_etal = gidden_etal[['Model',
                           'Scenario',
                           'Region',
                           'Variable',
                           'Unit',
                           'Category'] + numeric_cols].copy()

# compile the different land CDR related variables
cdr_array = ['Carbon Sequestration|Land Use']
land_seq = ar6_data.loc[ar6_data['Variable'].isin(cdr_array)]

cdr_array = ['AR6 Reanalysis|OSCARv3.2|Carbon Removal|Land|Direct']
gidden_seq = gidden_etal.loc[gidden_etal['Variable'].isin(cdr_array)]

# use net negative AFOLU emissions as conservative proxy for land sequestration
afolu_seq = ar6_data.loc[ar6_data['Variable'] == 'Emissions|CO2|AFOLU']
afolu_seq = afolu_seq.set_index(['Model',
                                 'Scenario',
                                 'Region',
                                 'Variable',
                                 'Unit',
                                 'Category'])
afolu_seq[afolu_seq > 0] = 0
afolu_seq = afolu_seq.abs()
afolu_seq.reset_index(inplace=True)
afolu_seq['Variable'] = 'netAFOLU_CDR'

# determine shared scenarios between the three dfs
shared_scen_df = pd.merge(land_seq[['Model', 'Scenario']],
                          gidden_seq[['Model', 'Scenario']],
                          on=['Model', 'Scenario'],
                          how='inner')

shared_scen_df = pd.merge(shared_scen_df,
                          afolu_seq[['Model', 'Scenario']],
                          on=['Model', 'Scenario'],
                          how='inner')

land_cdr_df = pd.concat([land_seq, gidden_seq, afolu_seq], axis=0)

# filter scenarios that report all required variables
land_cdr_df = pd.merge(land_cdr_df,
                       shared_scen_df[['Model', 'Scenario']],
                       on=['Model', 'Scenario'],
                       how='inner')

# plot compiled data
plot_df = pd.melt(land_cdr_df,
                  id_vars=['Category', 'Variable'],
                  value_vars=numeric_cols)

plot_df['Variable'].replace({'Carbon Sequestration|Land Use': 'AR6 Land CDR',
                             'AR6 Reanalysis|OSCARv3.2|Carbon Removal|Land|Direct':
                             'Gidden et al. Land CDR (direct)',
                             'netAFOLU_CDR':
                             'Net negative AFOLU CO$_2$'},
                            inplace=True)

plt.rcParams.update({'figure.dpi': 600})
g = sns.relplot(x='variable',
                y='value',
                data=plot_df,
                col='Category',
                col_order=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
                col_wrap=4,
                kind='line',
                linewidth=1.2,
                hue='Variable',
                hue_order=['AR6 Land CDR',
                           'Gidden et al. Land CDR (direct)',
                           'Net negative AFOLU CO$_2$'],
                palette=['blue', 'green', 'orange'],
                errorbar=('pi', 100),  # choose percentile range
                estimator='median',  # choose between mean and median
                height=2.5,
                aspect=0.8)

g.set_ylabels('Median land CDR\nwith min-max range\n[MtCO$_2$ yr$^{-1}$]',
              clear_inner=False,
              fontsize=12)
plt.xticks([0, 4, 8], ['2020', '2060', '2100'])
plt.xlim(0, 8)
plt.ylim(-5500, 10000)
g.set_xlabels('', clear_inner=False)
plt.subplots_adjust(wspace=0.35)

sns.move_legend(g, 'upper right', bbox_to_anchor=(0.735, 1.035),
                ncols=3, title='')

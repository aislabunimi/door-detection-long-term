import pandas as pd

results = pd.read_excel('./../results/epochs_and_backbone_analysis.xlsx')
results = results.groupby(by=['backbone', 'detector', 'epochs_gd', 'epochs', 'label']).agg({'AP': ['mean', 'std']})
results = results.xs('AP', axis=1, drop_level=True)
results.rename(columns={'mean':'AP'}, inplace=True)
with pd.ExcelWriter('../../results/epochsand_backbone_aggregation.xlsx') as writer:
    if not results.index.name:
        results.index.name = 'Index'
    results.to_excel(writer, sheet_name='s')
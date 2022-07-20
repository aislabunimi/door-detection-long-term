import pandas as pd

results = pd.read_excel('./../results/epochs_and_backbone_analysis.xlsx')
results =results.groupby(by=['backbone', 'detector', 'epochs_gd', 'epochs', 'label']).mean()
with pd.ExcelWriter('./../results/epochsand_backbone_aggregation.xlsx') as writer:
    if not results.index.name:
        results.index.name = 'Index'
    results[['AP']].to_excel(writer, sheet_name='s')
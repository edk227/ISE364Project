from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    names=columns, skipinitialspace=True, na_values='?'
)
df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1})

# GDP per capita values supplied for each country.
gdp_per_capita = {
    'United-States': 27694.85,
    'Cambodia': 302.13,
    'England': 21443,
    'Puerto-Rico': 10876.42,
    'Canada': 19997.62,
    'Germany': 27201.96,
    'Outlying-US(Guam-USVI-etc)': 7675,
    'India': 347.73,
    'Japan': 39933.52,
    'Greece': 10937.02,
    'South': 10385.39,
    'China': 475.68,
    'Cuba': 2613.29,
    'Iran': 1167.76,
    'Honduras': 832.06,
    'Philippines': 1059.98,
    'Italy': 19382.20,
    'Poland': 2874.83,
    'Jamaica': 2202.86,
    'Vietnam': 229.86,
    'Mexico': 6202.37,
    'Portugal': 9977.32,
    'Ireland': 15902.93,
    'France': 23359.90,
    'Dominican-Republic': 1891.67,
    'Laos': 320.97,
    'Ecuador': 1996.86,
    'Taiwan': 12009.75,
    'Haiti': 1806.82,
    'Columbia': 2325.98,
    'Hungary': 4187.03,
    'Guatemala': 1292.67,
    'Nicaragua': 849.18,
    'Scotland': 19709,
    'Thailand': 2517.71,
    'Yugoslavia': 2113,
    'El-Salvador': 1343.93,
    'Trinadad&Tobago': 3839.96,
    'Peru': 1879.37,
    'Hong': 22502.53,
    'Holand-Netherlands': 24682.59,
}
country = df['native-country'].str.strip().str.rstrip('.')
df['GDP-per-capita'] = country.map(gdp_per_capita)

numerical_features = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt',
    'GDP-per-capita'
]
categorical_features = [
    'workclass', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'education'
]

# Correlate only numeric features with income.
features = df[numerical_features]
correlations = features.corrwith(df['income']).sort_values()
correlations.plot.barh(figsize=(8, 5), color='seagreen')
plt.title('Numeric feature correlation with income')
plt.xlabel('Pearson correlation')
plt.tight_layout()
output = Path(__file__).with_name('feature_income_correlation.png')
plt.savefig(output, dpi=150)
plt.close()
print(f'Chart saved to: {output}')

# Count each income group within every category of the original features.
chart_folder = Path(__file__).with_name('categorical_income_charts')
chart_folder.mkdir(exist_ok=True)
for feature in ['education'] + categorical_features:
    counts = pd.crosstab(df[feature].fillna('Missing'), df['income'])
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    counts.columns = ['<=50K', '>50K']
    ax = counts.plot.bar(
        figsize=(max(9, len(counts) * 0.45), 6),
        color=['seagreen', 'steelblue'], rot=60
    )
    ax.set_title(f'Income counts by {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend(title='Income')
    plt.tight_layout()
    plt.savefig(chart_folder / f'{feature}_income_counts.png', dpi=150)
    plt.close()
print(f'Categorical income charts saved to: {chart_folder}')

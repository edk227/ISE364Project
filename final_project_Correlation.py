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

numerical_features = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt'
]
categorical_features = [
    'workclass', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'education'
]

# Use original numeric features; categorical columns need separate encoding.
features = pd.get_dummies(df.drop(columns='income'), dtype=int)
correlations = features.corrwith(df['income']).sort_values()
correlations.plot.barh(figsize=(8, 5), color='seagreen')
plt.title('Feature correlation with income')
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

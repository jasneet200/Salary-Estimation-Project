import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('india_job_market_2024_2026.csv')

# --- Normalize column names ---
df.columns = df.columns.str.strip().str.lower()



# --- Normalize text values for relevant columns ---
for col in ['company_type', 'location_tier', 'experience_level',
            'role', 'industry', 'job_type', 'education_required', 'skills_required']:
    df[col] = df[col].str.strip().str.lower()



# --- Skills encoding ---
def tech_skills(df):
    skills = df['skills_required'].apply(lambda x: [s.strip().lower() for s in x.split(',')])
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(skills)
    skills_df = pd.DataFrame(encoded, columns=mlb.classes_)
    df = pd.concat([df, skills_df], axis=1)
    return df

Tech_skills=FunctionTransformer(tech_skills)



# --- Ordinal encoding for company type and location ---
loc_category = ['remote', 'tier 2', 'tier 1']
company_category = ['startup', 'indian unicorn', 'psu/govt', 'mnc']

oe = OrdinalEncoder(categories=[company_category, loc_category])
df[['encoded_company', 'encoded_loc']] = oe.fit_transform(df[['company_type', 'location_tier']])

# --- Experience level split ---
post = df['experience_level'].str.split('(').str[0].str.strip().str.lower()
years = df['experience_level'].str.split('(').str[1].str.split(')').str[0].str.replace(r'yrs?', '', regex=True).str.strip()
years = years.str.replace('+', '')

# Split into two columns and convert to numbers
years_split = years.str.split('-', expand=True)
years_split = years_split.apply(pd.to_numeric, errors='coerce')

# Row-wise mean
df['years_avg'] = years_split.mean(axis=1)

# --- Ordinal encoding for post ---
cat_post = [['fresher', 'junior', 'mid', 'senior', 'lead']]
oe = OrdinalEncoder(categories=cat_post)
df['encoded_post'] = oe.fit_transform(post.to_frame())

# --- OneHotEncoder for other categorical columns ---
transformer = ColumnTransformer(transformers=[
    ('trf1', OneHotEncoder(sparse_output=False, drop='first'),
     ['role', 'industry', 'job_type', 'education_required'])
], remainder='passthrough')

transformed_data = transformer.fit_transform(df)

# Get new column names
named_colm = transformer.named_transformers_['trf1'].get_feature_names_out(
    ['role', 'industry', 'job_type', 'education_required']
)

# Remaining columns
passthrough_cols = [c for c in df.columns if c not in ['role', 'industry', 'job_type', 'education_required']]

# Combine
all_cols = list(named_colm) + passthrough_cols
transformed_df = pd.DataFrame(transformed_data, columns=all_cols)

# Drop unnecessary columns
transformed_df = transformed_df.drop([
    'unnamed: 1', 'company_type', 'unnamed: 4',
    'location_tier', 'experience_level',
    'unnamed: 8', 'skills_required'
], axis=1)

print(transformed_df.columns)

# Save to CSV
transformed_df.to_csv('Example.csv', index=True)

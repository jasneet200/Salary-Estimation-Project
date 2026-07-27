import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
import joblib
import re
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
df=pd.read_csv('indian_job_maket.csv')
df = df.loc[:, ~df.columns.str.contains("   ")]
def preprocess_text(df):
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Normalize text values for categorical columns
    for col in ['company_type', 'location_tier', 'years_avg','post', 'role', 'industry', 'job_type', 'education_required', 'skills_required']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
   
    return df

# Wrap as transformer
text_cleaner = FunctionTransformer(preprocess_text)
df = preprocess_text(df)


# -- making qualification user entry

# def education(x):
#     mapping={
#     'm.tech/m.e.':'m.tech/m.e.',
#     'master of technology' : 'm.tech/m.e.',
#     'master of engineering' :'m.tech/m.e.',
#     'b.tech/b.e.': 'b.tech/b.e.',
#     'bachelor of engineering' :'b.tech/b.e.',
#     'bachelor of technology': 'b.tech/b.e.',
#     'b.tech':'b.tech/b.e.',
#     'b.e.' : 'b.tech/b.e.',
#     'mca':'mca',
#     'master of computer application': 'mca',
#     'bca' : 'bca',
#     'bachelor of computer application' : 'bca',
#     'mba' : 'mba',
#     'master of business administration' : 'mba',
#     'b.sc (cs/it)' : 'b.sc (cs/it)',
#     'bsc cs': 'b.sc (cs/it)',
#     'bsc it' : 'b.sc (cs/it)',
#     'bachelor of science in computer science':'b.sc (cs/it)',
#     'bachelor of science in information technology': 'b.sc (cs/it)',
#     'b.com + certification' :'b.com + certification',
#     'bcom with certification' : 'b.com + certification',
#     'bachelor of commerce with certification' :'b.com + certification',
#     'phd' :'phd',
#     'doctor of philosophy': 'phd',
#     }
#     X=x.copy()
#     X['education_required']=X['education_required'].map(mapping).fillna(X['education_required'])
#     return X


def education(x):
    patterns = {
        r"(m\.tech|m\.e\.|master of technology|master of engineering)": "m.tech/m.e.",
        r"(b\.tech|btech|b\.e\.|be|bachelor of engineering|bachelor of technology)": "b.tech/b.e.",
        r"(mca|master of computer application)": "mca",
        r"(bca|bachelor of computer application)": "bca",
        r"(mba|master of business administration)": "mba",
        r"(bsc\s*(cs|it)|b\.sc\s*\(cs/it\)|bachelor of science in computer science|bachelor of science in information technology)": "b.sc (cs/it)",
        r"(bcom.*certification|b\.com.*certification|bachelor of commerce.*certification)": "b.com + certification",
        r"(phd|doctor of philosophy)": "phd"
    }

    X = x.copy()
    for pattern, replacement in patterns.items():
        X['education_required'] = X['education_required'].str.lower().str.replace(pattern, replacement, regex=True)
    return X

        
education_inp=FunctionTransformer(education)

# --- Skills encoding transformer ---




class SkillsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        skills = X['skills_required'].apply(lambda x: [s.strip().lower() for s in x.split(',')])
        self.mlb.fit(skills)
        return self

    def transform(self, X):
        skills = X['skills_required'].apply(lambda x: [s.strip().lower() for s in x.split(',')])
        encoded = self.mlb.transform(skills)
        skills_df = pd.DataFrame(encoded, columns=self.mlb.classes_, index=X.index)
        return pd.concat([X.drop(columns=['skills_required']), skills_df], axis=1)


# --- ColumnTransformer for categorical features ---
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore'),
         ['role', 'industry', 'job_type', 'education_required']),
        ('ordinal_company_loc', OrdinalEncoder(categories=[
            ['startup','indian unicorn','psu/govt','mnc'],
            ['remote','tier 2','tier 1']
        ]),
         ['company_type','location_tier']),
        ('ordinal_post', OrdinalEncoder(categories=[['fresher','junior','mid','senior','lead']]),
         ['post'])  # now using 'post' column directly
    ],
    remainder='passthrough'
)
target = 'salary_lpa'
X = df.drop(columns=[target])
y = df[target]

# ---- Step 2: Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# --- Full pipeline ---
pipeline = Pipeline(steps=[
    ('data_clean',text_cleaner),
    ('education_inp',education_inp),
    ('skills_transform', SkillsTransformer()),
    ('preprocessor_columntransform', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    ))
])

# Fit pipeline
pipeline.fit(X_train, y_train)
sample = df.drop(columns=[target]).iloc[[4988]]
# print(sample)   # keep 2D shape
# y_pred = pipeline.predict(X_test)

# # Save pipeline
# joblib.dump(pipeline, 'job_market_pipeline.pkl')

# Later: load and use on new user data
# pipeline = joblib.load('job_market_pipeline.pkl')
# predictions = pipeline.predict(new_df)
# manual input
roles = [
    "Android Developer", "QA Engineer", "Business Analyst", "Cybersecurity Analyst",
    "Python Developer", "Backend Developer", "Power BI Developer", "Java Developer",
    "Technical Lead", "UI/UX Designer", "Data Analyst", "Software Engineer",
    "Machine Learning Engineer", "Data Engineer", "Node.js Developer", "Computer Vision Engineer",
    "MLOps Engineer", "DevOps Engineer", "Product Manager", "Frontend Developer",
    "Full Stack Developer", "Cloud Engineer", "React Developer", "Blockchain Developer",
    "AI Engineer", "Research Scientist", "iOS Developer", "NLP Engineer",
    "Data Scientist", "Engineering Manager"
]

# print("Choose your role from the following list:")
# for i, role in enumerate(roles, 1):
#     print(f"{i}. {role}")

# choice = int(input("Enter the number corresponding to your role: "))
# inp_role = roles[choice- 1]
# print(f"You selected: {inp_role}")
# print("")
# company_type=['MNC', 'Indian Unicorn', 'PSU/Govt', 'Startup']
# for i,comp in enumerate(company_type,1):
#     print(i,comp)
# choice=int(input("choose company type"))
# inp_cp_type=company_type[choice-1]


# #input industry type
# indusType=['Information Technology',                 'EdTech',      'Banking & Finance', 'Media & Entertainment','FinTech','Government/PSU','HealthTech','E-Commerce','Logistics','Automobile','Gaming','Telecom','Manufacturing','Consulting','Retail']
# for i, ch in enumerate(indusType,1):
#     print(i,ch)
# choice=int(input("choose ur industry"))
# industry_type=indusType[choice-1]











# dataa = {
#     "Role": inp_role,
#     "Company_Type": inp_cp_type,
#     "Industry":industry_type ,
#     "Location_Tier": "tier 1",
#     "Job_Type": "Full-time",
#     "Skills_Required": "python, pandas,machine learning,c++,algorithms,scikit-learn ",
#     "Education_Required": "bca",
#     "years_avg": 20,
#     "post": "senior"
# }
# dff=pd.DataFrame([dataa])
# # Predict on the test set
# y_test_pred = pipeline.predict(X_test)
# output=pipeline.predict(dff)
# print("yor lpa= ",output)
# R² score (how well the model explains variance)
# r2 = r2_score(y_test, y_pred)

# RMSE (average prediction error in salary units)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print("R² Score:", r2)
# print("RMSE:", rmse)


# saving the model
joblib.dump(pipeline,'salary_estimator_project.pkl')

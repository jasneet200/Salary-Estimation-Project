import joblib
import pandas as pd
from pipelinee import preprocess_text,education
pipeline=joblib.load('salary_estimator_project.pkl')
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

print("Choose your role from the following list: ")
for i, role in enumerate(roles, 1):
    print(f"{i}. {role}")

choice = int(input("Enter the number corresponding to your role: "))
inp_role = roles[choice- 1]
print(f"You selected: {inp_role}")
print("")

#input company type
company_type=['MNC', 'Indian Unicorn', 'PSU/Govt', 'Startup']
for i,comp in enumerate(company_type,1):
    print(i,comp)
choice=int(input("choose company type : "))
inp_cp_type=company_type[choice-1]
print("")

#input industry type
indusType=['Information Technology',                 'EdTech',      'Banking & Finance', 'Media & Entertainment','FinTech','Government/PSU','HealthTech','E-Commerce','Logistics','Automobile','Gaming','Telecom','Manufacturing','Consulting','Retail']
for i, ch in enumerate(indusType,1):
    print(i,ch)
choice=int(input("choose ur industry : "))
industry_type=indusType[choice-1]
print("")

# input location tier
locType=['Remote', 'Tier 2', 'Tier 1']
for i, ch in enumerate(locType,1):
    print(i,ch)
choice=int(input("choose ur city location : "))
loc_type=locType[choice-1]
print("")

#input job type
jobType=['Full-Time', 'Part-Time', 'Internship', 'Contract']
for i, ch in enumerate(jobType,1):
    print(i,ch)
choice=int(input("choose ur job type : "))
job_type=jobType[choice-1]
print("")

#input skiils u have
skills=str(input("enter ur skills with comma-separation eg(python,pandas) : "))
print("")

# input education
educ=str(input("enter ur education without specalization : "))
print("")

#input of expperience
yrs=str(input("enter ur experience in years : "))
print("")

#input post lvl

postType=['senior', 'mid', 'junior', 'fresher', 'lead']
for i, ch in enumerate(postType,1):
    print(i,ch)
choice=int(input("choose ur job type : "))
post_type=postType[choice-1]
print("")





dataa = {
    "Role": inp_role,
    "Company_Type": inp_cp_type,
    "Industry":industry_type ,
    "Location_Tier": loc_type,
    "Job_Type": job_type,
    "Skills_Required": skills,
    "Education_Required": educ,
    "years_avg": yrs,
    "post": post_type
}


df=pd.DataFrame([dataa])
output=pipeline.predict(df)[0]
print(f'according to your specification your estimated salary is : {output}LPA')
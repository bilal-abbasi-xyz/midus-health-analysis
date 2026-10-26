import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Page Configuration
st.set_page_config(page_title="Midlife Marital Status & Health", layout="wide")

st.title("Impact of Marital Status on Health")

st.header("1. Target Audience + Problem")
st.markdown("""
* **Target Audience:** The target audience includes public health policy officials and workplace health workers.
* **Problem:** This presentation tries to argue that there is significant reason to research the potential negative effects of relationship
 separation on a person's health in the long-term. This is mainly done by comparing health attributes for both married and unmarried/divorced individuals.
 The hypothsis is that more research in this area may conclude that better health policies are needed to support such individuals.
""")

st.divider()

st.header("2. Data Source + Cleaning")
st.markdown("""
* **Sources:** MIDUS 2 Survey (ICPSR 04652, N=4,963) & Biomarker Study (ICPSR 29282, N=1,255).
* **Cleaning considerations:**
  1. Convert missing codes (`7`, `8`, `9`, `-1`) to `NaN`.
  2. Some participants are missing marital status.
  3. 'Poor' is at 5 while 'Excellent' is at 1 so graph may give wrong first impression.
  4. Grouped age as <55 and 55+ and also grouped education.
  5. Most people from the biomarker data are not present in the survey data.
""")

@st.cache_data
def load_and_clean():
    raw_survey = pd.read_csv("survey_filtered.csv")
    raw_bio = pd.read_csv("29282-0001-Data.tsv", sep='\t')
    
    cols = ['M2ID', 'B1PB19', 'B1PA1', 'B1PA15', 'B1PA2', 'B1PRSEX', 'B1PAGE_M2', 'B1PB1']
    df_s = raw_survey[cols].copy()
    
    for col in ['B1PB19', 'B1PA1', 'B1PA15', 'B1PA2', 'B1PB1']:
        df_s[col] = df_s[col].replace([7, 8, 9, 97, 98, 99, -1], np.nan)
        
    df_clean = df_s.dropna(subset=['B1PB19']).copy()
    
    marital_map = {1: 'Married', 2: 'Separated', 3: 'Divorced', 4: 'Widowed', 5: 'Never Married'}
    df_clean['marital_status'] = df_clean['B1PB19'].map(marital_map)
    df_clean['phys_health'] = 6 - df_clean['B1PA1']
    df_clean['mental_health'] = 6 - df_clean['B1PA15']
    df_clean['gender'] = df_clean['B1PRSEX'].map({1: 'Male', 2: 'Female'})
    df_clean['chronic_count'] = df_clean['B1PA2']
    df_clean['age'] = df_clean['B1PAGE_M2']
    df_clean['education_code'] = df_clean['B1PB1']
    
    df_clean['age_group'] = np.where(df_clean['age'] < 55, 'Midlife (<55)', 'Older Adult (55+)')
    df_clean['education_tier'] = pd.cut(df_clean['education_code'], 
                                        bins=[0, 5, 8, 12], 
                                        labels=['Low Education', 'Medium Education', 'High Education'])
    
    df_b = raw_bio[['M2ID', 'B4BFGN']].copy()
    df_b['B4BFGN'] = df_b['B4BFGN'].replace([999998, 999999, -1], np.nan)
    df_merged = pd.merge(df_clean, df_b, on='M2ID', how='inner')
    
    return raw_survey, df_clean, df_merged

raw_survey, clean_df, merged_df = load_and_clean()

m1, m2, m3 = st.columns(3)
m1.metric("Raw Survey N", f"{len(raw_survey):,}")
m2.metric("Cleaned Survey N", f"{len(clean_df):,}")
m3.metric("Merged Biomarker N", f"{len(merged_df):,}")

st.write("### Complete Cleaned Dataset")
st.dataframe(clean_df[['M2ID', 'marital_status', 'phys_health', 'mental_health', 'chronic_count', 'gender', 'age', 'age_group', 'education_tier']], use_container_width=True)

st.divider()

st.sidebar.header("Filters")
gender_filter = st.sidebar.selectbox("Gender", ["All Genders", "Male", "Female"])
age_filter = st.sidebar.selectbox("Age Group", ["All Ages", "Midlife (<55)", "Older Adult (55+)"])
edu_filter = st.sidebar.selectbox("Education Tier", ["All Education Tiers", "Low Education", "Medium Education", "High Education"])

filtered_df = clean_df.copy()
filtered_merged = merged_df.copy()

if gender_filter != "All Genders":
    filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
    filtered_merged = filtered_merged[filtered_merged['gender'] == gender_filter]

if age_filter != "All Ages":
    filtered_df = filtered_df[filtered_df['age_group'] == age_filter]
    filtered_merged = filtered_merged[filtered_merged['age_group'] == age_filter]

if edu_filter != "All Education Tiers":
    filtered_df = filtered_df[filtered_df['education_tier'] == edu_filter]
    filtered_merged = filtered_merged[filtered_merged['education_tier'] == edu_filter]

st.header("3. Visualizations & Statistical Results")

st.subheader("Self-Rated Physical Health")
fig1a, ax1a = plt.subplots(figsize=(8, 3.5))
sns.barplot(data=filtered_df, x="marital_status", y="phys_health", ci=None, palette="Blues_d", ax=ax1a)
ax1a.set_title("Self-Rated Physical Health (1 = Poor, 5 = Excellent)")
ax1a.set_xlabel("Marital Status")
ax1a.set_ylabel("Mean Score")
ax1a.set_ylim(1, 5)
st.pyplot(fig1a)

m_phys = clean_df[clean_df['marital_status']=='Married']['phys_health'].mean()
d_phys = clean_df[clean_df['marital_status']=='Divorced']['phys_health'].mean()
t_stat_phys, p_val_phys = stats.ttest_ind(clean_df[clean_df['marital_status']=='Married']['phys_health'].dropna(),
                                          clean_df[clean_df['marital_status']=='Divorced']['phys_health'].dropna())

st.markdown(f"""
* **Married Mean:** `{m_phys:.2f}` | **Divorced Mean:** `{d_phys:.2f}` (Diff: `{(m_phys - d_phys):.2f}`)
* **t-Test (Married vs Divorced):** $t = {t_stat_phys:.3f}$, $p = {p_val_phys:.5f}$ (Statistically significant deficit).
""")

st.divider()

st.subheader("Chronic Condition Counts")
fig1b, ax1b = plt.subplots(figsize=(8, 3.5))
sns.boxplot(data=filtered_df, x="marital_status", y="chronic_count", palette="Purples_d", ax=ax1b)
ax1b.set_title("Distribution of Chronic Conditions (Past 12 Months)")
ax1b.set_xlabel("Marital Status")
ax1b.set_ylabel("Condition Count")
st.pyplot(fig1b)

m_var_chr = clean_df[clean_df['marital_status']=='Married']['chronic_count'].var()
d_var_chr = clean_df[clean_df['marital_status']=='Divorced']['chronic_count'].var()
lev_stat, lev_pval = stats.levene(clean_df[clean_df['marital_status']=='Married']['chronic_count'].dropna(),
                                   clean_df[clean_df['marital_status']=='Divorced']['chronic_count'].dropna())

st.markdown(f"""
* **Married Variance:** `{m_var_chr:.3f}` | **Divorced Variance:** `{d_var_chr:.3f}` (`{((d_var_chr - m_var_chr)/m_var_chr * 100):.1f}%` higher variance)
* **Levene Variance Test:** $W = {lev_stat:.3f}$, $p = {lev_pval:.4e}$ (Significantly higher health outcome dispersion post-divorce).
""")

st.divider()

st.subheader("Blood Fibrinogen (Biomarker)")
fig2, ax2 = plt.subplots(figsize=(8, 3.5))
sns.barplot(data=filtered_merged, x="marital_status", y="B4BFGN", ci=None, palette="Reds_d", ax=ax2)
ax2.set_title("Mean Blood Fibrinogen Levels (mg/dL)")
ax2.set_xlabel("Marital Status")
ax2.set_ylabel("Fibrinogen (mg/dL)")
st.pyplot(fig2)

m_fib = merged_df[merged_df['marital_status']=='Married']['B4BFGN'].mean()
s_fib = merged_df[merged_df['marital_status']=='Separated']['B4BFGN'].mean()
t_fib, p_fib = stats.ttest_ind(merged_df[merged_df['marital_status']=='Married']['B4BFGN'].dropna(),
                               merged_df[merged_df['marital_status']=='Separated']['B4BFGN'].dropna())

st.markdown(f"""
* **Married Mean:** `{m_fib:.2f} mg/dL` | **Separated Mean:** `{s_fib:.2f} mg/dL` (Spike: `+{(s_fib - m_fib):.2f} mg/dL` / +17.4%)
* **t-Test (Married vs Separated):** $t = {t_fib:.3f}$, $p = {p_fib:.4f}$ (Significant biological inflammation spike).
""")

st.divider()

st.subheader("Self-Rated Mental Health by Gender")
fig3, ax3 = plt.subplots(figsize=(8, 3.5))
sns.barplot(data=filtered_df, x="marital_status", y="mental_health", hue="gender", ci=None, ax=ax3)
ax3.set_title("Self-Rated Mental Health by Gender")
ax3.set_xlabel("Marital Status")
ax3.set_ylabel("Mental Health Score (1-5)")
ax3.set_ylim(1, 5)
st.pyplot(fig3)

m_fem_mh = clean_df[(clean_df['marital_status']=='Married') & (clean_df['gender']=='Female')]['mental_health'].mean()
d_fem_mh = clean_df[(clean_df['marital_status']=='Divorced') & (clean_df['gender']=='Female')]['mental_health'].mean()
m_male_mh = clean_df[(clean_df['marital_status']=='Married') & (clean_df['gender']=='Male')]['mental_health'].mean()
d_male_mh = clean_df[(clean_df['marital_status']=='Divorced') & (clean_df['gender']=='Male')]['mental_health'].mean()

st.markdown(f"""
* **Married vs Divorced Males:** `{m_male_mh:.2f}` vs `{d_male_mh:.2f}` (Diff: `{(d_male_mh - m_male_mh):.2f}`)
* **Married vs Divorced Females:** `{m_fem_mh:.2f}` vs `{d_fem_mh:.2f}` (Diff: `{(d_fem_mh - m_fem_mh):.2f}`)
""")

st.divider()

st.header("4. Statistical Summary Table")
full_summary = clean_df.groupby("marital_status").agg(
    Sample_Size=('M2ID', 'size'),
    Phys_Health_Mean=('phys_health', 'mean'),
    Phys_Health_Std=('phys_health', 'std'),
    Phys_Health_Var=('phys_health', 'var'),
    Mental_Health_Mean=('mental_health', 'mean'),
    Mental_Health_Std=('mental_health', 'std'),
    Chronic_Count_Median=('chronic_count', 'median'),
    Chronic_Count_Var=('chronic_count', 'var')
).round(3).reset_index()

st.table(full_summary)

st.header("5. Limitations")
st.warning("""
1. **Correlation vs Causation:** This data does not prove causality or rule out pre-existing causes. 
Longitudinal studies would be more suitable for establishing causality.
2. **Clinic Selection Bias:** Biomarker sub-study required clinic travel while the survey study was conducted on phone.
So it's possible that impaired individuals are underrepresented.
3. **Individual Variance:** Standard deviations exceed mean group differences   .
4. **Self-Reported Bias:** Self-rated health is subjective.
5. "Marital Status" is not a very informative term. For instance, it does not measure relationship quality.
6. **Education Grouping:"" The education groups may not reflect meaningful differences in health.
 Another option was to group by income but income itself is also variable.
""")
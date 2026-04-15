# ============================================================
#  Crime Data Analysis - Maryland Counties (1975 to 2020)
#  Libraries: numpy, pandas, matplotlib, seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Set a clean style for all plots ──────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
#  STEP 1 ── LOAD DATA
# ============================================================
df = pd.read_csv("Violent_Crime___Property_Crime_by_County__1975_to_Present.csv")

print("=" * 60)
print("  CRIME DATA ANALYSIS  |  Maryland Counties  |  1975-2020")
print("=" * 60)
print(f"\nDataset Shape : {df.shape[0]} rows  x  {df.shape[1]} columns")
print(f"Year Range    : {df['YEAR'].min()} – {df['YEAR'].max()}")
print(f"Counties      : {df['JURISDICTION'].nunique()}")

# ============================================================
#  STEP 2 ── DATA CLEANING
# ============================================================
print("\n--- BEFORE CLEANING ---")
print("Missing values per column (only those with nulls):")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# The percent-change columns are NaN for 1975 (first year, no prior year)
# → Fill with 0.0 (no change for the baseline year)
pct_change_cols = [c for c in df.columns if 'CHANGE' in c]
df[pct_change_cols] = df[pct_change_cols].fillna(0.0)

# Strip extra whitespace from text columns
df['JURISDICTION'] = df['JURISDICTION'].str.strip()

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

print("\n--- AFTER CLEANING ---")
print(f"Missing values remaining : {df.isnull().sum().sum()}")
print(f"Rows after dedup         : {len(df)}")

# ============================================================
#  HELPER ── statewide yearly aggregates
# ============================================================
yearly = (df.groupby('YEAR')[['VIOLENT CRIME TOTAL',
                               'PROPERTY CRIME TOTALS',
                               'MURDER', 'ROBBERY',
                               'GRAND TOTAL', 'POPULATION']]
            .sum()
            .reset_index())

yearly['VIOLENT RATE']  = (yearly['VIOLENT CRIME TOTAL']  / yearly['POPULATION']) * 100_000
yearly['PROPERTY RATE'] = (yearly['PROPERTY CRIME TOTALS'] / yearly['POPULATION']) * 100_000
yearly['MURDER RATE']   = (yearly['MURDER'] / yearly['POPULATION']) * 100_000

# ============================================================
#  PLOT 1 ── LINE CHART
#  Objective: Overall crime trend across Maryland (1975–2020)
# ============================================================
fig, ax = plt.subplots()
ax.plot(yearly['YEAR'], yearly['VIOLENT RATE'],
        marker='o', markersize=4, label='Violent Crime Rate', color='crimson')
ax.plot(yearly['YEAR'], yearly['PROPERTY RATE'],
        marker='s', markersize=4, label='Property Crime Rate', color='steelblue')
ax.set_title('Crime Rate Trend in Maryland (1975–2020)\nper 100,000 People', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Crime Rate per 100,000 People')
ax.legend()
plt.tight_layout()
plt.savefig('plot1_line_trend.png', dpi=150)
plt.show()
print("\n[Plot 1 saved] Line chart – overall crime trend")

# ============================================================
#  PLOT 2 ── BAR CHART
#  Objective: Compare total violent crime by county (all years)
# ============================================================
county_total = (df.groupby('JURISDICTION')['VIOLENT CRIME TOTAL']
                  .sum()
                  .sort_values(ascending=False)
                  .reset_index())

fig, ax = plt.subplots(figsize=(14, 6))
colors = sns.color_palette("Reds_r", len(county_total))
ax.bar(county_total['JURISDICTION'], county_total['VIOLENT CRIME TOTAL'],
       color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Total Violent Crimes by County (1975–2020)', fontsize=14)
ax.set_xlabel('County')
ax.set_ylabel('Total Violent Crimes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plot2_bar_county.png', dpi=150)
plt.show()
print("[Plot 2 saved] Bar chart – violent crime by county")

# ============================================================
#  PLOT 3 ── HISTOGRAM
#  Objective: Distribution of violent crime rates across all
#             county-year records
# ============================================================
fig, ax = plt.subplots()
ax.hist(df['VIOLENT CRIME RATE PER 100,000 PEOPLE'],
        bins=30, color='tomato', edgecolor='white', alpha=0.85)
ax.axvline(df['VIOLENT CRIME RATE PER 100,000 PEOPLE'].mean(),
           color='black', linestyle='--', linewidth=1.5, label='Mean')
ax.set_title('Distribution of Violent Crime Rate (per 100,000 People)', fontsize=13)
ax.set_xlabel('Violent Crime Rate')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('plot3_histogram.png', dpi=150)
plt.show()
print("[Plot 3 saved] Histogram – distribution of violent crime rate")

# ============================================================
#  PLOT 4 ── SCATTER PLOT
#  Objective: Does higher population lead to more crime?
#             (Population vs Violent Crime Total per county-year)
# ============================================================
fig, ax = plt.subplots()
sc = ax.scatter(df['POPULATION'], df['VIOLENT CRIME TOTAL'],
                alpha=0.5, c=df['YEAR'], cmap='plasma',
                edgecolors='none', s=40)
plt.colorbar(sc, ax=ax, label='Year')
ax.set_title('Population vs. Violent Crime Total\n(color = year)', fontsize=13)
ax.set_xlabel('Population')
ax.set_ylabel('Violent Crime Total')
plt.tight_layout()
plt.savefig('plot4_scatter_pop_crime.png', dpi=150)
plt.show()
print("[Plot 4 saved] Scatter – population vs violent crime")

# ============================================================
#  PLOT 5 ── SCATTER PLOT
#  Objective: Relationship between violent and property crime rates
# ============================================================
fig, ax = plt.subplots()
sc = ax.scatter(df['VIOLENT CRIME RATE PER 100,000 PEOPLE'],
                df['PROPERTY CRIME RATE PER 100,000 PEOPLE'],
                alpha=0.5, c='teal', edgecolors='none', s=40)
# Trend line
m, b = np.polyfit(df['VIOLENT CRIME RATE PER 100,000 PEOPLE'],
                  df['PROPERTY CRIME RATE PER 100,000 PEOPLE'], 1)
x_line = np.linspace(df['VIOLENT CRIME RATE PER 100,000 PEOPLE'].min(),
                     df['VIOLENT CRIME RATE PER 100,000 PEOPLE'].max(), 100)
ax.plot(x_line, m * x_line + b, color='red', linewidth=2, label='Trend line')
ax.set_title('Violent Crime Rate vs. Property Crime Rate', fontsize=13)
ax.set_xlabel('Violent Crime Rate per 100,000')
ax.set_ylabel('Property Crime Rate per 100,000')
ax.legend()
plt.tight_layout()
plt.savefig('plot5_scatter_viol_prop.png', dpi=150)
plt.show()
print("[Plot 5 saved] Scatter – violent vs property crime rate")

# ============================================================
#  PLOT 6 ── HEATMAP (Correlation Matrix)
#  Objective: Which crime types are most correlated?
# ============================================================
crime_cols = ['MURDER', 'RAPE', 'ROBBERY', 'AGG. ASSAULT',
              'B & E', 'LARCENY THEFT', 'M/V THEFT']
corr = df[crime_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, ax=ax, square=True)
ax.set_title('Correlation Between Crime Types', fontsize=13)
plt.tight_layout()
plt.savefig('plot6_heatmap_corr.png', dpi=150)
plt.show()
print("[Plot 6 saved] Heatmap – crime type correlations")

# ============================================================
#  PLOT 7 ── BOX PLOT
#  Objective: Compare spread of murder rate across top-5 counties
# ============================================================
top5 = (df.groupby('JURISDICTION')['MURDER']
          .sum()
          .sort_values(ascending=False)
          .head(5)
          .index.tolist())

df_top5 = df[df['JURISDICTION'].isin(top5)]

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_top5, x='JURISDICTION', y='MURDER PER 100,000 PEOPLE',
            palette='Set2', ax=ax)
ax.set_title('Murder Rate Distribution – Top 5 Counties by Total Murders', fontsize=13)
ax.set_xlabel('County')
ax.set_ylabel('Murder Rate per 100,000 People')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('plot7_boxplot_murder.png', dpi=150)
plt.show()
print("[Plot 7 saved] Box plot – murder rate top-5 counties")

# ============================================================
#  PLOT 8 ── STACKED BAR CHART
#  Objective: How is crime composed (violent vs property) per decade?
# ============================================================
df['DECADE'] = (df['YEAR'] // 10) * 10
decade_sum = (df.groupby('DECADE')[['VIOLENT CRIME TOTAL', 'PROPERTY CRIME TOTALS']]
                .sum()
                .reset_index())

fig, ax = plt.subplots()
x = np.arange(len(decade_sum))
width = 0.5
ax.bar(x, decade_sum['VIOLENT CRIME TOTAL'],   width, label='Violent',  color='crimson')
ax.bar(x, decade_sum['PROPERTY CRIME TOTALS'], width,
       bottom=decade_sum['VIOLENT CRIME TOTAL'], label='Property', color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels([f"{d}s" for d in decade_sum['DECADE']])
ax.set_title('Total Crime by Decade: Violent vs Property', fontsize=13)
ax.set_xlabel('Decade')
ax.set_ylabel('Total Crimes')
ax.legend()
plt.tight_layout()
plt.savefig('plot8_stacked_bar_decade.png', dpi=150)
plt.show()
print("[Plot 8 saved] Stacked bar – crime by decade")

# ============================================================
#  PLOT 9 ── LINE CHART (Murder Rate Trend)
#  Objective: Has the murder rate declined over time?
# ============================================================
fig, ax = plt.subplots()
ax.plot(yearly['YEAR'], yearly['MURDER RATE'],
        color='darkred', marker='o', markersize=4, linewidth=2)
ax.fill_between(yearly['YEAR'], yearly['MURDER RATE'], alpha=0.2, color='red')
ax.set_title('Statewide Murder Rate per 100,000 People (1975–2020)', fontsize=13)
ax.set_xlabel('Year')
ax.set_ylabel('Murder Rate per 100,000')
plt.tight_layout()
plt.savefig('plot9_murder_rate.png', dpi=150)
plt.show()
print("[Plot 9 saved] Line + fill – murder rate over time")

# ============================================================
#  PLOT 10 ── SEABORN PAIRPLOT
#  Objective: Multi-variable relationships across major crime types
#             (uses a sampled subset for readability)
# ============================================================
pair_cols = ['MURDER PER 100,000 PEOPLE', 'ROBBERY PER 100,000 PEOPLE',
             'VIOLENT CRIME RATE PER 100,000 PEOPLE',
             'PROPERTY CRIME RATE PER 100,000 PEOPLE']
df_pair = df[pair_cols].dropna().sample(200, random_state=42)

pair_grid = sns.pairplot(df_pair, diag_kind='kde', plot_kws={'alpha': 0.4})
pair_grid.figure.suptitle('Pairplot of Key Crime Rates', y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig('plot10_pairplot.png', dpi=150)
plt.show()
print("[Plot 10 saved] Pairplot – multi-variable crime rates")

# ============================================================
#  SUMMARY STATISTICS (NumPy + Pandas)
# ============================================================
print("\n" + "=" * 60)
print("  SUMMARY STATISTICS (NumPy)")
print("=" * 60)
for col in ['VIOLENT CRIME TOTAL', 'PROPERTY CRIME TOTALS', 'MURDER']:
    arr = df[col].values
    print(f"\n{col}")
    print(f"  Mean   : {np.mean(arr):>10.1f}")
    print(f"  Median : {np.median(arr):>10.1f}")
    print(f"  Std Dev: {np.std(arr):>10.1f}")
    print(f"  Min    : {np.min(arr):>10.0f}")
    print(f"  Max    : {np.max(arr):>10.0f}")

print("\n" + "=" * 60)
print("  ALL PLOTS SAVED AS PNG IN THE CURRENT FOLDER")
print("=" * 60)
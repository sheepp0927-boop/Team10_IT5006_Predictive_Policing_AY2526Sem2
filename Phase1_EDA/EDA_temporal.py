import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dateutil.relativedelta import relativedelta, MO, TH
from matplotlib.patches import Patch

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
FILE_PATH = 'Crime_Dataset_Final.csv'

print("--- 1. Loading & Preprocessing Data ---")
try:
    df = pd.read_csv(FILE_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
except FileNotFoundError:
    print(f"ERROR: Could not find '{FILE_PATH}'. Please ensure the file exists.")
    exit()

# Theme Settings
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
cats = ['Violent', 'Property', 'Vice', 'Public Order', 'Sexual', 'Other']
colors = sns.color_palette("deep", 6)
cat_colors = dict(zip(cats, colors))

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Define Season Backgrounds
season_spans = [
    (10.5, 11.5, '#e6f3ff', 'Winter'), # Dec
    (-0.5, 1.5, '#e6f3ff', 'Winter'),  # Jan-Feb
    (1.5, 4.5, '#e6ffe6', 'Spring'),   # Mar-May
    (4.5, 7.5, '#fffce6', 'Summer'),   # Jun-Aug
    (7.5, 10.5, '#fff0e6', 'Autumn')   # Sep-Nov
]
legend_elements = [
    Patch(facecolor='#e6f3ff', label='Winter'),
    Patch(facecolor='#e6ffe6', label='Spring'),
    Patch(facecolor='#fffce6', label='Summer'),
    Patch(facecolor='#fff0e6', label='Autumn')
]

# ==========================================
# PART 1: ANNUAL TRENDS
# ==========================================
print("\n--- Generating Part 1: Annual Trends ---")
annual_trends = df.groupby(['Year', 'Crime_Category']).size().reset_index(name='Count')

for cat in cats:
    plt.figure(figsize=(10, 6))
    data = annual_trends[annual_trends['Crime_Category'] == cat]
    
    sns.lineplot(data=data, x='Year', y='Count', marker='o', markersize=10, linewidth=4, color=cat_colors[cat])
    
    plt.title(f"Annual Trend: {cat} Crimes (2014-2024)", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("Total Incidents")
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ==========================================
# PART 2: SEASONALITY
# ==========================================
print("\n--- Generating Part 2: Seasonality ---")
seasonal_agg = df.groupby(['Month', 'Crime_Category']).size().reindex(month_order, level=0).reset_index(name='Count')

for cat in cats:
    fig, ax = plt.subplots(figsize=(12, 7))
    data = seasonal_agg[seasonal_agg['Crime_Category'] == cat]
    
    # Add Backgrounds
    for start, end, color, label in season_spans:
        ax.axvspan(start, end, color=color, alpha=0.5, lw=0, zorder=0)

    # Plot Line
    sns.pointplot(data=data, x='Month', y='Count', color=cat_colors[cat], markers='o', scale=1.3, ax=ax, zorder=5)
    
    ax.set_title(f"Seasonal Pattern: {cat} Crimes", fontsize=18, fontweight='bold', pad=50)
    ax.set_ylabel("Avg. Monthly Incidents")
    ax.set_xlabel("")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=4, frameon=False, title="")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# PART 3: HEATMAPS (Generic Warning)
# ==========================================
print("\n--- Generating Part 3: Temporal Heatmaps ---")

for cat in cats:
    plt.figure(figsize=(12, 8))
    cat_data = df[df['Crime_Category'] == cat]
    
    heatmap_data = cat_data.pivot_table(index='Day_of_Week', columns='Hour', values='ID', aggfunc='count').reindex(days_order)
    
    sns.heatmap(heatmap_data, cmap='magma_r', linewidths=.5, linecolor='white', cbar_kws={'label': 'Incident Count'})
    
    plt.title(f"Time Fingerprint: {cat} Crimes", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Hour of Day (00:00 - 23:00)')
    plt.ylabel('')
    plt.xticks(ticks=range(0, 25, 2), labels=[f"{h:02d}:00" for h in range(0, 25, 2)], rotation=0)

    plt.figtext(0.5, 0.02, 
                "⚠️ DATA INTERPRETATION WARNING: Crime timestamps may reflect 'Time of Discovery' rather than exact occurrence.\n"
                "Please interpret distinct reporting spikes with caution.", 
                ha="center", fontsize=11, color="#8b0000", style='italic', 
                bbox=dict(facecolor='#ffdddd', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.subplots_adjust(bottom=0.20)
    plt.show()

# ==========================================
# PART 4: WEEKEND EFFECT RATIO
# ==========================================
print("\n--- Generating Part 4: Weekend Risk Ratio ---")
df['Is_Weekend'] = df['Day_of_Week'].isin(['Friday', 'Saturday', 'Sunday'])
daily_counts = df.groupby(['Is_Weekend', 'Crime_Category']).size().reset_index(name='Total_Count')

daily_counts['Daily_Avg'] = daily_counts.apply(lambda x: x['Total_Count']/3 if x['Is_Weekend'] else x['Total_Count']/4, axis=1)

risk_ratio = daily_counts.pivot(index='Crime_Category', columns='Is_Weekend', values='Daily_Avg')
risk_ratio['Ratio'] = risk_ratio[True] / risk_ratio[False]
risk_ratio = risk_ratio.sort_values('Ratio', ascending=False).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=risk_ratio, x='Ratio', y='Crime_Category', palette='coolwarm')
plt.axvline(x=1.0, color='black', linestyle='--', label='Neutral (1.0)')
plt.title('The "Weekend Effect": Relative Risk Ratio', fontsize=16, fontweight='bold')
plt.xlabel('Multiplier (Weekend Avg / Weekday Avg)')
plt.ylabel('')
plt.tight_layout()
plt.show()

# ==========================================
# PART 5: SHIFT ANALYSIS (Direct Labeling)
# ==========================================
print("\n--- Generating Part 5: Day vs. Night Shift Analysis ---")

df['Shift'] = df['Hour'].apply(lambda x: 'Day' if 6 <= x < 18 else 'Night')
shift_trends = df.groupby(['Year', 'Crime_Category', 'Shift']).size().reset_index(name='Count')

shift_colors = {'Day': '#ff9900', 'Night': '#003366'} 

for cat in cats:
    plt.figure(figsize=(12, 6))
    cat_data = shift_trends[shift_trends['Crime_Category'] == cat]
    
    # Plot
    sns.lineplot(data=cat_data, x='Year', y='Count', hue='Shift', style='Shift', 
                 markers=True, dashes=False, linewidth=3, palette=shift_colors, legend=False)
    
    # Direct Labeling
    for shift_type in ['Day', 'Night']:
        if not cat_data[cat_data['Shift'] == shift_type].empty:
            last_point = cat_data[cat_data['Shift'] == shift_type].iloc[-1]
            plt.text(x=last_point['Year'] + 0.2, y=last_point['Count'], s=shift_type, 
                     color=shift_colors[shift_type], fontsize=14, fontweight='bold', va='center')

    plt.title(f"{cat} Crimes: Day vs. Night Trends (2014-2024)", fontsize=18, fontweight='bold', pad=15)
    plt.ylabel("Total Incidents")
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ==========================================
# PART 6: HOLIDAY ANALYSIS
# ==========================================
print("\n--- Generating Part 6: Holiday Analysis ---")

holiday_list = []
for year in df['Date'].dt.year.unique():
    holiday_list.extend([
        {'Date': pd.Timestamp(f'{year}-01-01'), 'Name': 'New Years Day'},
        {'Date': pd.Timestamp(f'{year}-03-17'), 'Name': 'St Patricks Day'},
        {'Date': pd.Timestamp(f'{year}-07-04'), 'Name': 'Independence Day'},
        {'Date': pd.Timestamp(f'{year}-10-31'), 'Name': 'Halloween'},
        {'Date': pd.Timestamp(f'{year}-12-25'), 'Name': 'Christmas'}
    ])
    holiday_list.append({'Date': pd.Timestamp(f'{year}-01-01') + relativedelta(weekday=MO(3)), 'Name': 'MLK Day'})
    holiday_list.append({'Date': pd.Timestamp(f'{year}-05-31') + relativedelta(weekday=MO(-1)), 'Name': 'Memorial Day'})
    holiday_list.append({'Date': pd.Timestamp(f'{year}-09-01') + relativedelta(weekday=MO(1)), 'Name': 'Labor Day'})
    holiday_list.append({'Date': pd.Timestamp(f'{year}-11-01') + relativedelta(weekday=TH(4)), 'Name': 'Thanksgiving'})

holidays_df = pd.DataFrame(holiday_list)
holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])

df['Just_Date'] = df['Date'].dt.normalize()
merged = df.merge(holidays_df, left_on='Just_Date', right_on='Date', how='left')
merged['Name'] = merged['Name'].fillna('Normal Day')

holiday_order = ['Normal Day', 'New Years Day', 'MLK Day', 'St Patricks Day', 'Memorial Day', 'Independence Day', 'Labor Day', 'Halloween', 'Thanksgiving', 'Christmas']

for cat in cats:
    daily_vol = merged[merged['Crime_Category'] == cat].groupby(['Just_Date', 'Name']).size().reset_index(name='Count')
    
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=daily_vol, x='Name', y='Count', order=holiday_order, palette='Set3', showfliers=False)
    sns.stripplot(data=daily_vol, x='Name', y='Count', order=holiday_order, color='black', alpha=0.3, size=2)
    
    median_val = daily_vol[daily_vol['Name']=='Normal Day']['Count'].median()
    plt.axhline(median_val, color='red', linestyle='--', label=f'Normal Median ({int(median_val)})')
    
    plt.title(f'{cat} Crime Volume: Holidays vs. Normal Days', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Daily Incidents')
    plt.xlabel('')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# PART 7: YEAR-OVER-YEAR GROWTH HEATMAP (Fonts Fixed)
# ==========================================
print("\n--- Generating Part 7: Year-over-Year Growth Rate ---")

yoy_data = df.groupby(['Year', 'Crime_Category']).size().unstack()
yoy_pct = yoy_data.pct_change() * 100

years_filter = list(range(2015, 2025))
yoy_data = yoy_data.loc[years_filter].transpose()
yoy_pct = yoy_pct.loc[years_filter].transpose()

# Annotations (Pct + Count)
annot_df = yoy_pct.applymap(lambda x: f"{x:+.0f}%") + "\n" + yoy_data.applymap(lambda x: f"({int(x):,})")

plt.figure(figsize=(18, 10)) # Increased width for better spacing
sns.heatmap(yoy_pct, cmap='RdBu_r', center=0, annot=annot_df, fmt='', 
            linewidths=.5, cbar_kws={'label': '% Growth vs Prior Year'},
            annot_kws={"size": 10},
            yticklabels=True) # Ensure labels are shown

# FIX: Reduce Y-Axis Font Size and Alignment
plt.yticks(fontsize=11, rotation=0) 
plt.xticks(fontsize=12)

plt.title('Velocity of Crime: Year-over-Year % Change (with Incident Counts)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Year')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

print("\n✅ MASTER ANALYSIS COMPLETE (Version 8).")
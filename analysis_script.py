import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'social_media_campaign_data.csv'
df = pd.read_csv(file_path)

# --- 1. Data Cleaning and Preparation ---
# Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())
print("\n")

# Fill missing 'saves' values with 0, as not all platforms have this metric
df['saves'] = df['saves'].fillna(0)

# Convert `post_date` to datetime objects for time-series analysis
df['post_date'] = pd.to_datetime(df['post_date'])

# --- 2. Feature Engineering (Calculating Key Metrics) ---
# Calculate Engagement Rate by Reach
df['engagement_rate_reach'] = ((df['likes'] + df['comments'] + df['shares'] + df['saves']) / df['reach']) * 100

# Calculate Engagement Rate by Followers
df['engagement_rate_followers'] = ((df['likes'] + df['comments'] + df['shares'] + df['saves']) / df['follower_count']) * 100

# Calculate Virality Score (Shares per Impression)
df['virality_score'] = (df['shares'] / df['impressions']) * 100

# --- 3. Analysis and Insights ---
print("Descriptive Statistics of Key Metrics:")
print(df[['engagement_rate_reach', 'virality_score']].describe())
print("\n")

# Find Best-Performing Posts
top_5_posts = df.sort_values(by='engagement_rate_reach', ascending=False).head(5)
print("Top 5 Best-Performing Posts by Engagement Rate:")
print(top_5_posts[['post_id', 'platform', 'post_topic', 'engagement_rate_reach']].to_string(index=False))
print("\n")

# Performance by Platform
platform_performance = df.groupby('platform')[['engagement_rate_reach', 'virality_score', 'clicks']].mean().reset_index()
print("Average Performance by Platform:")
print(platform_performance.to_string(index=False))
print("\n")

# Performance by Content Type
content_performance = df.groupby('post_topic')[['engagement_rate_reach', 'virality_score', 'clicks']].mean().reset_index()
print("Average Performance by Content Topic:")
print(content_performance.sort_values(by='engagement_rate_reach', ascending=False).to_string(index=False))
print("\n")

# --- 4. Data Visualization ---
plt.style.use('ggplot')

# Plot 1: Average Engagement Rate by Platform
plt.figure(figsize=(10, 6))
sns.barplot(x='platform', y='engagement_rate_reach', data=platform_performance, palette='viridis')
plt.title('Average Engagement Rate by Platform')
plt.ylabel('Engagement Rate (%)')
plt.xlabel('Social Media Platform')
plt.show()

# Plot 2: Engagement Rate by Content Topic
plt.figure(figsize=(12, 7))
sns.barplot(x='post_topic', y='engagement_rate_reach', data=content_performance.sort_values(by='engagement_rate_reach', ascending=False), palette='plasma')
plt.title('Average Engagement Rate by Content Topic')
plt.ylabel('Engagement Rate (%)')
plt.xlabel('Content Topic')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 3: Virality vs. Impressions (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='impressions', y='virality_score', data=df, hue='platform', size='clicks', sizes=(20, 200), palette='tab10')
plt.title('Virality Score vs. Impressions (Sized by Clicks)')
plt.xlabel('Impressions')
plt.ylabel('Virality Score (%)')
plt.legend(title='Platform', loc='upper right')
plt.show()
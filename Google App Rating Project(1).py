#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns 

Load the data file using pandas. 
# In[2]:


df = pd.read_csv("1569582940_googleplaystore.zip")


# In[3]:


df.shape


# df.columns

# In[7]:


df.head(2)

Check for null values in the data. Get the number of null values for each column.
# In[9]:


df.isnull().sum()

Drop records with nulls in any of the columns.

Variables seem to have incorrect type and inconsistent formatting. You need to fix them:

Size column has sizes in Kb as well as Mb. To analyze, you’ll need to convert these to numeric.

Extract the numeric value from the column

Multiply the value by 1,000, if size is mentioned in Mb

Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).

Drop records with nulls in any of the columns.

Variables seem to have incorrect type and inconsistent formatting. You need to fix them:

Size column has sizes in Kb as well as Mb. To analyze, you’ll need to convert these to numeric.

Extract the numeric value from the column

Multiply the value by 1,000, if size is mentioned in Mb

Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).

Installs field is currently stored as string and has values like 1,000,000+.

Treat 1,000,000+ as 1,000,000

remove ‘+’, ‘,’ from the field, convert it to integer

Price field is a string and has 𝑠𝑦𝑚𝑏𝑜𝑙.𝑅𝑒𝑚𝑜𝑣𝑒‘
’ sign, and convert it to numeric.
# In[10]:


df1 = df.dropna()


# In[11]:


df1.shape


# In[12]:


df1.isna().sum()


# In[13]:


df1["Size"]


# In[14]:


df1 = df1[-df1["Size"].str.contains("Var")] #to get everything except rows where size=string starts with Var


# In[15]:


df1["Size"]


# In[16]:


df1.loc[:,"SizeNum"] = df1.Size.str.rstrip("Mk+") #search python w3 schools string functions 


# In[17]:


df1["SizeNum"]


# In[18]:


df1["SizeNum"]


# In[19]:


df1.SizeNum = pd.to_numeric(df1["SizeNum"])


# In[20]:


df1["SizeNum"].dtype


# In[22]:


import numpy as np
df1["SizeNum"] = np.where(df1["Size"].str.contains("M"),df1["SizeNum"]*1000,df1.SizeNum)


# In[23]:


df1["SizeNum"]


# In[24]:


df1.Size = df1.SizeNum
df1.drop("SizeNum",axis=1,inplace=True)


# In[25]:


#Convert reviews into numeric


# In[26]:


df1.Reviews = pd.to_numeric(df1.Reviews)


# In[27]:


df1.Reviews.dtype


# Installs field is currently stored as string and has values like 1,000,000+.
# 
# Treat 1,000,000+ as 1,000,000
# 
# remove ‘+’, ‘,’ from the field, convert it to integer
# 
# Price field is a string and has 𝑠𝑦𝑚𝑏𝑜𝑙.𝑅𝑒𝑚𝑜𝑣𝑒‘
# ’ sign, and convert it to numeric.

# In[28]:


df1["Installs"]=df1["Installs"].str.replace("+","")


# In[29]:


df1["Installs"]=df1["Installs"].str.replace(",","")


# In[30]:


df1["Installs"] = pd.to_numeric(df1.Installs)


# In[31]:


df1["Installs"].dtype


# In[32]:


df1["Installs"]


# In[34]:


df1["Price"]=df1["Price"].str.replace("$","")


# In[35]:


df1["Price"] = pd.to_numeric(df1.Price)


# In[36]:


df1.Price.dtype

|5. Sanity checks:

Average rating should be between 1 and 5 as only these values are allowed on the play store. Drop the rows that have a value outside this range.

Reviews should not be more than installs as only those who installed can review the app. If there are any such records, drop them.

For free apps (type = “Free”), the price should not be >0. Drop any such rows.
# In[38]:


df1 = df1[(df1.Rating>=1) & (df1.Rating<=5)]


# In[40]:


df1["Rating"]


# In[41]:


df1.head(2)


# In[42]:


len(df1.index)


# In[43]:


df1.drop(df1.index[df1.Reviews>df1.Installs],axis=0,inplace=True)


# In[44]:


len(df1.index)


# In[45]:


import warnings
warnings.filterwarnings('ignore')


# In[47]:


df1[(df1["Type"]=="Free") & (df1["Price"]>0)]


# In[48]:


# There are no free apps with price > 0


# 5. Performing univariate analysis: 
# 
# Boxplot for Price
# 
# Are there any outliers? Think about the price of usual apps on Play Store.
# 
# Boxplot for Reviews
# 
# Are there any apps with very high number of reviews? Do the values seem right?
# 
# Histogram for Rating
# 
# How are the ratings distributed? Is it more toward higher ratings?
# 
# Histogram for Size
# 
# Note down your observations for the plots made above. Which of these seem to have outliers?

# In[49]:


sns.boxplot(x="Price",data=df1)


# In[50]:


#greater than 100 might be consider as outliers


# In[51]:


std = np.std(df1.Price)


# In[52]:


mean = np.mean(df1.Price)


# In[53]:


outlier_uplimit = mean + 3*std


# In[54]:


outlier_uplimit = mean + 3*std


# In[58]:


outlier_uplimit


# In[61]:


len(df1[(df1["Price"]>outlier_uplimit)])

Boxplot for Reviews --- same way as with price Installs: There seems to be some outliers in this field too. Apps having very high number of
installs should be drepped from the analysis
Find out the different percentiles- 10,25,50,70,90,95,99 
Decide a threshold as cuttoff for outlier and drop records having values more than that.
# In[62]:


sns.boxplot(x="Installs",data=df1)


# In[63]:


np.percentile(df1["Installs"],10)


# In[64]:


np.percentile(df1["Installs"],25)


# In[65]:


np.percentile(df1["Installs"],50)


# In[66]:


np.percentile(df1["Installs"],70)


# In[67]:


np.percentile(df1["Installs"],90)


# In[70]:


np.percentile(df1["Installs"],99)


# In[72]:


sns.distplot(df1["Installs"])


# In[73]:


len(df1[df1.Installs>=100000000.0])


# In[74]:


df1.drop(df1.index[df1.Installs>=100000000.0],inplace=True)


# Bivariate analysis: Let’s look at how the available predictors relate to the variable of interest, i.e., our target variable rating. Make scatter plots (for numeric features) and box plots (for character features) to assess the relations between rating and the other features.
# 
# Make scatter plot/joinplot for Rating vs. Price
# 
# What pattern do you observe? Does rating increase with price?
# 
# Make scatter plot/joinplot for Rating vs. Size
# 
# Are heavier apps rated better?
# 
# Make scatter plot/joinplot for Rating vs. Reviews
# 
# Does more review mean a better rating always?
# 
# Make boxplot for Rating vs. Content Rating
# 
# Is there any difference in the ratings? Are some types liked better?
# 
# Make boxplot for Ratings vs. Category
# 
# Which genre has the best ratings?
# 
# For each of the plots above, note down your observation.
# 

# In[75]:


sns.jointplot(x="Price",y="Rating",data=df1)

Does rating increases with price?
----It seems like price has limited impact on rating.Make scatter plot/jointplot for Rating vs. Size
Are heavier apps rated better?
# In[78]:


sns.jointplot(x="Size",y="Rating",data=df1)


# In[79]:


sns.jointplot(x="Reviews",y="Rating",data=df1)


# Do more review mean a better rating always?

# In[80]:


df1.corr()


# Make boxplot for Rating vs. Content Rating
# Is there any difference in the ratings? Are some types liked better?

# In[81]:


df1["Content Rating"].unique()


# In[82]:


sns.boxplot(x="Rating",y="Content Rating",data=df1)


# In[83]:


sns.boxplot(x="Content Rating", y="Rating",data=df1)


# In[91]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.boxplot(x="Rating",y="Category",data=df1)


# 8. Data preprocessing
# 
# For the steps below, create a copy of the dataframe to make all the edits. Name it inp1.
# 
# Reviews and Install have some values that are still relatively very high. Before building a linear regression model,
# you need to reduce the skew. Apply log transformation (np.log1p) to Reviews and Installs.
# 
# Drop columns App, Last Updated, Current Ver, and Android Ver. These variables are not useful for our task.
# 
# Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not understand categorical 
# data, and all data should be numeric. Dummy encoding is one way to convert character fields to numeric. Name of dataframe 
# should be inp2.

# In[96]:


inp1 = df1.copy()


# In[97]:


sns.distplot(inp1["Reviews"])


# In[98]:


#plt.hist(inp1[["Reviews"]])
#plt.show()


# In[100]:


inp1.Reviews=inp1.Reviews.apply(np.log1p)


# In[101]:


inp1.Installs=inp1.Installs.apply(np.log1p)


# In[102]:


inp1.drop(['App','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)


# In[103]:


inp1.shape


# In[104]:


inp1.columns


# In[105]:


inp1["Type"].unique()


# In[106]:


inp2 = pd.get_dummies(inp1)


# In[107]:


inp2.shape


# 9. Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test.

# In[108]:


inp2.head(2)


# In[109]:


y = inp2.iloc[:,0] #target


# In[112]:


X = inp2.iloc[:,1:] #features


# In[115]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[117]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[118]:


lr.fit(x_train,y_train)


# In[119]:


y_pred = lr.predict(x_test)


# In[120]:


from sklearn.metrics import r2_score


# In[121]:


r2_score(y_test,y_pred)


# In[ ]:





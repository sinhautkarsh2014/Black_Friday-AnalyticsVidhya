import pandas as pd

train=pd.read_csv("train_avb.csv")
test=pd.read_csv("test_avb.csv")
print train.shape
test=test.drop("Comb", axis=1)
test=test.drop("Unnamed: 0", axis=1)
print test.shape

combin=pd.concat([train,test])
print combin.dtypes

combin["Age"]=combin["Age"].astype("category")
combin["Age"]=combin["Age"].cat.rename_categories(range(1,8))

combin["City_Category"]=combin["City_Category"].astype("category")
combin["City_Category"]=combin["City_Category"].cat.rename_categories(range(1,4))

combin["Gender"]=combin["Gender"].astype("category")
combin["Gender"]=combin["Gender"].cat.rename_categories(range(1,3))

combin["Marital_Status"]=combin["Marital_Status"].astype("category")
combin["Marital_Status"]=combin["Marital_Status"].cat.rename_categories(range(1,3))

combin["Occupation"]=combin["Occupation"].astype("category")
combin["Occupation"]=combin["Occupation"].cat.rename_categories(range(1,22))

combin["Product_Category_1"]=combin["Product_Category_1"].astype("category")
combin["Product_Category_1"]=combin["Product_Category_1"].cat.rename_categories(range(1,21))

combin["Product_Category_2"].fillna(value="0", inplace=True)
combin["Product_Category_2"]=combin["Product_Category_2"].astype("category")
combin["Product_Category_2"]=combin["Product_Category_2"].cat.rename_categories(range(1,19))

combin["Product_Category_3"].fillna(value="0", inplace=True)
combin["Product_Category_3"]=combin["Product_Category_3"].astype("category")
combin["Product_Category_3"]=combin["Product_Category_3"].cat.rename_categories(range(1,17))


combin["Stay_In_Current_City_Years"]=combin["Stay_In_Current_City_Years"].astype("category")
combin["Stay_In_Current_City_Years"]=combin["Stay_In_Current_City_Years"].cat.rename_categories(range(1,6))



df_train=combin[combin.Purchase.notnull()]
df_test=combin[combin.Purchase.isnull()]

df_test=df_test.drop("Purchase", axis=1)
key1=df_test["User_ID"]
key2=df_test["Product_ID"]


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
X_train=pd.DataFrame(enc.fit_transform(df_train[["Age", "City_Category", "Gender","Marital_Status", "Occupation", "Product_Category_1", "Product_Category_2", "Product_Category_3","Stay_In_Current_City_Years"]]).toarray())
X_test=pd.DataFrame(enc.transform(df_test[["Age", "City_Category", "Gender","Marital_Status", "Occupation", "Product_Category_1", "Product_Category_2", "Product_Category_3","Stay_In_Current_City_Years"]]).toarray())

print X_train.shape
print X_test.shape


y_train=df_train["Purchase"]
df_train=df_train.drop("Purchase", axis=1)

#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_regression
#sel = SelectKBest(f_regression, k=10)
#X_tr=pd.DataFrame(sel.fit_transform(X_train,y_train))
#X_tst=pd.DataFrame(sel.transform(X_test))

#print X_tr.shape
#print X_tst.shape

from sklearn.linear_model import ElasticNet
model=ElasticNet(alpha=0.001)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#print y_pred.shape
#print key1.shape
#print key2.shape


out=pd.DataFrame()
out["User_ID"]=key1
out["Product_ID"]=key2
out["Purchase"]=y_pred
out.to_csv('outavb.csv', index=False)

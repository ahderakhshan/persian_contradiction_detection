import pandas as pd
from contradiction_extractor import FeatureExtractor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm


to_num = {"e":0, "n":0, "c":1}
farstail_train = pd.read_csv("./dataset/farstail_train.csv")
farstail_test = pd.read_csv("./dataset/farstail_test.csv")
farstail_train["label"] = farstail_train["label"].apply(lambda x: to_num[x])
farstail_test["label"] = farstail_test["label"].apply(lambda x: to_num[x])

train_premise, train_hypothesis, train_label = farstail_train["premise"][:10],\
                                                farstail_train["hypothesis"][:10], farstail_train["label"][:10]
test_premise, test_hypothesis, test_label = farstail_test["premise"][:10], farstail_test["hypothesis"][:10], farstail_test["label"][:10]

train_features = []
for premise, hypothesis in tqdm(zip(train_premise, train_hypothesis)):
    feature_extractor = FeatureExtractor(premise, hypothesis)
    train_features.append(feature_extractor.feature_construction())

with open("./train_features.pkl", "wb") as f:
    pickle.dump(train_features, f)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)

with open("./scalar.pkl", "wb") as f:
    pickle.dump(scaler, f)

test_features = []
for premise, hypothesis in tqdm(zip(test_premise, test_hypothesis)):
    feature_extractor = FeatureExtractor(premise, hypothesis)
    test_features.append(feature_extractor.feature_construction())

with open("./test_features.pkl", "wb") as f:
    pickle.dump(test_features, f)

X_test_scaled = scaler.transform(test_features)

svm = SVC()
svm.fit(train_features, train_label)
y_train_pred = svm.predict(train_features)
print(accuracy_score(y_true=train_label, y_pred=y_train_pred))



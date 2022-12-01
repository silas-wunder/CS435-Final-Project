import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
import gc, warnings, json


spark = SparkSession.builder.appName("Steam Game Merge").getOrCreate()
with open("./input/steamspy/detailed/steam_spy_detailed.json", "r") as f:
    raw_file = json.load(f)

spy_data = pd.DataFrame.from_records(raw_file).T
del raw_file
gc.collect()
spy_data.drop(
    [
        "appid",
        "developer",
        "publisher",
        "score_rank",
        "userscore",
        "owners",
        "average_2weeks",
        "median_2weeks",
        "price",
        "discount",
        "ccu",
        "tags",
    ],
    axis=1,
    inplace=True,
)
spy_data.rename(
    {
        "name": "Name",
        "positive": "Positive Reviews",
        "negative": "Negative Reviews",
        "average_forever": "Average Playtime",
        "median_forever": "Median Playtime",
        "initialprice": "Price",
        "languages": "Languages",
        "genre": "Genres",
    },
    axis=1,
    inplace=True,
)

with open("./input/steam_charts/steam_charts.json", "r") as f:
    raw_player_count_file = json.load(f)

mean_dict = {}
for key in raw_player_count_file.keys():
    mean_dict[key] = (
        pd.DataFrame(raw_player_count_file[key], index=[key]).mean(axis=1).iloc[0]
    )
player_count_data = pd.DataFrame(
    mean_dict, index=["Mean Concurrent Players All Time"]
).T.sort_index()

all_data = spy_data.merge(
    player_count_data, how="inner", left_index=True, right_index=True
)

all_data["Languages"] = all_data["Languages"].str.split(", ")
all_data["Genres"] = all_data["Genres"].str.split(", ")
all_data = all_data.loc[all_data["Languages"].notna()]
all_data = all_data.loc[all_data["Mean Concurrent Players All Time"].notna()]

mlb = MultiLabelBinarizer()
oe = OrdinalEncoder(dtype="uint32")
column_transform = make_column_transformer(
    (
        oe,
        [
            "Positive Reviews",
            "Negative Reviews",
            "Average Playtime",
            "Median Playtime",
            "Price",
        ],
    )
)
X = column_transform.fit_transform(all_data)
X_languages = mlb.fit_transform(all_data["Languages"].tolist())
X_genres = mlb.fit_transform(all_data["Genres"].tolist())
Xtmp = [[] for _ in range(len(X))]
Xt = [[] for _ in range(len(X))]
for i in range(len(X)):
    Xtmp[i] = np.append(X[i], X_languages[i])
    Xt[i] = np.append(Xtmp[i], X_genres[i])
Xt = np.stack(Xt, axis=0)

y = np.array(all_data["Mean Concurrent Players All Time"])

X_train, X_test, y_train, y_test = train_test_split(
    Xt, y, test_size=0.50, shuffle=True, random_state=42
)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Lasso: {lasso.score(X_test, y_test)}")

enet = ElasticNet()
enet.fit(X_train, y_train)
print(f"Elastic Net: {enet.score(X_test, y_test)}")

ridge = Ridge()
ridge.fit(X_train, y_train)
print(f"Ridge: {ridge.score(X_test, y_test)}")

lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression: {lr.score(X_test, y_test)}")

svr = SVR()
svr.fit(X_train, y_train)
print(f"Support Vector Regression: {svr.score(X_test, y_test)}")

nsvr = NuSVR()
nsvr.fit(X_train, y_train)
print(f"NuSVR: {nsvr.score(X_test, y_test)}")

lsvr = LinearSVR(max_iter=1e9)
lsvr.fit(X_train, y_train)
print(f"Linear SVR: {lsvr.score(X_test, y_test)}")


class EvaluateCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        if "test loss" not in logs:
            logs["test loss"] = []
            logs["test acc"] = []
        logs["test loss"] += [loss]
        logs["test acc"] += [acc]
        print(f"Testing loss: {loss:6.4f}, Testing accuracy: {acc:6.4f}")


warnings.filterwarnings("ignore")
tf.get_logger().setLevel("WARNING")
linear_model = keras.Sequential()
linear_model.add(keras.layers.Dense(83, activation="softmax"))
linear_model.add(keras.layers.Dense(1, kernel_initializer="normal"))

linear_model.compile(
    loss="mean_squared_logarithmic_error", optimizer="adam", metrics=["accuracy"]
)
linear_history = linear_model.fit(
    X_train,
    y_train,
    batch_size=10,
    epochs=100,
    verbose=1,
    callbacks=[EvaluateCallback((X_test, y_test))],
)
print(linear_model.summary())

fig, ax = plt.subplots()
ax.plot(linear_history.history["accuracy"], label="Train")
ax.plot(linear_history.history["test acc"], label="Test")
ax.set_title("One Layer - SoftMax Activation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(loc="upper left")
ax.show()

one_layer = keras.Sequential(name="one_layer")
one_layer.add(keras.layers.Dense(83, activation="softmax"))
one_layer.add(keras.layers.Dense(83, activation="relu"))
one_layer.add(keras.layers.Dense(1, kernel_initializer="normal"))

one_layer.compile(
    loss="mean_squared_logarithmic_error", optimizer="adam", metrics=["accuracy"]
)
one_layer_history = one_layer.fit(
    X_train,
    y_train,
    batch_size=100,
    epochs=100,
    verbose=1,
    callbacks=[EvaluateCallback((X_test, y_test))],
)
print(one_layer.summary())

fig, ax = plt.subplots()
ax.plot(one_layer_history.history["accuracy"], label="Train")
ax.plot(one_layer_history.history["test acc"], label="Test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Two Layer - SoftMax Activation")
ax.legend(loc="upper left")
ax.show()

two_layer = keras.Sequential(name="two_layer")
two_layer.add(keras.layers.Dense(83, activation="tanh"))
two_layer.add(keras.layers.Dense(83, activation="tanh"))
two_layer.add(keras.layers.Dense(1, kernel_initializer="normal"))

two_layer.compile(
    loss="mean_squared_logarithmic_error", optimizer="adam", metrics=["accuracy"]
)
two_layer_history = two_layer.fit(
    X_train,
    y_train,
    batch_size=100,
    epochs=100,
    verbose=1,
    callbacks=[EvaluateCallback((X_test, y_test))],
)
print(two_layer.summary())

fig, ax = plt.subplots()
ax.plot(two_layer_history.history["accuracy"], label="Train")
ax.plot(two_layer_history.history["test acc"], label="Test")
ax.set_title("Two Layers - TanH Activation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(loc="upper left")
ax.show()

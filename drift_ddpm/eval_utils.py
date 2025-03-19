from copy import copy
from enum import Enum

import numpy as np
import pandas as pd
import pymysql as ms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from pandas.api.types import is_numeric_dtype, is_object_dtype
from sklearn import metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset


class Aggregation(Enum):
    COUNT = "count"
    AVG = "avg"
    SUM = "sum"

    def __str__(self):
        return self.value


class Operator(Enum):
    GE = ">"
    EQ = "="
    LE = "<"

    def __str__(self):
        return self.value


class AQPQuery:
    """Refer to the ReStore"""

    def __init__(
        self,
        aggregation_attribute=None,
        where_conditions=None,
        grouping_attributes=None,
    ):
        self.aggregation_attribute = aggregation_attribute

        self.where_conditions = where_conditions
        if where_conditions is None:
            self.where_conditions = []

        self.grouping_attributes = grouping_attributes
        if grouping_attributes is None:
            self.grouping_attributes = []

    def sql_string(self, aggregation, cat_value_dict=None):
        if cat_value_dict is None:
            cat_value_dict = dict()

        sql_string_components = ["SELECT"]
        if aggregation == Aggregation.COUNT:
            sql_string_components.append("COUNT(*)")
        else:
            sql_string_components.append(
                f"{str(aggregation).upper()}({self.aggregation_attribute})"
            )

        if len(self.where_conditions) > 0:
            sql_string_components.append("WHERE")
            where_conds = []
            for attr, op, lit in self.where_conditions:
                if attr in cat_value_dict.keys():
                    lit_replace = f"'{cat_value_dict[attr][int(lit)]}'"
                else:
                    lit_replace = lit

                where_conds.append(f"{attr}{str(op)}{lit_replace}")

            sql_string_components.append(" AND ".join(where_conds))

        if len(self.grouping_attributes) > 0:
            sql_string_components.append("GROUP BY")
            sql_string_components.append(", ".join(self.grouping_attributes))

        sql_string = " ".join(sql_string_components)
        sql_string += ";"
        return sql_string

    def compute(self, df, weights=None, upscale=1.0):
        df = copy(df)

        df["weights"] = 1
        if weights is not None:
            df["weights"] = weights
        for attribute, op, literal in self.where_conditions:
            if op == "=":
                df = df[df[attribute] == literal]
            elif op == ">":
                df = df[df[attribute] > literal]
            elif op == "<":
                df = df[df[attribute] < literal]

        counts = None
        averages = None

        if len(df) == 0:
            return [], [], []

        if len(self.grouping_attributes) > 0:
            # Count per Group
            df_count = (
                df[self.grouping_attributes + ["weights"]]
                .groupby(self.grouping_attributes)
                .sum()
                .reset_index()
            )
            df_count[df_count.columns[-1]] *= upscale
            # df_count.iloc[:, -1] *= upscale
            counts = [tuple(x) for x in df_count.to_numpy()]

            if self.aggregation_attribute is not None:
                df_agg = df[~df[self.aggregation_attribute].isna()]
                for agg in self.grouping_attributes:
                    df_agg = df[~df[agg].isna()]

                if len(df_agg) > 0:
                    try:
                        df_agg["weighted_aggregate"] = (
                            df_agg["weights"] * df_agg[self.aggregation_attribute]
                        )
                        df_agg = (
                            df_agg[
                                self.grouping_attributes
                                + ["weighted_aggregate", "weights"]
                            ]
                            .groupby(self.grouping_attributes)
                            .sum()
                        )
                        df_agg["weighted_aggregate"] /= df_agg["weights"]
                        df_agg = df_agg.reset_index()
                        df_agg.drop(columns=["weights"], inplace=True)
                        averages = [tuple(x) for x in df_agg.to_numpy()]
                    except:
                        averages = None

                sums = []
                if averages is not None:
                    for c, a in zip(counts, averages):
                        assert c[:-1] == a[:-1]
                        sums.append((c[:-1], c[-1] * a[-1]))
                else:
                    sums = None

        else:
            counts = df["weights"].sum() * upscale

            if self.aggregation_attribute is not None:
                df_agg = df[~df[self.aggregation_attribute].isna()]
                for agg in self.grouping_attributes:
                    df_agg = df[~df[agg].isna()]

                if len(df_agg) > 0:
                    try:
                        df_agg["weighted_aggregate"] = (
                            df_agg["weights"] * df_agg[self.aggregation_attribute]
                        )
                        averages = (
                            df_agg["weighted_aggregate"].sum() / df_agg["weights"].sum()
                        )
                    except:
                        averages = None

                sums = counts * averages if averages is not None else None

                counts = [("", counts)]
                sums = [("", sums)]
                averages = [("", averages)]

        return counts, sums, averages


def compute_relative_error(r_result, f_result):
    if len(f_result) == 0:
        return 1.0
    fake_group_num = 0
    error = 0
    for group in f_result.keys():
        if group not in r_result.keys():
            fake_group_num += 1
            continue
        error += abs(r_result[group] - f_result[group]) / (abs(r_result[group]) + 1)

    error += len(r_result) - (len(f_result) - fake_group_num) + fake_group_num
    error /= len(r_result) + fake_group_num
    return error


def AQP_error(queries, gen_data, eval_num, real_data):
    norm_gen_data = gen_data.copy()
    for col in norm_gen_data.columns:
        if is_numeric_dtype(norm_gen_data[col]):
            norm_gen_data.loc[:, col] = norm_gen_data[col] / real_data[col].max()
    all_errors = []
    for query in tqdm.tqdm(queries[:eval_num]):
        aqp = AQPQuery(query["agg_col"], query["where_conditions"], query["group_cols"])
        r_result = dict(query["result"])
        counts, sums, averages = aqp.compute(norm_gen_data)
        if query["agg_type"] == "sum":
            f_result = sums
        elif query["agg_type"] == "avg":
            f_result = averages
        elif query["agg_type"] == "count":
            f_result = counts
        f_result = dict(f_result)

        re = compute_relative_error(r_result, f_result)
        all_errors.append(re)
    all_errors = np.array(all_errors)
    return all_errors


def QError(actual_cards, est_cards):
    # [batch_size]
    bacth_ones = torch.ones(actual_cards.shape, dtype=torch.float32)
    fixed_actual_cards = torch.where(actual_cards == 0.0, bacth_ones, actual_cards)
    fixed_est_cards = torch.where(est_cards == 0.0, bacth_ones, est_cards)

    q_error = torch.where(
        actual_cards > est_cards,
        fixed_actual_cards / fixed_est_cards,
        fixed_est_cards / fixed_actual_cards,
    )

    return q_error


def QE_eval(actual_cards, workloads, gen_data, tbname):
    engine = create_engine("mysql+pymysql://lty:123456@localhost:2334/dbsyn")
    connection = ms.connect(
        host=None,
        user="lty",
        passwd="123456",
        db="dbsyn",
        port=2334,
        cursorclass=ms.cursors.DictCursor,
    )
    cursor = connection.cursor()
    gen_data.to_sql(tbname, engine, index=False, if_exists="replace")
    est_cards = []
    for sql in tqdm.tqdm(workloads):
        cursor.execute(sql)
        card = cursor.fetchone()["count(*)"]
        est_cards.append(card / len(gen_data))
    est_cards = torch.as_tensor(est_cards).float()
    test_errs = QError(actual_cards, est_cards)
    connection.close()
    return test_errs


# ML evaluation (MLP, classification/regression)
def clf_evaluation(y_true, y_pred, average="binary"):
    f1_score = metrics.f1_score(y_true, y_pred, average=average)
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average=average)
    return [f1_score]


def reg_evaluation(y_true, y_pred):
    r2_score = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    return [mse]


def feature_encoder(train_data, test_data, valid_data, label_col, task):
    if task == "Clf":
        train_data[label_col] = train_data[label_col].astype(str)
        test_data[label_col] = test_data[label_col].astype(str)
        if valid_data is not None:
            valid_data[label_col] = valid_data[label_col].astype(str)
    onehot_cols = []
    for ind, col in enumerate(train_data.columns):
        if is_object_dtype(train_data[col]):
            onehot_cols.append(col)
        if col == label_col:
            label_cols = [col]
    dt = DataTransformer(onehot_cols=onehot_cols)
    if valid_data is None:
        trans_datasets = dt.fit_transform([train_data, test_data], label_cols)
    else:
        trans_datasets = dt.fit_transform(
            [train_data, test_data, valid_data], label_cols
        )
    train_x, train_y = trans_datasets[0]
    test_x, test_y = trans_datasets[1]
    if valid_data is not None:
        valid_x, valid_y = trans_datasets[2]
    else:
        valid_x = None
        valid_y = None

    if task == "Clf":
        train_y = np.argmax(train_y, axis=1)
        test_y = np.argmax(test_y, axis=1)
        if valid_y is not None:
            valid_y = np.argmax(valid_y, axis=1)

    if valid_x is not None:
        raw_x = np.concatenate((train_x, test_x, valid_x), axis=0)
    else:
        raw_x = np.concatenate((train_x, test_x), axis=0)

    col_index = dt.get_col_index(train_data.columns)
    for ind, col in enumerate(train_data.columns):
        sta, end = col_index[ind]
        if col in onehot_cols:
            continue
        feature = raw_x[:, sta:end]
        if feature.shape[1] == 0:
            continue
        raw_x[:, sta:end] = preprocessing.scale(feature)

    train_x = raw_x[: len(train_x), :]
    test_x = raw_x[len(train_x) : (len(train_x) + len(test_x)), :]
    if valid_x is not None:
        valid_x = raw_x[(len(train_x) + len(test_x)) :, :]

    if task == "Reg":
        raw_y = np.concatenate((train_y, test_y, valid_y))
        raw_y = preprocessing.scale(raw_y)
        train_y = raw_y[: len(train_y), :]
        test_y = raw_y[len(train_y) : (len(train_y) + len(test_y)), :]
        valid_y = raw_y[(len(train_y) + len(test_y)) :, :]

    return train_x, train_y, test_x, test_y, valid_x, valid_y


class MLPClf(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class=2):
        super(MLPClf, self).__init__()
        net = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        ]
        for i in range(1, len(hidden_dim)):
            net = net + [
                nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
            ]
        if n_class == 2:
            net = net + [nn.Linear(hidden_dim[-1], 1)]
        else:
            net = net + [nn.Linear(hidden_dim[-1], n_class)]
        self.MLP = nn.Sequential(*net)
        self.n_class = n_class

    def forward(self, x):
        x = self.MLP(x)
        if self.n_class == 2:
            y = torch.sigmoid(x)
        else:
            y = torch.softmax(x, dim=1)
        return y


class MLPReg(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPReg, self).__init__()
        net = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        ]
        for i in range(1, len(hidden_dim)):
            net = net + [
                nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
            ]
        net = net + [nn.Linear(hidden_dim[-1], 1)]
        self.MLP = nn.Sequential(*net)

    def forward(self, x):
        x = self.MLP(x)
        y = F.leaky_relu(x)
        return y


def eval_sklearn(
    train_data, test_data, valid_data, label_col, task="Clf", average="binary"
):
    evaluators = {
        "DT10": DecisionTreeClassifier(max_depth=10, random_state=0),
        "DT30": DecisionTreeClassifier(max_depth=30, random_state=0),
        "RF10": RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0),
        "RF20": RandomForestClassifier(n_estimators=10, max_depth=20, random_state=0),
        "Adaboost": AdaBoostClassifier(random_state=0),
        "LR": LogisticRegression(random_state=0),
    }

    train_label = train_data[label_col]
    train_data = train_data.drop([label_col], axis=1)
    train_data.insert(len(train_data.columns), label_col, train_label)

    test_label = test_data[label_col]
    test_data = test_data.drop([label_col], axis=1)
    test_data.insert(len(test_data.columns), label_col, test_label)

    if valid_data is not None:
        valid_label = valid_data[label_col]
        valid_data = valid_data.drop([label_col], axis=1)
        valid_data.insert(len(valid_data.columns), label_col, valid_label)

    train_data = train_data.sample(frac=1.0, random_state=np.random.RandomState(0))
    test_data = test_data.sample(frac=1.0, random_state=np.random.RandomState(0))

    n_class = len(train_data[label_col].unique())
    multiclass = True if n_class > 2 else False
    # print(multiclass)
    train_x, train_y, test_x, test_y, _, _ = feature_encoder(
        train_data, test_data, valid_data, label_col, task
    )

    eval_results = []
    for evaluator in evaluators.keys():
        # print(evaluator+" ...")
        try:
            model = evaluators[evaluator]
            model.fit(train_x, train_y)

            pred_test_y = model.predict(test_x)
            # pred_valid_y = model.predict(valid_x)

            if task == "Clf":
                eval_test = clf_evaluation(test_y, pred_test_y, average)
                # eval_valid = clf_evaluation(valid_y, pred_valid_y, multiclass)
            else:
                eval_test = reg_evaluation(test_y, pred_test_y, average)
                # eval_valid = reg_evaluation(valid_y, pred_valid_y, multiclass)

            eval_results.append(eval_test[0])
        except:
            eval_results.append(0)
    return eval_results


def train_evaluation(
    model,
    train_dataloader,
    test_x,
    test_y,
    valid_x,
    valid_y,
    epochs,
    lr,
    task,
    multiclass=False,
    average="binary",
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    if task == "Clf":
        if multiclass:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
    elif task == "Reg":
        criterion = nn.MSELoss()
    mlp_vals = []
    num = epochs / 10

    for epoch in tqdm.tqdm(range(epochs)):
        for train_x, train_y in train_dataloader:
            if task == "Clf" and multiclass:
                train_y = train_y.long()
            y_ = model(train_x)
            if task == "Clf":
                y_ = y_.squeeze()
            loss = criterion(y_, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % num == 0:
            model.eval()
            # print("epochs {}, Loss:{}".format(epoch, loss.data))
            mlp_prob_valid_y = model(valid_x).cpu().detach().numpy()
            mlp_prob_test_y = model(test_x).cpu().detach().numpy()

            if task == "Clf":
                if multiclass:
                    mlp_pred_valid_y = np.argmax(mlp_prob_valid_y, axis=1)
                else:
                    mlp_pred_valid_y = (mlp_prob_valid_y > 0.5) + 0
                mlp_val_valid = clf_evaluation(valid_y, mlp_pred_valid_y, average)

                if multiclass:
                    mlp_pred_test_y = np.argmax(mlp_prob_test_y, axis=1)
                else:
                    mlp_pred_test_y = (mlp_prob_test_y > 0.5) + 0
                mlp_val_test = clf_evaluation(test_y, mlp_pred_test_y, average)

            elif task == "Reg":
                mlp_pred_valid_y = mlp_prob_valid_y
                mlp_pred_test_y = mlp_prob_test_y
                mlp_val_valid = reg_evaluation(valid_y, mlp_pred_valid_y)
                mlp_val_test = reg_evaluation(test_y, mlp_pred_test_y)

            mlp_vals.append(
                ["epoch={}".format(epoch + 1)] + mlp_val_valid + mlp_val_test
            )
            model.train()

    model.eval()
    return model, mlp_vals


def train_model(
    train_data,
    test_data,
    label_col,
    valid_data=None,
    batch_size=128,
    epochs=10,
    lr=0.0005,
    hidden_dims=[256, 128],
    task="Clf",
    average="binary",
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # print(train_data)
    train_label = train_data[label_col]
    train_data = train_data.drop([label_col], axis=1)
    train_data.insert(len(train_data.columns), label_col, train_label)

    test_label = test_data[label_col]
    test_data = test_data.drop([label_col], axis=1)
    test_data.insert(len(test_data.columns), label_col, test_label)

    train_data = train_data.sample(frac=1.0)

    if valid_data is None:
        length = len(train_data) // 4
        valid_data = train_data.iloc[:length, :]
        train_data = train_data.iloc[length:, :]
    else:
        valid_label = valid_data[label_col]
        valid_data = valid_data.drop([label_col], axis=1)
        valid_data.insert(len(valid_data.columns), label_col, valid_label)

    if task == "Clf":
        n_class = len(
            pd.concat([train_data, test_data, valid_data])[label_col].unique()
        )
        # print(pd.concat([train_data, test_data, valid_data])[label_col].unique())
        # print(n_class)
        multiclass = True if n_class > 2 else False

    train_x, train_y, test_x, test_y, valid_x, valid_y = feature_encoder(
        train_data, test_data, valid_data, label_col, task
    )
    # print(train_x.shape, valid_x.shape, test_x.shape)
    train_x = torch.from_numpy(train_x).float().cuda()
    train_y = torch.from_numpy(train_y).float().cuda()
    valid_x = torch.from_numpy(valid_x).float().cuda()
    test_x = torch.from_numpy(test_x).float().cuda()
    # print("train")
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    if task == "Clf":
        model = MLPClf(train_x.shape[1], hidden_dims, n_class)
    elif task == "Reg":
        model = MLPReg(train_x.shape[1], hidden_dims)
    model.cuda()
    # print("begin")
    if task == "Clf":
        model, mlp_val = train_evaluation(
            model,
            train_dataloader,
            test_x,
            test_y,
            valid_x,
            valid_y,
            epochs,
            lr,
            task,
            multiclass,
            average,
        )
    else:
        model, mlp_val = train_evaluation(
            model, train_dataloader, test_x, test_y, valid_x, valid_y, epochs, lr, task
        )
    return mlp_val, model


def F1_eval(
    train_data, test_data, valid_data, label_col, epochs, task="Clf", average="binary"
):
    print("MLP-based Classifier ...")
    try:
        mlp_val, model = train_model(
            train_data,
            test_data,
            valid_data=valid_data,
            label_col=label_col,
            epochs=epochs,
            hidden_dims=[512, 256, 128],
            task=task,
            average=average,
        )
        mlp_val = pd.DataFrame(mlp_val).sort_values([1], ascending=False)
        mlp_f1 = mlp_val[2][mlp_val.index[0]]
    except:
        mlp_f1 = 0.0
    print("DT RF Adaboost LR Classifier ...")
    sklearn_f1s = eval_sklearn(
        train_data, test_data, valid_data, label_col, task, average
    )

    all_f1s = sklearn_f1s + [mlp_f1]
    return all_f1s


class DataTransformer:
    def __init__(
        self, onehot_cols=[], gmm_cols=[], norm_cols=[], ordi_cols=[], norm_ordi=True
    ):
        self.onehot_cols = onehot_cols
        self.gmm_cols = gmm_cols
        self.ordi_cols = ordi_cols
        self.norm_cols = norm_cols
        self.col_dim = {}
        self.col_type = {}
        self.dec_cat = {}
        self.dec_num = {}
        self.norm_ordi = norm_ordi
        # print(self.label_cols,self.norm_cols,self.ordi_cols,self.gmm_cols)

    def fit(self, datasets):
        self.dtypes = {}
        all_columns = set()
        for dataset in datasets:
            all_columns = all_columns | set(dataset.columns)
        for col in all_columns:
            # print(col)
            data_values = []
            for dataset in datasets:
                if col in dataset.columns and dataset[col].notnull().sum() > 0:
                    dvalue = dataset.loc[dataset[col].notnull()][col]
                    self.dtypes[col] = dvalue.dtype
                    data_values.append(dvalue.values)
            if len(data_values) < 2:
                learn_data = data_values[0]
            else:
                learn_data = np.concatenate(data_values, axis=0)
            # print(col, learn_data[:10])
            if col in self.onehot_cols:
                gen_le = LabelEncoder()
                gen_labels = gen_le.fit_transform(learn_data.reshape(-1, 1))
                self.dec_cat[col] = gen_le
                self.col_dim[col] = gen_labels.max() + 1
                self.col_type[col] = "one-hot"
            elif col in self.ordi_cols:
                gen_le = LabelEncoder()
                gen_labels = gen_le.fit_transform(learn_data.reshape(-1, 1))
                self.dec_cat[col] = gen_le
                self.col_dim[col] = 1
                self.col_type[col] = "ordinal"
            elif col in self.norm_cols:
                self.dec_num[col] = [learn_data.min(), learn_data.max()]
                self.col_dim[col] = 1
                self.col_type[col] = "normalize"
            elif col in self.gmm_cols:
                model = GaussianMixture(5)
                fitdata = learn_data.reshape(-1, 1)
                model.fit(fitdata)
                self.dec_num[col] = model
                self.col_dim[col] = 6
                self.col_type[col] = "gmm"
            else:
                self.dec_num[col] = ""
                self.col_dim[col] = 1
                self.col_type[col] = "origin"

    def transform(self, data, conditions=[]):
        enc_data = data.copy()
        for i, col in enumerate(data.columns):
            if self.col_type[col] == "one-hot":
                gen_labels = self.dec_cat[col].transform(data[col])
                enc_data.loc[:, col] = gen_labels
            if self.col_type[col] == "ordinal":
                gen_labels = self.dec_cat[col].transform(data[col])
                enc_data.loc[:, col] = gen_labels
                if self.norm_ordi:
                    if gen_labels.max() == 0:
                        enc_data.loc[:, col] = 0.0
                    else:
                        enc_data.loc[:, col] = enc_data[col] / gen_labels.max()
            elif self.col_type[col] == "normalize":
                if self.dec_num[col][0] == self.dec_num[col][1]:
                    enc_data.loc[:, col] = 0.0
                else:
                    enc_data.loc[:, col] = (data[col] - self.dec_num[col][0]) / (
                        self.dec_num[col][1] - self.dec_num[col][0]
                    )
        trans_data = []
        trans_label = []
        dt = enc_data.values
        for i, col in enumerate(enc_data.columns):
            if col in self.norm_cols or col in self.ordi_cols:
                if col in conditions:
                    trans_label.append(dt[:, i].reshape(-1, 1))
                else:
                    trans_data.append(dt[:, i].reshape(-1, 1))
            elif col in self.gmm_cols:
                fitdata = dt[:, i].reshape(-1, 1)
                weights = self.dec_num[col].weights_
                means = self.dec_num[col].means_.reshape((1, 5))[0]
                stds = np.sqrt(self.dec_num[col].covariances_).reshape((1, 5))[0]
                features = (fitdata - means) / (2 * stds)
                probs = self.dec_num[col].predict_proba(fitdata)
                argmax = np.argmax(probs, axis=1)
                idx = np.arange(len(features))
                features = features[idx, argmax].reshape(-1, 1)
                features = np.concatenate((features, probs), axis=1)
                if col in conditions:
                    trans_label.append(features)
                else:
                    trans_data.append(features)
            elif col in self.onehot_cols:
                col_len = self.col_dim[col]
                features = np.zeros((dt.shape[0], col_len))
                idx = np.arange(len(features))
                features[idx, dt[:, i].astype(int).reshape(1, -1)] = 1
                if col in conditions:
                    trans_label.append(features)
                else:
                    trans_data.append(features)
            else:
                if col in conditions:
                    trans_label.append(dt[:, i].reshape(-1, 1))
                else:
                    trans_data.append(dt[:, i].reshape(-1, 1))

        trans_data = np.concatenate(trans_data, axis=1)
        try:
            trans_label = np.concatenate(trans_label, axis=1)
        except:
            pass
        return trans_data, trans_label

    def fit_transform(self, datasets, conditions=[]):
        self.fit(datasets)
        trans_datasets = []
        for data in datasets:
            trans_datasets.append((self.transform(data, conditions)))
        return trans_datasets

    def get_col_index(self, columns):
        col_index = {}
        sta = 0
        for i, col in enumerate(columns):
            col_len = self.col_dim[col]
            col_index[i] = [sta, sta + col_len]
            sta = sta + col_len
        return col_index

    def inverse_transform(self, data, columns):
        rev_data = []
        col_index = self.get_col_index(columns)
        for i, col in enumerate(columns):
            if col_index[i][0] >= data.shape[1]:
                break
            dc = data[:, col_index[i][0] : col_index[i][1]]
            if col in self.onehot_cols:
                dc = np.argmax(dc, axis=1).reshape(-1, 1).astype(np.int)
                rev_data.append(dc)
            elif col in self.ordi_cols:
                dc = dc.reshape(-1, 1)
                if self.norm_ordi:
                    dc = dc * (len(self.dec_cat[col].classes_) - 1)
                rev_data.append(dc)
            elif col in self.norm_cols:
                dc = dc.reshape(-1, 1)
                rev_data.append(dc)
            elif col in self.gmm_cols:
                v = dc[:, 0]
                u = dc[:, 1:6]
                argmax = np.argmax(u, axis=1)
                means = self.dec_num[col].means_.reshape((1, 5))[0]
                stds = np.sqrt(self.dec_num[col].covariances_).reshape((1, 5))[0]
                mean = means[argmax]
                std = stds[argmax]
                v_ = v * 2 * std + mean
                dc = v_.reshape(-1, 1)
                rev_data.append(dc)
            else:
                rev_data.append(dc.reshape(-1, 1))
        rev_data = np.concatenate(rev_data, axis=1)
        rev_data = pd.DataFrame(rev_data, columns=columns)
        for i, col in enumerate(rev_data.columns):
            if col in self.onehot_cols or col in self.ordi_cols:
                rev_data.loc[:, col] = self.dec_cat[col].inverse_transform(
                    list(rev_data[col].astype(int))
                )
            elif col in self.norm_cols:
                rev_data.loc[:, col] = (
                    rev_data[col] * (self.dec_num[col][1] - self.dec_num[col][0])
                    + self.dec_num[col][0]
                )
            if self.dtypes[col] == int:
                rev_data[col] = rev_data[col] + 0.5
                rev_data[col] = rev_data[col].astype(self.dtypes[col])
            else:
                rev_data[col] = rev_data[col].astype(self.dtypes[col])
        return rev_data


def RE_evaluate(real_data, gen_data, queries, query_num=None):
    # result = pd.DataFrame([], columns=["Setting", "50th", "75th", "90th", "99th", "Max", "Mean"])
    if query_num is None:
        query_num = len(queries)
    all_errors = AQP_error(queries, gen_data, query_num, real_data)
    result = [
        np.percentile(all_errors, 50),
        np.percentile(all_errors, 75),
        np.percentile(all_errors, 90),
        np.percentile(all_errors, 99),
        all_errors.max(),
        all_errors.mean(),
    ]
    return result


def error_reduction(incomplete, complete):
    return (incomplete - complete) / incomplete


def bias_reduction(real, incomplete, complete):
    return 1 - abs(complete - real) / abs(real - incomplete)

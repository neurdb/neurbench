import sys

sys.path.append("../")
import os

import numpy as np
import pandas as pd
import tqdm

import data_utils as du


class Schema:
    def __init__(self, directory):
        print("Database loading ...")
        self.directory = directory
        self.pk_names = []

    def remove_pk(self, table):
        remove_attrs = set(list(table.columns)).intersection(set(self.pk_names))
        return table.drop(list(remove_attrs), axis=1)

    def get_tf_condition(self, join_table, tf_table, exist_id, tf_id, tf_name):
        exist_tf = join_table[[exist_id]]
        exist_tf["tf"] = 1
        exist_tf = exist_tf.groupby(exist_id).sum()

        all_tf = tf_table[[tf_id]]
        all_tf["tf"] = tf_table[tf_name]
        all_tf.index = all_tf[tf_id]

        cond_tf = all_tf.copy()
        cond_tf.loc[cond_tf.index.isin(exist_tf.index), "tf"] = (
            cond_tf.loc[cond_tf.index.isin(exist_tf.index), "tf"] - exist_tf["tf"]
        )
        cond_tf.index = np.arange(len(cond_tf))
        cond_id = np.zeros(int(cond_tf["tf"].sum()))
        cond_id = pd.DataFrame(cond_id, columns=[tf_id])

        sta = 0
        end = 0
        for i in tqdm.tqdm(range(len(cond_tf))):
            sta = end
            end = sta + cond_tf["tf"][i]
            cond_id.loc[sta:end, tf_id] = cond_tf[tf_id][i]

        cond_data = pd.merge(left=cond_id, right=tf_table, on=tf_id, how="left")
        return cond_data


class AirbnbSchema(Schema):
    def __init__(self, directory):
        super().__init__(directory)

        self.apartment = pd.read_csv(os.path.join(self.directory, "apartment.csv"))
        self.neighborhoods = pd.read_csv(
            os.path.join(self.directory, "neighborhoods.csv")
        )
        self.landlord = pd.read_csv(os.path.join(self.directory, "landlord.csv"))

        self.pk_names = [
            "neighborhoods.neighborhood_id",
            "neighborhoods.tf_listings.neighborhood_id",
            "listings.id",
            "listings.neighborhood_id",
            "listings.host_id",
            "hosts.host_id",
            "hosts.tf_listings.host_id",
        ]

        self.ground_truth = pd.merge(
            left=self.apartment,
            right=self.neighborhoods,
            left_on="listings.neighborhood_id",
            right_on="neighborhoods.neighborhood_id",
        )
        self.ground_truth = pd.merge(
            left=self.ground_truth,
            right=self.landlord,
            left_on="listings.host_id",
            right_on="hosts.host_id",
        )

        self.apartment_wrapper = du.DataWrapper()
        self.apartment_wrapper.fit(self.remove_pk(self.apartment))
        self.landlord_wrapper = du.DataWrapper()
        self.landlord_wrapper.fit(self.remove_pk(self.landlord))
        self.neighborhoods_wrapper = du.DataWrapper()
        self.neighborhoods_wrapper.fit(self.remove_pk(self.neighborhoods))
        print("Complete.")

    def get_condition_tuples(self):
        outer_join = pd.merge(
            left=self.join_apartment_pk,
            right=self.join_landlord_pk,
            left_on="listings.host_id",
            right_on="hosts.host_id",
            how="left",
        )
        outer_join = pd.merge(
            left=outer_join,
            right=self.join_neighborhoods_pk,
            left_on="listings.neighborhood_id",
            right_on="neighborhoods.neighborhood_id",
            how="left",
        )

        self.cond_apartment = outer_join.loc[
            pd.isna(outer_join["hosts.host_id"])
            & pd.isna(outer_join["neighborhoods.neighborhood_id"])
        ][self.apartment.columns]
        self.cond_apartment = self.remove_pk(self.cond_apartment)
        self.cond_apartment_neighborhoods = outer_join.loc[
            pd.isna(outer_join["hosts.host_id"])
            & pd.notna(outer_join["neighborhoods.neighborhood_id"])
        ][list(self.apartment.columns) + list(self.neighborhoods.columns)]
        self.cond_apartment_neighborhoods = self.remove_pk(
            self.cond_apartment_neighborhoods
        )
        self.cond_apartment_landlord = outer_join.loc[
            pd.notna(outer_join["hosts.host_id"])
            & pd.isna(outer_join["neighborhoods.neighborhood_id"])
        ][list(self.apartment.columns) + list(self.landlord.columns)]
        self.cond_apartment_landlord = self.remove_pk(self.cond_apartment_landlord)

        self.cond_landlord = self.get_tf_condition(
            self.join_apartment_pk,
            self.join_landlord_pk,
            "listings.host_id",
            "hosts.host_id",
            "hosts.tf_listings.host_id",
        )
        self.cond_landlord = self.remove_pk(self.cond_landlord)
        self.cond_neighborhoods = self.get_tf_condition(
            self.join_apartment_pk,
            self.join_neighborhoods_pk,
            "listings.neighborhood_id",
            "neighborhoods.neighborhood_id",
            "neighborhoods.tf_listings.neighborhood_id",
        )
        self.cond_neighborhoods = self.remove_pk(self.cond_neighborhoods)

    def load_setup(self, incomplete_tables, keep_rate):
        self.incomplete_tables = incomplete_tables

        self.join_neighborhoods_pk = (
            pd.read_csv(
                os.path.join(self.directory, "incomplete_neighborhoods_kr_0.6.csv")
            )
            if "neighborhoods" in incomplete_tables
            else self.neighborhoods
        )
        self.join_apartment_pk = (
            pd.read_csv(
                os.path.join(self.directory, f"incomplete_apartment_kr_{keep_rate}.csv")
            )
            if "apartment" in incomplete_tables
            else self.apartment
        )
        self.join_landlord_pk = (
            pd.read_csv(os.path.join(self.directory, "incomplete_landlord_kr_0.6.csv"))
            if "landlord" in incomplete_tables
            else self.landlord
        )

        self.joined_data = pd.merge(
            left=self.join_apartment_pk,
            right=self.join_landlord_pk,
            left_on="listings.host_id",
            right_on="hosts.host_id",
        )
        self.joined_data = pd.merge(
            left=self.joined_data,
            right=self.join_neighborhoods_pk,
            left_on="listings.neighborhood_id",
            right_on="neighborhoods.neighborhood_id",
        )

        # self.get_condition_tuples()

        self.join_neighborhoods = self.remove_pk(self.join_neighborhoods_pk)
        self.join_apartment = self.remove_pk(self.join_apartment_pk)
        self.join_landlord = self.remove_pk(self.join_landlord_pk)


class HeartSchema(Schema):
    def __init__(self, directory):
        super().__init__(directory)

        self.patient = pd.read_csv(os.path.join(self.directory, "patient.csv"))
        self.cardio = pd.read_csv(os.path.join(self.directory, "cardio.csv"))

        self.pk_names = ["pid"]

        self.incomplete_patient = os.path.join(self.directory, "incomplete_patient")
        self.incomplete_cardio = os.path.join(self.directory, "incomplete_cardio")

        self.ground_truth = pd.merge(left=self.patient, right=self.cardio, on="pid")

        self.patient_wrapper = du.DataWrapper()
        self.patient_wrapper.fit(self.remove_pk(self.patient))
        self.cardio_wrapper = du.DataWrapper()
        self.cardio_wrapper.fit(self.remove_pk(self.cardio))
        print("Complete.")

    def load_setup(self, incomplete_tables, keep_rate):
        self.incomplete_tables = incomplete_tables
        self.join_patient = (
            pd.read_csv(os.path.join(self.directory, "incomplete_patient_kr_0.6.csv"))
            if "patient" in incomplete_tables
            else self.patient
        )
        self.join_cardio = (
            pd.read_csv(
                os.path.join(self.directory, f"incomplete_cardio_kr_{keep_rate}.csv")
            )
            if "cardio" in incomplete_tables
            else self.cardio
        )

        self.joined_data = pd.merge(
            left=self.join_patient, right=self.join_cardio, on="pid"
        )

        self.cond_patient = self.join_patient.loc[
            ~self.join_patient["pid"].isin(self.joined_data["pid"])
        ].drop(["pid"], axis=1)
        self.cond_patient.index = pd.Index(np.arange(len(self.cond_patient)))
        self.cond_cardio = self.join_cardio.loc[
            ~self.join_cardio["pid"].isin(self.joined_data["pid"])
        ].drop(["pid"], axis=1)
        self.cond_cardio.index = pd.Index(np.arange(len(self.cond_cardio)))

        self.join_patient = self.join_patient.drop(["pid"], axis=1)
        self.join_cardio = self.join_cardio.drop(["pid"], axis=1)


class ImdbSchema(Schema):
    def __init__(self, directory):
        super().__init__(directory)

        self.actor = pd.read_csv(os.path.join(self.directory, "actor.csv"))
        self.movie_actor = pd.read_csv(os.path.join(self.directory, "movie_actor.csv"))
        self.movie = pd.read_csv(os.path.join(self.directory, "movie.csv"))
        self.director = pd.read_csv(os.path.join(self.directory, "director.csv"))
        self.movie_director = pd.read_csv(
            os.path.join(self.directory, "movie_director.csv")
        )

        self.pk_names = [
            "movie.id",
            "movie.tf.movie_actor",
            "movie.tf.movie_director",
            "actor.id",
            "actor.tf.movie_actor",
            "movie_actor.id",
            "movie_actor.person_id",
            "movie_actor.movie_id",
            "movie_director.id",
            "movie_director.movie_id",
            "movie_director.person_id",
            "director.id",
            "director.tf.movie_director",
        ]

        self.ma_ground_truth = pd.merge(
            left=self.movie_actor,
            right=self.movie,
            left_on="movie_actor.movie_id",
            right_on="movie.id",
        )
        self.ma_ground_truth = pd.merge(
            left=self.ma_ground_truth,
            right=self.actor,
            left_on="movie_actor.person_id",
            right_on="actor.id",
        )

        self.md_ground_truth = pd.merge(
            left=self.movie_director,
            right=self.movie,
            left_on="movie_director.movie_id",
            right_on="movie.id",
        )
        self.md_ground_truth = pd.merge(
            left=self.md_ground_truth,
            right=self.director,
            left_on="movie_director.person_id",
            right_on="director.id",
        )

        self.movie_wrapper = du.DataWrapper()
        self.movie_wrapper.fit(self.remove_pk(self.movie))
        self.actor_wrapper = du.DataWrapper()
        self.actor_wrapper.fit(self.remove_pk(self.actor))
        self.director_wrapper = du.DataWrapper()
        self.director_wrapper.fit(self.remove_pk(self.director))
        print("Complete.")

    def get_condition_tuples(self):
        # outer_join = pd.merge(left=self.movie_actor, right=self.join_movie_pk, left_on='movie_actor.movie_id', right_on='movie.id', how='left')
        # outer_join = pd.merge(left=outer_join, right=self.join_actor_pk, left_on='movie_actor.person_id', right_on='actor.id', how='left')

        self.cond_actor = self.get_tf_condition(
            self.ma_joined_data[self.movie_actor.columns],
            self.actor,
            "movie_actor.person_id",
            "actor.id",
            "actor.tf.movie_actor",
        )
        self.cond_actor = self.remove_pk(self.cond_actor)

        self.cond_director = self.get_tf_condition(
            self.md_joined_data[self.movie_director.columns],
            self.director,
            "movie_director.person_id",
            "director.id",
            "director.tf.movie_director",
        )
        self.cond_director = self.remove_pk(self.cond_director)

        self.ma_cond_movie = self.get_tf_condition(
            self.ma_joined_data[self.movie_actor.columns],
            self.join_movie_pk,
            "movie_actor.movie_id",
            "movie.id",
            "movie.tf.movie_actor",
        )
        self.ma_cond_movie = self.remove_pk(self.ma_cond_movie)

        self.md_cond_movie = self.get_tf_condition(
            self.md_joined_data[self.movie_director.columns],
            self.join_movie_pk,
            "movie_director.movie_id",
            "movie.id",
            "movie.tf.movie_director",
        )
        self.md_cond_movie = self.remove_pk(self.md_cond_movie)

    def load_setup(self, keep_rate):
        # self.setup_id = setup_id
        self.join_movie_pk = pd.read_csv(
            os.path.join(self.directory, f"incomplete_movie_kr_{keep_rate}.csv")
        )
        # @self.join_actor_pk = self.actor

        self.ma_joined_data = pd.merge(
            left=self.movie_actor,
            right=self.join_movie_pk,
            left_on="movie_actor.movie_id",
            right_on="movie.id",
        )
        self.ma_joined_data = pd.merge(
            left=self.ma_joined_data,
            right=self.actor,
            left_on="movie_actor.person_id",
            right_on="actor.id",
        )

        self.md_joined_data = pd.merge(
            left=self.movie_director,
            right=self.join_movie_pk,
            left_on="movie_director.movie_id",
            right_on="movie.id",
        )
        self.md_joined_data = pd.merge(
            left=self.md_joined_data,
            right=self.director,
            left_on="movie_director.person_id",
            right_on="director.id",
        )

        self.join_movie = self.remove_pk(self.join_movie_pk)

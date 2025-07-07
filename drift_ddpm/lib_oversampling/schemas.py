import sys
sys.path.append('../')
import data_utils as du
import numpy as np
import pandas as pd
import os
import tqdm

class Schema:
	def __init__(self, directory):
		self.directory = directory

	def remove_pk(self, table):
		remove_attrs = set(list(table.columns)).intersection(set(self.pk_names))
		return table.drop(list(remove_attrs), axis=1)

	def get_tf_condition(self, join_table, tf_table, exist_id, tf_id, tf_name):
		exist_tf = join_table[[exist_id]]
		exist_tf['tf'] = 1 
		exist_tf = exist_tf.groupby(exist_id).sum()

		all_tf = tf_table[[tf_id]]
		all_tf['tf'] = tf_table[tf_name]
		all_tf.index = all_tf[tf_id]

		cond_tf = all_tf.copy()
		cond_tf.loc[cond_tf.index.isin(exist_tf.index), 'tf'] = cond_tf.loc[cond_tf.index.isin(exist_tf.index), 'tf'] - exist_tf['tf']
		cond_tf.index = np.arange(len(cond_tf))
		cond_id = np.zeros(int(cond_tf['tf'].sum()))
		cond_id = pd.DataFrame(cond_id, columns=[tf_id])

		sta = 0
		end = 0
		for i in tqdm.tqdm(range(len(cond_tf))):
			sta = end
			end = sta + cond_tf['tf'][i]
			cond_id.loc[sta:end, tf_id] = cond_tf[tf_id][i]

		cond_data = pd.merge(left=cond_id, right=tf_table, on=tf_id, how='left')
		return cond_data

class ImdbSchema(Schema):

	def __init__(self, directory):
		print('Database loading ...')
		self.directory = directory
		
		self.actor = pd.read_csv(os.path.join(self.directory, "actor.csv"))
		self.movie_actor = pd.read_csv(os.path.join(self.directory,"movie_actor.csv"))
		self.movie = pd.read_csv(os.path.join(self.directory,"movie.csv"))
		self.director = pd.read_csv(os.path.join(self.directory, 'director.csv'))
		self.movie_director = pd.read_csv(os.path.join(self.directory, 'movie_director.csv'))


		self.pk_names = ['movie.id', 'movie.tf.movie_actor', 'movie.tf.movie_director', 'actor.id', 'actor.tf.movie_actor', 
						 'movie_actor.id', 'movie_actor.person_id', 'movie_actor.movie_id',
						 'movie_director.id', 'movie_director.movie_id', 'movie_director.person_id',
						 'director.id', 'director.tf.movie_director']
						 
		self.ma_ground_truth = pd.merge(left=self.movie_actor, right=self.movie, left_on='movie_actor.movie_id', right_on='movie.id')
		self.ma_ground_truth = pd.merge(left=self.ma_ground_truth, right=self.actor, left_on='movie_actor.person_id', right_on='actor.id')

		self.md_ground_truth = pd.merge(left=self.movie_director, right=self.movie, left_on='movie_director.movie_id', right_on='movie.id')
		self.md_ground_truth = pd.merge(left=self.md_ground_truth, right=self.director, left_on='movie_director.person_id', right_on='director.id')

		self.movie_wrapper = du.DataWrapper()
		self.movie_wrapper.fit(self.remove_pk(self.movie))
		self.actor_wrapper = du.DataWrapper()
		self.actor_wrapper.fit(self.remove_pk(self.actor))
		self.director_wrapper = du.DataWrapper()
		self.director_wrapper.fit(self.remove_pk(self.director))
		print('Complete.')

	def get_condition_tuples(self):
		#outer_join = pd.merge(left=self.movie_actor, right=self.join_movie_pk, left_on='movie_actor.movie_id', right_on='movie.id', how='left')
		#outer_join = pd.merge(left=outer_join, right=self.join_actor_pk, left_on='movie_actor.person_id', right_on='actor.id', how='left')
	
		self.cond_actor = self.get_tf_condition(self.ma_joined_data[self.movie_actor.columns], self.actor, 'movie_actor.person_id', 'actor.id', 'actor.tf.movie_actor')
		self.cond_actor = self.remove_pk(self.cond_actor)

		self.cond_director = self.get_tf_condition(self.md_joined_data[self.movie_director.columns], self.director, 'movie_director.person_id', 'director.id', 'director.tf.movie_director')
		self.cond_director = self.remove_pk(self.cond_director)

		self.ma_cond_movie = self.get_tf_condition(self.ma_joined_data[self.movie_actor.columns], self.join_movie_pk, 'movie_actor.movie_id', 'movie.id', 'movie.tf.movie_actor')
		self.ma_cond_movie = self.remove_pk(self.ma_cond_movie)

		self.md_cond_movie = self.get_tf_condition(self.md_joined_data[self.movie_director.columns], self.join_movie_pk, 'movie_director.movie_id', 'movie.id', 'movie.tf.movie_director')
		self.md_cond_movie = self.remove_pk(self.md_cond_movie)

	def load_setup(self, keep_rate):

		#self.setup_id = setup_id
		self.join_movie_pk = pd.read_csv(os.path.join(self.directory, f'incomplete_movie_kr_{keep_rate}.csv'))
		#@self.join_actor_pk = self.actor

		self.ma_joined_data = pd.merge(left=self.movie_actor, right=self.join_movie_pk, left_on='movie_actor.movie_id', right_on='movie.id')
		self.ma_joined_data = pd.merge(left=self.ma_joined_data, right=self.actor, left_on='movie_actor.person_id', right_on='actor.id')

		self.md_joined_data = pd.merge(left=self.movie_director, right=self.join_movie_pk, left_on='movie_director.movie_id', right_on='movie.id')
		self.md_joined_data = pd.merge(left=self.md_joined_data, right=self.director, left_on='movie_director.person_id', right_on='director.id')
		
		self.join_movie = self.remove_pk(self.join_movie_pk)
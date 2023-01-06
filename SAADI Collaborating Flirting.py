import os
import gc
import time
import argparse
from math import sqrt


# data science imports
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class NeoRecommender:
    """
    Il s'agit d'un outil de recommandation de filtrage collaboratif basé sur 
    les éléments
    """
    def __init__(self, path_movies, path_ratings,
                 movie_rating_thres=50, user_rating_thres=50, n_neighbors=2):
        """
        Le recommendeur a besoin du path vers les données sur les films, 
        sur les notations. 
        On defini un seuil afin de filtrer les films  les moins notés 
        et les utilisateurs les moins actifs.
        Enfin, on défini le nombre de voisin.
        
        Parametres
        ----------
        path_movies: str, path du fichier de données des films
        
        path_ratings: str, path du fichier de données des notations
        
        movie_rating_thres: int, nombre minimum de notation reçu par les 
        utilisateurs
        
        user_rating_thres: int, nombre minimum de notation donné par les 
        utilisateurs
        
        n_neighbors: int, nombre de voisin
        
        """
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres
        self.n_neighbors = n_neighbors

    def preparation_data(self):
        """
        Préparer les données pour le recommendeur
        """
        
        #On lit les fichiers csv
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        
        #On filtre les données
        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))  # noqa
        movies_filter = df_ratings.movieId.isin(popular_movies).values
        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)

        #On nettoie
        del df_movies_cnt, df_users_cnt
        del df_ratings, df_ratings_filtered
        gc.collect()
        return movie_user_mat, df_movies

    @staticmethod
    def make_mean_table(df):
        """
        On crée un DataFrame contenant les moyennes des notes des utilisateurs
        
        Parameters
        ----------
        df: pandas.DataFrame, dataframe contenant les notes des utlisateurs
        
        Return
        ------
        DataFrame contenant les moyennes des notes des utilisateurs      
        
        """
        mean_table = []
        df1 = df.replace(0.0, np.NaN)
        for user in df.columns.values:
            mean_table.append(df1[int(user)].mean(skipna=True))

        df_mean = pd.DataFrame(data=mean_table, index=df.columns.values)
        return df_mean

    @staticmethod
    def pearson(objet1, objet2, mean_table, table):
        """
        On calcule la similarité entre objet1 et objet2

        Parameters
        ----------
        objet1: int, id du premier objet qu'on compare
        objet2: int, id du deuxieme objet qu'on compare
        mean_table: pandas.DataFrame, dataframe contenant les moyennes des 
        notes des utilisateurs
        table: pandas.DataFrame, dataframe contenant les notes des utlisateurs

        Returns
        -------
        p: float, score de Pearson entre objet1 et objet2

        """
        summation1 = 0
        summation2 = 0
        summation3 = 0
        for movie in table.index.values:
            ruk = table.loc[int(movie), int(objet1)]
            muu = mean_table.loc[int(objet1), 0]
            rvk = table.loc[int(movie), int(objet2)]
            muv = mean_table.loc[int(objet2), 0]

            if ruk != 0 and rvk != 0:
                summation1 += ((ruk - muu) * (rvk - muv))
                summation2 += pow(ruk - muu, 2)
                summation3 += pow(rvk - muv, 2)

        if summation2 != 0 or summation3 != 0:
            p = summation1 / (sqrt(summation2) * sqrt(summation3))
        else:
            p = 0
        return p

    def person_tab(self, user, moy, tab):
        """
        On crée une liste contenant les scores de Pearson entre user et les autres 
        utilisateurs
        
        Parameters
        ----------
        user: int, id de l'utilisateur avec qui on va calculer le score de Pearson
        
        moy: pandas.DataFrame, dataframe contenant les moyennes des notes 
        des utilisateurs
        
        tab: pandas.DataFrame, dataframe contenant les notes des utlisateurs
        
        Return
        ------
        pearson_table: list, liste contenant les scores de Pearson entre user 
        et les autres utilisateurs      
        
        """
        pearson_table = list()
        for i in tab.columns.values:
            if str(i) != user:
                pears = list()
                pears.append(i)
                pears.append(self.pearson(user, i, moy, tab))

                pearson_table.append(pears)
        #On ordonne le tableau du plus au moins similaire
        pearson_table.sort(key=lambda x: x[1], reverse=True)
        return pearson_table

    @staticmethod
    def predicted_score(user, item, mean_table, table, neighbour_number, pearson_table):
        """
        On prédit le score qu'aurait mis user pour "item" à l'aide des autres
        utilisateurs similaires

        Parameters
        ----------
        user : int, id de l'utilisateur avec qui on va prédire le score
        
        item : int, id de l'item pour lequel on va prédire le score
        
        mean_table : pandas.DataFrame, dataframe contenant les moyennes des notes 
        des utilisateurs
        
        table : pandas.DataFrame, dataframe contenant les notes des utlisateurs
        
        neighbour_number : int, nombre d'utilisateur similaire avec lesquels on
        va prédire le score
        pearson_table : list, liste contenant les scores de Pearson entre user et les autres 
        utilisateurs

        Return
        -------
        result : float, le score qu'aurait mis l'utilisateur pour l'item

        """
        #On supprime les utilisateurs n'ayant pas noté l'objet
        neo_pearson_table = pearson_table.copy()
        remove_tab = list()
        for pears in neo_pearson_table:
            if table.loc[int(item), int(pears[0])] == 0.0:
                remove_tab.append(pears)
        for pears in remove_tab:
            neo_pearson_table.remove(pears)
        if len(neo_pearson_table) < neighbour_number:
            return -1
        #On garde seulement les utilisateurs les plus similaires à user
        neo_pearson_table = neo_pearson_table[:neighbour_number]
        #On calcule le score
        summation1 = 0
        summation2 = 0
        muu = mean_table.loc[int(user), 0]
        for pearson in neo_pearson_table:
            simuv = pearson[1]
            rvj = table.loc[int(item), int(pearson[0])]
            muv = mean_table.loc[int(pearson[0]), 0]
            svj = rvj - muv
            summation1 += (simuv * svj)
            summation2 += abs(simuv)
        result = muu + (summation1 / summation2)
        return result

    def inference(self, data, id_user, n_recommendations):
        """
        On prédit les notes qu'aurait mis l'user et renvoie les 
        n_recommendations films ayant les meilleurs notes

        Parameters
        ----------
        data : pandas.DataFrame, dataframe contenant les notes des utlisateurs
        
        id_user : int, id de l'utilisateur à qui on va recommender des films
        
        n_recommendations : int, nombre de film qu'on va recommander

        Returns
        -------
        inference_table : list, liste contenant l'id et la notation des films
        que l'on recommende à l'user

        """
        print('You have input user number ' + str(id_user))
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        inference_table = list()
        #On crée un tableau contenant la note moyenne pour chaque utilisateur
        mean_table = self.make_mean_table(data)
        #On crée un tableau contenant le score de Pearson pour chaque utilisateur
        pearson_table = self.person_tab(id_user, mean_table, data)
        #On crée un tableau contenant l'id et la notation des films que l'on 
        #recommende à l'user
        for i in data.index:
            if data.loc[int(i),int(id_user)] == 0.0:
                inf = list()
                inf.append(self.predicted_score(id_user, i, mean_table, data, self.n_neighbors, pearson_table))
                inf.append(i)
                inference_table.append(inf)
        #On ordonne la liste du film du plus grand au plus petit 
        inference_table.sort(key=lambda x: x[0], reverse=True)
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        return inference_table[:n_recommendations]

    def make_recommendations(self, id_user, top_n):
        """
        On fait un top n des film à recommender

        Parameters
        ----------
        id_user : int, id de l'utilisateur à qui on va recommender des films
        
        top_n : int, nombre de film qu'on va recommander


        """
        movie_user_mat, df_movies = self.preparation_data()
        raw_recommends = self.inference(movie_user_mat, id_user, top_n)
        print('Recommendations for user number {}'.format(id_user))
        for i, (dist, idx) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i + 1, df_movies.loc[df_movies['movieId'] == idx, 'title'].values[0], dist))

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Movie Recommender",
        description="Run KNN Movie Recommender")
    parser.add_argument('--path', nargs='?', default='ml-latest-small',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--id_user', nargs='?', default='1',
                        help='provide the favorite movie name for user')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    id_user = args.id_user
    top_n = args.top_n
    recommender = NeoRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename),
        50, 50)
    try:
        recommender.make_recommendations(id_user, top_n)
        #On anticipe le fait que l'on demande un id d'user pas présent dans la 
        #base de donnée
    except KeyError:
        print("L'utlisateur {} n'a pas donné plus de 50 notaions ou n'est pas présent dans la base de données.".format(
            id_user))

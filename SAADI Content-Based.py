import pandas as pd
from math import sqrt
import os
import argparse
import gc

class ContentBasedRecommender :
    """Il s'agit d'un outil de recommandation basé sur le contenu"""
    def __init__(self, path_movies, path_ratings, userId) :
        """
        Le recommendeur a besoin du path vers les données sur les films et sur
        les notation.

        Parameters
        ----------
        path_movies : str, path du fichier de données des films.
        path_ratings : str, path du fichier de données des notations.
        """
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.userId = userId
        
    def _preparation_data(self) :
        """
        Préparer les données pour le recommendeur

         Return
         ------
         df_genres : pandas.DataFrame, DataFrame contenant pour chaque film 
         un 1 si le genre est présent, 0 sinon
         
         df_movies : pandas.DataFrame, DataFrame contenant pour chaque film 
         le nom du film
        """
        #On lit le csv contenant les films, on créé un dataframe contenant le
        #le genre des films et une autres les noms.
        df_genres = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'genres'],
            dtype={'movieId': 'int32', 'genres': 'str'})
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        
        #On met 'movieId' en index
        df_genres = df_genres.set_index('movieId')
        df_movies = df_movies.set_index('movieId')
        
        #On créé une colonne pour chaque genre en mettant 1 si le genre est
        #présent, 0 sinon 
        for raw in df_genres.index :
            for genre in df_genres.loc[raw,'genres'].split('|') :
                df_genres.loc[raw,genre]=1
        df_genres = df_genres.fillna(0)
        
        #On supprime la colonne 'genres' qui nous est p^lus utile
        df_genres = df_genres.drop(columns=['genres'])
        
        #On crée une colonne 'Like/Dislike' et on initialise la colonne à -1
        #signifiant "je n'ai pas vu le film
        df_genres['Like/Dislike']=-1
        
        #On lit le csv contenant les notations de tous les utilisateurs
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        #On garde seulement les notations de user
        df_ratings_by_user =  df_ratings[df_ratings['userId']==self.userId]

        #On rempli la colonne 'Like/Dislike' avec 1 si le film a eu une note 
        #supérieur à 3.5, dans ce cas il a aimé le film, sinon par 0
        for row in df_ratings_by_user.index :
            if df_ratings_by_user.loc[row, 'rating'] > 3.5 :
                df_genres.loc[df_ratings_by_user.loc[row, 'movieId'], 'Like/Dislike'] = 1
            else :
                df_genres.loc[df_ratings_by_user.loc[row, 'movieId'], 'Like/Dislike'] = 0
  
        #On nettoie
        del df_ratings, df_ratings_by_user
        gc.collect()
        return df_genres, df_movies
  
    @staticmethod
    def cosinus (table, row, liste) :
        """
        On calcule la similarité entre l'objet de la ligne 'raw' de la table
        la liste.

        Parameters
        ----------
        table : pandas.DataFrame, DataFrame contenant pour chaque film 
        un 1 si le genre est présent, 0 sinon.
        row : int, id du film avec lequel on veut calculé la similarité.
        liste : list, list composée de 0 et 1 avec avec lequel on veut
        calculé la similarité.

        Returns
        -------
        result : float, resultat de la fonction cosinus entre la liste et
        le vecteur à la ligne 'row' de 'table'

        """
        table= table.drop(['Like/Dislike'], axis=1)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i, c in enumerate (table.columns) :
            sum1 += table.loc[row,c] * liste[i]
            sum2 += pow(table.loc[row,c], 2)
            sum3 += pow(liste[i], 2)
        
        result = sum1/ (sqrt(sum2) * sqrt(sum3))
        return result

    def _inference(self, df_genres, top_n) :
        """
        On calcule la similitude entre le vecteur moyen des films que 
        l'utilisateur apprecie et les films qu'il n'a pas vu.
        
        Parameters
        ----------
        df_genres : pandas.DataFrame, DataFrame contenant pour chaque film 
        un 1 si le genre est présent, 0 sinon.

        Returns
        -------
        pandas.DataFrame, DataFrame contenant les 10 films ayant le plus de
        similarité avec le vecteur moyen des films que l'utilisateur 
        apprecie.

        """
        #On créé une liste contenant la présence moyenne d'un genre dans 
        #les film que l'utilisateur apprecie 
        movie_like = list(df_genres[df_genres['Like/Dislike']==1].mean(axis = 0))
        
        #On supprime le dernière élément de la liste contenant la moyenne 
        #de la colonne 'Like/Dislike'
        movie_like.pop()
        
        #On crée une dataframe contenant seulement les films qu'il n'a pas
        #vu, on supprime la colonne 'Like/Dislike' et on crée la colonne
        #cosinus en l'initialisant à 0
        movie_unwatch =df_genres[df_genres['Like/Dislike']==-1]
        movie_unwatch = movie_unwatch.drop(['Like/Dislike'], axis=1)
        movie_unwatch['cosinus'] = 0
        
        #On calcule la similitude entre le vecteur moyen des films que 
        #l'utilisateur apprecie et les films qu'il n'a pas vu
        for row in movie_unwatch.index :
            movie_unwatch.loc[row,'cosinus' ] = self.cosinus(df_genres, row, movie_like)
        
        return movie_unwatch.sort_values(by='cosinus', ascending=False).head(top_n)
        
    def make_recommendations(self, top_n) :
        """
        On fait un top_n des films à recommender

        Parameters
        ----------
        top_n : int, le nombre de film que l'on va recommender'
        
        """
        df_genres, df_movies = self._preparation_data()
        movie_unwatch = self._inference(df_genres, top_n)
        for i, movie in enumerate(movie_unwatch.index) :
            print ('{}. {} avec une similarité de {}'.format(i+1, df_movies.loc[movie, 'title'], movie_unwatch.loc[movie,'cosinus' ]) )
    
def parse_args():
    parser = argparse.ArgumentParser(
        prog="Content-Based Recommender",
        description="Run Content-Based Recommender")
    parser.add_argument('--path', nargs='?', default='ml-latest-small',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--user_id', type=int, default=4,
                        help='provide the favorite movie name for user')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()    
    
    
if __name__ == '__main__':
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    user_id = args.user_id
    top_n = args.top_n
    recommender = ContentBasedRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename),
        user_id)
    try:
        recommender.make_recommendations(top_n)
        #On anticipe le fait que l'on demande un id d'utlisateur pas présent dans la 
        #base de donnée
    except KeyError:
        print("L'utlisateur {} n'est pas présent dans la base de données.".format(
            user_id))    
    
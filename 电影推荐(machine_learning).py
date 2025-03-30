import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class HybridRecommender:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.user_features = None
        self.movie_features = None
        self.tfidf = None
        self.svd = None

    def load_data(self, data_path):
        """加载并预处理数据"""
        try:
            self.movies = pd.read_excel(f'{data_path}/movies.xlsx', engine='openpyxl')
            self.ratings = pd.read_excel(f'{data_path}/ratings.xlsx', engine='openpyxl')

            # 数据清洗
            self.ratings['USER_MD5'] = self.ratings['USER_MD5'].astype(str)
            self.ratings['MOVIE_ID'] = self.ratings['MOVIE_ID'].astype(str)
            self.movies['MOVIE_ID'] = self.movies['MOVIE_ID'].astype(str)

            self.ratings = self.ratings.dropna(subset=['USER_MD5', 'MOVIE_ID', 'RATING'])
            self.ratings = self.ratings[pd.to_numeric(self.ratings['RATING'], errors='coerce').notnull()]
            self.ratings['RATING'] = self.ratings['RATING'].astype(float)

            if 'GENRES' in self.movies.columns:
                self.movies['GENRES'] = self.movies['GENRES'].fillna('未知')

            return True
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False

    def _extract_cf_features(self, sample_size=50000):
        """提取协同过滤特征（带采样）"""
        # 对评分数据进行采样
        ratings_sample = self.ratings.sample(min(sample_size, len(self.ratings)))

        # 用户-物品矩阵
        user_movie_matrix = ratings_sample.pivot_table(
            index='USER_MD5',
            columns='MOVIE_ID',
            values='RATING'
        ).fillna(0)

        # 矩阵分解获取隐因子
        svd = TruncatedSVD(n_components=20)
        user_factors = svd.fit_transform(user_movie_matrix)
        movie_factors = svd.components_.T

        # 创建特征DataFrame
        self.user_features = pd.DataFrame(
            user_factors,
            index=user_movie_matrix.index,
            columns=[f'cf_user_{i}' for i in range(user_factors.shape[1])]
        )

        self.movie_features = pd.DataFrame(
            movie_factors,
            index=user_movie_matrix.columns,
            columns=[f'cf_movie_{i}' for i in range(movie_factors.shape[1])]
        )

    def _extract_cb_features(self):
        """提取基于内容的特征"""
        # 电影类型TF-IDF
        self.tfidf = TfidfVectorizer(max_features=50)
        genres_features = self.tfidf.fit_transform(self.movies['GENRES'])

        # 降维
        self.svd = TruncatedSVD(n_components=10)
        genres_reduced = self.svd.fit_transform(genres_features)

        # 添加到电影特征
        genres_df = pd.DataFrame(
            genres_reduced,
            index=self.movies['MOVIE_ID'],
            columns=[f'cb_genre_{i}' for i in range(genres_reduced.shape[1])]
        )

        self.movie_features = pd.concat([self.movie_features, genres_df], axis=1)
        self.movie_features = self.movie_features.fillna(0)

    def prepare_features(self):
        """准备特征数据集"""
        self._extract_cf_features()
        self._extract_cb_features()

        # 合并特征
        data = []
        for _, row in self.ratings.iterrows():
            user_id = row['USER_MD5']
            movie_id = row['MOVIE_ID']
            rating = row['RATING']

            if user_id in self.user_features.index and movie_id in self.movie_features.index:
                features = pd.concat([
                    self.user_features.loc[user_id],
                    self.movie_features.loc[movie_id]
                ]).to_dict()
                features['RATING'] = rating
                data.append(features)

        feature_df = pd.DataFrame(data)

        # 划分特征和标签
        X = feature_df.drop('RATING', axis=1)
        y = feature_df['RATING']

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_model(self, test_size=0.2):
        """训练回归模型"""
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        self.model.fit(X_train, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"模型训练完成，测试集RMSE: {rmse:.4f}")

    def recommend(self, user_id, top_n=5):
        """为用户推荐电影"""
        if user_id not in self.user_features.index:
            print(f"用户 {user_id} 不存在")
            return None

        # 获取用户未评分的电影
        rated_movies = set(self.ratings[self.ratings['USER_MD5'] == user_id]['MOVIE_ID'])
        candidate_movies = set(self.movie_features.index) - rated_movies

        if not candidate_movies:
            print("没有可推荐的候选电影")
            return None

        # 准备预测数据
        X_predict = []
        movie_ids = []
        for movie_id in candidate_movies:
            features = pd.concat([
                self.user_features.loc[user_id],
                self.movie_features.loc[movie_id]
            ]).values
            X_predict.append(features)
            movie_ids.append(movie_id)

        X_predict = self.scaler.transform(np.array(X_predict))

        # 预测评分
        pred_ratings = self.model.predict(X_predict)

        # 获取Top-N推荐
        recommendations = pd.DataFrame({
            'MOVIE_ID': movie_ids,
            'PRED_RATING': pred_ratings
        }).sort_values('PRED_RATING', ascending=False).head(top_n)

        # 添加电影信息
        recommendations = pd.merge(
            recommendations,
            self.movies[['MOVIE_ID', 'NAME', 'GENRES']],
            on='MOVIE_ID',
            how='left'
        )

        return recommendations.reset_index(drop=True)


# 使用示例
if __name__ == "__main__":
    recommender = HybridRecommender()

    # 加载数据
    data_path = r'/kaggle/input/movie-data/豆瓣数据'
    if not recommender.load_data(data_path):
        exit()

    # 准备特征并训练模型
    print("正在准备特征...")
    recommender.prepare_features()
    print("训练模型中...")
    recommender.train_model()

    # 测试用户推荐
    test_user = '47e69de0d68e6a4db159bc29301caece'
    print(f"\n为用户 {test_user} 生成推荐:")
    recs = recommender.recommend(test_user, top_n=5)
    if recs is not None:
        print(recs[['NAME', 'GENRES', 'PRED_RATING']])
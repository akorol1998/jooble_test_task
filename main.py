from pandas import DataFrame
from pandas import Series
from typing import Any
import pandas as pd
import os


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_PATH = os.path.join(DIR_PATH, 'test.tsv')
TRAIN_DATA_PATH = os.path.join(DIR_PATH, 'train.tsv')


class DocumentHandler:
    def __init__(self):
        self.df: DataFrame = None
        self.vac_df: DataFrame = None
        self.stat_df: DataFrame = None
        self.job_ids: Series= None
        self.features: DataFrame = None
        self.stat_features: DataFrame = None
        self.feature_codes: Series = None

    def load(self, test_path: str, train_path: str):
        """Loads to the object files using provided paths."""

        self.vac_df = pd.read_csv(
            filepath_or_buffer=test_path,
            sep='\t',
        )

        self.stat_df = pd.read_csv(
            filepath_or_buffer=train_path,
            sep='\t',
        )

    def format(self):
        """Format loaded data to the convenient form"""

        self.job_ids = self.vac_df['id_job']

        feat = self.vac_df['features']
        self.features: DataFrame = feat.str.split(
            pat=',',
            expand=True)
        self.feature_codes = self.features[0]
        self.features.drop(labels=0, axis=1, inplace=True)

        stat_feat = self.stat_df['features']
        self.stat_features: DataFrame = stat_feat.str.split(
            pat=',',
            expand=True)
        self.stat_features.drop(labels=0, axis=1, inplace=True)
        self._create_dataframe()


    def _create_dataframe(self) -> None:
        """Create empty Dataframe."""

        feature_code_set = set(self.feature_codes)
        code_columns = [f'feature_{code}_stand' for code in feature_code_set]
        columns = [column + f'_{col_index}' for col_index in self.features.columns for column in code_columns]
        columns.extend([f'max_feature_{code}_index' for code in feature_code_set])
        columns.extend([f'max_feature_{code}_abs_mean_diff' for code in feature_code_set])
        columns.insert(0, 'id_job')
        self.df = DataFrame(
            columns=columns,
        )
        self.df['id_job'] = self.job_ids

    def _z_score(self, x: float, column: Any) -> float:
        """Z - score function."""

        x_mean = self.stat_features[column].astype(float).mean()
        x_std = self.stat_features[column].astype(float).std()
        return (x - x_mean)/x_std

    def load_to_file(self, file_name: str='test_proc.tsv'):
        """Load dataframe data nto the file."""

        self.df.to_csv(
            path_or_buf=file_name,
            index=False
        )

    def aggregate_data(self):
        """Aggregate data according to requirements."""

        for index in self.vac_df.index:
            code = self.feature_codes[index]
            for idx, column in enumerate(self.features.columns):
                x = float(self.features.at[index, column])
                normalized = self._z_score(x, column)
                stand_column = f'feature_{code}_stand_{column}'
                self.df.at[index, stand_column] = normalized

            # Evaluating max_feature_index_column
            max_feature_index_column = f'max_feature_{code}_index'
            max_el_idx = self.features.loc[index].astype(float).idxmax()
            self.df.at[index, max_feature_index_column] = max_el_idx

            # Evaluating asb_mean_diff column
            abs_mean_diff_column = f'max_feature_{code}_abs_mean_diff'
            mean_value_of_max_feature_column = self.features[max_el_idx].astype(float).mean()
            abs_mean_diff = abs(mean_value_of_max_feature_column - float(self.features.at[index, max_el_idx]))
            self.df.at[index, abs_mean_diff_column] = abs_mean_diff


if __name__ == '__main__':
    handler = DocumentHandler()
    handler.load(
        TEST_DATA_PATH,
        TRAIN_DATA_PATH
    )
    handler.format()
    handler.aggregate_data()
    handler.load_to_file('test_proc.tsv')
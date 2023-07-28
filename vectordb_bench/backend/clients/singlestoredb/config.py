from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, EmptyDBCaseConfig

SINGLESTOREDB_URL_PLACEHOLDER = "singlestoredb://%s:%s@%s/%s"

class SingleStoreDBConfig(DBConfig):
    user_name: SecretStr = "admin"
    password: SecretStr
    url: SecretStr
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        url_str = self.url.get_secret_value()
        return {
            "url" : SINGLESTOREDB_URL_PLACEHOLDER%(user_str, pwd_str, url_str, self.db_name)
        }

class SingleStoreDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return ""
        elif self.metric_type == MetricType.IP:
            return ""
        return ""

    def parse_metric_fun_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "euclidian_distance"
        elif self.metric_type == MetricType.IP:
            return "dot_product"
        return "dot_product"

    def index_param(self) -> dict:
        return {
            "metric" : self.parse_metric()
        }

    def search_param(self) -> dict:
        return {
            "metric_fun" : self.parse_metric_fun_str()
        }

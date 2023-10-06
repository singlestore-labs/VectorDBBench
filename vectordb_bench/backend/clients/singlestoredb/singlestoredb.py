"""Wrapper around the SingleStoreDB vector database over VectorDB"""

import logging
import numpy as np
import time
from contextlib import contextmanager
from typing import Any, Type
from functools import wraps

import sqlalchemy as sa
from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import SingleStoreDBConfig, SingleStoreDBIndexConfig
from sqlalchemy import (
    MetaData,
    create_engine,
    insert,
    select,
    Index,
    Table,
    text,
    Column,
    Float, 
    Integer,
    LargeBinary
)
from sqlalchemy.orm import (
    declarative_base, 
    Session
)

log = logging.getLogger(__name__) 

class SingleStoreDB(VectorDB):
    """ Use SQLAlchemy instructions"""
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "SingleStoreDBCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "singlestoredb_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        s2_engine = create_engine(**self.db_config)
        Base = declarative_base()
        s2_metadata = Base.metadata
        s2_metadata.reflect(s2_engine) 
        
        self.s2_table = self._get_table_schema(s2_metadata)
        if drop_old and self.table_name in s2_metadata.tables:
            log.info(f"SingleStoreDB client drop table : {self.table_name}")
            # self.s2_table.drop(s2_engine, checkfirst=True)
            s2_metadata.drop_all(s2_engine)
            self._create_table(dim, s2_engine)

    
    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return SingleStoreDBConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return SingleStoreDBIndexConfig

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.s2_engine = create_engine(**self.db_config)

        Base = declarative_base()
        s2_metadata = Base.metadata
        s2_metadata.reflect(self.s2_engine) 
        self.s2_session = Session(self.s2_engine)
        self.s2_table = self._get_table_schema(s2_metadata)
        yield 
        self.s2_session = None
        self.s2_engine = None 
        del (self.s2_session)
        del (self.s2_engine)
    
    def ready_to_load(self):
        pass

    def optimize(self):
        pass

    def ready_to_search(self):
        pass

    def _get_table_schema(self, s2_metadata):
        return Table(
            self.table_name,
            s2_metadata,
            Column(self._primary_field, Integer, primary_key=True, autoincrement=False),
            Column(self._vector_field, LargeBinary(self.dim)),
            extend_existing=True
        )
    
    def _create_table(self, dim, s2_engine : int):
        try:
            # create table
            self.s2_table.create(bind=s2_engine, checkfirst=True)
        except Exception as e:
            log.warning(f"Failed to create singlestoredb table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        try:
            items = [dict(id=metadata[i], embedding=np.array(embeddings[i], dtype='<f4')) for i in range(len(metadata))]
            self.s2_session.execute(insert(self.s2_table), items)
            self.s2_session.commit()
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into singlestoredb table ({self.table_name}), error: {e}")   
            return 0, e

    def search_embedding(        
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.s2_table is not None
        search_param = self.case_config.search_param()
        if search_param["metric_fun"] == "cosine_distance":
            op_fun = (lambda x, y: sa.func.dot_product(x, y) / 
                                   (sa.func.sqrt(sa.func.dot_product(x, x) *
                                    sa.func.dot_product(y, y))))
        else:
            op_fun = getattr(sa.func, search_param["metric_fun"])
        if filters:
            res = self.s2_session.scalars(select(self.s2_table).order_by(op_fun(self.s2_table.c.embedding, np.array(query, dtype='<f4'))).filter(self.s2_table.c.id > filters.get('id')).limit(k))
        else: 
            res = self.s2_session.scalars(select(self.s2_table).order_by(op_fun(self.s2_table.c.embedding, np.array(query, dtype='<f4'))).limit(k))
        return list(res)
        

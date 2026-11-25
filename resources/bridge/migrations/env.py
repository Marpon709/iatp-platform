"""Alembic 迁移环境，支持程序内连接和手工命令。"""
import os

from alembic import context
from sqlalchemy import create_engine, pool
from history_store import Base

config = context.config
target_metadata = Base.metadata


def migrate(connection):
    context.configure(connection=connection, target_metadata=target_metadata,
                      compare_type=True, render_as_batch=connection.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


connection = config.attributes.get("connection")
if connection is not None:
    migrate(connection)
else:
    url = os.environ.get("IATP_DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if not url:
        raise ValueError("手工迁移前请设置 IATP_DATABASE_URL。")
    if context.is_offline_mode():
        context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
    else:
        engine = create_engine(url, poolclass=pool.NullPool)
        with engine.connect() as connection:
            migrate(connection)

"""分析运行历史：只保存记录，不负责调度或恢复计算。

三个业务表分别保存模型版本、运行记录和结果。每个方法都创建自己的
Session，后台计算线程之间不共享数据库会话。默认 SQLite 无需额外服务。
"""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import threading
import time

from alembic import command
from alembic.config import Config
from sqlalchemy import (Boolean, Column, Float, ForeignKey, Integer, JSON, String,
                        create_engine, event, func, select)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from result_checks import validate_result


Base = declarative_base()
STATUSES = {"running", "completed", "failed", "cancelled", "timed_out", "interrupted"}
TERMINAL_STATUSES = STATUSES - {"running"}


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    version = Column(String(64), nullable=False, unique=True)
    created_at = Column(String(40), nullable=False)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    job_id = Column(String(64), primary_key=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    kind = Column(String(40), nullable=False, index=True)
    request_key = Column(String(128), nullable=False)
    parameters = Column(JSON, nullable=False)
    snapshot = Column(JSON, nullable=False)
    status = Column(String(24), nullable=False, index=True)
    created_at = Column(String(40), nullable=False)
    created_epoch = Column(Float, nullable=False, index=True)
    finished_at = Column(String(40))
    updated_at = Column(String(40), nullable=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    run_id = Column(String(64), ForeignKey("analysis_runs.job_id"), primary_key=True)
    payload = Column(JSON, nullable=False)
    summary = Column(JSON, nullable=False)
    is_partial = Column(Boolean, nullable=False, default=False)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_copy(value):
    """在提交事务前检查 JSON，拒绝 NaN 和不可序列化的对象。"""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def result_summary(result):
    """列表只展示轻量摘要，不传输完整路线和 GeoJSON。"""
    if not isinstance(result, dict):
        return {}
    summary = {}
    for field in ("totalCarbonG", "totalTravelMin", "totalDistanceKm",
                  "validatedCandidateCount", "feasibleCandidateCount", "hasActivitySpace"):
        if field in result:
            summary[field] = result[field]
    if isinstance(result.get("candidates"), list):
        summary["candidateCount"] = len(result["candidates"])
    if isinstance(result.get("categories"), list):
        summary["categoryCount"] = len(result["categories"])
    if result.get("message"):
        summary["message"] = result["message"]
    return summary


class HistoryStore:
    def __init__(self, work_root, model_version, database_url=None):
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.model_version = str(model_version)
        self.database_url = (database_url or os.environ.get("IATP_DATABASE_URL")
                             or "sqlite:///" + (self.work_root / "history.sqlite3").as_posix())
        self._write_lock = threading.RLock()
        options = {"pool_pre_ping": True}
        if self.database_url.startswith("sqlite:"):
            options["connect_args"] = {"check_same_thread": False, "timeout": 15}
            if self.database_url in ("sqlite://", "sqlite:///:memory:"):
                options["poolclass"] = StaticPool
        self.engine = create_engine(self.database_url, **options)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", self._configure_sqlite)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._migrate()

    @staticmethod
    def _configure_sqlite(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    def _migrate(self):
        """真实执行 Alembic 迁移；不使用 create_all 代替迁移历史。"""
        directory = Path(__file__).resolve().parent
        config = Config(str(directory / "alembic.ini"))
        config.set_main_option("script_location", str(directory / "migrations"))
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    @staticmethod
    def _snapshot_without_result(snapshot):
        value = copy.deepcopy(snapshot)
        value.pop("result", None)
        value.pop("partialResult", None)
        return value

    @staticmethod
    def _save_result(session, run, snapshot):
        result = snapshot.get("result")
        partial = bool(isinstance(result, dict) and result.get("partial"))
        if result is None:
            result = snapshot.get("partialResult")
            partial = result is not None
        if result is None:
            return  # 新快照没有结果时，保留此前已保存的部分或完整结果。
        saved = session.get(AnalysisResult, run.job_id)
        if saved is not None and not saved.is_partial and partial:
            return  # 迟到的部分结果不能覆盖完整结果。
        if saved is None:
            saved = AnalysisResult(run_id=run.job_id)
            session.add(saved)
        saved.payload = result
        saved.summary = result_summary(result)
        saved.is_partial = partial

    def create_run(self, job_id, kind, request_key, parameters, snapshot):
        parameters, snapshot = json_copy(parameters), json_copy(snapshot)
        status = snapshot.get("status", "running")
        if status not in STATUSES:
            raise ValueError("不支持的任务状态。")
        created = snapshot.get("created_at") or utc_now()
        with self._write_lock, self.Session.begin() as session:
            if session.get(AnalysisRun, job_id) is not None:
                raise ValueError("任务编号已存在。")
            version = session.scalar(select(ModelVersion).where(ModelVersion.version == self.model_version))
            if version is None:
                version = ModelVersion(version=self.model_version, created_at=utc_now())
                session.add(version)
                session.flush()
            run = AnalysisRun(job_id=job_id, model_version_id=version.id,
                              kind=kind, request_key=request_key, parameters=parameters,
                              snapshot=self._snapshot_without_result(snapshot), status=status,
                              created_at=created, created_epoch=snapshot.get("createdEpoch", time.time()),
                              finished_at=snapshot.get("finished_at"), updated_at=utc_now())
            session.add(run)
            session.flush()
            self._save_result(session, run, snapshot)

    def save_snapshot(self, snapshot):
        snapshot = json_copy(snapshot)
        job_id, status = snapshot["jobId"], snapshot["status"]
        if status not in STATUSES:
            raise ValueError("不支持的任务状态。")
        with self._write_lock, self.Session.begin() as session:
            run = session.get(AnalysisRun, job_id)
            if run is None:
                raise ValueError("计算任务不存在。")
            # 计算完成、取消和超时的竞态不能把终态改回运行中或另一种终态。
            if run.status in TERMINAL_STATUSES and run.status != status:
                return False
            run.snapshot = self._snapshot_without_result(snapshot)
            run.status = status
            run.finished_at = snapshot.get("finished_at") or run.finished_at
            run.updated_at = utc_now()
            self._save_result(session, run, snapshot)
        return True

    def mark_interrupted(self):
        """启动时标记上次未结束的任务。只恢复记录，不重新执行计算。"""
        with self._write_lock, self.Session.begin() as session:
            rows = session.scalars(select(AnalysisRun).where(AnalysisRun.status == "running")).all()
            finished = utc_now()
            for run in rows:
                snapshot = dict(run.snapshot)
                snapshot.update(status="interrupted", stage="服务重启，任务已中断",
                                error="服务重启前计算未完成，请重新提交。",
                                finished_at=finished, backendRequest=None)
                run.status = "interrupted"
                run.finished_at = finished
                run.updated_at = finished
                run.snapshot = snapshot
            return len(rows)

    def list_runs(self, page=1, page_size=10, status=None, kind=None):
        if (not isinstance(page, int) or isinstance(page, bool) or page < 1
                or not isinstance(page_size, int) or isinstance(page_size, bool)
                or not 1 <= page_size <= 50):
            raise ValueError("页码须大于等于 1，每页条数须在 1 至 50 之间。")
        if status is not None and status not in STATUSES:
            raise ValueError("不支持的任务状态。")
        filters = []
        if status is not None:
            filters.append(AnalysisRun.status == status)
        if kind is not None:
            filters.append(AnalysisRun.kind == kind)
        # 这个查询展示运行记录、模型版本和结果摘要三个表之间的关联。
        query = (select(AnalysisRun, ModelVersion.version, AnalysisResult)
                 .join(ModelVersion, AnalysisRun.model_version_id == ModelVersion.id)
                 .outerjoin(AnalysisResult, AnalysisRun.job_id == AnalysisResult.run_id)
                 .where(*filters).order_by(AnalysisRun.created_epoch.desc(), AnalysisRun.job_id.desc())
                 .offset((page - 1) * page_size).limit(page_size))
        with self.Session() as session:
            total = session.scalar(select(func.count()).select_from(AnalysisRun).where(*filters))
            items = []
            for run, version, result in session.execute(query):
                items.append({"jobId": run.job_id, "kind": run.kind, "status": run.status,
                              "stage": run.snapshot.get("stage"), "progress": run.snapshot.get("progress"),
                              "created_at": run.created_at, "finished_at": run.finished_at,
                              "modelVersion": version, "summary": result.summary if result else {},
                              "hasResult": result is not None, "isPartial": bool(result and result.is_partial),
                              "error": run.snapshot.get("error")})
            return {"items": items, "total": total, "page": page, "pageSize": page_size}

    def get_run(self, job_id):
        with self.Session() as session:
            row = session.execute(
                select(AnalysisRun, ModelVersion.version, AnalysisResult)
                .join(ModelVersion, AnalysisRun.model_version_id == ModelVersion.id)
                .outerjoin(AnalysisResult, AnalysisRun.job_id == AnalysisResult.run_id)
                .where(AnalysisRun.job_id == job_id)).first()
            if row is None:
                return None
            run, version, result = row
            snapshot = copy.deepcopy(run.snapshot)
            snapshot.update(jobId=run.job_id, kind=run.kind, status=run.status,
                            modelVersion=version, parameters=run.parameters,
                            created_at=run.created_at, finished_at=run.finished_at,
                            summary=result.summary if result else {}, persisted=True)
            if result is not None:
                snapshot["result"] = result.payload
                snapshot["validation"] = validate_result('route-chain' if run.kind == 'result-review' else run.kind, result.payload)
                snapshot["isPartial"] = result.is_partial
                if result.is_partial:
                    snapshot["partialResult"] = result.payload
            return snapshot

    def close(self):
        self.engine.dispose()

"""建立模型版本、分析运行和结果表。"""
from alembic import op
import sqlalchemy as sa

revision = "0001_analysis_history"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("model_versions",
                    sa.Column("id", sa.Integer(), primary_key=True),
                    sa.Column("version", sa.String(64), nullable=False, unique=True),
                    sa.Column("created_at", sa.String(40), nullable=False))
    op.create_table("analysis_runs",
                    sa.Column("job_id", sa.String(64), primary_key=True),
                    sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("model_versions.id"), nullable=False),
                    sa.Column("kind", sa.String(40), nullable=False),
                    sa.Column("request_key", sa.String(128), nullable=False),
                    sa.Column("parameters", sa.JSON(), nullable=False),
                    sa.Column("snapshot", sa.JSON(), nullable=False),
                    sa.Column("status", sa.String(24), nullable=False),
                    sa.Column("created_at", sa.String(40), nullable=False),
                    sa.Column("created_epoch", sa.Float(), nullable=False),
                    sa.Column("finished_at", sa.String(40)),
                    sa.Column("updated_at", sa.String(40), nullable=False))
    op.create_index("ix_analysis_runs_kind", "analysis_runs", ["kind"])
    op.create_index("ix_analysis_runs_status", "analysis_runs", ["status"])
    op.create_index("ix_analysis_runs_created_epoch", "analysis_runs", ["created_epoch"])
    op.create_table("analysis_results",
                    sa.Column("run_id", sa.String(64), sa.ForeignKey("analysis_runs.job_id"), primary_key=True),
                    sa.Column("payload", sa.JSON(), nullable=False),
                    sa.Column("summary", sa.JSON(), nullable=False),
                    sa.Column("is_partial", sa.Boolean(), nullable=False))


def downgrade():
    op.drop_table("analysis_results")
    op.drop_index("ix_analysis_runs_created_epoch", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_status", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_kind", table_name="analysis_runs")
    op.drop_table("analysis_runs")
    op.drop_table("model_versions")

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    time: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)

    v1: Mapped[float] = mapped_column(Float, nullable=False)
    v2: Mapped[float] = mapped_column(Float, nullable=False)
    v3: Mapped[float] = mapped_column(Float, nullable=False)
    v4: Mapped[float] = mapped_column(Float, nullable=False)
    v5: Mapped[float] = mapped_column(Float, nullable=False)
    v6: Mapped[float] = mapped_column(Float, nullable=False)
    v7: Mapped[float] = mapped_column(Float, nullable=False)
    v8: Mapped[float] = mapped_column(Float, nullable=False)
    v9: Mapped[float] = mapped_column(Float, nullable=False)
    v10: Mapped[float] = mapped_column(Float, nullable=False)
    v11: Mapped[float] = mapped_column(Float, nullable=False)
    v12: Mapped[float] = mapped_column(Float, nullable=False)
    v13: Mapped[float] = mapped_column(Float, nullable=False)
    v14: Mapped[float] = mapped_column(Float, nullable=False)
    v15: Mapped[float] = mapped_column(Float, nullable=False)
    v16: Mapped[float] = mapped_column(Float, nullable=False)
    v17: Mapped[float] = mapped_column(Float, nullable=False)
    v18: Mapped[float] = mapped_column(Float, nullable=False)
    v19: Mapped[float] = mapped_column(Float, nullable=False)
    v20: Mapped[float] = mapped_column(Float, nullable=False)
    v21: Mapped[float] = mapped_column(Float, nullable=False)
    v22: Mapped[float] = mapped_column(Float, nullable=False)
    v23: Mapped[float] = mapped_column(Float, nullable=False)
    v24: Mapped[float] = mapped_column(Float, nullable=False)
    v25: Mapped[float] = mapped_column(Float, nullable=False)
    v26: Mapped[float] = mapped_column(Float, nullable=False)
    v27: Mapped[float] = mapped_column(Float, nullable=False)
    v28: Mapped[float] = mapped_column(Float, nullable=False)

    actual_class: Mapped[int] = mapped_column(Integer, nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="transaction")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.id"), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    fraud_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transaction: Mapped["Transaction"] = relationship(back_populates="predictions")


def create_tables(engine):
    Base.metadata.create_all(engine)

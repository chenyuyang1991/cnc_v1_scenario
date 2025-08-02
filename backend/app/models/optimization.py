from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from app.core.database import Base


class Optimization(Base):
    __tablename__ = "optimizations"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="pending")
    optimization_type = Column(String)
    original_config = Column(JSON)
    optimized_config = Column(JSON)
    results = Column(JSON)
    time_reduction = Column(Float)
    quality_improvement = Column(Float)
    cost_savings = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True)) 
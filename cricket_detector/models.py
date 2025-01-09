from sqlalchemy import create_engine, Column, Integer, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class CricketCount(Base):
    __tablename__ = 'cricket_counts'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    count = Column(Integer)
    confidence = Column(Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'count': self.count,
            'confidence': self.confidence
        }

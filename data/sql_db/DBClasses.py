import sys, os
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import configure_mappers, relation, relationship
from DatabaseConnection import DatabaseConnection

dbc = DatabaseConnection()

# ========================
# Define database classes
# ========================
Base = declarative_base(bind=dbc.engine)

class DBFlare(Base):
    __tablename__ = 'flares'
    id = Column(Integer, primary_key=True)
    start_time = Column(String)
    end_time = Column(String)
    peak_time = Column(String)
    location = Column(String)
    fclass = Column(String)
    peak_flux = Column(Float)
    integ_flux = Column(Float)
    region_id = Column(Integer)

    # region_id = Column(Integer, ForeignKey('regions.id'))
    # region = relationship('DBRegion', back_populates='flares')

# class DBRegion(Base):
#     __tablename__ = 'regions'
#     id = Column(Integer, primary_key=True)
#     flares = relationship('DBFlare', back_populates='region')

try:
    configure_mappers()
except RuntimeError:
    print("""
An error occurred when verifying the relations between the database tables.
Most likely this is an error in the definition of the SQLAlchemy relations -
see the error message below for details.
""")
    print("Error type: %s" % sys.exc_info()[0])
    print("Error value: %s" % sys.exc_info()[1])
    print("Error trace: %s" % sys.exc_info()[2])
    sys.exit(1)

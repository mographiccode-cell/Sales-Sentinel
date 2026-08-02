from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker

class Base(DeclarativeBase): pass
_engine = None
SessionLocal = scoped_session(sessionmaker(autoflush=False, expire_on_commit=False))

def init_engine(url: str):
    global _engine
    connect_args = {'check_same_thread': False} if url.startswith('sqlite') else {}
    _engine = create_engine(url, future=True, connect_args=connect_args)
    SessionLocal.configure(bind=_engine)
    return _engine

def get_engine(): return _engine

def create_all():
    from . import models
    Base.metadata.create_all(bind=_engine)

@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback(); raise
    finally:
        db.close()

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker


class Base(DeclarativeBase):
    pass


_engine = None
SessionLocal = scoped_session(sessionmaker(autoflush=False, expire_on_commit=False))


def init_engine(url: str):
    global _engine
    # A scoped_session keeps a thread-local Session in its registry even after
    # Session.close(). Remove it before rebinding so tests, CLI jobs, and any
    # controlled reinitialization cannot accidentally keep using the old DB.
    SessionLocal.remove()
    if _engine is not None:
        _engine.dispose()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _engine = create_engine(url, future=True, connect_args=connect_args)
    SessionLocal.configure(bind=_engine)
    return _engine


def get_engine():
    return _engine


def create_all():
    from . import models  # noqa: F401
    if _engine is None:
        raise RuntimeError("Database engine is not initialized")
    Base.metadata.create_all(bind=_engine)


@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        # remove() closes the current scoped Session and discards it from the
        # registry; a later scope receives a fresh Session bound to the engine.
        SessionLocal.remove()

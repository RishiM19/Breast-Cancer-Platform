"""Test-only SQL isolation and application loading, not production routing."""
import importlib.util
import sqlite3
import threading
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/processed/lumina_integration_stage2"


class SQLGuard:
    def __init__(self,path:Path):
        self.path=path.resolve();self.original=sqlite3.connect;self.connections=[];self.denied=[];self.statements={};self.lock=threading.Lock()
    def connect(self,database,*args,**kwargs):
        candidate=Path(database).resolve()
        if candidate!=self.path:
            self.denied.append(str(candidate));raise AssertionError("Database isolation refused a non-test connection")
        with self.lock:self.connections.append(str(candidate))
        conn=self.original(database,*args,**kwargs)
        def trace(statement):
            verb=statement.lstrip().split(maxsplit=1)[0].upper() if statement.strip() else "EMPTY"
            with self.lock:self.statements[verb]=self.statements.get(verb,0)+1
        conn.set_trace_callback(trace)
        return conn
    def report(self):return dict(allowed_database=str(self.path),connections=len(self.connections),
        actual_database_paths=sorted(set(self.connections)),denied_connections=self.denied,sql_verb_counts=self.statements,
        patient_data_logged=False)


def load_main():
    spec=importlib.util.spec_from_file_location("stage2_main_"+uuid.uuid4().hex,ROOT/"main.py")
    main=importlib.util.module_from_spec(spec);spec.loader.exec_module(main);return main

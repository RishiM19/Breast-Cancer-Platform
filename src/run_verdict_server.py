"""Loopback-only browser test server using an explicit synthetic-only database."""
import argparse
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

from .stage2_test_support import ROOT, SQLGuard, load_main
OUT=ROOT/"data/processed/lumina_ui_verdict_upgrade"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--db",default="browser.db");parser.add_argument("--port",type=int,default=8765);parser.add_argument("--reuse",action="store_true");parser.add_argument("--missing-segmentation",action="store_true");args=parser.parse_args()
    if Path(args.db).name!=args.db:raise ValueError("Pass only a test database filename")
    path=OUT/"databases"/args.db
    if path.exists() and not args.reuse:raise FileExistsError("Use --reuse explicitly to reopen this synthetic-only test database")
    existed=path.exists()
    os.environ["LUMINA_TEST_MODE"]="1";os.environ["LUMINA_TEST_DB_PATH"]=str(path)
    guard=SQLGuard(path);sqlite3.connect=guard.connect
    from .research_inference import ResearchEngine
    from .research_api import create_research_router
    from unittest.mock import patch
    engine=ResearchEngine()
    if args.missing_segmentation:engine.segmentation.path=OUT/"missing_browser_checkpoint.pth"
    with patch("src.research_api.create_research_router",side_effect=lambda:create_research_router(engine)):
        app=load_main().app
    events=[];stop_file=OUT/("browser_server_"+uuid.uuid4().hex+".stop")
    def save():
        (OUT/"browser_server_isolation.json").write_text(json.dumps(guard.report()|dict(events=events,existed_at_start=existed,stop_file=str(stop_file),missing_segmentation_simulation=args.missing_segmentation,segmentation_loads=engine.segmentation.load_count,classifiers_cached=list(engine._models),hook_count=sum(len(m._forward_hooks)+len(m._backward_hooks) for model in engine._models.values() for m in model.modules())),indent=2),encoding="utf-8")
    @app.middleware("http")
    async def audit_request(request,call_next):
        try:return await call_next(request)
        finally:save()
    def started():
        conn=guard.connect(str(path))
        try:count=conn.execute("SELECT count(*) FROM patients").fetchone()[0]
        finally:conn.close()
        events.append(dict(event="startup",initial_patient_count=count));save()
    def stopped():events.append(dict(event="shutdown"));save()
    app.add_event_handler("startup",started);app.add_event_handler("shutdown",stopped)
    import uvicorn
    server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=args.port,log_level="warning"))
    def watch_stop():
        wait=threading.Event()
        while not server.should_exit:
            if stop_file.exists():server.should_exit=True;return
            wait.wait(.2)
    threading.Thread(target=watch_stop,daemon=True).start()
    try:server.run()
    finally:save();sqlite3.connect=guard.original


if __name__=="__main__":main()

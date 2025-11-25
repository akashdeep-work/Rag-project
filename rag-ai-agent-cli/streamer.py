import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from indexer import Indexer
from threading import Thread

class _Handler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def on_created(self, event):
        if not event.is_directory:
            print("New file detected:", event.src_path)
            # index in a thread to avoid blocking watch events
            t = Thread(target=self.indexer.index_all)
            t.start()

def start_watcher(path: str, indexer: Indexer):
    ev = _Handler(indexer)
    obs = Observer()
    obs.schedule(ev, path, recursive=True)
    obs.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

from pathlib import Path
import os
import json

class QueueManager:
    """Handles all operations related to .qml queue files."""

    def __init__(self, queues_dir: str = "queues"):
        self.queues_dir = Path(queues_dir)
        self.queues_dir.mkdir(exist_ok=True)
        self.progress_dir = self.queues_dir / ".progress"
        self.progress_dir.mkdir(exist_ok=True)

    def _get_qml_path(self, queue_name: str) -> Path:
        """Ensures the queue name ends with .qml and returns its full path."""
        if not queue_name.endswith(".qml"):
            queue_name += ".qml"
        return self.queues_dir / queue_name

    def list_queues(self):
        """Returns a list of all .qml files in the queues directory."""
        return [f.name for f in self.queues_dir.glob("*.qml")]

    def create_queue(self, queue_name: str):
        """Creates a new, empty .qml file."""
        qml_path = self._get_qml_path(queue_name)
        if qml_path.exists():
            print(f"⚠️ Queue '{queue_name}' already exists.")
            return
        qml_path.touch()
        print(f"✅ Created new queue: {qml_path}")

    def add_to_queue(self, queue_name: str, config_path: str):
        """Appends a config file path to a .qml file."""
        qml_path = self._get_qml_path(queue_name)
        if not Path(config_path).exists():
            print(f"❌ Error: Config file not found at '{config_path}'")
            return
            
        with open(qml_path, 'a') as f:
            f.write(f"{os.path.abspath(config_path)}\n")
        print(f"✅ Added '{config_path}' to queue '{queue_name}'.")

    def get_queue_contents(self, queue_name: str) -> list[str]:
        """Reads and returns the list of config paths from a queue."""
        qml_path = self._get_qml_path(queue_name)
        if not qml_path.exists():
            return []
        with open(qml_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
            
    def delete_queue(self, queue_name: str):
        """Deletes a .qml file."""
        qml_path = self._get_qml_path(queue_name)
        if qml_path.exists():
            qml_path.unlink()
            # Also delete progress file
            progress_path = self._get_progress_path(queue_name)
            if progress_path.exists():
                progress_path.unlink()
            print(f"🗑️ Deleted queue '{queue_name}'.")
        else:
            print(f"❌ Queue '{queue_name}' not found.")
    
    def _get_progress_path(self, queue_name: str) -> Path:
        """Get the progress file path for a queue."""
        if not queue_name.endswith(".qml"):
            queue_name += ".qml"
        return self.progress_dir / f"{queue_name}.progress"
    
    def save_progress(self, queue_name: str, completed_index: int):
        """Save progress for a queue."""
        progress_path = self._get_progress_path(queue_name)
        progress_data = {
            'queue_name': queue_name,
            'completed_index': completed_index,
            'timestamp': os.path.getmtime(self._get_qml_path(queue_name))
        }
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f)
    
    def get_progress(self, queue_name: str) -> int:
        """Get the last completed index for a queue. Returns -1 if no progress."""
        progress_path = self._get_progress_path(queue_name)
        if not progress_path.exists():
            return -1
        
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
            
            # Check if queue file has been modified since progress was saved
            qml_path = self._get_qml_path(queue_name)
            if qml_path.exists():
                current_mtime = os.path.getmtime(qml_path)
                if current_mtime > data['timestamp']:
                    print(f"⚠️ Queue '{queue_name}' has been modified since last run. Starting from beginning.")
                    return -1
            
            return data['completed_index']
        except:
            return -1
    
    def clear_progress(self, queue_name: str):
        """Clear progress for a queue."""
        progress_path = self._get_progress_path(queue_name)
        if progress_path.exists():
            progress_path.unlink()
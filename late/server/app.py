from flask import Flask, render_template, request, redirect, url_for, flash
from waitress import serve
from late.engine.queue_manager import QueueManager
import os

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages
qm = QueueManager()

@app.route('/')
def index():
    queues = qm.list_queues()
    queue_data = {}
    for q_name in queues:
        queue_data[q_name] = qm.get_queue_contents(q_name)
    return render_template('index.html', queue_data=queue_data)

@app.route('/queue/create', methods=['POST'])
def create_queue():
    queue_name = request.form.get('queue_name')
    if queue_name:
        qm.create_queue(queue_name)
        flash(f"Queue '{queue_name}' created successfully!", "success")
    else:
        flash("Queue name cannot be empty.", "error")
    return redirect(url_for('index'))

@app.route('/queue/delete', methods=['POST'])
def delete_queue():
    queue_name = request.form.get('queue_name')
    qm.delete_queue(queue_name)
    flash(f"Queue '{queue_name}' deleted.", "info")
    return redirect(url_for('index'))

@app.route('/queue/add_job', methods=['POST'])
def add_job_to_queue():
    queue_name = request.form.get('queue_name')
    config_path = request.form.get('config_path')
    if not os.path.exists(config_path):
        flash(f"Error: Config file not found at '{config_path}'.", "error")
    else:
        qm.add_to_queue(queue_name, config_path)
        flash(f"Added '{config_path}' to '{queue_name}'.", "success")
    return redirect(url_for('index'))

def run_server(host='0.0.0.0', port=8080):
    # Use Waitress, a production-ready WSGI server
    serve(app, host=host, port=port)
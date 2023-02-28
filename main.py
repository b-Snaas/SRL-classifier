import subprocess

# Check if python3 is available, otherwise check for python
try:
    subprocess.check_output(['which', 'python3'])
    python_cmd = 'python3'
except subprocess.CalledProcessError:
    python_cmd = 'python'

# Run process.py first
subprocess.run([python_cmd, 'process.py'], check=True)

# Then run model.py
subprocess.run([python_cmd, 'model.py'], check=True)
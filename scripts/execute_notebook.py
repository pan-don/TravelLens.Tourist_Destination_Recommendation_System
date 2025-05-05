import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(path, as_version=4)
        
    ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
    ep.preprocess(nb,{'metadata': {'path': './'}})
    
    for _, cell in enumerate(nb.cells):
        print(f"run code: {cell.source}...")
        start_runtime = time.time()
        try:
            ep.preprocess_cell(cell,{'metadata': {'path': './'}}, _)
            end_runtime = time.time()
            print(f"completed (runtime: {end_runtime-start_runtime} s)")
        except Exception as e:
            print(e)
    
    total_runtime = time.time()
    print(f"total runtime notebook: {total_runtime-start_runtime:.4f} s")